defmodule LangChain.Utils.HarmonyRenderer do
  @moduledoc """
  Renders LangChain messages and tools into OpenAI Harmony response format.

  The Harmony format uses special tokens to structure conversations for
  llama.cpp completions API, enabling chat-like behavior from raw completions.
  """

  require Logger

  alias LangChain.Message
  alias LangChain.Function
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult

  @special_tokens %{
    start: "<|start|>",
    end: "<|end|>",
    message: "<|message|>",
    channel: "<|channel|>",
    constrain: "<|constrain|>",
    call: "<|call|>",
    return: "<|return|>"
  }

  @doc """
  Renders a complete Harmony-formatted prompt from messages and tools.

  Returns a string ready to be sent as the "prompt" field to llama.cpp /completions.

  According to OpenAI Harmony cookbook: once the model gives a final response,
  subsequent messages should not include the reasoning/tool calls from previous interactions.
  Only final responses should be included in the conversation history.
  """
  @spec render_conversation([Message.t()], [Function.t()], map()) :: String.t()
  def render_conversation(messages, tools \\ [], options \\ %{}) do
    # Default options
    defaults = %{
      reasoning_effort: :medium,
      knowledge_cutoff: "2024-06",
      current_date: Date.utc_today() |> Date.to_string(),
      identity: "You are ChatGPT, a large language model trained by OpenAI.",
      instructions: "You are a helpful AI assistant."
    }

    opts = Map.merge(defaults, options)

    # Build system message
    system = build_system_message(opts, tools)

    # Build developer message (includes tool definitions)
    developer = build_developer_message(opts, tools)

    # Filter messages to only include final responses (not intermediate reasoning/tool calls)
    show_reasoning = options[:show_reasoning] || false
    {filtered_messages, discard_stats} = filter_final_messages(messages, show_reasoning)

    # Convert filtered messages to Harmony format
    conversation_parts = [system, developer | Enum.map(filtered_messages, &message_to_harmony/1)]

    # Join all parts
    prompt = Enum.join(conversation_parts, "")

    # Log metrics
    context_tokens = estimate_tokens(prompt)

    Logger.debug(
      "🤖 Context metrics: #{context_tokens} tokens (#{discard_stats.discarded_messages} msgs discarded, #{discard_stats.discarded_tokens} tokens saved)"
    )

    prompt
  end

  @doc """
  Builds the system message with reasoning settings.
  """
  @spec build_system_message(map(), [Function.t()]) :: String.t()
  def build_system_message(opts, tools) do
    defaults = %{
      identity: "You are ChatGPT, a large language model trained by OpenAI.",
      reasoning_effort: :medium,
      knowledge_cutoff: "2024-06",
      current_date: Date.utc_today() |> Date.to_string()
    }

    opts = Map.merge(defaults, opts)

    reasoning = opts.reasoning_effort |> Atom.to_string() |> String.capitalize()

    system_content =
      """
      #{opts.identity}
      Knowledge cutoff: #{opts.knowledge_cutoff}
      Current date: #{opts.current_date}

      Reasoning: #{reasoning}

      # Valid channels: analysis, commentary, final. Channel must be included for every message.
      """ <> build_tools_section(tools)

    build_message("system", system_content)
  end

  @doc """
  Builds the developer message with instructions and tool definitions.
  """
  @spec build_developer_message(map(), [Function.t()]) :: String.t()
  def build_developer_message(opts, tools) do
    instructions = opts[:instructions] || "You are a helpful AI assistant."

    developer_content =
      """
      # Instructions

      #{instructions}
      """ <> build_functions_section(tools)

    build_message("developer", developer_content)
  end

  @doc """
  Builds the functions section for the developer message.
  """
  @spec build_functions_section([Function.t()]) :: String.t()
  def build_functions_section([]), do: ""

  def build_functions_section(tools) do
    tool_definitions =
      tools
      |> Enum.map_join("\n\n", &render_tool/1)
      |> String.trim_trailing()

    """

    # Tools

    ## functions

    namespace functions {

    #{tool_definitions}

    } // namespace functions
    """
  end

  @doc """
  Builds the tools section for the system message.
  """
  @spec build_tools_section([Function.t()]) :: String.t()
  def build_tools_section([]), do: ""

  def build_tools_section(_tools) do
    # For function tools, always use 'functions' as per Harmony format
    """
    Calls to these tools must go to the commentary channel: 'functions'.
    """
  end

  @doc """
  Converts a LangChain Message to Harmony format.
  """
  @spec message_to_harmony(Message.t()) :: String.t()
  def message_to_harmony(%Message{role: :system, content: content}) do
    text = extract_text_content(content)
    build_message("system", text)
  end

  def message_to_harmony(%Message{role: :user, content: content}) do
    text = extract_text_content(content)
    build_message("user", text)
  end

  def message_to_harmony(%Message{role: :assistant} = msg) do
    tool_call_segments =
      msg.tool_calls
      |> List.wrap()
      |> Enum.reject(&is_nil/1)
      |> Enum.map(&render_tool_call_segment/1)

    text_content =
      msg
      |> Map.get(:content)
      |> extract_text_content()
      |> String.trim()

    text_segments =
      if text_content == "" do
        []
      else
        channel = assistant_channel(msg)
        [build_message_with_channel("assistant", channel, text_content)]
      end

    (text_segments ++ tool_call_segments)
    |> Enum.join("")
  end

  def message_to_harmony(%Message{role: :tool} = msg) do
    channel = get_channel(msg) || "commentary"

    results =
      msg.tool_results
      |> List.wrap()
      |> Enum.reject(&is_nil/1)

    cond do
      results != [] ->
        results
        |> Enum.map_join("", fn %ToolResult{} = result ->
          tool_name =
            ensure_functions_namespace(result.name || msg.metadata[:tool_name] || msg.name)

          content = extract_text_content(result.content)
          header = "#{tool_name} to=assistant#{@special_tokens.channel}#{channel}"
          build_message(header, content)
        end)

      true ->
        tool_name = ensure_functions_namespace(get_tool_name(msg))
        content = extract_text_content(msg.content)
        header = "#{tool_name} to=assistant#{@special_tokens.channel}#{channel}"
        build_message(header, content)
    end
  end

  @doc """
  Extracts text content from a list of ContentParts.
  """
  @spec extract_text_content([any()]) :: String.t()
  def extract_text_content(content_parts) when is_list(content_parts) do
    content_parts
    |> Enum.filter(&match?(%{type: :text}, &1))
    |> Enum.map(& &1.content)
    |> Enum.join("")
  end

  def extract_text_content(content) when is_binary(content), do: content
  def extract_text_content(_), do: ""

  @doc """
  Gets the channel from a message's metadata.
  """
  @spec get_channel(Message.t()) :: String.t() | nil
  def get_channel(%Message{metadata: metadata}) when is_map(metadata), do: metadata[:channel]
  def get_channel(_), do: nil

  @doc """
  Gets the tool name from a tool message.
  """
  @spec get_tool_name(Message.t()) :: String.t()
  def get_tool_name(%Message{role: :tool, metadata: metadata} = msg) do
    cond do
      is_map(metadata) && metadata[:tool_name] -> metadata[:tool_name]
      msg.name -> msg.name
      true -> "tool"
    end
  end

  @doc """
  Builds a Harmony message with the given role and content.
  """
  @spec build_message(String.t(), String.t(), :end | :call | :return) :: String.t()
  def build_message(header, content, stop_token \\ :end) do
    stop =
      case stop_token do
        :call -> @special_tokens.call
        :return -> @special_tokens.return
        _ -> @special_tokens.end
      end

    "#{@special_tokens.start}#{header}#{@special_tokens.message}#{content}#{stop}"
  end

  @doc """
  Renders a single tool definition in TypeScript-like syntax.
  """
  @spec render_tool(Function.t()) :: String.t()
  def render_tool(%Function{name: name, description: description, parameters_schema: schema}) do
    comment =
      case String.trim(to_string(description || "")) do
        "" -> ""
        desc -> "// #{desc}\n"
      end

    type_def =
      if schema do
        build_type_definition(name, schema)
      else
        "type #{name} = () => any;"
      end

    comment <> type_def
  end

  @doc """
  Builds TypeScript-like type definition from JSON schema.
  """
  @spec build_type_definition(String.t(), map()) :: String.t()
  def build_type_definition(name, %{"type" => "object", "properties" => properties} = schema) do
    required = Map.get(schema, "required", [])
    params = build_parameters(properties, required)

    """
    type #{name} = (_: {
    #{params}
    }) => any;
    """
    |> String.trim_leading()
  end

  def build_type_definition(name, _schema) do
    "type #{name} = () => any;"
  end

  @doc """
  Builds parameter definitions from properties.
  """
  @spec build_parameters(map(), [String.t()]) :: String.t()
  def build_parameters(properties, _required) when properties == %{}, do: "    // no parameters\n"

  def build_parameters(properties, required) do
    properties
    |> Enum.sort_by(fn {field, _} -> field end)
    |> Enum.map_join("\n", fn {field_name, spec} ->
      required_mark = if field_name in required, do: "", else: "?"
      type = map_json_type_to_ts(spec)
      desc = String.trim(to_string(Map.get(spec, "description", "")))

      default_comment =
        case Map.get(spec, "default") do
          nil -> ""
          default -> " // default: #{inspect(default)}"
        end

      comment =
        case desc do
          "" -> ""
          _ -> "    // #{desc}\n"
        end

      "#{comment}    #{field_name}#{required_mark}: #{type},#{default_comment}"
    end)
  end

  @doc """
  Formats a tool result message in Harmony format for inclusion in conversation.
  """
  @spec format_tool_result(String.t(), String.t()) :: String.t()
  def format_tool_result(tool_name, result) do
    header = "functions.#{tool_name} to=assistant#{@special_tokens.channel}commentary"
    build_message(header, result)
  end

  @doc """
  Maps JSON schema types to TypeScript-like types.
  """
  @spec map_json_type_to_ts(map()) :: String.t()
  def map_json_type_to_ts(%{"enum" => enum}) when is_list(enum) do
    Enum.map_join(enum, " | ", fn
      value when is_binary(value) -> ~s("#{value}")
      value -> inspect(value)
    end)
  end

  def map_json_type_to_ts(%{"type" => "string"}), do: "string"
  def map_json_type_to_ts(%{"type" => "number"}), do: "number"
  def map_json_type_to_ts(%{"type" => "integer"}), do: "number"
  def map_json_type_to_ts(%{"type" => "boolean"}), do: "boolean"

  def map_json_type_to_ts(%{"type" => "array", "items" => items}),
    do: "#{map_json_type_to_ts(items)}[]"

  def map_json_type_to_ts(_), do: "any"

  defp assistant_channel(%Message{metadata: metadata}) when is_map(metadata) do
    metadata[:channel] || "final"
  end

  defp assistant_channel(_), do: "final"

  defp render_tool_call_segment(%ToolCall{} = call) do
    recipient =
      call.name
      |> Kernel.||("tool")
      |> ensure_functions_namespace()

    arguments =
      case call.arguments do
        args when is_binary(args) -> args
        args -> Jason.encode!(args)
      end

    header =
      "assistant#{@special_tokens.channel}commentary to=#{recipient} #{@special_tokens.constrain}json"

    build_message(header, arguments, :call)
  end

  defp build_message_with_channel(role, nil, content), do: build_message(role, content)

  defp build_message_with_channel(role, channel, content) do
    build_message("#{role}#{@special_tokens.channel}#{channel}", content)
  end

  defp ensure_functions_namespace(name) when is_binary(name) do
    if String.starts_with?(name, "functions."), do: name, else: "functions.#{name}"
  end

  defp ensure_functions_namespace(_), do: "functions.tool"

  @doc """
  Filters messages to drop assistant analysis-channel messages unless explicitly requested.
  """
  @spec filter_final_messages([Message.t()], boolean()) ::
          {[Message.t()], %{discarded_messages: integer(), discarded_tokens: integer()}}
  def filter_final_messages(messages, show_reasoning \\ false) do
    {filtered, discarded} =
      Enum.split_with(messages, fn message ->
        # Keep messages that are NOT assistant messages in "analysis" channel (unless show_reasoning is true)
        not (message.role == :assistant and get_channel(message) == "analysis" and
               not show_reasoning)
      end)

    discarded_count = length(discarded)

    discarded_tokens =
      Enum.reduce(discarded, 0, fn msg, acc ->
        acc + estimate_tokens(message_to_harmony(msg))
      end)

    {filtered, %{discarded_messages: discarded_count, discarded_tokens: discarded_tokens}}
  end

  @doc """
  Estimates token count for a given text string.

  Uses a simple approximation of ~4 characters per token.
  This is a rough estimate for monitoring purposes.
  """
  @spec estimate_tokens(String.t()) :: integer()
  def estimate_tokens(text) when is_binary(text) do
    # Rough approximation: ~4 characters per token
    # This works well for English text and code
    div(String.length(text), 4)
  end

  def estimate_tokens(_), do: 0
end
