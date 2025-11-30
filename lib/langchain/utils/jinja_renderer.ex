defmodule LangChain.Utils.JinjaRenderer do
  @moduledoc """
  Renders LangChain messages and tools into the Qwen 3 Jinja chat template.
  """

  require Logger

  alias LangChain.Message
  alias LangChain.Function
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.Utils.HarmonyRenderer, as: ContentHelper

  @assistant_prompt "<|im_start|>assistant\n"
  @default_system_message "You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks."

  @tool_call_instructions """
  \n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>
  """

  @doc """
  Build a full prompt suitable for llama.cpp /completions when running with the
  Qwen 3 Jinja template configured on the server.
  """
  @spec render_conversation([Message.t()], [Function.t()], map()) :: String.t()
  def render_conversation(messages, tools \\ [], opts \\ %{}) do
    add_generation_prompt = Map.get(opts, :add_generation_prompt, true)

    normalized_tools = normalize_tools(tools)
    {system_message, loop_messages} = extract_system_message(messages, opts)

    prompt =
      render_system_block(system_message, normalized_tools) <>
        render_loop_messages(loop_messages)

    if add_generation_prompt do
      prompt <> @assistant_prompt
    else
      prompt
    end
  end

  defp extract_system_message([%Message{role: :system} = system | rest], _opts) do
    {ContentHelper.extract_text_content(system.content), rest}
  end

  defp extract_system_message(messages, opts) do
    system_message = Map.get(opts, :system_message)
    {system_message, messages}
  end

  defp render_system_block(nil, []), do: ""

  defp render_system_block(system_message, tools) do
    initial_block =
      cond do
        present?(system_message) -> "<|im_start|>system\n" <> system_message
        tools != [] -> "<|im_start|>system\n" <> @default_system_message
        true -> ""
      end

    block =
      if tools != [] and initial_block != "" do
        initial_block <> build_tools_section(tools)
      else
        initial_block
      end

    if block == "" do
      ""
    else
      block <> "<|im_end|>\n"
    end
  end

  defp render_loop_messages(messages) do
    {parts, pending_tools} =
      Enum.reduce(messages, {[], []}, fn
        %Message{role: :tool} = msg, {processed, buffer} ->
          {processed, buffer ++ [msg]}

        %Message{} = msg, {processed, buffer} ->
          updated = flush_tool_buffer(processed, buffer)
          {updated ++ [render_message(msg)], []}
      end)

    final_parts = flush_tool_buffer(parts, pending_tools)

    final_parts
    |> Enum.reject(&(&1 == ""))
    |> Enum.join("")
  end

  defp flush_tool_buffer(parts, []), do: parts

  defp flush_tool_buffer(parts, buffer) do
    parts ++ [render_tool_block(buffer)]
  end

  defp render_tool_block(tool_messages) do
    body =
      tool_messages
      |> Enum.map_join("", fn msg ->
        content = tool_message_content(msg)
        "\n<tool_response>\n" <> content <> "\n</tool_response>\n"
      end)

    "<|im_start|>user" <> body <> "<|im_end|>\n"
  end

  defp tool_message_content(%Message{tool_results: results} = msg)
       when is_list(results) and results != [] do
    results
    |> Enum.map_join("\n", fn %ToolResult{} = result ->
      tool_name = result.name || msg.metadata[:tool_name] || msg.name
      content = ContentHelper.extract_text_content(result.content)

      if present?(tool_name) do
        "<function=#{tool_name}>\n" <> String.trim(content) <> "\n</function>"
      else
        String.trim(content)
      end
    end)
  end

  defp tool_message_content(%Message{} = msg) do
    msg.content
    |> ContentHelper.extract_text_content()
    |> String.trim()
  end

  defp render_message(%Message{role: :assistant, tool_calls: tool_calls} = msg)
       when is_list(tool_calls) and tool_calls != [] do
    text =
      msg.content
      |> ContentHelper.extract_text_content()
      |> String.trim()

    header = "<|im_start|>assistant"

    content_segment =
      if text == "" do
        ""
      else
        "\n" <> text <> "\n"
      end

    calls =
      tool_calls
      |> Enum.reject(&is_nil/1)
      |> Enum.map_join("", &render_tool_call/1)

    header <> content_segment <> calls <> "<|im_end|>\n"
  end

  defp render_message(%Message{role: :assistant} = msg) do
    render_standard_message("assistant", msg)
  end

  defp render_message(%Message{role: :user} = msg) do
    render_standard_message("user", msg)
  end

  defp render_message(%Message{role: :system} = msg) do
    render_standard_message("system", msg)
  end

  defp render_message(_), do: ""

  defp render_standard_message(role, %Message{} = msg) do
    content =
      msg.content
      |> ContentHelper.extract_text_content()
      |> String.trim()

    "<|im_start|>" <> role <> "\n" <> content <> "\n<|im_end|>\n"
  end

  defp render_tool_call(%ToolCall{} = call) do
    name = call.name || "tool"

    params =
      call.arguments
      |> normalize_arguments()
      |> Enum.map_join("", fn {key, value} ->
        formatted_value = format_argument_value(value)
        "<parameter=#{key}>\n" <> formatted_value <> "\n</parameter>\n"
      end)

    "\n<tool_call>\n<function=" <> name <> ">\n" <> params <> "</function>\n</tool_call>\n"
  end

  defp normalize_arguments(nil), do: []

  defp normalize_arguments(map) when is_map(map) do
    map
    |> Enum.map(fn {key, value} -> {to_string(key), value} end)
    |> Enum.sort_by(&elem(&1, 0))
  end

  defp normalize_arguments(list) when is_list(list) do
    list
    |> Enum.with_index()
    |> Enum.map(fn {value, idx} -> {"arg_#{idx}", value} end)
  end

  defp normalize_arguments(binary) when is_binary(binary) do
    case Jason.decode(binary) do
      {:ok, %{} = decoded} -> Enum.sort_by(decoded, fn {key, _} -> to_string(key) end)
      _ -> [{"value", String.trim(binary)}]
    end
  end

  defp normalize_arguments(other), do: [{"value", to_string(other)}]

  defp format_argument_value(value) when is_map(value) or is_list(value) do
    Jason.encode!(value)
  end

  defp format_argument_value(value) when is_binary(value), do: String.trim(value)
  defp format_argument_value(value), do: to_string(value)

  defp build_tools_section([]), do: ""

  defp build_tools_section(tools) do
    definitions = Enum.map_join(tools, "", &render_tool_definition/1)

    "\n\nYou have access to the following functions:\n\n<tools>" <>
      definitions <>
      "\n</tools>" <>
      @tool_call_instructions
  end

  defp render_tool_definition(%{"name" => name} = tool) when not is_nil(name) do
    description = tool |> Map.get("description", "") |> String.trim()
    parameters = tool |> Map.get("parameters", %{}) |> stringify_keys()
    properties = Map.get(parameters, "properties", %{}) |> stringify_keys()

    rendered_parameters =
      properties
      |> Enum.map_join("", &render_parameter/1)

    required_params = render_item_list(Map.get(parameters, "required"), "required")
    return_block = render_return_block(Map.get(tool, "return"))

    "\n<function>\n<name>" <>
      name <>
      "</name>" <>
      "\n<description>" <>
      description <>
      "</description>" <>
      "\n<parameters>" <>
      rendered_parameters <>
      required_params <>
      "\n</parameters>" <>
      return_block <>
      "\n</function>"
  end

  defp render_tool_definition(_), do: ""

  defp render_parameter({param_name, param_fields}) do
    fields = stringify_keys(param_fields || %{})
    type_block = optional_block("type", Map.get(fields, "type"))
    description_block = optional_block("description", Map.get(fields, "description"), trim?: true)
    enum_block = render_item_list(Map.get(fields, "enum"), "enum")

    handled = MapSet.new(["type", "description", "enum", "required"])

    extras =
      fields
      |> Enum.reject(fn {key, _} -> MapSet.member?(handled, key) end)
      |> Enum.map_join("", fn {key, value} ->
        normalized_key = key |> String.replace(~r/[-\s$]/, "_")

        rendered_value =
          if is_map(value) or is_list(value) do
            Jason.encode!(value)
          else
            to_string(value)
          end

        "\n<" <> normalized_key <> ">" <> rendered_value <> "</" <> normalized_key <> ">"
      end)

    required_block = render_item_list(Map.get(fields, "required"), "required")

    "\n<parameter>" <>
      "\n<name>" <>
      param_name <>
      "</name>" <>
      type_block <>
      description_block <>
      enum_block <>
      extras <>
      required_block <>
      "\n</parameter>"
  end

  defp optional_block(tag, value, opts \\ [])
  defp optional_block(_tag, nil, _opts), do: ""
  defp optional_block(_tag, "", _opts), do: ""

  defp optional_block(tag, value, opts) do
    content =
      if Keyword.get(opts, :trim?, false),
        do: String.trim(to_string(value)),
        else: to_string(value)

    "\n<" <> tag <> ">" <> content <> "</" <> tag <> ">"
  end

  defp render_return_block(nil), do: ""

  defp render_return_block(value) do
    rendered =
      cond do
        is_map(value) or is_list(value) -> Jason.encode!(value)
        true -> to_string(value)
      end

    "\n<return>" <> rendered <> "</return>"
  end

  defp render_item_list(value, _tag_name) when is_nil(value), do: ""
  defp render_item_list(value, _tag_name) when value == [], do: ""

  defp render_item_list(value, tag_name) do
    list = List.wrap(value)

    formatted_items =
      list
      |> Enum.map_join(", ", fn item ->
        if is_binary(item), do: "`" <> item <> "`", else: to_string(item)
      end)

    "\n<" <> tag_name <> ">[" <> formatted_items <> "]</" <> tag_name <> ">"
  end

  defp normalize_tools(tools) do
    tools
    |> Enum.map(&normalize_tool/1)
    |> Enum.reject(&is_nil/1)
  end

  defp normalize_tool(%Function{} = function) do
    %{
      "name" => function.name,
      "description" => function.description || "",
      "parameters" => function.parameters_schema || %{}
    }
  end

  defp normalize_tool(%{function: %{} = fn_info}), do: normalize_tool(fn_info)
  defp normalize_tool(%{"function" => %{} = fn_info}), do: normalize_tool(fn_info)

  defp normalize_tool(%{} = map) do
    cond do
      Map.has_key?(map, "name") -> stringify_keys(map)
      Map.has_key?(map, :name) -> stringify_keys(map)
      true -> nil
    end
  end

  defp normalize_tool(_), do: nil

  defp stringify_keys(map) when is_map(map) do
    map
    |> Enum.reduce(%{}, fn {key, value}, acc ->
      new_key =
        cond do
          is_binary(key) -> key
          is_atom(key) -> Atom.to_string(key)
          true -> to_string(key)
        end

      Map.put(acc, new_key, value)
    end)
  end

  defp stringify_keys(value), do: value

  defp present?(nil), do: false
  defp present?(value) when is_binary(value), do: String.trim(value) != ""
  defp present?(_), do: true
end
