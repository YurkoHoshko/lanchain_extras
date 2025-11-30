defmodule LangChain.ChatModels.ChatLlamaJinja do
  @moduledoc """
  Represents a chat model backend that uses llama.cpp's /completions endpoint
  with the Qwen 3 Jinja chat template response format for sampling tokens.

  This backend converts LangChain messages to Jinja-formatted prompts,
  sends them to llama.cpp's completions API, and parses the streamed responses
  back into LangChain-compatible messages and tool calls.
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.ChatModels.ChatModel
  alias LangChain.Message
  alias LangChain.Message.ToolResult
  alias LangChain.LangChainError

  @primary_key false
  embedded_schema do
    field(:endpoint, :string, default: "http://localhost:8000/v1/completions")
    field(:model, :string, default: "Qwen3-Coder-30B-A3B-Instruct")
    field(:temperature, :float, default: 0.7)
    field(:stream, :boolean, default: true)
    field(:receive_timeout, :integer, default: 60_000)
    field(:max_tokens, :integer, default: 16384)
    field(:seed, :integer)
    field(:api_key, :string)
    field(:verbose_api, :boolean, default: false)
    field(:callbacks, {:array, :any}, default: [])
    field(:json_response, :boolean, default: false)
    field(:json_schema, :map)
    field(:req_config, :map, default: %{})
    field(:reasoning_effort, Ecto.Enum, values: [:low, :medium, :high], default: :medium)
    field(:reasoning_mode, :boolean, default: false)
    # Prompt-specific fields
    field(:knowledge_cutoff, :string, default: "2024-06")
    field(:current_date, :string)
    # Additional sampling parameters
    field(:top_p, :float, default: 0.8)
    field(:top_k, :integer, default: 20)
    field(:repetition_penalty, :float, default: 1.05)
  end

  @type t :: %ChatLlamaJinja{}

  @create_fields [
    :endpoint,
    :model,
    :temperature,
    :stream,
    :receive_timeout,
    :max_tokens,
    :seed,
    :api_key,
    :verbose_api,
    :callbacks,
    :json_response,
    :json_schema,
    :req_config,
    :reasoning_effort,
    :reasoning_mode,
    :knowledge_cutoff,
    :current_date,
    :top_p,
    :top_k,
    :repetition_penalty
  ]
  @required_fields []

  @doc """
  Setup a ChatLlamaJinja client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %ChatLlamaJinja{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatLlamaJinja client configuration and return it or raise an error if invalid.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, chain} ->
        chain

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  defp common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_number(:temperature, greater_than_or_equal_to: 0, less_than_or_equal_to: 2)
    |> validate_number(:max_tokens, greater_than: 0)
    |> validate_number(:receive_timeout, greater_than: 0)
    |> validate_number(:repetition_penalty, greater_than: 0)
  end

  @doc """
  Calls the llama.cpp completions API with the Qwen Jinja-formatted prompt.

  Converts LangChain messages to the template format, sends to /completions endpoint,
  and parses the streamed response back into LangChain structures.
  """
  @spec call(t(), ChatModel.prompt(), ChatModel.tools()) :: {:ok, [Message.t()]} | {:error, any()}
  def call(%ChatLlamaJinja{} = model, prompt, tools \\ []) do
    # Convert messages to Jinja prompt
    jinja_prompt =
      LangChain.Utils.JinjaRenderer.render_conversation(prompt, tools, %{
        reasoning_effort: model.reasoning_effort,
        knowledge_cutoff: model.knowledge_cutoff,
        current_date: model.current_date || Date.utc_today() |> Date.to_string()
      })

    # Prepare request payload
    payload =
      %{
        prompt: jinja_prompt,
        stream: model.stream,
        max_tokens: model.max_tokens,
        temperature: model.temperature,
        seed: model.seed,
        top_p: model.top_p,
        top_k: model.top_k,
        repetition_penalty: model.repetition_penalty
      }
      |> Map.reject(fn {_k, v} -> is_nil(v) end)

    # Make HTTP request
    if model.stream do
      # Handle streaming response
      case make_streaming_request(model, payload) do
        {:ok, response_body, _stream_text} ->
          parse_response_and_convert(model, response_body, prompt)

        {:error, reason} ->
          {:error, reason}
      end
    else
      # Handle non-streaming response
      case make_request(model, payload) do
        {:ok, response_body} ->
          parse_response_and_convert(model, response_body, prompt)

        {:error, reason} ->
          {:error, reason}
      end
    end
  end

  @doc """
  Makes the HTTP request to llama.cpp completions endpoint.
  """
  @spec make_request(t(), map()) :: {:ok, String.t()} | {:error, any()}
  def make_request(%ChatLlamaJinja{} = model, payload) do
    url = "#{model.endpoint}"

    headers = [{"Content-Type", "application/json"}]

    headers =
      if model.api_key,
        do: [{"Authorization", "Bearer #{model.api_key}"} | headers],
        else: headers

    Logger.debug("Making HTTP request to: #{url}")
    Logger.debug("Payload: #{inspect(payload)}")
    Logger.debug("Headers: #{inspect(headers)}")

    case Req.post(url,
           json: payload,
           headers: headers,
           receive_timeout: model.receive_timeout,
           retry: false
         ) do
      {:ok, %Req.Response{status: 200, body: body}} ->
        Logger.debug("HTTP 200 response body: #{inspect(body)}")
        {:ok, body}

      {:ok, %Req.Response{status: status, body: body}} ->
        Logger.error("HTTP #{status} error: #{inspect(body)}")
        {:error, "HTTP #{status}: #{inspect(body)}"}

      {:error, error} ->
        Logger.error("HTTP request error: #{inspect(error)}")
        {:error, error}
    end
  end

  @doc """
  Makes a streaming HTTP request to llama.cpp completions endpoint.
  """
  @spec make_streaming_request(t(), map()) :: {:ok, String.t()} | {:error, any()}
  def make_streaming_request(%ChatLlamaJinja{} = model, payload) do
    url = "#{model.endpoint}"

    headers = [{"Content-Type", "application/json"}]

    headers =
      if model.api_key do
        [{"Authorization", "Bearer #{model.api_key}"} | headers]
      else
        headers
      end

    Logger.debug("Making streaming HTTP request to: #{url}")
    Logger.debug("Payload: #{inspect(payload)}")
    Logger.debug("Headers: #{inspect(headers)}")

    stream_fun = fn
      {:data, chunk}, {req, %Req.Response{} = resp} ->
        updated_resp =
          case parse_sse_data(chunk) do
            {:text, text} ->
              Logger.debug("Received jinja chunk: #{inspect(text)}")

              Req.Response.update_private(
                resp,
                :jinja_stream,
                %{buffer: "", finish: nil},
                fn state ->
                  %{state | buffer: state.buffer <> text}
                end
              )

            {:done, finish_reason} ->
              Logger.debug(
                "Jinja stream signaled completion with finish_reason=#{inspect(finish_reason)}"
              )

              Req.Response.update_private(
                resp,
                :jinja_stream,
                %{buffer: "", finish: nil},
                fn state ->
                  %{state | finish: finish_reason}
                end
              )

            :skip ->
              resp
          end

        {:cont, {req, updated_resp}}

      {:done, _}, acc ->
        {:cont, acc}
    end

    case Req.post(url,
           json: payload,
           headers: headers,
           receive_timeout: model.receive_timeout,
           retry: false,
           into: stream_fun
         ) do
      {:ok, %Req.Response{status: 200} = response} ->
        %{buffer: buffer, finish: finish_reason} =
          Req.Response.get_private(response, :jinja_stream, %{buffer: "", finish: nil})

        finalized_buffer = finalize_stream_buffer(buffer)
        Logger.debug("Final jinja buffer length=#{String.length(finalized_buffer)}")

        body = %{"choices" => [%{"text" => finalized_buffer, "finish_reason" => finish_reason}]}

        jinja_text =
          body
          |> Map.fetch!("choices")
          |> List.first()
          |> Map.get("text", "")

        {:ok, Jason.encode!(body), jinja_text}

      {:ok, %Req.Response{status: status, body: body}} ->
        {:error, "HTTP #{status}: #{inspect(body)}"}

      {:error, error} ->
        {:error, error}
    end
  end

  @doc """
  Parses SSE data chunk to extract text.
  """
  @spec parse_sse_data(String.t()) :: {:text, String.t()} | {:done, any()} | :skip
  def parse_sse_data("data: " <> json_str) do
    json_str = String.trim_trailing(json_str, "\n")

    case Jason.decode(json_str) do
      {:ok, %{"choices" => [%{"text" => _text, "finish_reason" => finish}]}}
      when not is_nil(finish) ->
        {:done, finish}

      {:ok, %{"choices" => [%{"text" => text}]}} ->
        {:text, text}

      _ ->
        :skip
    end
  end

  def parse_sse_data(_), do: :skip

  @doc """
  Parses the llama.cpp response and converts Jinja messages to LangChain messages.
  """
  @spec parse_response_and_convert(t(), String.t() | map(), ChatModel.prompt()) ::
          {:ok, [Message.t()]} | {:error, any()}
  def parse_response_and_convert(%ChatLlamaJinja{} = _model, response_body, _prompt) do
    Logger.debug("Parsing response body: #{inspect(response_body)}")

    # Handle both JSON string and already-decoded map
    response_map =
      case response_body do
        %{} = map ->
          map

        json_string when is_binary(json_string) ->
          Logger.debug("Decoding JSON string: #{json_string}")

          case Jason.decode(json_string) do
            {:ok, decoded} ->
              decoded

            {:error, error} ->
              Logger.error("JSON decode error: #{inspect(error)}")
              {:error, "JSON decode error: #{inspect(error)}"}
          end

        _ ->
          Logger.error("Invalid response format: #{inspect(response_body)}")
          {:error, "Invalid response format"}
      end

    case response_map do
      {:error, reason} ->
        {:error, reason}

      %{"choices" => [%{"text" => jinja_content} | _]} ->
        Logger.debug("Extracted jinja_content from choices: #{inspect(jinja_content)}")
        {:ok, convert_jinja_content(jinja_content)}

      %{"content" => jinja_content} ->
        Logger.debug("Using content fallback: #{inspect(jinja_content)}")
        {:ok, convert_jinja_content(jinja_content)}

      response ->
        Logger.error("Unexpected response format: #{inspect(response)}")
        {:error, "Unexpected response format: #{inspect(response)}"}
    end
  end

  defp convert_jinja_content(content) do
    jinja_messages = LangChain.Utils.JinjaParser.parse_response(content)
    Logger.debug("Parsed jinja messages: #{inspect(jinja_messages)}")

    langchain_messages =
      jinja_messages
      |> Enum.map(&jinja_to_langchain_message/1)
      |> Enum.reject(&is_nil/1)

    Logger.debug("Converted to LangChain messages: #{inspect(langchain_messages)}")

    combined_messages = combine_assistant_messages(langchain_messages)
    Logger.debug("Combined messages: #{inspect(combined_messages)}")
    combined_messages
  end

  @doc """
  Filters out intermediate analysis messages while preserving response order.
  """
  @spec combine_assistant_messages([Message.t()]) :: [Message.t()]
  def combine_assistant_messages(messages) do
    filtered =
      Enum.reject(messages, fn
        %Message{metadata: %{channel: "analysis"}} -> true
        _ -> false
      end)

    if filtered == [] do
      messages
    else
      filtered
    end
  end

  @doc """
  Converts a parsed Jinja message map to a LangChain Message struct.
  """
  @spec jinja_to_langchain_message(map()) :: Message.t() | nil
  def jinja_to_langchain_message(nil), do: nil

  def jinja_to_langchain_message(%{role: "assistant", type: :tool_call, tool_calls: tool_calls}) do
    calls =
      tool_calls
      |> List.wrap()
      |> Enum.reject(&is_nil/1)
      |> Enum.map(&build_tool_call/1)
      |> Enum.reject(&is_nil/1)

    if calls == [] do
      nil
    else
      Message.new_assistant!(%{content: [], tool_calls: calls})
    end
  end

  def jinja_to_langchain_message(%{role: "assistant"} = msg) do
    content = Map.get(msg, :content) || Map.get(msg, "content") || ""
    Message.new_assistant!(String.trim(content))
  end

  def jinja_to_langchain_message(%{role: "user"} = msg) do
    content = Map.get(msg, :content) || Map.get(msg, "content") || ""
    Message.new_user!(content)
  end

  def jinja_to_langchain_message(%{role: "tool"} = msg) do
    content = Map.get(msg, :content) || Map.get(msg, "content") || ""
    tool_name = Map.get(msg, :tool_name) || Map.get(msg, "tool_name") || "tool"

    tool_result =
      ToolResult.new!(%{
        tool_call_id:
          Map.get(msg, :tool_call_id) || "call_result_#{System.unique_integer([:positive])}",
        name: tool_name,
        content: content
      })

    Message.new_tool_result!(%{
      content: nil,
      tool_results: [tool_result],
      metadata: %{tool_name: tool_name}
    })
  end

  def jinja_to_langchain_message(message) when is_map(message) do
    content = Map.get(message, :content) || Map.get(message, "content") || ""
    Message.new_assistant!(content)
  end

  def jinja_to_langchain_message(_), do: nil

  defp build_tool_call(%{name: name, arguments: arguments}) do
    LangChain.Message.ToolCall.new!(%{
      call_id: "call_#{System.unique_integer([:positive])}",
      name: sanitize_tool_name(name),
      arguments: normalize_arguments(arguments)
    })
  rescue
    _ -> nil
  end

  defp sanitize_tool_name(nil), do: "tool"
  defp sanitize_tool_name(name) when is_binary(name), do: String.trim(name)
  defp sanitize_tool_name(name), do: to_string(name)

  defp normalize_arguments(%{} = args) do
    args
    |> Enum.map(fn {key, value} -> {to_string(key), value} end)
    |> Enum.into(%{})
  end

  defp normalize_arguments(args) when is_binary(args) do
    case Jason.decode(args) do
      {:ok, decoded} -> decoded
      {:error, _} -> %{"value" => String.trim(args)}
    end
  end

  defp normalize_arguments(_), do: %{}

  defp finalize_stream_buffer(buffer), do: buffer
end
