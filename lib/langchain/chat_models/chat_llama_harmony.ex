defmodule LangChain.ChatModels.ChatLlamaHarmony do
  @moduledoc """
  Represents a chat model backend that uses llama.cpp's /completions endpoint
  with OpenAI Harmony response format for sampling tokens.

  This backend converts LangChain messages to Harmony-formatted prompts,
  sends them to llama.cpp's completions API, and parses the streamed responses
  back into LangChain-compatible messages and tool calls.
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.ChatModels.ChatModel
  alias LangChain.Message
  alias LangChain.LangChainError

  @call_token "<|call|>"
  @return_token "<|return|>"

  @primary_key false
  embedded_schema do
    field(:endpoint, :string, default: "http://localhost:8000/v1/completions")
    field(:model, :string, default: "llama-3.1-8b")
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
    # Harmony-specific fields
    field(:knowledge_cutoff, :string, default: "2024-06")
    field(:current_date, :string)
    # Additional sampling parameters
    field(:top_p, :float)
    field(:top_k, :integer)
  end

  @type t :: %ChatLlamaHarmony{}

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
    :top_k
  ]
  @required_fields []

  @doc """
  Setup a ChatLlamaHarmony client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %ChatLlamaHarmony{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatLlamaHarmony client configuration and return it or raise an error if invalid.
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
  end

  @doc """
  Calls the llama.cpp completions API with Harmony-formatted prompt.

  Converts LangChain messages to Harmony format, sends to /completions endpoint,
  and parses the streamed response back into LangChain structures.
  """
  @spec call(t(), ChatModel.prompt(), ChatModel.tools()) :: {:ok, [Message.t()]} | {:error, any()}
  def call(%ChatLlamaHarmony{} = model, prompt, tools \\ []) do
    # Convert messages to Harmony prompt
    harmony_prompt =
      LangChain.Utils.HarmonyRenderer.render_conversation(prompt, tools, %{
        reasoning_effort: model.reasoning_effort,
        knowledge_cutoff: model.knowledge_cutoff,
        current_date: model.current_date || Date.utc_today() |> Date.to_string()
      })

    # Prepare request payload
    payload =
      %{
        # llama-swap raw completions route still accepts model for routing
        model: model.model,
        prompt: harmony_prompt,
        stream: model.stream,
        # llama.cpp /completion expects n_predict instead of OpenAI's max_tokens
        n_predict: model.max_tokens,
        temperature: model.temperature,
        seed: model.seed,
        top_p: model.top_p,
        top_k: model.top_k
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
  def make_request(%ChatLlamaHarmony{} = model, payload) do
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
  def make_streaming_request(%ChatLlamaHarmony{} = model, payload) do
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
              Logger.debug("Received harmony chunk: #{inspect(text)}")

              Req.Response.update_private(
                resp,
                :harmony_stream,
                %{buffer: "", finish: nil},
                fn state ->
                  %{state | buffer: state.buffer <> text}
                end
              )

            {:done, finish_reason} ->
              Logger.debug(
                "Harmony stream signaled completion with finish_reason=#{inspect(finish_reason)}"
              )

              Req.Response.update_private(
                resp,
                :harmony_stream,
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
          Req.Response.get_private(response, :harmony_stream, %{buffer: "", finish: nil})

        finalized_buffer = finalize_stream_buffer(buffer)
        Logger.debug("Final harmony buffer length=#{String.length(finalized_buffer)}")

        body = %{"choices" => [%{"text" => finalized_buffer, "finish_reason" => finish_reason}]}

        harmony_text =
          body
          |> Map.fetch!("choices")
          |> List.first()
          |> Map.get("text", "")

        {:ok, Jason.encode!(body), harmony_text}

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

      # llama.cpp /completion streaming shape
      {:ok, %{"content" => _text, "stop" => true}} ->
        {:done, :stop}

      {:ok, %{"content" => text}} when is_binary(text) ->
        {:text, text}

      {:ok, %{"token" => %{"text" => text}, "stop" => stop}}
      when is_binary(text) and stop in [false, nil] ->
        {:text, text}

      {:ok, %{"token" => %{"text" => _text}, "stop" => true}} ->
        {:done, :stop}

      _ ->
        :skip
    end
  end

  def parse_sse_data(_), do: :skip

  defp finalize_stream_buffer(buffer) do
    cond do
      buffer == "" -> buffer
      String.ends_with?(buffer, @call_token) -> buffer
      String.ends_with?(buffer, @return_token) -> buffer
      true -> buffer
    end
  end

  @doc """
  Parses the llama.cpp response and converts Harmony messages to LangChain messages.
  """
  @spec parse_response_and_convert(t(), String.t() | map(), ChatModel.prompt()) ::
          {:ok, [Message.t()]} | {:error, any()}
  def parse_response_and_convert(%ChatLlamaHarmony{} = _model, response_body, _prompt) do
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

      %{"choices" => [%{"text" => harmony_content} | _]} ->
        Logger.debug("Extracted harmony_content from choices: #{inspect(harmony_content)}")
        # Parse the Harmony content from OpenAI completions format
        harmony_messages = LangChain.Utils.HarmonyParser.parse_response(harmony_content)
        Logger.debug("Parsed harmony messages: #{inspect(harmony_messages)}")

        # Convert to LangChain messages
        langchain_messages = Enum.map(harmony_messages, &harmony_to_langchain_message/1)
        Logger.debug("Converted to LangChain messages: #{inspect(langchain_messages)}")

        # Combine consecutive assistant messages
        combined_messages = combine_assistant_messages(langchain_messages)
        Logger.debug("Combined messages: #{inspect(combined_messages)}")

        {:ok, combined_messages}

      %{"content" => harmony_content} ->
        Logger.debug("Using content fallback: #{inspect(harmony_content)}")
        # Fallback for direct content format
        harmony_messages = LangChain.Utils.HarmonyParser.parse_response(harmony_content)
        Logger.debug("Parsed harmony messages: #{inspect(harmony_messages)}")

        # Convert to LangChain messages
        langchain_messages = Enum.map(harmony_messages, &harmony_to_langchain_message/1)
        Logger.debug("Converted to LangChain messages: #{inspect(langchain_messages)}")

        # Combine consecutive assistant messages
        combined_messages = combine_assistant_messages(langchain_messages)
        Logger.debug("Combined messages: #{inspect(combined_messages)}")

        {:ok, combined_messages}

      response ->
        Logger.error("Unexpected response format: #{inspect(response)}")
        {:error, "Unexpected response format: #{inspect(response)}"}
    end
  end

  @doc """
  Filters out intermediate analysis messages while preserving response order.
  """
  @spec combine_assistant_messages([Message.t()]) :: [Message.t()]
  def combine_assistant_messages(messages) do
    # Preserve messages exactly as returned by the model; callers can decide how to handle channels.
    messages
  end

  @doc """
  Converts a parsed Harmony message to a LangChain Message struct.
  """
  @spec harmony_to_langchain_message(map()) :: Message.t()
  def harmony_to_langchain_message(harmony_msg) do
    role =
      case harmony_msg[:role] do
        "assistant" -> :assistant
        "tool" -> :tool
        _ -> :assistant
      end

    content = harmony_msg[:content] || ""
    msg_type = harmony_msg[:type] || :message

    # Create message based on type
    message =
      case {role, msg_type} do
        {:assistant, :tool_call} ->
          # Tool call from assistant
          # Parse JSON arguments - find JSON in content if content_type is json
          arguments =
            if harmony_msg[:content_type] == "json" and is_binary(content) do
              # Find the JSON object in the content
              case Regex.run(~r/\{.*\}/s, content) do
                [json_match] ->
                  case Jason.decode(json_match) do
                    {:ok, parsed} -> parsed
                    {:error, _} -> %{}
                  end

                _ ->
                  %{}
              end
            else
              case Jason.decode(content) do
                {:ok, parsed} -> parsed
                {:error, _} -> %{}
              end
            end

          # Create tool call
          name =
            case harmony_msg[:recipient] do
              nil ->
                "unknown"

              recipient ->
                recipient |> String.split(" ") |> List.first() |> String.split(".") |> List.last()
            end

          tool_call =
            LangChain.Message.ToolCall.new!(%{
              call_id: "call_#{:erlang.unique_integer([:positive])}",
              name: name,
              arguments: arguments
            })

          # Create assistant message with tool calls
          Message.new_assistant!(%{content: [], tool_calls: [tool_call]})

        {:tool, _} ->
          # Tool result - should be handled by the renderer, not parsed back
          # This represents tool outputs that come back from function calls
          tool_name = tool_name_from_recipient(harmony_msg[:recipient])

          Message.new_tool_result!(%{
            content: content,
            tool_call_id: nil,
            tool_name: tool_name
          })

        {:assistant, _} ->
          # Regular assistant message
          Message.new_assistant!(content)

        _ ->
          Message.new_assistant!(content)
      end

    # Add metadata
    metadata = %{}

    metadata =
      if harmony_msg[:channel],
        do: Map.put(metadata, :channel, harmony_msg[:channel]),
        else: metadata

    metadata =
      if harmony_msg[:recipient],
        do: Map.put(metadata, :recipient, harmony_msg[:recipient]),
        else: metadata

    metadata =
      if harmony_msg[:content_type],
        do: Map.put(metadata, :content_type, harmony_msg[:content_type]),
        else: metadata

    metadata =
      if harmony_msg[:recipient] do
        Map.put(metadata, :tool_name, tool_name_from_recipient(harmony_msg[:recipient]))
      else
        metadata
      end

    %{message | metadata: metadata}
  end

  defp tool_name_from_recipient(nil), do: "unknown"

  defp tool_name_from_recipient(recipient) do
    recipient
    |> String.split(" ")
    |> List.first()
    |> String.split(".")
    |> List.last()
  end
end
