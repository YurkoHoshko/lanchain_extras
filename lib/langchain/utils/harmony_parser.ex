defmodule LangChain.Utils.HarmonyParser do
  @moduledoc """
  Parses OpenAI Harmony response format from streamed tokens.

  1. Use STREAMING REQ; 2. leverage known tokens to parse streamed tokens.

  Handles the special tokens and structure defined in the Harmony specification
  to extract messages, tool calls, and channel information from llama.cpp responses.
  """

  require Logger

  @header_regex ~r/^(?<role>[^\s<]+)(?:<\|channel\|>(?<channel>[^\s<]+))?(?:\s+to=(?<recipient>[^\s<]+))?(?:\s*<\|constrain\|>(?<content_type>[^\s<]+))?/

  defstruct [
    :buffer,
    :messages,
    :in_message,
    :parsing_header,
    :current_role,
    :current_channel,
    :current_content,
    :current_recipient,
    :current_content_type
  ]

  @doc """
  Creates a new streaming parser instance with initial state.
  """
  @spec new() :: %__MODULE__{}
  def new() do
    %__MODULE__{
      buffer: "",
      messages: [],
      in_message: false,
      parsing_header: false,
      current_role: nil,
      current_channel: nil,
      current_content: nil,
      current_recipient: nil,
      current_content_type: nil
    }
  end

  @doc """
  Parses a complete Harmony response string into messages.
  Useful for testing or non-streaming scenarios.
  """
  @spec parse_response(String.t()) :: [map()]
  def parse_response(response) do
    Logger.debug("HarmonyParser.parse_response input: #{inspect(response)}")

    # Split by <|start|> and parse each message segment
    segments = String.split(response, "<|start|>", trim: true)
    Logger.debug("Segments after split: #{inspect(segments)}")

    messages =
      segments
      |> Enum.map(&parse_message_segment/1)
      |> Enum.reject(&is_nil/1)

    Logger.debug("Parsed messages: #{inspect(messages)}")
    messages
  end

  @doc """
  Parse a single message segment.
  """
  @spec parse_message_segment(String.t()) :: map() | nil
  def parse_message_segment(segment) do
    Logger.debug("parse_message_segment input: #{inspect(segment)}")

    unless String.contains?(segment, "<|message|>") do
      Logger.debug("No <|message|> in segment")
      nil
    else
      [raw_header, raw_content] = String.split(segment, "<|message|>", parts: 2)
      header = String.trim(raw_header)
      raw_content = String.trim(raw_content)

      {role, channel, recipient, content_type} = parse_header(header)

      {role, channel} =
        cond do
          role in ["analysis", "commentary", "final"] and is_nil(channel) ->
            {"assistant", role}

          true ->
            {role, channel}
        end

      {content, stop_token} = extract_stop_token(raw_content)

      type =
        cond do
          stop_token == :call -> :tool_call
          recipient -> :tool_call
          stop_token == :return -> :final_response
          true -> :message
        end

      message = %{
        role: role,
        channel: channel,
        content: String.trim(content),
        recipient: recipient,
        content_type: content_type,
        type: type
      }

      Logger.debug("Parsed message: #{inspect(message)}")
      message
    end
  end

  @doc """
  Processes incoming text chunk and extracts any complete messages.
  """
  @spec process_text(%__MODULE__{}, String.t()) :: %__MODULE__{}
  def process_text(%__MODULE__{} = parser, text) do
    new_buffer = parser.buffer <> text
    {messages, remaining_buffer} = extract_complete_messages(new_buffer, parser.messages)
    %{parser | buffer: remaining_buffer, messages: messages}
  end

  @doc """
  Returns completed messages and resets the parser's message list.
  """
  @spec get_completed_messages(%__MODULE__{}) :: {[map()], %__MODULE__{}}
  def get_completed_messages(%__MODULE__{} = parser) do
    {parser.messages, %{parser | messages: []}}
  end

  @doc """
  Checks if there are any completed messages.
  """
  @spec has_completed_messages?(%__MODULE__{}) :: boolean()
  def has_completed_messages?(%__MODULE__{} = parser) do
    parser.messages != []
  end

  @doc """
  Extracts complete messages from the buffer.
  """
  @spec extract_complete_messages(String.t(), [map()]) :: {[map()], String.t()}
  def extract_complete_messages(buffer, messages) do
    case String.split(buffer, "<|end|>", parts: 2) do
      [complete_part, rest] ->
        complete_segment = complete_part <> "<|end|>"

        case parse_message_segment(complete_segment) do
          nil ->
            # Skip invalid segments
            extract_complete_messages(rest, messages)

          message ->
            extract_complete_messages(rest, messages ++ [message])
        end

      [_] ->
        {messages, buffer}
    end
  end

  # Helper functions

  @doc """
  Extracts content after a token in the buffer.
  """
  @spec extract_after_token(String.t(), String.t()) :: {String.t(), String.t()}
  def extract_after_token(buffer, token) do
    case String.split(buffer, token, parts: 2) do
      [_, after_token] ->
        trimmed = String.trim(after_token)

        case String.split(trimmed, " ", parts: 2) do
          [first] -> {first, ""}
          [first, rest] -> {first, " " <> rest}
        end

      [_] ->
        {"", String.trim(buffer)}
    end
  end

  @doc """
  Removes a token from the buffer.
  """
  @spec remove_token(String.t(), String.t()) :: String.t()
  def remove_token(buffer, token) do
    String.replace(buffer, token, "", global: false)
  end

  defp parse_header(header) do
    captures = Regex.named_captures(@header_regex, String.trim(header)) || %{}

    role = normalize_field(captures["role"]) || "assistant"
    channel = normalize_field(captures["channel"])
    recipient = normalize_field(captures["recipient"])
    content_type = normalize_field(captures["content_type"])

    {role, channel, recipient, content_type}
  end

  defp extract_stop_token(content) do
    trimmed = String.trim(content)

    cond do
      String.ends_with?(trimmed, "<|call|>") ->
        {strip_terminal_token(trimmed, "<|call|>"), :call}

      String.ends_with?(trimmed, "<|return|>") ->
        {strip_terminal_token(trimmed, "<|return|>"), :return}

      String.ends_with?(trimmed, "<|end|>") ->
        {strip_terminal_token(trimmed, "<|end|>"), :end}

      true ->
        {trimmed, :none}
    end
  end

  defp strip_terminal_token(content, token) do
    content
    |> String.trim_trailing(token)
    |> String.trim()
  end

  defp normalize_field(nil), do: nil
  defp normalize_field(""), do: nil
  defp normalize_field(value), do: String.trim(value)
end
