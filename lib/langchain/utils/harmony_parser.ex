defmodule LangChain.Utils.HarmonyParser do
  @moduledoc """
  NimbleParsec-based parser for OpenAI Harmony message format.

  What it handles
  ---------------
  * `<|start|>...<|message|>...<|end|>` message framing
  * Optional channel (`<|channel|>`), recipient (`to=`) and content type (`<|constrain|>`)
  * Stop markers `<|call|>`, `<|return|>`, `<|end|>` mapped to `:tool_call` / `:final_response`

  How it is structured
  --------------------
  * **Header combinator** (`header`): parses role + optional channel / recipient / content_type.
  * **Content combinator** (`content`): consumes UTF‑8 until a stop tag is seen.
  * **Stop combinator** (`stop_token`): captures which terminal token closed the message.
  * **Message combinator** (`message`): `<|start|>` + header + `<|message|>` + content + stop.
  * **message_list**: top-level parser; `process_text/2` reuses it for streaming buffers.

  Streaming use
  -------------
  Call `process_text/2` with new chunks; completed messages accumulate on `parser.messages`,
  and any trailing partial message remains in `parser.buffer`.
  """

  require Logger
  import NimbleParsec

  defstruct [:buffer, :messages]

  @start "<|start|>"
  @message "<|message|>"
  @channel "<|channel|>"
  @constrain "<|constrain|>"
  @call "<|call|>"
  @return "<|return|>"
  @end_token "<|end|>"

  # ---------- Public API ----------

  @doc """
  Creates a new streaming parser instance with initial state.
  """
  @spec new() :: %__MODULE__{}
  def new, do: %__MODULE__{buffer: "", messages: []}

  @doc """
  Parses a complete Harmony response string into messages.
  """
  @spec parse_response(String.t()) :: [map()]
  def parse_response(text) when is_binary(text) do
    case message_list(text) do
      {:ok, messages, _rest, _ctx, _loc, _offset} -> messages
      {:error, reason, _rest, _ctx, _loc, _offset} ->
        Logger.debug("HarmonyParser error: #{inspect(reason)}")
        []
    end
  end

  @doc """
  Processes streaming text, accumulating complete messages and keeping remainder in the buffer.
  """
  @spec process_text(%__MODULE__{}, String.t()) :: %__MODULE__{}
  def process_text(%__MODULE__{} = parser, chunk) do
    buffer = parser.buffer <> chunk

    case message_list(buffer) do
      {:ok, messages, rest, _ctx, _loc, _offset} ->
        %{parser | buffer: rest, messages: parser.messages ++ messages}

      {:error, _reason, _rest, _ctx, _loc, _offset} ->
        # Keep buffer intact if we cannot parse yet (likely incomplete data)
        %{parser | buffer: buffer}
    end
  end

  @doc """
  Returns completed messages and resets the internal list.
  """
  @spec get_completed_messages(%__MODULE__{}) :: {[map()], %__MODULE__{}}
  def get_completed_messages(%__MODULE__{} = parser) do
    {parser.messages, %{parser | messages: []}}
  end

  @doc """
  Checks if there are any completed messages ready to be consumed.
  """
  @spec has_completed_messages?(%__MODULE__{}) :: boolean()
  def has_completed_messages?(%__MODULE__{} = parser), do: parser.messages != []

  # ---------- NimbleParsec definitions ----------

  whitespace = ignore(repeat(ascii_char([?\s, ?\t])))
  bare_token = ascii_string([not: ?<, not: ?\s], min: 1)

  header =
    tag(bare_token, :role)
    |> optional(ignore(string(@channel)) |> tag(bare_token, :channel))
    |> optional(whitespace |> ignore(string("to=")) |> tag(bare_token, :recipient))
    |> optional(whitespace |> ignore(string(@constrain)) |> tag(bare_token, :content_type))
    |> reduce({__MODULE__, :normalize_header, []})

  stop_tag = choice([string(@call), string(@return), string(@end_token)])

  content =
    repeat(
      lookahead_not(stop_tag)
      |> utf8_char([])
    )
    |> reduce({List, :to_string, []})
    |> tag(:content)

  stop_token =
    choice([
      string(@call) |> replace(:call),
      string(@return) |> replace(:return),
      string(@end_token) |> replace(:end)
    ])
    |> tag(:stop)

  message =
    ignore(string(@start))
    |> concat(header)
    |> ignore(string(@message))
    |> concat(content)
    |> optional(stop_token)
    |> reduce({__MODULE__, :build_message, []})

  defparsec(:message_list, repeat(message))

  # ---------- Reducers ----------

  def normalize_header(parts) do
    parts
    |> Enum.into(%{})
    |> Map.put_new(:channel, nil)
    |> Map.put_new(:recipient, nil)
    |> Map.put_new(:content_type, nil)
  end

  def build_message(parts) do
    {header_map, rest} =
      case Enum.split_with(parts, &is_map/1) do
        {[%{} = header | _], tail} -> {header, tail}
        {_, tail} -> {%{}, tail}
      end

    attrs =
      Enum.reduce(rest, header_map, fn
        {:content, c}, acc -> Map.put(acc, :content, c)
        {:stop, s}, acc -> Map.put(acc, :stop, s)
        {:channel, c}, acc -> Map.put(acc, :channel, c)
        {:role, r}, acc -> Map.put(acc, :role, r)
        {:recipient, r}, acc -> Map.put(acc, :recipient, r)
        {:content_type, ct}, acc -> Map.put(acc, :content_type, ct)
        _, acc -> acc
      end)

    role = normalize(attrs.role, "assistant")
    channel = normalize(attrs.channel, nil)
    recipient = normalize(attrs.recipient, nil)
    content_type = normalize(attrs.content_type, nil)
    stop = normalize_stop(attrs[:stop])

    {role, channel} =
      if role in ["analysis", "commentary", "final"] and is_nil(channel),
        do: {"assistant", role},
        else: {role, channel}

    type =
      cond do
        stop == :call or not is_nil(recipient) -> :tool_call
        stop == :return -> :final_response
        true -> :message
      end

    %{
      role: role,
      channel: channel,
      recipient: recipient,
      content_type: content_type,
      content: attrs.content |> normalize("") |> String.trim(),
      type: type
    }
  end

  defp normalize(nil, default), do: default
  defp normalize(value, _default) when is_list(value), do: value |> Enum.join()
  defp normalize(value, _default), do: value

  defp normalize_stop([value]) when is_atom(value), do: value
  defp normalize_stop(value) when is_atom(value), do: value
  defp normalize_stop(_), do: nil
end
