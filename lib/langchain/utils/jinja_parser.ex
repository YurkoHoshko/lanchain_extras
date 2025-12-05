defmodule LangChain.Utils.JinjaParser do
  @moduledoc """
  NimbleParsec-based parser for Qwen Jinja completions (`<|im_start|> ... <|im_end|>`).

  What it handles
  ---------------
  * Segmented messages with roles (assistant, user, system, etc.)
  * Assistant tool calls: `<tool_call><function=...><parameter=...>...</function></tool_call>`
  * User tool responses: `<tool_response><function=...>JSON/text</function></tool_response>`
  * Plain text segments when no tool tags are present

  How it is structured
  --------------------
  * **Segment combinator** (`segment`): parses one `<|im_start|>role ... <|im_end|>` block.
  * **Assistant parts** (`assistant_parts`): alternates between `tool_call_block` and `text_chunk`.
  * **User parts** (`user_parts`): alternates between `tool_response_block` and `user_text_chunk`.
  * **Function block** (`function_block`): grabs function name and raw inner body; parameters are
    parsed separately so we can both keep raw text and produce an arguments map.
  * **Parameter block** (`parameter_block`): `<parameter=key>value</parameter>` with JSON-or-text decode.

  Fallbacks
  ---------
  If no tool tags are detected, content is emitted as a simple `:message`. Tool names are inferred
  from the body if the generator omits `function=` in responses.
  """

  import NimbleParsec
  require Logger

  @im_start "<|im_start|>"
  @im_end "<|im_end|>"
  @tool_call "<tool_call>"
  @tool_call_close "</tool_call>"
  @tool_response "<tool_response>"
  @tool_response_close "</tool_response>"
  @function_open "<function="
  @function_close "</function>"
  @parameter_open "<parameter="
  @parameter_close "</parameter>"

  # ---------- Public API ----------

  @spec parse_response(String.t()) :: [map()]
  def parse_response(text) when is_binary(text) do
    trimmed = String.trim(text)

    case segments(text) do
      {:ok, segs, _rest, _ctx, _loc, _offset} ->
        messages = segs |> Enum.flat_map(&segment_to_messages/1)

        cond do
          messages != [] -> messages
          trimmed != "" -> [%{role: "assistant", type: :message, content: trimmed}]
          true -> []
        end

      {:error, reason, _rest, _ctx, _loc, _offset} ->
        Logger.debug("JinjaParser error: #{inspect(reason)}")
        if trimmed == "" do
          []
        else
          [%{role: "assistant", type: :message, content: trimmed}]
        end
    end
  end

  def parse_response(_), do: []

  # ---------- Segment parser ----------

  defp not_im_end(<<@im_end, _::binary>>, ctx, _, _), do: {:halt, ctx}
  defp not_im_end(<<>>, ctx, _, _), do: {:halt, ctx}
  defp not_im_end(_, ctx, _, _), do: {:cont, ctx}

  defp not_tool_or_end(<<@tool_call, _::binary>>, ctx, _, _), do: {:halt, ctx}
  defp not_tool_or_end(<<@im_end, _::binary>>, ctx, _, _), do: {:halt, ctx}
  defp not_tool_or_end(<<>>, ctx, _, _), do: {:halt, ctx}
  defp not_tool_or_end(_, ctx, _, _), do: {:cont, ctx}

  defp not_tool_response_or_end(<<@tool_response, _::binary>>, ctx, _, _), do: {:halt, ctx}
  defp not_tool_response_or_end(<<@im_end, _::binary>>, ctx, _, _), do: {:halt, ctx}
  defp not_tool_response_or_end(<<>>, ctx, _, _), do: {:halt, ctx}
  defp not_tool_response_or_end(_, ctx, _, _), do: {:cont, ctx}

  defp not_role_terminator(<<"\n", _::binary>>, ctx, _, _), do: {:halt, ctx}
  defp not_role_terminator(<<@im_end, _::binary>>, ctx, _, _), do: {:halt, ctx}
  defp not_role_terminator(_, ctx, _, _), do: {:cont, ctx}

  role_line =
    repeat_while(utf8_char([]), {:not_role_terminator, []})
    |> reduce({List, :to_string, []})

  segment =
    ignore(string(@im_start))
    |> concat(role_line |> tag(:role))
    |> optional(ignore(string("\n")))
    |> concat(
      repeat_while(utf8_char([]), {:not_im_end, []})
      |> reduce({List, :to_string, []})
      |> tag(:content)
    )
    |> ignore(string(@im_end))
    |> reduce({__MODULE__, :build_segment, []})

  defparsec(:segments, repeat(segment))

  # ---------- Assistant content parser ----------

  ws = ignore(repeat(ascii_char([?\s, ?\t, ?\n])))

  param_value =
    repeat_while(utf8_char([]), {:not_param_close, []})
    |> reduce({List, :to_string, []})

  defp not_param_close(<<@parameter_close, _::binary>>, ctx, _, _), do: {:halt, ctx}
  defp not_param_close(<<>>, ctx, _, _), do: {:halt, ctx}
  defp not_param_close(_, ctx, _, _), do: {:cont, ctx}

  parameter_block =
    ws
    |> ignore(string(@parameter_open))
    |> ascii_string([not: ?>], min: 1)
    |> ignore(string(">"))
    |> concat(param_value)
    |> ignore(string(@parameter_close))
    |> reduce({__MODULE__, :build_parameter, []})

  defparsecp(:parameter_list, repeat(parameter_block))

  defp not_function_close(<<@function_close, _::binary>>, ctx, _, _), do: {:halt, ctx}
  defp not_function_close(<<>>, ctx, _, _), do: {:halt, ctx}
  defp not_function_close(_, ctx, _, _), do: {:cont, ctx}

  function_block =
    ignore(string(@function_open))
    |> ascii_string([not: ?>], min: 1)
    |> tag(:fn_name)
    |> ignore(string(">"))
    |> repeat_while(utf8_char([]), {:not_function_close, []})
    |> tag(:fn_body)
    |> ignore(string(@function_close))
    |> reduce({__MODULE__, :build_function_block, []})

  tool_call_block =
    ignore(string(@tool_call))
    |> optional(ignore(string("\n")))
    |> concat(function_block)
    |> optional(ignore(string("\n")))
    |> ignore(string(@tool_call_close))
    |> tag(:tool_call)

  tool_response_block =
    ignore(string(@tool_response))
    |> optional(ignore(string("\n")))
    |> concat(function_block)
    |> optional(ignore(string("\n")))
    |> ignore(string(@tool_response_close))
    |> reduce({__MODULE__, :build_tool_response, []})

  text_chunk =
    lookahead_not(string(@tool_call))
    |> lookahead_not(string(@im_end))
    |> utf8_char([])
    |> repeat_while(utf8_char([]), {:not_tool_or_end, []})
    |> reduce({List, :to_string, []})
    |> tag(:text)

  user_text_chunk =
    lookahead_not(string(@tool_response))
    |> lookahead_not(string(@im_end))
    |> utf8_char([])
    |> repeat_while(utf8_char([]), {:not_tool_response_or_end, []})
    |> reduce({List, :to_string, []})
    |> tag(:text)

  defparsecp(
    :assistant_parts,
    repeat(choice([tool_call_block, text_chunk]))
  )

  defparsecp(
    :user_parts,
    repeat(choice([tool_response_block, user_text_chunk]))
  )

  # ---------- Reducers / builders ----------

  def build_segment(parts) do
    Enum.into(parts, %{role: "assistant", content: ""})
  end

  def build_parameter([name, value]) do
    {String.trim(name), parse_json_or_string(value)}
  end

  def build_function_block(parts) do
    attrs = Enum.into(parts, %{})
    raw_name = attrs[:fn_name] |> to_flat_string() |> String.trim()
    body = attrs[:fn_body] |> to_flat_string()

    name =
      case raw_name do
        "" -> body |> String.trim_leading() |> String.split(~r/[\n<]/, parts: 2) |> List.first() |> to_string() |> String.trim()
        other -> other
      end

    param_source =
      case String.split(body, "\n", parts: 2) do
        [_, rest] -> String.trim_leading(rest)
        _ -> String.trim_leading(body)
      end

    params =
      case parameter_list(param_source) do
        {:ok, plist, _rest, _ctx, _loc, _offset} -> Map.new(plist)
        _ -> %{}
      end

    %{name: name, arguments: params, body: String.trim(body)}
  end

  def build_tool_response(%{name: name, body: body, arguments: args}) do
    inferred_name =
      case String.trim(name) do
        "" ->
          body
          |> String.trim_leading()
          |> String.split(~r/[\n<]/, parts: 2)
          |> List.first()
          |> to_string()
          |> String.trim()

        other ->
          other
      end

    content_body =
      case String.split(body, "\n", parts: 2) do
        [first, rest] ->
          if String.trim(first) == inferred_name or String.trim(name) == "" do
            rest
          else
            body
          end

        _ ->
          body
      end

    content =
      case String.trim(content_body) do
        "" -> args |> Map.get("result", "") |> to_string() |> String.trim()
        other -> other
      end

    %{
      role: "tool",
      tool_name: inferred_name,
      content: content,
      type: :tool_result
    }
  end
  def build_tool_response([%{} = fun]), do: build_tool_response(fun)

  defp parse_json_or_string(value) do
    trimmed = value |> to_flat_string() |> String.trim()

    case Jason.decode(trimmed) do
      {:ok, decoded} -> decoded
      _ -> trimmed
    end
  end

  defp segment_to_messages(%{role: role, content: content}) do
    role = role |> to_flat_string() |> String.trim()
    content = content |> to_flat_string()

    case role do
      "assistant" -> parse_assistant(content)
      "user" -> parse_user(content)
      other -> [%{role: other, content: String.trim(content), type: :message}]
    end
  end

  defp to_flat_string(nil), do: ""
  defp to_flat_string(value) when is_binary(value), do: value

  defp to_flat_string(value) when is_list(value) do
    value
    |> Enum.map(fn
      {_, v} -> to_flat_string(v)
      v -> v
    end)
    |> IO.iodata_to_binary()
  end

  defp parse_assistant(content) do
    if not String.contains?(content, @tool_call) do
      [%{role: "assistant", content: String.trim(content), type: :message}]
    else
      parse_assistant_with_tools(content)
    end
  end

  defp parse_assistant_with_tools(content) do
    case assistant_parts(content) do
      {:ok, parts, _rest, _ctx, _loc, _offset} ->
        {tool_calls, texts} =
          parts
          |> Enum.reduce({[], []}, fn
            {:tool_call, tc}, {calls, ts} -> {List.wrap(tc) ++ calls, ts}
            {:text, t}, {calls, ts} -> {calls, [t | ts]}
          end)

        text =
          texts
          |> Enum.reverse()
          |> Enum.join()
          |> String.trim()

        base =
          case text do
            "" -> []
            other -> [%{role: "assistant", content: other, type: :message}]
          end

        case tool_calls do
          [] ->
            base

          calls ->
            base ++
              [
                %{
                  role: "assistant",
                  type: :tool_call,
                  tool_calls: Enum.reverse(calls),
                  content: ""
                }
              ]
        end

      {:error, _, _, _, _, _} ->
        [%{role: "assistant", content: String.trim(content), type: :message}]
    end
  end

  defp parse_user(content) do
    if not String.contains?(content, @tool_response) do
      [%{role: "user", content: String.trim(content), type: :message}]
    else
      parse_user_with_tools(content)
    end
  end

  defp parse_user_with_tools(content) do
    case user_parts(content) do
      {:ok, parts, _rest, _ctx, _loc, _offset} ->
        {tool_responses, texts} =
          parts
          |> Enum.reduce({[], []}, fn
            %{} = resp, {trs, ts} -> {[resp | trs], ts}
            {:text, t}, {trs, ts} -> {trs, [t | ts]}
          end)

        text =
          texts
          |> Enum.reverse()
          |> Enum.join()
          |> String.trim()

        cond do
          tool_responses == [] -> [%{role: "user", content: text, type: :message}]
          text == "" -> Enum.reverse(tool_responses)
          true -> Enum.reverse(tool_responses) ++ [%{role: "user", content: text, type: :message}]
        end

      {:error, _, _, _, _, _} ->
        [%{role: "user", content: String.trim(content), type: :message}]
    end
  end
end
