defmodule LangChain.Utils.JinjaParser do
  @moduledoc """
  Parses Qwen 3 Jinja-formatted completions into structured message maps
  without relying on regular expressions. We manually walk the token stream
  emitted by the template (<|im_start|> ... <|im_end|>) and decode
  tool-call / tool-response blocks via binary pattern matching.
  """

  @im_start "<|im_start|>"
  @im_end "<|im_end|>"
  @tool_call_open "<tool_call>"
  @tool_call_close "</tool_call>"
  @tool_response_open "<tool_response>"
  @tool_response_close "</tool_response>"
  @function_open "<function="
  @function_close "</function>"
  @parameter_open "<parameter="
  @parameter_close "</parameter>"

  @doc """
  Parse a raw completion string into message maps ready for downstream
  conversion to LangChain structs.
  """
  @spec parse_response(String.t()) :: [map()]
  def parse_response(text) when is_binary(text) do
    segments = collect_segments(text, [])
    messages = segments |> Enum.flat_map(&segment_to_messages/1)

    cond do
      messages == [] and String.trim(to_string(text)) != "" ->
        parse_assistant_segment(text)

      true ->
        messages
    end
  end

  def parse_response(_), do: []

  defp collect_segments(text, acc) do
    case next_segment(text) do
      :done -> Enum.reverse(acc)
      {:ok, segment, rest} -> collect_segments(rest, [segment | acc])
    end
  end

  defp next_segment(text) do
    case split_on(text, @im_start) do
      nil ->
        :done

      {_, ""} ->
        :done

      {_, after_start} ->
        {role_line, remainder} = take_line(after_start)
        role = role_line |> to_string() |> String.trim()
        {content, rest} = take_until(remainder, @im_end)
        {:ok, %{"role" => role, "content" => content}, rest}
    end
  end

  defp take_line(text) do
    case split_on(text, "\n") do
      nil -> {text, ""}
      {line, rest} -> {line, rest}
    end
  end

  defp take_until(text, pattern) do
    case split_on(text, pattern) do
      nil -> {text, ""}
      {before, remainder} -> {before, remainder}
    end
  end

  defp split_on(text, pattern) do
    case :binary.match(text, pattern) do
      :nomatch ->
        nil

      {idx, len} ->
        prefix = binary_part(text, 0, idx)
        suffix = binary_part(text, idx + len, byte_size(text) - idx - len)
        {prefix, suffix}
    end
  end

  defp segment_to_messages(%{"role" => role, "content" => content}) do
    case String.trim(role) do
      "assistant" -> parse_assistant_segment(content)
      "user" -> parse_user_segment(content)
      other -> [build_basic_message(other, content)]
    end
  end

  defp build_basic_message(role, content) do
    %{
      role: role,
      content: content |> to_string() |> String.trim(),
      type: :message
    }
  end

  defp parse_assistant_segment(content) do
    {tool_calls, remaining} = extract_tool_calls(content)
    text = remaining |> String.trim()

    base_messages =
      case text do
        "" -> []
        other -> [build_basic_message("assistant", other)]
      end

    case tool_calls do
      [] ->
        base_messages

      calls ->
        base_messages ++
          [
            %{
              role: "assistant",
              type: :tool_call,
              tool_calls: calls,
              content: ""
            }
          ]
    end
  end

  defp parse_user_segment(content) do
    {tool_responses, remaining} = extract_tool_responses(content)
    trimmed = remaining |> String.trim()

    cond do
      tool_responses == [] -> [build_basic_message("user", trimmed)]
      trimmed == "" -> tool_responses
      true -> tool_responses ++ [build_basic_message("user", trimmed)]
    end
  end

  defp extract_tool_calls(content),
    do:
      extract_blocks(content, @tool_call_open, @tool_call_close, [], "", &parse_tool_call_body/1)

  defp extract_tool_responses(content),
    do:
      extract_blocks(
        content,
        @tool_response_open,
        @tool_response_close,
        [],
        "",
        &parse_tool_response_body/1
      )

  defp extract_blocks(content, open_tag, close_tag, acc, kept, parser_fun) do
    case split_on(content, open_tag) do
      nil ->
        {Enum.reverse(acc), kept <> content}

      {before, after_open} ->
        kept = kept <> before

        case split_on(after_open, close_tag) do
          nil ->
            {Enum.reverse(acc), kept <> open_tag <> after_open}

          {body, after_close} ->
            case parser_fun.(body) do
              nil ->
                extract_blocks(after_close, open_tag, close_tag, acc, kept, parser_fun)

              parsed ->
                extract_blocks(after_close, open_tag, close_tag, [parsed | acc], kept, parser_fun)
            end
        end
    end
  end

  defp parse_tool_call_body(body) do
    with {name, inner} when name != "" <- extract_function_block(body) do
      %{
        name: name,
        arguments: extract_parameters(inner, %{})
      }
    else
      _ -> nil
    end
  end

  defp parse_tool_response_body(body) do
    case extract_function_block(body) do
      {"", inner} ->
        build_tool_response(nil, inner)

      {name, inner} ->
        build_tool_response(name, inner)
    end
  end

  defp build_tool_response(name, inner_content) do
    trimmed = inner_content |> to_string() |> String.trim()

    cond do
      trimmed == "" ->
        nil

      true ->
        %{
          role: "tool",
          tool_name: name,
          content: trimmed,
          type: :tool_result
        }
    end
  end

  defp extract_function_block(text) do
    case split_on(text, @function_open) do
      nil ->
        {"", text}

      {_, after_open} ->
        {name_part, remainder} = take_until(after_open, ">")

        clean_name =
          name_part
          |> to_string()
          |> String.trim()
          |> String.trim("\"")

        {inner, _rest} = take_until(remainder, @function_close)
        {clean_name, inner}
    end
  end

  defp extract_parameters(text, acc) do
    case split_on(text, @parameter_open) do
      nil ->
        acc

      {_, after_open} ->
        {key_part, remainder} = take_until(after_open, ">")
        key = key_part |> to_string() |> String.trim()
        {value_body, after_close} = take_until(remainder, @parameter_close)
        value = parse_parameter_value(value_body)
        extract_parameters(after_close, Map.put(acc, key, value))
    end
  end

  defp parse_parameter_value(value) do
    trimmed = value |> to_string() |> String.trim()

    case Jason.decode(trimmed) do
      {:ok, decoded} -> decoded
      {:error, _} -> trimmed
    end
  end
end
