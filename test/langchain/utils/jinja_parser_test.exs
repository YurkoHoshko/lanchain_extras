defmodule LangChain.Utils.JinjaParserTest do
  use ExUnit.Case, async: true

  alias LangChain.Utils.JinjaParser

  test "parses plain assistant message" do
    text = "<|im_start|>assistant\nHello world<|im_end|>"

    assert [%{role: "assistant", content: "Hello world", type: :message}] =
             JinjaParser.parse_response(text)
  end

  test "parses assistant tool calls with parameters" do
    text = """
    <|im_start|>assistant
    <tool_call>
    <function=get_weather>
    <parameter=location>
    Seattle
    </parameter>
    <parameter=days>
    3
    </parameter>
    </function>
    </tool_call>
    <|im_end|>
    """

    assert [
             %{
               role: "assistant",
               type: :tool_call,
               tool_calls: [
                 %{name: "get_weather", arguments: %{"location" => "Seattle", "days" => 3}}
               ]
             }
           ] = JinjaParser.parse_response(text)
  end

  test "parses tool responses embedded in user segment" do
    text = """
    <|im_start|>user
    <tool_response>
    <function=get_weather>
    {"temperature":22}
    </function>
    </tool_response>
    <|im_end|>
    """

    assert [
             %{
               role: "tool",
               tool_name: "get_weather",
               content: "{\"temperature\":22}",
               type: :tool_result
             }
           ] = JinjaParser.parse_response(text)
  end
end
