defmodule LangChain.Utils.HarmonyParserTest do
  use ExUnit.Case
  alias LangChain.Utils.HarmonyParser

  describe "new/0" do
    test "creates parser with initial state" do
      parser = HarmonyParser.new()

      assert parser.current_role == nil
      assert parser.current_channel == nil
      assert parser.current_content == nil
      assert parser.buffer == ""
      assert parser.messages == []
      assert parser.in_message == false
      assert parser.parsing_header == false
    end
  end

  describe "parse_response/1" do
    test "handles basic assistant message" do
      response = "<|start|>assistant<|message|>Hello world<|end|>"
      messages = HarmonyParser.parse_response(response)

      assert length(messages) == 1
      message = hd(messages)
      assert message.role == "assistant"
      assert message.content == "Hello world"
      assert message.channel == nil
    end

    test "handles message with channel" do
      response = "<|start|>assistant<|channel|>final<|message|>Response<|end|>"
      messages = HarmonyParser.parse_response(response)

      assert length(messages) == 1
      message = hd(messages)
      assert message.role == "assistant"
      assert message.channel == "final"
      assert message.content == "Response"
    end

    test "handles tool call" do
      response =
        "<|start|>assistant<|channel|>commentary to=functions.weather<|constrain|>json<|message|>{\"location\":\"NYC\"}<|call|>"

      messages = HarmonyParser.parse_response(response)

      assert length(messages) == 1
      message = hd(messages)
      assert message.role == "assistant"
      assert message.channel == "commentary"
      assert message.recipient == "functions.weather"
      assert message.content_type == "json"
      assert message.content == "{\"location\":\"NYC\"}"
      assert message.type == :tool_call
    end

    test "handles final response" do
      response = "<|start|>assistant<|channel|>final<|message|>The weather is sunny<|return|>"
      messages = HarmonyParser.parse_response(response)

      assert length(messages) == 1
      message = hd(messages)
      assert message.role == "assistant"
      assert message.channel == "final"
      assert message.content == "The weather is sunny"
      assert message.type == :final_response
    end

    test "handles multiple messages" do
      response =
        "<|start|>assistant<|channel|>analysis<|message|>Thinking...<|end|><|start|>assistant<|channel|>final<|message|>Answer<|return|>"

      messages = HarmonyParser.parse_response(response)

      assert length(messages) == 2
      [msg1, msg2] = messages

      assert msg1.role == "assistant"
      assert msg1.channel == "analysis"
      assert msg1.content == "Thinking..."

      assert msg2.role == "assistant"
      assert msg2.channel == "final"
      assert msg2.content == "Answer"
      assert msg2.type == :final_response
    end
  end

  describe "get_completed_messages/1" do
    test "returns messages and resets parser" do
      parser = HarmonyParser.new()
      # Simulate a completed message
      parser = %{parser | messages: [%{role: "assistant", content: "test"}]}

      {messages, new_parser} = HarmonyParser.get_completed_messages(parser)

      assert messages == [%{role: "assistant", content: "test"}]
      assert new_parser.messages == []
    end
  end

  describe "has_completed_messages?/1" do
    test "returns true when messages exist" do
      parser = %{HarmonyParser.new() | messages: [%{role: "test"}]}
      assert HarmonyParser.has_completed_messages?(parser)
    end

    test "returns false when no messages" do
      parser = HarmonyParser.new()
      refute HarmonyParser.has_completed_messages?(parser)
    end
  end

  describe "parse_response/1 additional tests" do
    test "parses complete response string" do
      response =
        "<|start|>assistant<|message|>Hello<|end|><|start|>assistant<|channel|>final<|message|>World<|return|>"

      messages = HarmonyParser.parse_response(response)

      assert length(messages) == 2
      [msg1, msg2] = messages

      assert msg1.role == "assistant"
      assert msg1.content == "Hello"

      assert msg2.role == "assistant"
      assert msg2.channel == "final"
      assert msg2.content == "World"
      assert msg2.type == :final_response
    end
  end

  describe "extract_after_token/2" do
    test "extracts content after token" do
      buffer = "assistant<|message|>Hello world"
      assert HarmonyParser.extract_after_token(buffer, "<|message|>") == {"Hello", " world"}
    end

    test "handles end of buffer" do
      buffer = "assistant<|message|>Hello"
      assert HarmonyParser.extract_after_token(buffer, "<|message|>") == {"Hello", ""}
    end
  end

  describe "remove_token/1" do
    test "removes first occurrence of token" do
      buffer = "start<|end|>content<|end|>"
      result = HarmonyParser.remove_token(buffer, "<|end|>")
      assert result == "startcontent<|end|>"
    end
  end
end
