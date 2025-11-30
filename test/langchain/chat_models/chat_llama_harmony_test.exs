defmodule LangChain.ChatModels.ChatLlamaHarmonyTest do
  use ExUnit.Case
  alias LangChain.ChatModels.ChatLlamaHarmony

  describe "new/1" do
    test "creates a valid ChatLlamaHarmony struct with default values" do
      assert {:ok, %ChatLlamaHarmony{} = model} = ChatLlamaHarmony.new()
      assert model.endpoint == "http://localhost:8000/v1/completions"
      assert model.model == "llama-3.1-8b"
      assert model.temperature == 0.7
      assert model.stream == true
      assert model.receive_timeout == 60_000
      assert model.reasoning_effort == :medium
      assert model.reasoning_mode == false
      assert model.knowledge_cutoff == "2024-06"
    end

    test "creates a valid struct with custom values" do
      attrs = %{
        endpoint: "http://custom:8080/v1/completions",
        model: "custom-model",
        temperature: 0.5,
        stream: false,
        max_tokens: 100,
        reasoning_effort: :high,
        knowledge_cutoff: "2025-01"
      }

      assert {:ok, %ChatLlamaHarmony{} = model} = ChatLlamaHarmony.new(attrs)
      assert model.endpoint == "http://custom:8080/v1/completions"
      assert model.model == "custom-model"
      assert model.temperature == 0.5
      assert model.stream == false
      assert model.max_tokens == 100
      assert model.reasoning_effort == :high
      assert model.knowledge_cutoff == "2025-01"
    end

    test "validates temperature range" do
      assert {:error, changeset} = ChatLlamaHarmony.new(%{temperature: -1})
      assert {"must be greater than or equal to %{number}", _} = changeset.errors[:temperature]

      assert {:error, changeset} = ChatLlamaHarmony.new(%{temperature: 3})
      assert {"must be less than or equal to %{number}", _} = changeset.errors[:temperature]
    end

    test "validates max_tokens" do
      assert {:error, changeset} = ChatLlamaHarmony.new(%{max_tokens: 0})
      assert {"must be greater than %{number}", _} = changeset.errors[:max_tokens]
    end

    test "validates receive_timeout" do
      assert {:error, changeset} = ChatLlamaHarmony.new(%{receive_timeout: 0})
      assert {"must be greater than %{number}", _} = changeset.errors[:receive_timeout]
    end
  end

  describe "new!/1" do
    test "returns the struct when valid" do
      model = ChatLlamaHarmony.new!(%{model: "test-model"})
      assert %ChatLlamaHarmony{} = model
      assert model.model == "test-model"
    end

    test "raises an error when invalid" do
      assert_raise LangChain.LangChainError, fn ->
        ChatLlamaHarmony.new!(%{temperature: -1})
      end
    end
  end

  describe "call/3" do
    test "attempts to make HTTP request and fails with connection error" do
      model = ChatLlamaHarmony.new!()
      {:error, %Req.TransportError{reason: :econnrefused}} = ChatLlamaHarmony.call(model, [], [])
    end
  end

  describe "parse_response_and_convert/3" do
    test "parses llama.cpp response and converts to LangChain messages" do
      model = ChatLlamaHarmony.new!()

      # Mock llama.cpp response
      response_body =
        Jason.encode!(%{
          "content" => "<|start|>assistant<|message|>Hello world<|end|>",
          "usage" => %{"prompt_tokens" => 10, "completion_tokens" => 2}
        })

      prompt = [LangChain.Message.new_user!("test")]

      {:ok, messages} = ChatLlamaHarmony.parse_response_and_convert(model, response_body, prompt)

      assert length(messages) == 1
      message = hd(messages)
      assert message.role == :assistant
      assert length(message.content) == 1
      content_part = hd(message.content)
      assert content_part.type == :text
      assert content_part.content == "Hello world"
      assert message.metadata[:channel] == nil
    end

    test "handles tool call messages" do
      model = ChatLlamaHarmony.new!()

      response_body =
        Jason.encode!(%{
          "content" =>
            "<|start|>assistant<|channel|>commentary to=functions.weather<|constrain|>json<|message|>{\"location\":\"NYC\"}<|call|>"
        })

      prompt = []

      {:ok, messages} = ChatLlamaHarmony.parse_response_and_convert(model, response_body, prompt)

      assert length(messages) == 1
      message = hd(messages)
      assert message.role == :assistant
      assert length(message.tool_calls) == 1
      tool_call = hd(message.tool_calls)
      assert tool_call.name == "weather"
      assert tool_call.arguments == %{"location" => "NYC"}
      assert message.metadata[:recipient] == "functions.weather"
      assert message.metadata[:channel] == "commentary"
      assert message.metadata[:recipient] == "functions.weather"
      assert message.metadata[:content_type] == "json"
    end

    test "returns error for invalid JSON" do
      model = ChatLlamaHarmony.new!()
      prompt = []

      {:error, "JSON decode error:" <> _} =
        ChatLlamaHarmony.parse_response_and_convert(model, "invalid json", prompt)
    end

    test "returns error for missing content field" do
      model = ChatLlamaHarmony.new!()
      response_body = Jason.encode!(%{"usage" => %{}})
      prompt = []

      {:error, "Unexpected response format:" <> _} =
        ChatLlamaHarmony.parse_response_and_convert(model, response_body, prompt)
    end
  end

  describe "harmony_to_langchain_message/1" do
    test "converts assistant message" do
      harmony_msg = %{
        role: "assistant",
        content: "Hello!",
        channel: "final"
      }

      message = ChatLlamaHarmony.harmony_to_langchain_message(harmony_msg)

      assert message.role == :assistant
      assert length(message.content) == 1
      content_part = hd(message.content)
      assert content_part.type == :text
      assert content_part.content == "Hello!"
      assert message.metadata[:channel] == "final"
    end

    test "converts tool call message" do
      harmony_msg = %{
        role: "assistant",
        content: "{\"location\":\"NYC\"}",
        recipient: "functions.weather",
        type: :tool_call
      }

      message = ChatLlamaHarmony.harmony_to_langchain_message(harmony_msg)

      assert message.role == :assistant
      assert length(message.tool_calls) == 1
      tool_call = hd(message.tool_calls)
      assert tool_call.name == "weather"
      assert tool_call.arguments == %{"location" => "NYC"}
    end

    test "handles missing content" do
      harmony_msg = %{"role" => "assistant"}

      message = ChatLlamaHarmony.harmony_to_langchain_message(harmony_msg)

      assert length(message.content) == 1
      content_part = hd(message.content)
      assert content_part.type == :text
      assert content_part.content == ""
    end
  end
end
