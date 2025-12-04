defmodule LangChain.ChatModels.ChatLlamaJinjaTest do
  use ExUnit.Case

  alias LangChain.ChatModels.ChatLlamaJinja

  describe "new/1" do
    test "returns struct with defaults" do
      assert {:ok, %ChatLlamaJinja{} = model} = ChatLlamaJinja.new()
      assert model.model == "Qwen3-Coder-30B-A3B-Instruct"
      assert model.stream == true
      assert model.max_tokens == 16_384
      assert model.top_p == 0.8
      assert model.top_k == 20
      assert model.repetition_penalty == 1.05
    end

    test "validates temperature bounds" do
      assert {:error, _} = ChatLlamaJinja.new(%{temperature: -0.1})
      assert {:error, _} = ChatLlamaJinja.new(%{temperature: 3.0})
    end

    test "validates repetition_penalty is positive" do
      assert {:error, _} = ChatLlamaJinja.new(%{repetition_penalty: 0})
    end
  end

  describe "call/3" do
    test "returns error when endpoint unavailable" do
      model = ChatLlamaJinja.new!(%{endpoint: "http://127.0.0.1:9"})
      {:error, %Req.TransportError{reason: :econnrefused}} = ChatLlamaJinja.call(model, [], [])
    end
  end

  describe "parse_response_and_convert/3" do
    setup do
      {:ok, model: ChatLlamaJinja.new!(), prompt: [LangChain.Message.new_user!("test")]}
    end

    test "parses assistant message", %{model: model, prompt: prompt} do
      response_body =
        Jason.encode!(%{
          "content" => "<|im_start|>assistant\nHello from Qwen<|im_end|>"
        })

      assert {:ok, [message]} =
               ChatLlamaJinja.parse_response_and_convert(model, response_body, prompt)

      assert message.role == :assistant
      assert [%{type: :text, content: "Hello from Qwen"}] = message.content
    end

    test "parses tool call message", %{model: model, prompt: prompt} do
      response_body =
        Jason.encode!(%{
          "content" => """
          <|im_start|>assistant
          <tool_call>
          <function=get_weather>
          <parameter=location>
          Seattle
          </parameter>
          </function>
          </tool_call>
          <|im_end|>
          """
        })

      assert {:ok, [message]} =
               ChatLlamaJinja.parse_response_and_convert(model, response_body, prompt)

      assert message.role == :assistant
      assert length(message.tool_calls) == 1
      tool_call = hd(message.tool_calls)
      assert tool_call.name == "get_weather"
      assert tool_call.arguments == %{"location" => "Seattle"}
    end

    test "returns error for invalid JSON", %{model: model, prompt: prompt} do
      assert {:error, "JSON decode error:" <> _} =
               ChatLlamaJinja.parse_response_and_convert(model, "invalid json", prompt)
    end

    test "parses tool result messages", %{model: model, prompt: prompt} do
      response_body =
        Jason.encode!(%{
          "content" => """
          <|im_start|>user
          <tool_response>
          <function=get_weather>
          {"temperature":22}
          </function>
          </tool_response>
          <|im_end|>
          """
        })

      assert {:ok, [message]} =
               ChatLlamaJinja.parse_response_and_convert(model, response_body, prompt)

      assert message.role == :tool
      [result] = message.tool_results
      assert result.name == "get_weather"

      assert LangChain.Utils.HarmonyRenderer.extract_text_content(result.content) ==
               "{\"temperature\":22}"
    end
  end
end
