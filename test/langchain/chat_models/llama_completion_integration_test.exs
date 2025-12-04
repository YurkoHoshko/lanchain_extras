defmodule LangChain.ChatModels.LlamaCompletionIntegrationTest do
  use ExUnit.Case, async: true

  alias LangChain.ChatModels.{ChatLlamaHarmony, ChatLlamaJinja}
  alias LangChain.Message

  describe "harmony /completion payload" do
    test "sends n_predict and model and parses content" do
      bypass = Bypass.open()

      Bypass.expect_once(bypass, "POST", "/completion", fn conn ->
        {:ok, body, conn} = Plug.Conn.read_body(conn)
        params = Jason.decode!(body)

        assert params["model"] == "test-model"
        assert params["n_predict"] == 50
        assert is_binary(params["prompt"])

        Plug.Conn.resp(
          conn,
          200,
          ~s({"content":"<|start|>assistant<|message|>Hello!<|end|>"})
        )
      end)

      prompt = [Message.new_user!("hi")]

      model =
        ChatLlamaHarmony.new!(%{
          endpoint: "http://localhost:#{bypass.port}/completion",
          model: "test-model",
          max_tokens: 50,
          stream: false
        })

      assert {:ok, [reply]} = ChatLlamaHarmony.call(model, prompt, [])
      assert reply.role == :assistant
      assert Enum.any?(reply.content, &(&1.content == "Hello!"))
    end
  end

  describe "jinja /completion payload" do
    test "sends n_predict and repeat_penalty and parses content" do
      bypass = Bypass.open()

      Bypass.expect_once(bypass, "POST", "/completion", fn conn ->
        {:ok, body, conn} = Plug.Conn.read_body(conn)
        params = Jason.decode!(body)

        assert params["model"] == "qwen"
        assert params["n_predict"] == 32
        assert params["repeat_penalty"] == 1.5
        assert is_binary(params["prompt"])

        body = Jason.encode!(%{"content" => "<|im_start|>assistant\nHello from jinja<|im_end|>"})
        Plug.Conn.resp(conn, 200, body)
      end)

      prompt = [Message.new_user!("ping")]

      model =
        ChatLlamaJinja.new!(%{
          endpoint: "http://localhost:#{bypass.port}/completion",
          model: "qwen",
          max_tokens: 32,
          repetition_penalty: 1.5,
          stream: false
        })

      assert {:ok, [reply]} = ChatLlamaJinja.call(model, prompt, [])
      assert reply.role == :assistant
      assert Enum.any?(reply.content, &(&1.content =~ "Hello from jinja"))
    end
  end
end
