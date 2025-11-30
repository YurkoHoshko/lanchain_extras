defmodule LangChain.Utils.JinjaRendererTest do
  use ExUnit.Case, async: true

  alias LangChain.Utils.JinjaRenderer
  alias LangChain.Message
  alias LangChain.Function
  alias LangChain.Message.ToolCall

  test "renders default system message when no system prompt exists but tools do" do
    tools = [
      %Function{
        name: "search_files",
        description: "Search for content inside the repository",
        parameters_schema: %{
          "type" => "object",
          "properties" => %{
            "query" => %{"type" => "string", "description" => "Search term"}
          },
          "required" => ["query"]
        }
      }
    ]

    messages = [Message.new_user!("Hi there")]

    rendered = JinjaRenderer.render_conversation(messages, tools, %{add_generation_prompt: false})

    assert rendered =~ "<|im_start|>system"
    assert rendered =~ "You are Qwen, a helpful AI assistant"
    assert rendered =~ "You have access to the following functions"
    assert rendered =~ "<tools>"
    assert rendered =~ "<function>"
    assert rendered =~ "search_files"

    assert rendered =~
             "If you choose to call a function ONLY reply in the following format with NO suffix"
  end

  test "renders assistant tool call block" do
    tool_call =
      ToolCall.new!(%{
        call_id: "call_001",
        name: "get_weather",
        arguments: %{"location" => "Paris"}
      })

    message =
      Message.new!(%{
        role: :assistant,
        tool_calls: [tool_call]
      })

    rendered = JinjaRenderer.render_conversation([message], [], %{add_generation_prompt: false})

    assert rendered =~ "<tool_call>"
    assert rendered =~ "<function=get_weather>"
    assert rendered =~ "<parameter=location>"
    assert rendered =~ "Paris"
  end
end
