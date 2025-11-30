defmodule LangChain.Utils.HarmonyRendererTest do
  use ExUnit.Case
  alias LangChain.Utils.HarmonyRenderer
  alias LangChain.Message
  alias LangChain.Function

  describe "render_conversation/3" do
    test "renders basic conversation with system and user messages" do
      messages = [
        Message.new_system!("You are a helpful assistant."),
        Message.new_user!("Hello!")
      ]

      result = HarmonyRenderer.render_conversation(messages)

      assert String.contains?(result, "<|start|>system<|message|>You are ChatGPT")
      assert String.contains?(result, "You are a helpful assistant.<|end|>")
      assert String.contains?(result, "<|start|>user<|message|>Hello!<|end|>")
    end

    test "includes tools in system message when provided" do
      tools = [
        %Function{
          name: "get_weather",
          description: "Gets the weather",
          parameters_schema: %{
            "type" => "object",
            "properties" => %{
              "location" => %{"type" => "string", "description" => "City name"}
            },
            "required" => ["location"]
          }
        }
      ]

      messages = [Message.new_user!("What's the weather?")]
      result = HarmonyRenderer.render_conversation(messages, tools)

      assert String.contains?(
               result,
               "Calls to these tools must go to the commentary channel: 'functions'"
             )
    end

    test "customizes reasoning effort and dates" do
      messages = [Message.new_user!("Test")]

      options = %{
        reasoning_effort: :high,
        knowledge_cutoff: "2025-01",
        current_date: "2025-10-29"
      }

      result = HarmonyRenderer.render_conversation(messages, [], options)

      assert String.contains?(result, "Reasoning: High")
      assert String.contains?(result, "Knowledge cutoff: 2025-01")
      assert String.contains?(result, "Current date: 2025-10-29")
    end
  end

  describe "build_system_message/2" do
    test "builds basic system message" do
      opts = %{reasoning_effort: :medium, knowledge_cutoff: "2024-06", current_date: "2024-10-29"}
      result = HarmonyRenderer.build_system_message(opts, [])

      assert String.starts_with?(result, "<|start|>system<|message|>")
      assert String.contains?(result, "Reasoning: Medium")
      assert String.contains?(result, "Knowledge cutoff: 2024-06")
      assert String.contains?(result, "Current date: 2024-10-29")
      assert String.ends_with?(result, "<|end|>")
    end

    test "includes tool names in system message" do
      tools = [
        %Function{name: "tool1"},
        %Function{name: "tool2"}
      ]

      opts = %{reasoning_effort: :low}
      result = HarmonyRenderer.build_system_message(opts, tools)

      assert String.contains?(
               result,
               "Calls to these tools must go to the commentary channel: 'functions'"
             )
    end
  end

  describe "message_to_harmony/1" do
    test "converts system message" do
      msg = Message.new_system!("System prompt")
      result = HarmonyRenderer.message_to_harmony(msg)

      assert result == "<|start|>system<|message|>System prompt<|end|>"
    end

    test "converts user message" do
      msg = Message.new_user!("User input")
      result = HarmonyRenderer.message_to_harmony(msg)

      assert result == "<|start|>user<|message|>User input<|end|>"
    end

    test "converts assistant message" do
      msg = Message.new_assistant!("Assistant response")
      result = HarmonyRenderer.message_to_harmony(msg)

      assert result == "<|start|>assistant<|channel|>final<|message|>Assistant response<|end|>"
    end

    test "converts tool message" do
      # Create a tool result message manually since new_tool! may not exist
      msg = %LangChain.Message{
        role: :tool,
        content: [%{type: :text, content: "Tool result"}]
      }

      result = HarmonyRenderer.message_to_harmony(msg)

      assert result ==
               "<|start|>functions.tool to=assistant<|channel|>commentary<|message|>Tool result<|end|>"
    end

    test "handles tool_results with ContentPart content (replicates crash)" do
      # This replicates the crash when chain auto-executes unknown tools
      tool_result = %LangChain.Message.ToolResult{
        type: :function,
        tool_call_id: "call_123",
        name: "unknown",
        content: [
          %LangChain.Message.ContentPart{
            type: :text,
            content: "Tool call made to unknown but tool not found",
            options: []
          }
        ],
        is_error: true
      }

      msg = %LangChain.Message{
        role: :tool,
        tool_results: [tool_result]
      }

      # This should not crash
      result = HarmonyRenderer.message_to_harmony(msg)

      # Should render the error message as text
      expected =
        "<|start|>functions.unknown to=assistant<|channel|>commentary<|message|>Tool call made to unknown but tool not found<|end|>"

      assert result == expected
    end
  end

  describe "build_functions_section/1" do
    test "renders empty tools" do
      assert HarmonyRenderer.build_functions_section([]) == ""
    end

    test "renders tools with parameters" do
      tools = [
        %Function{
          name: "get_weather",
          description: "Gets the current weather",
          parameters_schema: %{
            "type" => "object",
            "properties" => %{
              "location" => %{"type" => "string", "description" => "The city name"},
              "unit" => %{"type" => "string", "enum" => ["celsius", "fahrenheit"]}
            },
            "required" => ["location"]
          }
        }
      ]

      result = HarmonyRenderer.build_functions_section(tools)

      assert String.contains?(result, "## functions")
      assert String.contains?(result, "namespace functions {")
      assert String.contains?(result, "// Gets the current weather")
      assert String.contains?(result, "location: string")
      assert String.contains?(result, "unit?: \"celsius\" | \"fahrenheit\"")
    end
  end

  describe "build_type_definition/2" do
    test "builds type definition for object with properties" do
      name = "test_function"

      schema = %{
        "type" => "object",
        "properties" => %{
          "param1" => %{"type" => "string", "description" => "A string param"},
          "param2" => %{"type" => "number"}
        },
        "required" => ["param1"]
      }

      result = HarmonyRenderer.build_type_definition(name, schema)

      assert String.contains?(result, "type test_function = (_: {")
      assert String.contains?(result, "param1: string")
      assert String.contains?(result, "param2?: number")
    end

    test "handles array types" do
      name = "test_function"

      schema = %{
        "type" => "object",
        "properties" => %{
          "items" => %{"type" => "array", "items" => %{"type" => "string"}}
        }
      }

      result = HarmonyRenderer.build_type_definition(name, schema)

      assert String.contains?(result, "items?: string[]")
    end
  end
end
