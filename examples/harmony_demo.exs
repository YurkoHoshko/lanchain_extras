# Only install deps when run outside this project (e.g., `elixir examples/harmony_demo.exs`)
if not function_exported?(Mix.Project, :get, 0) or Mix.Project.get() == nil do
  Mix.install([
    {:langchain, "~> 0.4.0"},
    {:langchain_extras, path: "."}
  ])
end

defmodule HarmonyDemo do
  alias LangChain.ChatModels.ChatLlamaHarmony
  alias LangChain.Message
  alias LangChain.Chains.LLMChain
  alias LangChain.Utils.ChainResult

  def run do
    endpoint = System.get_env("LLAMA_SWAP_URL", "http://localhost:8000/completion")
    model = System.get_env("LLAMA_SWAP_MODEL", "gpt-oss-20b")

    {:ok, llm} =
      ChatLlamaHarmony.new(%{
        endpoint: endpoint,
        model: model,
        stream: false,
        max_tokens: 512
      })

    prompt = [
      Message.new_system!("You are a concise assistant."),
      Message.new_user!("Name three Ukrainian cities near the Dnipro river.")
    ]

    chain =
      LLMChain.new!(%{llm: llm, verbose: true})
      |> LLMChain.add_messages(prompt)

    case LLMChain.run(chain, mode: :while_needs_response) do
      {:ok, final_chain} ->
        IO.puts("\nConversation:")
        Enum.each(final_chain.messages, &render_message/1)

        IO.puts(color(:green) <> "\nAnswer:\n" <> color(:reset) <> ChainResult.to_string!(final_chain))

      {:error, reason} ->
        IO.puts(color(:red) <> "Error: " <> color(:reset) <> inspect(reason))
    end
  end

  defp render_message(%Message{role: :assistant, metadata: %{channel: "analysis"}} = msg),
    do: print(msg, "🧠", :blue, "analysis")

  defp render_message(%Message{role: :assistant} = msg), do: print(msg, "💬", :green, "final")
  defp render_message(%Message{role: :user} = msg), do: print(msg, "🙋", :cyan, "user")
  defp render_message(%Message{role: :tool} = msg), do: print(msg, "🛠️", :yellow, "tool")
  defp render_message(msg), do: print(msg, "…", :default, "other")

  defp print(%Message{} = msg, icon, color, label) do
    text =
      msg.content
      |> List.wrap()
      |> Enum.map_join("", fn
        %{content: c} -> c
        other -> inspect(other)
      end)
      |> String.trim()

    IO.puts("#{color(color)}#{icon} #{label}:#{color(:reset)} #{text}")
  end

  defp color(:blue), do: IO.ANSI.blue()
  defp color(:green), do: IO.ANSI.green()
  defp color(:cyan), do: IO.ANSI.cyan()
  defp color(:yellow), do: IO.ANSI.yellow()
  defp color(:red), do: IO.ANSI.red()
  defp color(:default), do: ""
  defp color(:reset), do: IO.ANSI.reset()
end

HarmonyDemo.run()
