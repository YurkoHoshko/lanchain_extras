# LangChain Extras

LangChain Extras packages llama.cpp integrations that speak LangChain's chat APIs. It includes:

- `LangChain.ChatModels.ChatLlamaHarmony` – Harmony prompt renderer/parser for llama.cpp `/completions`.
- `LangChain.ChatModels.ChatLlamaJinja` – Qwen 3 Jinja prompt renderer/parser with tool-call support.
- `LangChain.Utils.HarmonyRenderer` / `HarmonyParser` and `JinjaRenderer` / `JinjaParser`.

Use it alongside `:langchain` to add local llama.cpp backends without forking the upstream library.

## Installation

Until the package is published on Hex, point your app at this repo/path:

```elixir
def deps do
  [
    {:langchain, "~> 0.4.0"},
    {:langchain_extras, path: "../langchain_extras"} # or "./langchain_extras" inside a monorepo
  ]
end
```

## Usage

```elixir
alias LangChain.ChatModels.ChatLlamaHarmony
alias LangChain.Message

{:ok, llm} =
  ChatLlamaHarmony.new(%{
    endpoint: "http://localhost:8000/v1/completions",
    model: "llama-3.1-8b",
    stream: true
  })

messages = [
  Message.new_system!("You are a concise assistant."),
  Message.new_user!("Give me a fun fact about Elixir.")
]

{:ok, replies} = ChatLlamaHarmony.call(llm, messages)
IO.inspect(replies, label: "llama.cpp replies")
```

Swap in `ChatLlamaJinja` to use the Qwen 3 Jinja template; both models support tool calls and JSON-mode parsing.

## Developing

- `mix deps.get` – fetch dependencies.
- `mix test` – run the focused test suite for the package.
- `mix docs` – build local docs.
