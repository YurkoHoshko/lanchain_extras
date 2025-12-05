defmodule LangchainExtras.MixProject do
  use Mix.Project

  def project do
    [
      app: :langchain_extras,
      version: "0.1.0",
      elixir: "~> 1.18",
      build_path: "../_build",
      deps_path: "../deps",
      lockfile: "../mix.lock",
      start_permanent: Mix.env() == :prod,
      description: "LangChain adapters for llama.cpp Harmony and Qwen Jinja chat formats",
      deps: deps(),
      package: package(),
      docs: docs()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:langchain, "~> 0.4.0"},
      {:ecto, "~> 3.10"},
      {:req, "~> 0.5"},
      {:jason, "~> 1.4"},
      {:nimble_parsec, "~> 1.4"},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:bypass, "~> 2.1", only: :test}
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      files: ~w(lib mix.exs README.md LICENSE CHANGELOG.md),
      links: %{}
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: ["README.md"],
      source_ref: "main"
    ]
  end
end
