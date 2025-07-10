# 🖥️ MCP サーバー

このディレクトリには、医療対話プラットフォーム（MCP）で使用される、さまざまな大規模言語モデル（LLM）のサーバーサイドの実装が含まれています。

各サーバーは、特定のLLMと通信し、FHIR形式のデータを処理するように設計されています。

## 📂 ディレクトリ構成

- **gemini/**: GoogleのGeminiモデルを使用するサーバーです。
- **medgemma/**: GoogleのMedGemmaモデルを使用するサーバーです。
- **medgemma-bnb/**: 4-bit NormalFloat (BNB) で量子化されたMedGemmaモデルを使用するサーバーです。
- **sample/**: 開発やテストのためのサンプルサーバーです。

## 🐳 Docker

各サーバーは、Dockerfileを持っており、Dockerコンテナとして独立してビルド・実行することができます。
`docker-compose.yml` を使うことで、すべてのサーバーを一度に起動することも可能です。
