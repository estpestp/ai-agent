# AI Agent Server

A small Flask JSON API for a local Python client.

## Endpoints
- `GET /health`: health status and whether an AI provider key is configured.
- `POST /api/ask`: accepts `{"message":"...", "page":{"title":"...", "url":"...", "summary":"..."}}`, requires `Authorization: Bearer <API_TOKEN>`, and returns JSON.

## Environment variables
- `API_TOKEN` (required): shared secret used by the local Python client.
- `AI_API_KEY` (optional until an AI provider is chosen): API key for an OpenAI-compatible chat-completions endpoint.
- `AI_BASE_URL` (optional): defaults to `https://api.openai.com/v1`.
- `AI_MODEL` (optional): defaults to `gpt-4o-mini`.

The API can be deployed without an AI key to verify request/response wiring. It will clearly say that AI is not configured rather than pretending to generate a model answer. Configure an actual provider key before using AI responses. Never commit API keys to this repository or send private browser data unless necessary.
