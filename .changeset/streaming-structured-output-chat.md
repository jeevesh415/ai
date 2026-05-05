---
'@tanstack/ai': minor
---

feat: `chat({ outputSchema, stream: true })` returns `AsyncIterable<StreamChunk>` with raw JSON deltas plus a final `CUSTOM` `structured-output.complete` event carrying the validated parsed object. The existing `chat({ outputSchema })` (non-streaming) path is unchanged. Adapters expose this via a new optional `structuredOutputStream` method on `TextAdapter`; `BaseTextAdapter` provides a default that wraps the non-streaming `structuredOutput` so adapters without native streaming JSON support still satisfy the new combination.
