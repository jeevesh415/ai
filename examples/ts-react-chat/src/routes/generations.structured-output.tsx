import { useRef, useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'

const SAMPLE_PROMPT =
  'I play indie rock and have a $1500 budget. Recommend two electric guitars and one acoustic to round out my rig.'

type Provider = 'openai' | 'grok' | 'groq' | 'openrouter'

const PROVIDER_MODELS: Record<
  Provider,
  Array<{ value: string; label: string }>
> = {
  openai: [
    { value: 'gpt-4o', label: 'gpt-4o' },
    { value: 'gpt-4o-mini', label: 'gpt-4o-mini' },
  ],
  grok: [
    { value: 'grok-3', label: 'grok-3' },
    { value: 'grok-4-0709', label: 'grok-4-0709' },
  ],
  groq: [
    { value: 'llama-3.3-70b-versatile', label: 'llama-3.3-70b-versatile' },
    { value: 'llama-3.1-8b-instant', label: 'llama-3.1-8b-instant' },
  ],
  openrouter: [
    { value: 'openai/gpt-5.2', label: 'OpenRouter / GPT-5.2' },
    { value: 'openai/gpt-5.1', label: 'OpenRouter / GPT-5.1' },
    { value: 'anthropic/claude-sonnet-4.6', label: 'OpenRouter / Sonnet 4.6' },
    { value: 'x-ai/grok-4.1-fast', label: 'OpenRouter / Grok 4.1 Fast' },
  ],
}

interface RecommendationResult {
  title: string
  summary: string
  recommendations: Array<{
    name: string
    brand: string
    type: 'acoustic' | 'electric' | 'bass' | 'classical'
    priceRangeUsd: { min: number; max: number }
    reason: string
  }>
  nextSteps: Array<string>
}

interface StreamChunk {
  type: string
  delta?: string
  content?: string
  name?: string
  value?: { object?: unknown; raw?: string; reasoning?: string }
  message?: string
}

function StructuredOutputPage() {
  const [prompt, setPrompt] = useState(SAMPLE_PROMPT)
  const [provider, setProvider] = useState<Provider>('openrouter')
  const [model, setModel] = useState<string>(
    PROVIDER_MODELS.openrouter[0].value,
  )
  const [stream, setStream] = useState(true)
  const [result, setResult] = useState<RecommendationResult | null>(null)
  const [streamingText, setStreamingText] = useState<string>('')
  const [deltaCount, setDeltaCount] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const onProviderChange = (next: Provider) => {
    setProvider(next)
    setModel(PROVIDER_MODELS[next][0].value)
  }

  const handleGenerate = async () => {
    if (!prompt.trim()) return
    setIsLoading(true)
    setError(null)
    setResult(null)
    setStreamingText('')
    setDeltaCount(0)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const response = await fetch('/api/structured-output', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          provider,
          model,
          stream,
        }),
        signal: controller.signal,
      })

      if (!response.ok) {
        const errPayload = await response.json().catch(() => ({}))
        throw new Error(
          errPayload.error || `Request failed (${response.status})`,
        )
      }

      if (!stream) {
        const payload = await response.json()
        setResult(payload.data as RecommendationResult)
        return
      }

      // Streaming path: parse SSE, accumulate text deltas live, and capture
      // the terminal `structured-output.complete` CUSTOM event.
      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let accumulated = ''
      let deltas = 0

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        // SSE frames are separated by "\n\n"
        let sepIdx = buffer.indexOf('\n\n')
        while (sepIdx !== -1) {
          const frame = buffer.slice(0, sepIdx)
          buffer = buffer.slice(sepIdx + 2)
          sepIdx = buffer.indexOf('\n\n')

          for (const line of frame.split('\n')) {
            if (!line.startsWith('data: ')) continue
            const json = line.slice(6).trim()
            if (!json) continue
            let chunk: StreamChunk
            try {
              chunk = JSON.parse(json) as StreamChunk
            } catch {
              continue
            }

            if (chunk.type === 'TEXT_MESSAGE_CONTENT' && chunk.delta) {
              accumulated += chunk.delta
              deltas += 1
              setStreamingText(accumulated)
              setDeltaCount(deltas)
            } else if (
              chunk.type === 'CUSTOM' &&
              chunk.name === 'structured-output.complete' &&
              chunk.value?.object
            ) {
              setResult(chunk.value.object as RecommendationResult)
            } else if (chunk.type === 'RUN_ERROR') {
              throw new Error(chunk.message || 'Stream failed')
            }
          }
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        setError('Aborted')
      } else {
        setError(err instanceof Error ? err.message : 'Unknown error')
      }
    } finally {
      setIsLoading(false)
      abortRef.current = null
    }
  }

  const handleAbort = () => abortRef.current?.abort()

  return (
    <div className="flex flex-col h-[calc(100vh-72px)] bg-gray-900 text-white">
      <div className="border-b border-orange-500/20 bg-gray-800 px-6 py-4">
        <h2 className="text-xl font-semibold">Structured Output</h2>
        <p className="text-sm text-gray-400 mt-1">
          Calls <code className="text-orange-400">chat()</code> with an{' '}
          <code className="text-orange-400">outputSchema</code>. Toggle{' '}
          <code className="text-orange-400">stream</code> to exercise{' '}
          <code className="text-orange-400">structuredOutputStream</code> on the
          selected provider; deltas render live while the final{' '}
          <code className="text-orange-400">structured-output.complete</code>{' '}
          event populates the parsed result.
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-2xl mx-auto space-y-6">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <label className="text-sm text-gray-400">Provider</label>
              <select
                value={provider}
                onChange={(e) => onProviderChange(e.target.value as Provider)}
                disabled={isLoading}
                className="w-full rounded-lg border border-orange-500/20 bg-gray-800/50 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-orange-500/50 disabled:opacity-50"
              >
                <option value="openai">OpenAI</option>
                <option value="grok">Grok (xAI)</option>
                <option value="groq">Groq</option>
                <option value="openrouter">OpenRouter</option>
              </select>
            </div>
            <div className="space-y-2">
              <label className="text-sm text-gray-400">Model</label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={isLoading}
                className="w-full rounded-lg border border-orange-500/20 bg-gray-800/50 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-orange-500/50 disabled:opacity-50"
              >
                {PROVIDER_MODELS[provider].map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <label className="flex items-center gap-2 text-sm text-gray-300">
            <input
              type="checkbox"
              checked={stream}
              onChange={(e) => setStream(e.target.checked)}
              disabled={isLoading}
              className="accent-orange-500"
            />
            Stream (single-request{' '}
            <code className="text-orange-400">stream: true</code> +{' '}
            <code className="text-orange-400">
              response_format: json_schema
            </code>
            )
          </label>

          <div className="space-y-3">
            <label className="text-sm text-gray-400">Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe what you want recommendations for..."
              className="w-full rounded-lg border border-orange-500/20 bg-gray-800/50 px-4 py-3 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-orange-500/50 resize-none"
              rows={6}
              disabled={isLoading}
            />
          </div>

          <div className="flex gap-3">
            <button
              onClick={handleGenerate}
              disabled={!prompt.trim() || isLoading}
              className="px-6 py-2 bg-orange-600 hover:bg-orange-700 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg text-sm font-medium transition-colors"
            >
              {isLoading
                ? stream
                  ? 'Streaming...'
                  : 'Generating...'
                : 'Generate'}
            </button>
            {isLoading && stream && (
              <button
                onClick={handleAbort}
                className="px-6 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm font-medium transition-colors"
              >
                Abort
              </button>
            )}
            {(result || streamingText) && !isLoading && (
              <button
                onClick={() => {
                  setResult(null)
                  setStreamingText('')
                  setDeltaCount(0)
                }}
                className="px-6 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm font-medium transition-colors"
              >
                Clear
              </button>
            )}
          </div>

          {error && (
            <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}

          {streamingText && !result && (
            <div className="p-4 bg-gray-800/30 border border-gray-700/50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs text-gray-400 uppercase tracking-wider">
                  Streaming JSON
                </p>
                <p className="text-xs text-orange-400">{deltaCount} deltas</p>
              </div>
              <pre className="text-xs text-gray-300 whitespace-pre-wrap break-words">
                {streamingText}
              </pre>
            </div>
          )}

          {result && (
            <div className="space-y-4">
              {stream && deltaCount > 0 && (
                <p className="text-xs text-gray-500">
                  Reassembled from {deltaCount} streamed deltas.
                </p>
              )}
              <div className="p-4 bg-gray-800/50 border border-gray-700 rounded-lg">
                <h3 className="text-lg font-semibold text-white">
                  {result.title}
                </h3>
                <p className="text-gray-300 mt-2 text-sm">{result.summary}</p>
              </div>

              <div className="space-y-3">
                {result.recommendations.map((rec, i) => (
                  <div
                    key={i}
                    className="p-4 bg-gray-800/50 border border-gray-700 rounded-lg"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-white font-medium">
                          {rec.brand} {rec.name}
                        </p>
                        <p className="text-xs text-orange-400 uppercase tracking-wider mt-0.5">
                          {rec.type}
                        </p>
                      </div>
                      <p className="text-sm text-gray-400 whitespace-nowrap">
                        ${rec.priceRangeUsd.min} – ${rec.priceRangeUsd.max}
                      </p>
                    </div>
                    <p className="text-sm text-gray-300 mt-2">{rec.reason}</p>
                  </div>
                ))}
              </div>

              {result.nextSteps.length > 0 && (
                <div className="p-4 bg-gray-800/50 border border-gray-700 rounded-lg">
                  <p className="text-sm text-gray-400 mb-2">Next Steps</p>
                  <ul className="list-disc list-inside text-sm text-gray-200 space-y-1">
                    {result.nextSteps.map((step, i) => (
                      <li key={i}>{step}</li>
                    ))}
                  </ul>
                </div>
              )}

              <details className="p-4 bg-gray-800/30 border border-gray-700/50 rounded-lg">
                <summary className="text-sm text-gray-400 cursor-pointer">
                  Raw JSON
                </summary>
                <pre className="text-xs text-gray-300 mt-3 overflow-x-auto">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </details>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export const Route = createFileRoute('/generations/structured-output')({
  component: StructuredOutputPage,
})
