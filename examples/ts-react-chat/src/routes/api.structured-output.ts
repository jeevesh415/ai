import { createFileRoute } from '@tanstack/react-router'
import { chat, toServerSentEventsResponse } from '@tanstack/ai'
import { openaiText } from '@tanstack/ai-openai'
import { grokText } from '@tanstack/ai-grok'
import { groqText } from '@tanstack/ai-groq'
import { openRouterText } from '@tanstack/ai-openrouter'
import { z } from 'zod'
import type { AnyTextAdapter, StreamChunk } from '@tanstack/ai'

const GuitarRecommendationSchema = z.object({
  title: z.string().describe('Short headline for the recommendation'),
  summary: z.string().describe('One paragraph summary'),
  recommendations: z
    .array(
      z.object({
        name: z.string(),
        brand: z.string(),
        type: z.enum(['acoustic', 'electric', 'bass', 'classical']),
        priceRangeUsd: z.object({ min: z.number(), max: z.number() }),
        reason: z.string(),
      }),
    )
    .min(1)
    .describe('Guitar recommendations with reasons'),
  nextSteps: z.array(z.string()).describe('Practical follow-up actions'),
})

type Provider = 'openai' | 'grok' | 'groq' | 'openrouter'

function adapterFor(provider: Provider, model?: string): AnyTextAdapter {
  switch (provider) {
    case 'openai':
      return openaiText((model || 'gpt-4o') as 'gpt-4o')
    case 'grok':
      return grokText((model || 'grok-3') as 'grok-3')
    case 'groq':
      return groqText(
        (model || 'llama-3.3-70b-versatile') as 'llama-3.3-70b-versatile',
      )
    case 'openrouter':
      return openRouterText((model || 'openai/gpt-5.2') as 'openai/gpt-5.2')
  }
}

export const Route = createFileRoute('/api/structured-output')({
  server: {
    handlers: {
      POST: async ({ request }) => {
        const body = await request.json()
        const { prompt, provider, model, stream } = body as {
          prompt: string
          provider?: Provider
          model?: string
          stream?: boolean
        }
        const resolvedProvider: Provider = provider || 'openrouter'

        try {
          if (stream) {
            const abortController = new AbortController()
            request.signal.addEventListener('abort', () =>
              abortController.abort(),
            )
            const streamIterable = chat({
              adapter: adapterFor(resolvedProvider, model),
              messages: [{ role: 'user', content: prompt }],
              outputSchema: GuitarRecommendationSchema,
              stream: true,
              abortController,
            }) as AsyncIterable<StreamChunk>
            return toServerSentEventsResponse(streamIterable, {
              abortController,
            })
          }

          const result = await chat({
            adapter: adapterFor(resolvedProvider, model),
            messages: [{ role: 'user', content: prompt }],
            outputSchema: GuitarRecommendationSchema,
          })

          return new Response(JSON.stringify({ data: result }), {
            headers: { 'Content-Type': 'application/json' },
          })
        } catch (error: unknown) {
          const message =
            error instanceof Error ? error.message : 'An error occurred'
          console.error('[api/structured-output] Error:', error)
          return new Response(JSON.stringify({ error: message }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' },
          })
        }
      },
    },
  },
})
