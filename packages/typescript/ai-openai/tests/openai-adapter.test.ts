import { describe, it, expect, beforeEach, vi } from 'vitest'
import { chat, type Tool, type StreamChunk } from '@tanstack/ai'
import { OpenAITextAdapter } from '../src/adapters/text'
import type { OpenAITextProviderOptions } from '../src/adapters/text'

const createAdapter = <TModel extends 'gpt-4o-mini' | 'gpt-4o'>(
  model: TModel,
) => new OpenAITextAdapter({ apiKey: 'test-key' }, model)

const toolArguments = JSON.stringify({ location: 'Berlin' })

const weatherTool: Tool = {
  name: 'lookup_weather',
  description: 'Return the forecast for a location',
}

function createMockChatCompletionsStream(
  chunks: Array<Record<string, unknown>>,
): AsyncIterable<Record<string, unknown>> {
  return {
    async *[Symbol.asyncIterator]() {
      for (const chunk of chunks) {
        yield chunk
      }
    },
  }
}

describe('OpenAI adapter option mapping', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('maps options into the Responses API payload', async () => {
    // Mock the Responses API event stream format
    const mockStream = createMockChatCompletionsStream([
      {
        type: 'response.created',
        response: {
          id: 'resp-123',
          model: 'gpt-4o-mini',
          status: 'in_progress',
          created_at: 1234567890,
        },
      },
      {
        type: 'response.content_part.added',
        part: {
          type: 'output_text',
          text: 'It is sunny',
        },
      },
      {
        type: 'response.done',
        response: {
          id: 'resp-123',
          model: 'gpt-4o-mini',
          status: 'completed',
          created_at: 1234567891,
          usage: {
            input_tokens: 12,
            output_tokens: 4,
          },
        },
      },
    ])

    const responsesCreate = vi.fn().mockResolvedValueOnce(mockStream)

    const adapter = createAdapter('gpt-4o-mini')
    // Replace the internal OpenAI SDK client with our mock
    ;(adapter as any).client = {
      responses: {
        create: responsesCreate,
      },
    }

    const modelOptions: OpenAITextProviderOptions = {
      tool_choice: 'required',
    }

    const chunks: StreamChunk[] = []
    for await (const chunk of chat({
      adapter,
      messages: [
        { role: 'system', content: 'Stay concise' },
        { role: 'user', content: 'How is the weather?' },
        {
          role: 'assistant',
          content: 'Let me check',
          toolCalls: [
            {
              id: 'call_weather',
              type: 'function',
              function: { name: 'lookup_weather', arguments: toolArguments },
            },
          ],
        },
        { role: 'tool', toolCallId: 'call_weather', content: '{"temp":72}' },
      ],
      tools: [weatherTool],
      temperature: 0.25,
      topP: 0.6,
      maxTokens: 1024,
      metadata: { requestId: 'req-42' },
      modelOptions,
    })) {
      chunks.push(chunk)
    }

    expect(responsesCreate).toHaveBeenCalledTimes(1)
    const [payload] = responsesCreate.mock.calls[0]

    // Responses API uses different field names and structure
    expect(payload).toMatchObject({
      model: 'gpt-4o-mini',
      temperature: 0.25,
      top_p: 0.6,
      max_output_tokens: 1024, // Responses API uses max_output_tokens instead of max_tokens
      stream: true,
      tool_choice: 'required', // From modelOptions
    })

    // Responses API uses 'input' instead of 'messages'
    expect(payload.input).toBeDefined()

    // Verify tools are included
    expect(payload.tools).toBeDefined()
    expect(Array.isArray(payload.tools)).toBe(true)
    expect(payload.tools.length).toBeGreaterThan(0)
  })
})

describe('OpenAI structuredOutputStream', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  const setupAdapter = (
    chunks: Array<Record<string, unknown>>,
    overrides?: { responsesCreate?: ReturnType<typeof vi.fn> },
  ) => {
    const responsesCreate =
      overrides?.responsesCreate ??
      vi.fn().mockResolvedValueOnce(createMockChatCompletionsStream(chunks))
    const adapter = createAdapter('gpt-4o-mini')
    ;(adapter as any).client = {
      responses: { create: responsesCreate },
    }
    return { adapter, responsesCreate }
  }

  it('issues a single streaming Responses API request with text.format json_schema and emits parsed object', async () => {
    const chunks: Array<Record<string, unknown>> = [
      {
        type: 'response.created',
        response: {
          id: 'resp-1',
          model: 'gpt-4o-mini',
          status: 'in_progress',
          created_at: 1,
        },
      },
      {
        type: 'response.output_text.delta',
        delta: '{"name":"Ali',
      },
      {
        type: 'response.output_text.delta',
        delta: 'ce","age":30}',
      },
      {
        type: 'response.completed',
        response: {
          id: 'resp-1',
          model: 'gpt-4o-mini',
          status: 'completed',
          created_at: 2,
          output: [],
          usage: { input_tokens: 5, output_tokens: 9, total_tokens: 14 },
        },
      },
    ]

    const { adapter, responsesCreate } = setupAdapter(chunks)

    const outputSchema = {
      type: 'object',
      properties: {
        name: { type: 'string' },
        age: { type: 'number' },
      },
      required: ['name', 'age'],
    }

    const collected: Array<StreamChunk> = []
    for await (const chunk of adapter.structuredOutputStream({
      chatOptions: {
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'Give me a person' }],
        logger: (
          await import('@tanstack/ai/adapter-internals')
        ).resolveDebugOption(false),
      },
      outputSchema,
    })) {
      collected.push(chunk)
    }

    expect(responsesCreate).toHaveBeenCalledTimes(1)
    const [payload] = responsesCreate.mock.calls[0]
    expect(payload.stream).toBe(true)
    expect(payload.text).toEqual({
      format: {
        type: 'json_schema',
        name: 'structured_output',
        schema: expect.objectContaining({
          type: 'object',
          additionalProperties: false,
        }),
        strict: true,
      },
    })
    expect(payload.tools).toBeUndefined()

    const types: Array<string> = collected.map((c) => c.type)
    const idx = (t: string) => types.indexOf(t)
    expect(idx('RUN_STARTED')).toBeGreaterThanOrEqual(0)
    expect(idx('TEXT_MESSAGE_START')).toBeGreaterThan(idx('RUN_STARTED'))
    expect(idx('TEXT_MESSAGE_CONTENT')).toBeGreaterThan(
      idx('TEXT_MESSAGE_START'),
    )
    expect(idx('TEXT_MESSAGE_END')).toBeGreaterThan(idx('TEXT_MESSAGE_CONTENT'))
    expect(idx('CUSTOM')).toBeGreaterThan(idx('TEXT_MESSAGE_END'))
    expect(idx('RUN_FINISHED')).toBeGreaterThan(idx('CUSTOM'))

    const contentChunks = collected.filter(
      (c): c is Extract<StreamChunk, { type: 'TEXT_MESSAGE_CONTENT' }> =>
        c.type === 'TEXT_MESSAGE_CONTENT',
    )
    expect(contentChunks).toHaveLength(2)
    expect(contentChunks[0]!.delta).toBe('{"name":"Ali')
    expect(contentChunks[1]!.delta).toBe('ce","age":30}')

    const customChunks = collected.filter(
      (c): c is Extract<StreamChunk, { type: 'CUSTOM' }> => c.type === 'CUSTOM',
    )
    expect(customChunks).toHaveLength(1)
    expect(customChunks[0]!.name).toBe('structured-output.complete')
    expect(customChunks[0]!.value).toEqual({
      object: { name: 'Alice', age: 30 },
      raw: '{"name":"Alice","age":30}',
    })
  })

  it('emits RUN_ERROR when accumulated content is not valid JSON', async () => {
    const { resolveDebugOption } =
      await import('@tanstack/ai/adapter-internals')
    const chunks: Array<Record<string, unknown>> = [
      { type: 'response.output_text.delta', delta: 'not json' },
      {
        type: 'response.completed',
        response: {
          id: 'resp-bad',
          model: 'gpt-4o-mini',
          status: 'completed',
          created_at: 1,
          output: [],
          usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
        },
      },
    ]
    const { adapter } = setupAdapter(chunks)

    const collected: Array<StreamChunk> = []
    for await (const chunk of adapter.structuredOutputStream({
      chatOptions: {
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'Give me a person' }],
        logger: resolveDebugOption(false),
      },
      outputSchema: {
        type: 'object',
        properties: { name: { type: 'string' } },
        required: ['name'],
      },
    })) {
      collected.push(chunk)
    }

    const errors = collected.filter((c) => c.type === 'RUN_ERROR')
    expect(errors).toHaveLength(1)
    expect(errors[0]).toMatchObject({ code: 'parse-error' })
    expect(collected.filter((c) => c.type === 'CUSTOM')).toHaveLength(0)
  })

  it('emits empty-response RUN_ERROR when no content is streamed', async () => {
    const { resolveDebugOption } =
      await import('@tanstack/ai/adapter-internals')
    const chunks: Array<Record<string, unknown>> = [
      {
        type: 'response.completed',
        response: {
          id: 'resp-empty',
          model: 'gpt-4o-mini',
          status: 'completed',
          created_at: 1,
          output: [],
          usage: { input_tokens: 1, output_tokens: 0, total_tokens: 1 },
        },
      },
    ]
    const { adapter } = setupAdapter(chunks)

    const collected: Array<StreamChunk> = []
    for await (const chunk of adapter.structuredOutputStream({
      chatOptions: {
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'Give me a person' }],
        logger: resolveDebugOption(false),
      },
      outputSchema: {
        type: 'object',
        properties: { name: { type: 'string' } },
        required: ['name'],
      },
    })) {
      collected.push(chunk)
    }

    const errors = collected.filter((c) => c.type === 'RUN_ERROR')
    expect(errors).toHaveLength(1)
    expect(errors[0]).toMatchObject({ code: 'empty-response' })
    expect(collected.filter((c) => c.type === 'CUSTOM')).toHaveLength(0)
  })

  it('finalizes the run when upstream stream closes without response.completed', async () => {
    const { resolveDebugOption } =
      await import('@tanstack/ai/adapter-internals')
    const chunks: Array<Record<string, unknown>> = [
      { type: 'response.output_text.delta', delta: '{"name":"Alice"}' },
    ]
    const { adapter } = setupAdapter(chunks)

    const collected: Array<StreamChunk> = []
    for await (const chunk of adapter.structuredOutputStream({
      chatOptions: {
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'Give me a person' }],
        logger: resolveDebugOption(false),
      },
      outputSchema: {
        type: 'object',
        properties: { name: { type: 'string' } },
        required: ['name'],
      },
    })) {
      collected.push(chunk)
    }

    const customChunks = collected.filter((c) => c.type === 'CUSTOM')
    expect(customChunks).toHaveLength(1)
    expect(customChunks[0]).toMatchObject({
      name: 'structured-output.complete',
      value: { object: { name: 'Alice' }, raw: '{"name":"Alice"}' },
    })
    expect(collected.filter((c) => c.type === 'RUN_FINISHED')).toHaveLength(1)
    expect(collected.filter((c) => c.type === 'RUN_ERROR')).toHaveLength(0)
  })

  it('terminates on response.failed without emitting RUN_FINISHED', async () => {
    const { resolveDebugOption } =
      await import('@tanstack/ai/adapter-internals')
    const chunks: Array<Record<string, unknown>> = [
      { type: 'response.output_text.delta', delta: '{"name":"Al' },
      {
        type: 'response.failed',
        response: {
          id: 'resp-err',
          model: 'gpt-4o-mini',
          status: 'failed',
          created_at: 1,
          error: { message: 'Upstream rate limit', code: 'rate_limit' },
          output: [],
        },
      },
    ]
    const { adapter } = setupAdapter(chunks)

    const collected: Array<StreamChunk> = []
    for await (const chunk of adapter.structuredOutputStream({
      chatOptions: {
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'Give me a person' }],
        logger: resolveDebugOption(false),
      },
      outputSchema: {
        type: 'object',
        properties: { name: { type: 'string' } },
        required: ['name'],
      },
    })) {
      collected.push(chunk)
    }

    const errors = collected.filter((c) => c.type === 'RUN_ERROR')
    expect(errors).toHaveLength(1)
    expect(collected.filter((c) => c.type === 'RUN_FINISHED')).toHaveLength(0)
    expect(collected.filter((c) => c.type === 'CUSTOM')).toHaveLength(0)
  })

  it('transforms null values to undefined on the parsed object', async () => {
    const { resolveDebugOption } =
      await import('@tanstack/ai/adapter-internals')
    const chunks: Array<Record<string, unknown>> = [
      {
        type: 'response.output_text.delta',
        delta: '{"name":"Alice","nickname":null}',
      },
      {
        type: 'response.completed',
        response: {
          id: 'resp-null',
          model: 'gpt-4o-mini',
          status: 'completed',
          created_at: 1,
          output: [],
          usage: { input_tokens: 5, output_tokens: 9, total_tokens: 14 },
        },
      },
    ]
    const { adapter } = setupAdapter(chunks)

    const collected: Array<StreamChunk> = []
    for await (const chunk of adapter.structuredOutputStream({
      chatOptions: {
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'Give me a person' }],
        logger: resolveDebugOption(false),
      },
      outputSchema: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          nickname: { type: ['string', 'null'] },
        },
        required: ['name', 'nickname'],
      },
    })) {
      collected.push(chunk)
    }

    const customChunks = collected.filter(
      (c): c is Extract<StreamChunk, { type: 'CUSTOM' }> => c.type === 'CUSTOM',
    )
    expect(customChunks).toHaveLength(1)
    const value = customChunks[0]!.value as { object: Record<string, unknown> }
    expect(value.object.name).toBe('Alice')
    expect(value.object.nickname).toBeUndefined()
    expect((customChunks[0]!.value as { raw: string }).raw).toBe(
      '{"name":"Alice","nickname":null}',
    )
  })
})
