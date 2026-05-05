import {
  describe,
  it,
  expect,
  vi,
  afterEach,
  beforeEach,
  type Mock,
} from 'vitest'
import { resolveDebugOption } from '@tanstack/ai/adapter-internals'
import { createGroqText, groqText } from '../src/adapters/text'
import type { StreamChunk, Tool } from '@tanstack/ai'

// Test helper: a silent logger for test chatStream calls.
const testLogger = resolveDebugOption(false)

// Declare mockCreate at module level
let mockCreate: Mock<(...args: Array<unknown>) => unknown>

// Mock the Groq SDK
vi.mock('groq-sdk', () => {
  class APIUserAbortError extends Error {
    constructor() {
      super('Request aborted')
      this.name = 'APIUserAbortError'
    }
  }
  return {
    default: class {
      chat = {
        completions: {
          create: (...args: Array<unknown>) => mockCreate(...args),
        },
      }
    },
    APIUserAbortError,
  }
})

// Helper to create async iterable from chunks
function createAsyncIterable<T>(chunks: Array<T>): AsyncIterable<T> {
  return {
    [Symbol.asyncIterator]() {
      let index = 0
      return {
        async next() {
          if (index < chunks.length) {
            return { value: chunks[index++]!, done: false }
          }
          return { value: undefined as T, done: true }
        },
      }
    },
  }
}

// Helper to setup the mock SDK client for streaming responses
function setupMockSdkClient(
  streamChunks: Array<Record<string, unknown>>,
  nonStreamResponse?: Record<string, unknown>,
) {
  mockCreate = vi.fn().mockImplementation((params) => {
    if (params.stream) {
      return Promise.resolve(createAsyncIterable(streamChunks))
    }
    return Promise.resolve(nonStreamResponse)
  })
}

const weatherTool: Tool = {
  name: 'lookup_weather',
  description: 'Return the forecast for a location',
}

describe('Groq adapters', () => {
  afterEach(() => {
    vi.unstubAllEnvs()
  })

  describe('Text adapter', () => {
    it('creates a text adapter with explicit API key', () => {
      const adapter = createGroqText('llama-3.3-70b-versatile', 'test-api-key')

      expect(adapter).toBeDefined()
      expect(adapter.kind).toBe('text')
      expect(adapter.name).toBe('groq')
      expect(adapter.model).toBe('llama-3.3-70b-versatile')
    })

    it('creates a text adapter from environment variable', () => {
      vi.stubEnv('GROQ_API_KEY', 'env-api-key')

      const adapter = groqText('llama-3.1-8b-instant')

      expect(adapter).toBeDefined()
      expect(adapter.kind).toBe('text')
      expect(adapter.model).toBe('llama-3.1-8b-instant')
    })

    it('throws if GROQ_API_KEY is not set when using groqText', () => {
      vi.stubEnv('GROQ_API_KEY', '')

      expect(() => groqText('llama-3.3-70b-versatile')).toThrow(
        'GROQ_API_KEY is required',
      )
    })

    it('allows custom baseURL override', () => {
      const adapter = createGroqText(
        'llama-3.3-70b-versatile',
        'test-api-key',
        {
          baseURL: 'https://custom.api.example.com/v1',
        },
      )

      expect(adapter).toBeDefined()
    })
  })
})

describe('Groq AG-UI event emission', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.unstubAllEnvs()
  })

  it('emits RUN_STARTED as the first event', async () => {
    const streamChunks = [
      {
        id: 'chatcmpl-123',
        model: 'llama-3.3-70b-versatile',
        choices: [
          {
            delta: { content: 'Hello' },
            finish_reason: null,
          },
        ],
      },
      {
        id: 'chatcmpl-123',
        model: 'llama-3.3-70b-versatile',
        choices: [
          {
            delta: {},
            finish_reason: 'stop',
          },
        ],
        x_groq: {
          usage: {
            prompt_tokens: 5,
            completion_tokens: 1,
            total_tokens: 6,
          },
        },
      },
    ]

    setupMockSdkClient(streamChunks)
    const adapter = createGroqText('llama-3.3-70b-versatile', 'test-api-key')
    const chunks: Array<StreamChunk> = []

    for await (const chunk of adapter.chatStream({
      model: 'llama-3.3-70b-versatile',
      messages: [{ role: 'user', content: 'Hello' }],
      logger: testLogger,
    })) {
      chunks.push(chunk)
    }

    expect(chunks[0]?.type).toBe('RUN_STARTED')
    if (chunks[0]?.type === 'RUN_STARTED') {
      expect(chunks[0].runId).toBeDefined()
      expect(chunks[0].model).toBe('llama-3.3-70b-versatile')
    }
  })

  it('emits TEXT_MESSAGE_START before TEXT_MESSAGE_CONTENT', async () => {
    const streamChunks = [
      {
        id: 'chatcmpl-123',
        model: 'llama-3.3-70b-versatile',
        choices: [
          {
            delta: { content: 'Hello' },
            finish_reason: null,
          },
        ],
      },
      {
        id: 'chatcmpl-123',
        model: 'llama-3.3-70b-versatile',
        choices: [
          {
            delta: {},
            finish_reason: 'stop',
          },
        ],
        x_groq: {
          usage: {
            prompt_tokens: 5,
            completion_tokens: 1,
            total_tokens: 6,
          },
        },
      },
    ]

    setupMockSdkClient(streamChunks)
    const adapter = createGroqText('llama-3.3-70b-versatile', 'test-api-key')
    const chunks: Array<StreamChunk> = []

    for await (const chunk of adapter.chatStream({
      model: 'llama-3.3-70b-versatile',
      messages: [{ role: 'user', content: 'Hello' }],
      logger: testLogger,
    })) {
      chunks.push(chunk)
    }

    const textStartIndex = chunks.findIndex(
      (c) => c.type === 'TEXT_MESSAGE_START',
    )
    const textContentIndex = chunks.findIndex(
      (c) => c.type === 'TEXT_MESSAGE_CONTENT',
    )

    expect(textStartIndex).toBeGreaterThan(-1)
    expect(textContentIndex).toBeGreaterThan(-1)
    expect(textStartIndex).toBeLessThan(textContentIndex)

    const textStart = chunks[textStartIndex]
    if (textStart?.type === 'TEXT_MESSAGE_START') {
      expect(textStart.messageId).toBeDefined()
      expect(textStart.role).toBe('assistant')
    }
  })

  it('emits TEXT_MESSAGE_END and RUN_FINISHED at the end', async () => {
    const streamChunks = [
      {
        id: 'chatcmpl-123',
        model: 'llama-3.3-70b-versatile',
        choices: [
          {
            delta: { content: 'Hello' },
            finish_reason: null,
          },
        ],
      },
      {
        id: 'chatcmpl-123',
        model: 'llama-3.3-70b-versatile',
        choices: [
          {
            delta: {},
            finish_reason: 'stop',
          },
        ],
        x_groq: {
          usage: {
            prompt_tokens: 5,
            completion_tokens: 1,
            total_tokens: 6,
          },
        },
      },
    ]

    setupMockSdkClient(streamChunks)
    const adapter = createGroqText('llama-3.3-70b-versatile', 'test-api-key')
    const chunks: Array<StreamChunk> = []

    for await (const chunk of adapter.chatStream({
      model: 'llama-3.3-70b-versatile',
      messages: [{ role: 'user', content: 'Hello' }],
      logger: testLogger,
    })) {
      chunks.push(chunk)
    }

    const textEndChunk = chunks.find((c) => c.type === 'TEXT_MESSAGE_END')
    expect(textEndChunk).toBeDefined()
    if (textEndChunk?.type === 'TEXT_MESSAGE_END') {
      expect(textEndChunk.messageId).toBeDefined()
    }

    const runFinishedChunk = chunks.find((c) => c.type === 'RUN_FINISHED')
    expect(runFinishedChunk).toBeDefined()
    if (runFinishedChunk?.type === 'RUN_FINISHED') {
      expect(runFinishedChunk.runId).toBeDefined()
      expect(runFinishedChunk.finishReason).toBe('stop')
      expect(runFinishedChunk.usage).toMatchObject({
        promptTokens: 5,
        completionTokens: 1,
        totalTokens: 6,
      })
    }
  })

  it('emits AG-UI tool call events', async () => {
    const streamChunks = [
      {
        id: 'chatcmpl-456',
        model: 'llama-3.3-70b-versatile',
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: 'call_abc123',
                  type: 'function',
                  function: {
                    name: 'lookup_weather',
                    arguments: '{"location":',
                  },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      },
      {
        id: 'chatcmpl-456',
        model: 'llama-3.3-70b-versatile',
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  function: {
                    arguments: '"Berlin"}',
                  },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      },
      {
        id: 'chatcmpl-456',
        model: 'llama-3.3-70b-versatile',
        choices: [
          {
            delta: {},
            finish_reason: 'tool_calls',
          },
        ],
        x_groq: {
          usage: {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
          },
        },
      },
    ]

    setupMockSdkClient(streamChunks)
    const adapter = createGroqText('llama-3.3-70b-versatile', 'test-api-key')
    const chunks: Array<StreamChunk> = []

    for await (const chunk of adapter.chatStream({
      model: 'llama-3.3-70b-versatile',
      messages: [{ role: 'user', content: 'Weather in Berlin?' }],
      tools: [weatherTool],
      logger: testLogger,
    })) {
      chunks.push(chunk)
    }

    // Check AG-UI tool events
    const toolStartChunk = chunks.find((c) => c.type === 'TOOL_CALL_START')
    expect(toolStartChunk).toBeDefined()
    if (toolStartChunk?.type === 'TOOL_CALL_START') {
      expect(toolStartChunk.toolCallId).toBe('call_abc123')
      expect(toolStartChunk.toolName).toBe('lookup_weather')
    }

    const toolArgsChunks = chunks.filter((c) => c.type === 'TOOL_CALL_ARGS')
    expect(toolArgsChunks.length).toBeGreaterThan(0)

    const toolEndChunk = chunks.find((c) => c.type === 'TOOL_CALL_END')
    expect(toolEndChunk).toBeDefined()
    if (toolEndChunk?.type === 'TOOL_CALL_END') {
      expect(toolEndChunk.toolCallId).toBe('call_abc123')
      expect(toolEndChunk.toolName).toBe('lookup_weather')
      expect(toolEndChunk.input).toEqual({ location: 'Berlin' })
    }

    // Check finish reason
    const runFinishedChunk = chunks.find((c) => c.type === 'RUN_FINISHED')
    if (runFinishedChunk?.type === 'RUN_FINISHED') {
      expect(runFinishedChunk.finishReason).toBe('tool_calls')
    }
  })

  it('emits RUN_ERROR on stream error', async () => {
    const streamChunks = [
      {
        id: 'chatcmpl-123',
        model: 'llama-3.3-70b-versatile',
        choices: [
          {
            delta: { content: 'Hello' },
            finish_reason: null,
          },
        ],
      },
    ]

    // Create an async iterable that throws mid-stream
    const errorIterable = {
      [Symbol.asyncIterator]() {
        let index = 0
        return {
          async next() {
            if (index < streamChunks.length) {
              return { value: streamChunks[index++]!, done: false }
            }
            throw new Error('Stream interrupted')
          },
        }
      },
    }

    mockCreate = vi.fn().mockResolvedValue(errorIterable)

    const adapter = createGroqText('llama-3.3-70b-versatile', 'test-api-key')
    const chunks: Array<StreamChunk> = []

    for await (const chunk of adapter.chatStream({
      model: 'llama-3.3-70b-versatile',
      messages: [{ role: 'user', content: 'Hello' }],
      logger: testLogger,
    })) {
      chunks.push(chunk)
    }

    // Should emit RUN_ERROR
    const runErrorChunk = chunks.find((c) => c.type === 'RUN_ERROR')
    expect(runErrorChunk).toBeDefined()
    if (runErrorChunk?.type === 'RUN_ERROR') {
      expect(runErrorChunk.error.message).toBe('Stream interrupted')
    }
  })

  it('emits proper AG-UI event sequence', async () => {
    const streamChunks = [
      {
        id: 'chatcmpl-123',
        model: 'llama-3.3-70b-versatile',
        choices: [
          {
            delta: { content: 'Hello world' },
            finish_reason: null,
          },
        ],
      },
      {
        id: 'chatcmpl-123',
        model: 'llama-3.3-70b-versatile',
        choices: [
          {
            delta: {},
            finish_reason: 'stop',
          },
        ],
        x_groq: {
          usage: {
            prompt_tokens: 5,
            completion_tokens: 2,
            total_tokens: 7,
          },
        },
      },
    ]

    setupMockSdkClient(streamChunks)
    const adapter = createGroqText('llama-3.3-70b-versatile', 'test-api-key')
    const chunks: Array<StreamChunk> = []

    for await (const chunk of adapter.chatStream({
      model: 'llama-3.3-70b-versatile',
      messages: [{ role: 'user', content: 'Hello' }],
      logger: testLogger,
    })) {
      chunks.push(chunk)
    }

    // Verify proper AG-UI event sequence
    const eventTypes = chunks.map((c) => c.type)

    // Should start with RUN_STARTED
    expect(eventTypes[0]).toBe('RUN_STARTED')

    // Should have TEXT_MESSAGE_START before TEXT_MESSAGE_CONTENT
    const textStartIndex = eventTypes.indexOf('TEXT_MESSAGE_START')
    const textContentIndex = eventTypes.indexOf('TEXT_MESSAGE_CONTENT')
    expect(textStartIndex).toBeGreaterThan(-1)
    expect(textContentIndex).toBeGreaterThan(textStartIndex)

    // Should have TEXT_MESSAGE_END before RUN_FINISHED
    const textEndIndex = eventTypes.indexOf('TEXT_MESSAGE_END')
    const runFinishedIndex = eventTypes.indexOf('RUN_FINISHED')
    expect(textEndIndex).toBeGreaterThan(-1)
    expect(runFinishedIndex).toBeGreaterThan(textEndIndex)

    // Verify RUN_FINISHED has proper data
    const runFinishedChunk = chunks.find((c) => c.type === 'RUN_FINISHED')
    if (runFinishedChunk?.type === 'RUN_FINISHED') {
      expect(runFinishedChunk.finishReason).toBe('stop')
      expect(runFinishedChunk.usage).toBeDefined()
    }
  })

  it('streams content with correct accumulated values', async () => {
    const streamChunks = [
      {
        id: 'chatcmpl-stream',
        model: 'llama-3.3-70b-versatile',
        choices: [
          {
            delta: { content: 'Hello ' },
            finish_reason: null,
          },
        ],
      },
      {
        id: 'chatcmpl-stream',
        model: 'llama-3.3-70b-versatile',
        choices: [
          {
            delta: { content: 'world' },
            finish_reason: null,
          },
        ],
      },
      {
        id: 'chatcmpl-stream',
        model: 'llama-3.3-70b-versatile',
        choices: [
          {
            delta: {},
            finish_reason: 'stop',
          },
        ],
        x_groq: {
          usage: {
            prompt_tokens: 5,
            completion_tokens: 2,
            total_tokens: 7,
          },
        },
      },
    ]

    setupMockSdkClient(streamChunks)
    const adapter = createGroqText('llama-3.3-70b-versatile', 'test-api-key')
    const chunks: Array<StreamChunk> = []

    for await (const chunk of adapter.chatStream({
      model: 'llama-3.3-70b-versatile',
      messages: [{ role: 'user', content: 'Say hello' }],
      logger: testLogger,
    })) {
      chunks.push(chunk)
    }

    // Check TEXT_MESSAGE_CONTENT events have correct accumulated content
    const contentChunks = chunks.filter(
      (c) => c.type === 'TEXT_MESSAGE_CONTENT',
    )
    expect(contentChunks.length).toBe(2)

    const firstContent = contentChunks[0]
    if (firstContent?.type === 'TEXT_MESSAGE_CONTENT') {
      expect(firstContent.delta).toBe('Hello ')
      expect(firstContent.content).toBe('Hello ')
    }

    const secondContent = contentChunks[1]
    if (secondContent?.type === 'TEXT_MESSAGE_CONTENT') {
      expect(secondContent.delta).toBe('world')
      expect(secondContent.content).toBe('Hello world')
    }
  })
})

describe('Groq structuredOutputStream', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  const createAdapter = () =>
    createGroqText('llama-3.3-70b-versatile', 'test-api-key')

  it('issues a single streaming request with response_format json_schema and emits parsed object', async () => {
    const streamChunks = [
      {
        id: 'chatcmpl-stream-1',
        model: 'llama-3.3-70b-versatile',
        choices: [{ delta: { content: '{"name":"Ali' }, finish_reason: null }],
      },
      {
        id: 'chatcmpl-stream-1',
        model: 'llama-3.3-70b-versatile',
        choices: [
          { delta: { content: 'ce","age":30}' }, finish_reason: 'stop' },
        ],
        x_groq: {
          usage: {
            prompt_tokens: 5,
            completion_tokens: 9,
            total_tokens: 14,
          },
        },
      },
    ]

    setupMockSdkClient(streamChunks)
    const adapter = createAdapter()

    const outputSchema = {
      type: 'object',
      properties: {
        name: { type: 'string' },
        age: { type: 'number' },
      },
      required: ['name', 'age'],
    }

    const chunks: Array<StreamChunk> = []
    for await (const chunk of adapter.structuredOutputStream({
      chatOptions: {
        model: 'llama-3.3-70b-versatile',
        messages: [{ role: 'user', content: 'Give me a person' }],
        logger: testLogger,
      },
      outputSchema,
    })) {
      chunks.push(chunk)
    }

    expect(mockCreate).toHaveBeenCalledTimes(1)
    const [params] = mockCreate.mock.calls[0]! as Array<any>
    expect(params.stream).toBe(true)
    expect(params.response_format).toEqual({
      type: 'json_schema',
      json_schema: {
        name: 'structured_output',
        schema: expect.objectContaining({
          type: 'object',
          additionalProperties: false,
        }),
        strict: true,
      },
    })
    expect(params.tools).toBeUndefined()

    const types: Array<string> = chunks.map((c) => c.type)
    const idx = (t: string) => types.indexOf(t)
    expect(idx('RUN_STARTED')).toBeGreaterThanOrEqual(0)
    expect(idx('TEXT_MESSAGE_START')).toBeGreaterThan(idx('RUN_STARTED'))
    expect(idx('TEXT_MESSAGE_CONTENT')).toBeGreaterThan(
      idx('TEXT_MESSAGE_START'),
    )
    expect(idx('TEXT_MESSAGE_END')).toBeGreaterThan(idx('TEXT_MESSAGE_CONTENT'))
    expect(idx('CUSTOM')).toBeGreaterThan(idx('TEXT_MESSAGE_END'))
    expect(idx('RUN_FINISHED')).toBeGreaterThan(idx('CUSTOM'))

    const contentChunks = chunks.filter(
      (c): c is Extract<StreamChunk, { type: 'TEXT_MESSAGE_CONTENT' }> =>
        c.type === 'TEXT_MESSAGE_CONTENT',
    )
    expect(contentChunks).toHaveLength(2)
    expect(contentChunks[0]!.delta).toBe('{"name":"Ali')
    expect(contentChunks[1]!.delta).toBe('ce","age":30}')

    const customChunks = chunks.filter(
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
    const streamChunks = [
      {
        id: 'chatcmpl-bad',
        model: 'llama-3.3-70b-versatile',
        choices: [{ delta: { content: 'not json' }, finish_reason: 'stop' }],
        x_groq: {
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        },
      },
    ]

    setupMockSdkClient(streamChunks)
    const adapter = createAdapter()

    const chunks: Array<StreamChunk> = []
    for await (const chunk of adapter.structuredOutputStream({
      chatOptions: {
        model: 'llama-3.3-70b-versatile',
        messages: [{ role: 'user', content: 'Give me a person' }],
        logger: testLogger,
      },
      outputSchema: {
        type: 'object',
        properties: { name: { type: 'string' } },
        required: ['name'],
      },
    })) {
      chunks.push(chunk)
    }

    const errorChunks = chunks.filter((c) => c.type === 'RUN_ERROR')
    expect(errorChunks).toHaveLength(1)
    expect(errorChunks[0]).toMatchObject({
      type: 'RUN_ERROR',
      code: 'parse-error',
      message: expect.stringContaining('Failed to parse structured output'),
    })
    expect(chunks.filter((c) => c.type === 'CUSTOM')).toHaveLength(0)
  })

  it('emits empty-response RUN_ERROR when no content is streamed', async () => {
    const streamChunks = [
      {
        id: 'chatcmpl-empty',
        model: 'llama-3.3-70b-versatile',
        choices: [{ delta: { content: '' }, finish_reason: 'stop' }],
        x_groq: {
          usage: { prompt_tokens: 1, completion_tokens: 0, total_tokens: 1 },
        },
      },
    ]

    setupMockSdkClient(streamChunks)
    const adapter = createAdapter()

    const chunks: Array<StreamChunk> = []
    for await (const chunk of adapter.structuredOutputStream({
      chatOptions: {
        model: 'llama-3.3-70b-versatile',
        messages: [{ role: 'user', content: 'Give me a person' }],
        logger: testLogger,
      },
      outputSchema: {
        type: 'object',
        properties: { name: { type: 'string' } },
        required: ['name'],
      },
    })) {
      chunks.push(chunk)
    }

    const errorChunks = chunks.filter((c) => c.type === 'RUN_ERROR')
    expect(errorChunks).toHaveLength(1)
    expect(errorChunks[0]).toMatchObject({
      type: 'RUN_ERROR',
      code: 'empty-response',
    })
    expect(chunks.filter((c) => c.type === 'CUSTOM')).toHaveLength(0)
  })

  it('finalizes the run when upstream stream closes without finish_reason', async () => {
    const streamChunks = [
      {
        id: 'chatcmpl-trunc',
        model: 'llama-3.3-70b-versatile',
        choices: [
          { delta: { content: '{"name":"Alice"}' }, finish_reason: null },
        ],
      },
    ]

    setupMockSdkClient(streamChunks)
    const adapter = createAdapter()

    const chunks: Array<StreamChunk> = []
    for await (const chunk of adapter.structuredOutputStream({
      chatOptions: {
        model: 'llama-3.3-70b-versatile',
        messages: [{ role: 'user', content: 'Give me a person' }],
        logger: testLogger,
      },
      outputSchema: {
        type: 'object',
        properties: { name: { type: 'string' } },
        required: ['name'],
      },
    })) {
      chunks.push(chunk)
    }

    const customChunks = chunks.filter((c) => c.type === 'CUSTOM')
    expect(customChunks).toHaveLength(1)
    expect(customChunks[0]).toMatchObject({
      name: 'structured-output.complete',
      value: { object: { name: 'Alice' }, raw: '{"name":"Alice"}' },
    })
    expect(chunks.filter((c) => c.type === 'RUN_FINISHED')).toHaveLength(1)
    expect(chunks.filter((c) => c.type === 'RUN_ERROR')).toHaveLength(0)
  })

  it('terminates on iterator-thrown provider error without emitting RUN_FINISHED', async () => {
    // Groq surfaces in-stream provider errors as iterator throws (not inline
    // error chunks), so the test mirrors that contract.
    mockCreate = vi.fn().mockImplementation(() => {
      const chunks = [
        {
          id: 'chatcmpl-err',
          model: 'llama-3.3-70b-versatile',
          choices: [{ delta: { content: '{"name":"Al' }, finish_reason: null }],
        },
      ] as Array<unknown>
      return Promise.resolve({
        [Symbol.asyncIterator]() {
          let i = 0
          return {
            // eslint-disable-next-line @typescript-eslint/require-await
            async next() {
              if (i < chunks.length) return { value: chunks[i++], done: false }
              throw Object.assign(new Error('Upstream rate limit'), {
                code: '429',
              })
            },
          }
        },
      })
    })
    const adapter = createAdapter()

    const chunks: Array<StreamChunk> = []
    for await (const chunk of adapter.structuredOutputStream({
      chatOptions: {
        model: 'llama-3.3-70b-versatile',
        messages: [{ role: 'user', content: 'Give me a person' }],
        logger: testLogger,
      },
      outputSchema: {
        type: 'object',
        properties: { name: { type: 'string' } },
        required: ['name'],
      },
    })) {
      chunks.push(chunk)
    }

    const errorChunks = chunks.filter((c) => c.type === 'RUN_ERROR')
    expect(errorChunks).toHaveLength(1)
    expect(chunks.filter((c) => c.type === 'RUN_FINISHED')).toHaveLength(0)
    expect(chunks.filter((c) => c.type === 'CUSTOM')).toHaveLength(0)
  })

  it('transforms null values to undefined on the parsed object', async () => {
    const streamChunks = [
      {
        id: 'chatcmpl-null',
        model: 'llama-3.3-70b-versatile',
        choices: [
          {
            delta: { content: '{"name":"Alice","nickname":null}' },
            finish_reason: 'stop',
          },
        ],
        x_groq: {
          usage: { prompt_tokens: 5, completion_tokens: 9, total_tokens: 14 },
        },
      },
    ]

    setupMockSdkClient(streamChunks)
    const adapter = createAdapter()

    const chunks: Array<StreamChunk> = []
    for await (const chunk of adapter.structuredOutputStream({
      chatOptions: {
        model: 'llama-3.3-70b-versatile',
        messages: [{ role: 'user', content: 'Give me a person' }],
        logger: testLogger,
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
      chunks.push(chunk)
    }

    const customChunks = chunks.filter(
      (c): c is Extract<StreamChunk, { type: 'CUSTOM' }> => c.type === 'CUSTOM',
    )
    expect(customChunks).toHaveLength(1)
    // nickname becomes undefined (not present) after transform — matches the
    // non-streaming structuredOutput contract.
    const value = customChunks[0]!.value as { object: Record<string, unknown> }
    expect(value.object.name).toBe('Alice')
    expect(value.object.nickname).toBeUndefined()
    // Raw JSON preserves the original null
    expect((customChunks[0]!.value as { raw: string }).raw).toBe(
      '{"name":"Alice","nickname":null}',
    )
  })
})
