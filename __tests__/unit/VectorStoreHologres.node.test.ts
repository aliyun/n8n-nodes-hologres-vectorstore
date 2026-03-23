/**
 * Unit tests for VectorStoreHologres node
 * Tests execute() and supplyData() methods
 */

import { jest, describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { Document } from '@langchain/core/documents';
import { NodeConnectionTypes, NodeOperationError } from 'n8n-workflow';
import {
  VectorStoreHologres,
  processDocumentInput,
  getColumnOptions,
  getHGraphIndexSettings,
  getMetadataFiltersValues,
} from '../../nodes/VectorStoreHologres/VectorStoreHologres.node';
import {
  createMockExecuteContext,
  createMockSupplyDataContext,
  createMockInputData,
} from '../mocks/n8n-context.mock';
import { FakeEmbeddings } from '../mocks/embeddings.mock';
import {
  createMockHologresVectorStore,
  createMockInitialize,
  createMockPool,
} from '../mocks/hologres-store.mock';

// Mock HologresVectorStore module
jest.mock('../../nodes/VectorStoreHologres/HologresVectorStore', () => {
  const actual = jest.requireActual('../../nodes/VectorStoreHologres/HologresVectorStore');
  return {
    ...actual,
    HologresVectorStore: jest.fn().mockImplementation(() => createMockHologresVectorStore()),
    createPoolFromCredentials: jest.fn().mockImplementation(() => createMockPool()),
  };
});

// ─────────────────────────────────────────────────────────────────────────────
// processDocumentInput Tests
// ─────────────────────────────────────────────────────────────────────────────

describe('processDocumentInput', () => {
  it('should handle Document array input', async () => {
    const documents = [
      new Document({ pageContent: 'doc 1', metadata: { key: 'value1' } }),
      new Document({ pageContent: 'doc 2', metadata: { key: 'value2' } }),
    ];
    const inputItem = { json: {} };
    const itemIndex = 0;

    const result = await processDocumentInput(documents, inputItem as any, itemIndex);

    expect(result.processedDocuments).toEqual(documents);
    expect(result.serializedDocuments).toHaveLength(2);
    expect(result.serializedDocuments[0]).toEqual({
      json: { pageContent: 'doc 1', metadata: { key: 'value1' } },
      pairedItem: { item: 0 },
    });
  });

  it('should handle loader object with processItem', async () => {
    const documents = [new Document({ pageContent: 'loaded doc', metadata: {} })];
    const loader = {
      processItem: jest.fn().mockResolvedValue(documents),
    };
    const inputItem = { json: { data: 'test' } };
    const itemIndex = 1;

    const result = await processDocumentInput(loader, inputItem as any, itemIndex);

    expect(loader.processItem).toHaveBeenCalledWith(inputItem, itemIndex);
    expect(result.processedDocuments).toEqual(documents);
    expect(result.serializedDocuments).toHaveLength(1);
  });

  it('should throw error for unsupported type', async () => {
    const invalidInput = 'not a document or loader';
    const inputItem = { json: {} };
    const itemIndex = 0;

    await expect(
      processDocumentInput(invalidInput, inputItem as any, itemIndex),
    ).rejects.toThrow('Unsupported document input type');
  });

  it('should serialize documents with pairedItem', async () => {
    const documents = [
      new Document({ pageContent: 'doc', metadata: { source: 'test' } }),
    ];
    const inputItem = { json: { id: 123 } };
    const itemIndex = 5;

    const result = await processDocumentInput(documents, inputItem as any, itemIndex);

    expect(result.serializedDocuments[0].pairedItem).toEqual({ item: 5 });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions Tests
// ─────────────────────────────────────────────────────────────────────────────

describe('getColumnOptions', () => {
  it('should return column options from parameters', () => {
    const context = {
      getNodeParameter: jest.fn().mockReturnValue({
        idColumnName: 'custom_id',
        vectorColumnName: 'vec',
        contentColumnName: 'content',
        metadataColumnName: 'meta',
      }),
    };

    const result = getColumnOptions(context);

    expect(result).toEqual({
      idColumnName: 'custom_id',
      vectorColumnName: 'vec',
      contentColumnName: 'content',
      metadataColumnName: 'meta',
    });
  });

  it('should return default values when not specified', () => {
    const context = {
      getNodeParameter: jest.fn().mockReturnValue({
        idColumnName: 'id',
        vectorColumnName: 'embedding',
        contentColumnName: 'text',
        metadataColumnName: 'metadata',
      }),
    };

    const result = getColumnOptions(context);

    expect(result.idColumnName).toBe('id');
    expect(result.vectorColumnName).toBe('embedding');
  });
});

describe('getHGraphIndexSettings', () => {
  it('should return HGraph index settings from parameters', () => {
    const context = {
      getNodeParameter: jest.fn().mockReturnValue({
        baseQuantizationType: 'sq8',
        useReorder: false,
        maxDegree: 32,
        efConstruction: 200,
      }),
    };

    const result = getHGraphIndexSettings(context);

    expect(result.baseQuantizationType).toBe('sq8');
    expect(result.useReorder).toBe(false);
    expect(result.maxDegree).toBe(32);
    expect(result.efConstruction).toBe(200);
  });
});

describe('getMetadataFiltersValues', () => {
  it('should return undefined when no filter provided', () => {
    const context = {
      getNodeParameter: jest.fn().mockReturnValue({}),
    };

    const result = getMetadataFiltersValues(context as any, 0);

    expect(result).toBeUndefined();
  });

  it('should return metadata values from options', () => {
    const context = {
      getNodeParameter: jest.fn((name: string) => {
        if (name === 'options') {
          return {
            metadata: {
              metadataValues: [
                { name: 'category', value: 'tech' },
                { name: 'author', value: 'john' },
              ],
            },
          };
        }
        return {};
      }),
    };

    const result = getMetadataFiltersValues(context as any, 0);

    expect(result).toEqual({ category: 'tech', author: 'john' });
  });

  it('should return filter from searchFilterJson', () => {
    const context = {
      getNodeParameter: jest.fn((name: string, _index: number, fallback?: unknown, options?: any) => {
        if (name === 'options') {
          return { searchFilterJson: '{"status":"active"}' };
        }
        if (name === 'options.searchFilterJson') {
          return { status: 'active' };
        }
        return fallback;
      }),
    };

    const result = getMetadataFiltersValues(context as any, 0);

    expect(result).toEqual({ status: 'active' });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// execute() Tests - load mode
// ─────────────────────────────────────────────────────────────────────────────

describe('execute() - load mode', () => {
  let node: VectorStoreHologres;
  let mockStore: ReturnType<typeof createMockHologresVectorStore>;
  let embeddings: FakeEmbeddings;

  beforeEach(() => {
    jest.clearAllMocks();
    node = new VectorStoreHologres();
    mockStore = createMockHologresVectorStore();
    embeddings = new FakeEmbeddings(4);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('should query and return serialized results', async () => {
    const context = createMockExecuteContext(
      {
        mode: 'load',
        tableName: 'test_table',
        prompt: 'test query',
        topK: 4,
        includeDocumentMetadata: true,
        options: {},
      },
      createMockInputData([{}]),
    );

    // Mock embeddings
    context.getInputConnectionData = jest.fn().mockResolvedValue(embeddings);

    // Mock the HologresVectorStore constructor
    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    const result = await node.execute.call(context);

    expect(mockStore._initializeClient).toHaveBeenCalled();
    expect(mockStore.similaritySearchVectorWithScore).toHaveBeenCalled();
    expect(mockStore.client.release).toHaveBeenCalled();
    // Pool is now managed externally and closed once at the end
    expect(result[0]).toHaveLength(2);
  });

  it('should include metadata when flag is true', async () => {
    const context = createMockExecuteContext(
      {
        mode: 'load',
        tableName: 'test_table',
        prompt: 'test query',
        topK: 4,
        includeDocumentMetadata: true,
        options: {},
      },
      createMockInputData([{}]),
    );

    context.getInputConnectionData = jest.fn().mockResolvedValue(embeddings);

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    const result = await node.execute.call(context);

    expect(result[0][0].json.document).toHaveProperty('metadata');
  });

  it('should exclude metadata when flag is false', async () => {
    const context = createMockExecuteContext(
      {
        mode: 'load',
        tableName: 'test_table',
        prompt: 'test query',
        topK: 4,
        includeDocumentMetadata: false,
        options: {},
      },
      createMockInputData([{}]),
    );

    context.getInputConnectionData = jest.fn().mockResolvedValue(embeddings);

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    const result = await node.execute.call(context);

    expect(result[0][0].json.document).not.toHaveProperty('metadata');
  });

  it('should apply metadata filter', async () => {
    const context = createMockExecuteContext(
      {
        mode: 'load',
        tableName: 'test_table',
        prompt: 'test query',
        topK: 4,
        includeDocumentMetadata: true,
        options: {
          metadata: {
            metadataValues: [{ name: 'category', value: 'tech' }],
          },
        },
      },
      createMockInputData([{}]),
    );

    context.getInputConnectionData = jest.fn().mockResolvedValue(embeddings);

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    await node.execute.call(context);

    expect(mockStore.similaritySearchVectorWithScore).toHaveBeenCalledWith(
      expect.any(Array),
      4,
      { category: 'tech' },
    );
  });

  it('should cleanup resources in finally block', async () => {
    const context = createMockExecuteContext(
      {
        mode: 'load',
        tableName: 'test_table',
        prompt: 'test query',
        topK: 4,
        includeDocumentMetadata: true,
        options: {},
      },
      createMockInputData([{}]),
    );

    context.getInputConnectionData = jest.fn().mockResolvedValue(embeddings);

    // Make similaritySearchVectorWithScore throw an error
    mockStore.similaritySearchVectorWithScore.mockRejectedValueOnce(new Error('Query failed'));

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    await expect(node.execute.call(context)).rejects.toThrow('Query failed');

    // Still should cleanup client
    expect(mockStore.client.release).toHaveBeenCalled();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// execute() Tests - insert mode
// ─────────────────────────────────────────────────────────────────────────────

describe('execute() - insert mode', () => {
  let node: VectorStoreHologres;
  let mockStore: ReturnType<typeof createMockHologresVectorStore>;
  let embeddings: FakeEmbeddings;

  beforeEach(() => {
    jest.clearAllMocks();
    node = new VectorStoreHologres();
    mockStore = createMockHologresVectorStore();
    embeddings = new FakeEmbeddings(4);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('should insert documents', async () => {
    const documents = [new Document({ pageContent: 'test doc', metadata: {} })];

    const context = createMockExecuteContext(
      {
        mode: 'insert',
        tableName: 'test_table',
        dimensions: 4,
        embeddingBatchSize: 10,
        options: {},
      },
      createMockInputData([{}]),
    );

    context.getInputConnectionData = jest.fn((type: string) => {
      if (type === NodeConnectionTypes.AiEmbedding) {
        return Promise.resolve(embeddings);
      }
      if (type === NodeConnectionTypes.AiDocument) {
        return Promise.resolve(documents);
      }
      return Promise.resolve(null);
    });

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.initialize = jest.fn().mockResolvedValue(mockStore);

    const result = await node.execute.call(context);

    expect(MockHologresVectorStore.initialize).toHaveBeenCalled();
    expect(mockStore.addDocuments).toHaveBeenCalled();
    expect(mockStore.client.release).toHaveBeenCalled();
    // Pool is now managed externally
    expect(result[0]).toHaveLength(1);
  });

  it('should process in batches', async () => {
    const documents = [
      new Document({ pageContent: 'doc 1', metadata: {} }),
      new Document({ pageContent: 'doc 2', metadata: {} }),
      new Document({ pageContent: 'doc 3', metadata: {} }),
    ];

    const context = createMockExecuteContext(
      {
        mode: 'insert',
        tableName: 'test_table',
        dimensions: 4,
        embeddingBatchSize: 2, // Process 2 at a time
        options: {},
      },
      createMockInputData([{}]),
    );

    context.getInputConnectionData = jest.fn((type: string) => {
      if (type === NodeConnectionTypes.AiEmbedding) {
        return Promise.resolve(embeddings);
      }
      if (type === NodeConnectionTypes.AiDocument) {
        return Promise.resolve(documents);
      }
      return Promise.resolve(null);
    });

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.initialize = jest.fn().mockResolvedValue(mockStore);

    await node.execute.call(context);

    // Should be called twice (batch 1: 2 docs, batch 2: 1 doc)
    expect(mockStore.addDocuments).toHaveBeenCalledTimes(2);
  });

  it('should respect abort signal', async () => {
    const documents = [new Document({ pageContent: 'test doc', metadata: {} })];

    const context = createMockExecuteContext(
      {
        mode: 'insert',
        tableName: 'test_table',
        dimensions: 4,
        embeddingBatchSize: 10,
        options: {},
      },
      createMockInputData([{}]),
    );

    context.getInputConnectionData = jest.fn((type: string) => {
      if (type === NodeConnectionTypes.AiEmbedding) {
        return Promise.resolve(embeddings);
      }
      if (type === NodeConnectionTypes.AiDocument) {
        return Promise.resolve(documents);
      }
      return Promise.resolve(null);
    });

    // Mock abort signal
    context.getExecutionCancelSignal = jest.fn().mockReturnValue({ aborted: true });

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.initialize = jest.fn().mockResolvedValue(mockStore);

    const result = await node.execute.call(context);

    // Should not process any items due to abort
    expect(result[0]).toHaveLength(0);
  });

  it('should break batch processing on abort signal', async () => {
    const documents = [
      new Document({ pageContent: 'doc 1', metadata: {} }),
      new Document({ pageContent: 'doc 2', metadata: {} }),
      new Document({ pageContent: 'doc 3', metadata: {} }),
    ];

    const context = createMockExecuteContext(
      {
        mode: 'insert',
        tableName: 'test_table',
        dimensions: 4,
        embeddingBatchSize: 1, // Process 1 at a time
        options: {},
      },
      createMockInputData([{}]),
    );

    context.getInputConnectionData = jest.fn((type: string) => {
      if (type === NodeConnectionTypes.AiEmbedding) {
        return Promise.resolve(embeddings);
      }
      if (type === NodeConnectionTypes.AiDocument) {
        return Promise.resolve(documents);
      }
      return Promise.resolve(null);
    });

    // Mock abort signal to be false initially, then true
    let callCount = 0;
    context.getExecutionCancelSignal = jest.fn().mockImplementation(() => {
      callCount++;
      return { aborted: callCount > 2 }; // Abort after 2 calls
    });

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.initialize = jest.fn().mockResolvedValue(mockStore);

    await node.execute.call(context);

    // Should have called addDocuments only once (first batch before abort in loop)
    expect(mockStore.addDocuments).toHaveBeenCalledTimes(1);
  });

  it('should cleanup resources', async () => {
    const documents = [new Document({ pageContent: 'test doc', metadata: {} })];

    const context = createMockExecuteContext(
      {
        mode: 'insert',
        tableName: 'test_table',
        dimensions: 4,
        embeddingBatchSize: 10,
        options: {},
      },
      createMockInputData([{}]),
    );

    context.getInputConnectionData = jest.fn((type: string) => {
      if (type === NodeConnectionTypes.AiEmbedding) {
        return Promise.resolve(embeddings);
      }
      if (type === NodeConnectionTypes.AiDocument) {
        return Promise.resolve(documents);
      }
      return Promise.resolve(null);
    });

    // Make addDocuments throw
    mockStore.addDocuments.mockRejectedValueOnce(new Error('Insert failed'));

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.initialize = jest.fn().mockResolvedValue(mockStore);

    await expect(node.execute.call(context)).rejects.toThrow('Insert failed');

    expect(mockStore.client.release).toHaveBeenCalled();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// execute() Tests - update mode
// ─────────────────────────────────────────────────────────────────────────────

describe('execute() - update mode', () => {
  let node: VectorStoreHologres;
  let mockStore: ReturnType<typeof createMockHologresVectorStore>;
  let embeddings: FakeEmbeddings;

  beforeEach(() => {
    jest.clearAllMocks();
    node = new VectorStoreHologres();
    mockStore = createMockHologresVectorStore();
    embeddings = new FakeEmbeddings(4);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('should update document by ID', async () => {
    const documents = [new Document({ pageContent: 'updated doc', metadata: { updated: true } })];

    const context = createMockExecuteContext(
      {
        mode: 'update',
        id: 'doc-123',
        tableName: 'test_table',
        options: {},
      },
      createMockInputData([{}]),
    );

    context.getInputConnectionData = jest.fn((type: string) => {
      if (type === NodeConnectionTypes.AiEmbedding) {
        return Promise.resolve(embeddings);
      }
      if (type === NodeConnectionTypes.AiDocument) {
        return Promise.resolve(documents);
      }
      return Promise.resolve(null);
    });

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    const result = await node.execute.call(context);

    expect(mockStore.update).toHaveBeenCalledWith({
      id: 'doc-123',
      document: documents[0],
    });
    expect(mockStore.client.release).toHaveBeenCalled();
    // Pool is now managed externally
    expect(result[0]).toHaveLength(1);
  });

  it('should throw error when no document provided', async () => {
    const documents: Document[] = []; // Empty array

    const context = createMockExecuteContext(
      {
        mode: 'update',
        id: 'doc-123',
        tableName: 'test_table',
        options: {},
      },
      createMockInputData([{}]),
    );

    context.getInputConnectionData = jest.fn((type: string) => {
      if (type === NodeConnectionTypes.AiEmbedding) {
        return Promise.resolve(embeddings);
      }
      if (type === NodeConnectionTypes.AiDocument) {
        return Promise.resolve(documents);
      }
      return Promise.resolve(null);
    });

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    await expect(node.execute.call(context)).rejects.toThrow('No document provided for update');
  });

  it('should cleanup resources', async () => {
    const documents = [new Document({ pageContent: 'updated doc', metadata: {} })];

    const context = createMockExecuteContext(
      {
        mode: 'update',
        id: 'doc-123',
        tableName: 'test_table',
        options: {},
      },
      createMockInputData([{}]),
    );

    context.getInputConnectionData = jest.fn((type: string) => {
      if (type === NodeConnectionTypes.AiEmbedding) {
        return Promise.resolve(embeddings);
      }
      if (type === NodeConnectionTypes.AiDocument) {
        return Promise.resolve(documents);
      }
      return Promise.resolve(null);
    });

    // Make update throw
    mockStore.update.mockRejectedValueOnce(new Error('Update failed'));

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    await expect(node.execute.call(context)).rejects.toThrow('Update failed');

    expect(mockStore.client.release).toHaveBeenCalled();
  });

  it('should respect abort signal in loop', async () => {
    const documents = [new Document({ pageContent: 'updated doc', metadata: {} })];

    const context = createMockExecuteContext(
      {
        mode: 'update',
        id: 'doc-123',
        tableName: 'test_table',
        options: {},
      },
      createMockInputData([{}, {}]), // 2 items
    );

    context.getInputConnectionData = jest.fn((type: string) => {
      if (type === NodeConnectionTypes.AiEmbedding) {
        return Promise.resolve(embeddings);
      }
      if (type === NodeConnectionTypes.AiDocument) {
        return Promise.resolve(documents);
      }
      return Promise.resolve(null);
    });

    // Abort after first item
    let callCount = 0;
    context.getExecutionCancelSignal = jest.fn().mockImplementation(() => {
      callCount++;
      return { aborted: callCount > 1 };
    });

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    const result = await node.execute.call(context);

    // Should have processed only 1 item before abort
    expect(mockStore.update).toHaveBeenCalledTimes(1);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// execute() Tests - retrieve-as-tool mode
// ─────────────────────────────────────────────────────────────────────────────

describe('execute() - retrieve-as-tool mode', () => {
  let node: VectorStoreHologres;
  let mockStore: ReturnType<typeof createMockHologresVectorStore>;
  let embeddings: FakeEmbeddings;

  beforeEach(() => {
    jest.clearAllMocks();
    node = new VectorStoreHologres();
    mockStore = createMockHologresVectorStore();
    embeddings = new FakeEmbeddings(4);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('should use chatInput as query', async () => {
    const context = createMockExecuteContext(
      {
        mode: 'retrieve-as-tool',
        tableName: 'test_table',
        topK: 4,
        includeDocumentMetadata: true,
        options: {},
      },
      createMockInputData([{ chatInput: 'my query' }]),
    );

    context.getInputConnectionData = jest.fn().mockResolvedValue(embeddings);

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    const result = await node.execute.call(context);

    expect(mockStore.similaritySearchVectorWithScore).toHaveBeenCalled();
    expect(result[0]).toHaveLength(2);
  });

  it('should use query field as fallback', async () => {
    const context = createMockExecuteContext(
      {
        mode: 'retrieve-as-tool',
        tableName: 'test_table',
        topK: 4,
        includeDocumentMetadata: true,
        options: {},
      },
      createMockInputData([{ query: 'fallback query' }]),
    );

    context.getInputConnectionData = jest.fn().mockResolvedValue(embeddings);

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    await node.execute.call(context);

    expect(mockStore.similaritySearchVectorWithScore).toHaveBeenCalled();
  });

  it('should throw error when no query', async () => {
    const context = createMockExecuteContext(
      {
        mode: 'retrieve-as-tool',
        tableName: 'test_table',
        topK: 4,
        includeDocumentMetadata: true,
        options: {},
      },
      createMockInputData([{}]), // No chatInput or query
    );

    context.getInputConnectionData = jest.fn().mockResolvedValue(embeddings);

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    await expect(node.execute.call(context)).rejects.toThrow(
      'No query found in input item. Expected "chatInput" or "query" field.',
    );
  });

  it('should respect abort signal in loop', async () => {
    const context = createMockExecuteContext(
      {
        mode: 'retrieve-as-tool',
        tableName: 'test_table',
        topK: 4,
        includeDocumentMetadata: true,
        options: {},
      },
      createMockInputData([{ chatInput: 'test' }, { chatInput: 'test2' }]),
    );

    context.getInputConnectionData = jest.fn().mockResolvedValue(embeddings);

    // Abort after first item
    let callCount = 0;
    context.getExecutionCancelSignal = jest.fn().mockImplementation(() => {
      callCount++;
      return { aborted: callCount > 1 };
    });

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    const result = await node.execute.call(context);

    // Should have processed only 1 item before abort
    expect(mockStore.similaritySearchVectorWithScore).toHaveBeenCalledTimes(1);
    expect(result[0]).toHaveLength(2); // 2 docs from the single search
  });

  it('should cleanup resources after execution', async () => {
    const context = createMockExecuteContext(
      {
        mode: 'retrieve-as-tool',
        tableName: 'test_table',
        topK: 4,
        includeDocumentMetadata: true,
        options: {},
      },
      createMockInputData([{ chatInput: 'test' }]),
    );

    context.getInputConnectionData = jest.fn().mockResolvedValue(embeddings);

    // Make search throw
    mockStore.similaritySearchVectorWithScore.mockRejectedValueOnce(new Error('Search failed'));

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    await expect(node.execute.call(context)).rejects.toThrow('Search failed');

    expect(mockStore.client.release).toHaveBeenCalled();
  });

  it('should exclude metadata when flag is false', async () => {
    const context = createMockExecuteContext(
      {
        mode: 'retrieve-as-tool',
        tableName: 'test_table',
        topK: 4,
        includeDocumentMetadata: false,
        options: {},
      },
      createMockInputData([{ chatInput: 'test query' }]),
    );

    context.getInputConnectionData = jest.fn().mockResolvedValue(embeddings);

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    const result = await node.execute.call(context);

    expect(result[0][0].json.document).not.toHaveProperty('metadata');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// execute() Tests - error cases
// ─────────────────────────────────────────────────────────────────────────────

describe('execute() - error cases', () => {
  it('should throw for invalid mode', async () => {
    const node = new VectorStoreHologres();
    const context = createMockExecuteContext(
      {
        mode: 'invalid_mode',
        tableName: 'test_table',
      },
      createMockInputData([{}]),
    );

    await expect(node.execute.call(context)).rejects.toThrow(
      'The operation mode "invalid_mode" is not supported in execute',
    );
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// supplyData() Tests - retrieve mode
// ─────────────────────────────────────────────────────────────────────────────

describe('supplyData() - retrieve mode', () => {
  let node: VectorStoreHologres;
  let mockStore: ReturnType<typeof createMockHologresVectorStore>;
  let embeddings: FakeEmbeddings;

  beforeEach(() => {
    jest.clearAllMocks();
    node = new VectorStoreHologres();
    mockStore = createMockHologresVectorStore();
    embeddings = new FakeEmbeddings(4);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('should return VectorStore with closeFunction', async () => {
    const context = createMockSupplyDataContext({
      mode: 'retrieve',
      tableName: 'test_table',
      options: {},
    });

    context.getInputConnectionData = jest.fn().mockResolvedValue(embeddings);

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    const result = await node.supplyData.call(context, 0);

    expect(result.response).toBe(mockStore);
    expect(result.closeFunction).toBeDefined();
    expect(mockStore._initializeClient).toHaveBeenCalled();

    // Test closeFunction - now calls store.close()
    await result.closeFunction!();
    expect(mockStore.close).toHaveBeenCalled();
  });

  it('should pass filter to config', async () => {
    const context = createMockSupplyDataContext({
      mode: 'retrieve',
      tableName: 'test_table',
      options: {
        metadata: {
          metadataValues: [{ name: 'category', value: 'tech' }],
        },
      },
    });

    context.getInputConnectionData = jest.fn().mockResolvedValue(embeddings);

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    await node.supplyData.call(context, 0);

    // The constructor should be called with filter
    expect(MockHologresVectorStore).toHaveBeenCalledWith(
      embeddings,
      expect.objectContaining({
        filter: { category: 'tech' },
      }),
    );
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// supplyData() Tests - retrieve-as-tool mode
// ─────────────────────────────────────────────────────────────────────────────

describe('supplyData() - retrieve-as-tool mode', () => {
  let node: VectorStoreHologres;
  let mockStore: ReturnType<typeof createMockHologresVectorStore>;
  let embeddings: FakeEmbeddings;

  beforeEach(() => {
    jest.clearAllMocks();
    node = new VectorStoreHologres();
    mockStore = createMockHologresVectorStore();
    embeddings = new FakeEmbeddings(4);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('should return DynamicTool', async () => {
    const context = createMockSupplyDataContext({
      mode: 'retrieve-as-tool',
      tableName: 'test_table',
      toolName: 'test_tool',
      toolDescription: 'A test tool',
      topK: 4,
      includeDocumentMetadata: true,
      options: {},
    });

    context.getInputConnectionData = jest.fn().mockResolvedValue(embeddings);

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    const result = await node.supplyData.call(context, 0);

    expect(result.response).toBeDefined();
    expect(result.response.name).toBe('test_tool');
    expect(result.response.description).toBe('A test tool');
  });

  it('should execute tool func and return JSON', async () => {
    const context = createMockSupplyDataContext({
      mode: 'retrieve-as-tool',
      tableName: 'test_table',
      toolName: 'test_tool',
      toolDescription: 'A test tool',
      topK: 4,
      includeDocumentMetadata: true,
      options: {},
    });

    context.getInputConnectionData = jest.fn().mockResolvedValue(embeddings);

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    const result = await node.supplyData.call(context, 0);

    // Execute the tool function
    const toolResult = await result.response.func('test query');

    expect(mockStore.similaritySearchVectorWithScore).toHaveBeenCalled();

    // Should be valid JSON
    const parsed = JSON.parse(toolResult);
    expect(Array.isArray(parsed)).toBe(true);
  });

  it('should cleanup resources after tool execution', async () => {
    const context = createMockSupplyDataContext({
      mode: 'retrieve-as-tool',
      tableName: 'test_table',
      toolName: 'test_tool',
      toolDescription: 'A test tool',
      topK: 4,
      includeDocumentMetadata: true,
      options: {},
    });

    context.getInputConnectionData = jest.fn().mockResolvedValue(embeddings);

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    const result = await node.supplyData.call(context, 0);

    // Execute the tool function
    await result.response.func('test query');

    expect(mockStore.client.release).toHaveBeenCalled();
    // Pool is now managed externally via closeFunction
  });

  it('should exclude metadata when flag is false', async () => {
    const context = createMockSupplyDataContext({
      mode: 'retrieve-as-tool',
      tableName: 'test_table',
      toolName: 'test_tool',
      toolDescription: 'A test tool',
      topK: 4,
      includeDocumentMetadata: false,
      options: {},
    });

    context.getInputConnectionData = jest.fn().mockResolvedValue(embeddings);

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    const result = await node.supplyData.call(context, 0);

    // Execute the tool function
    const toolResult = await result.response.func('test query');
    const parsed = JSON.parse(toolResult);

    // Should not have metadata
    expect(parsed[0]).not.toHaveProperty('metadata');
  });

  it('should have closeFunction that ends pool', async () => {
    const mockPool = createMockPool();
    const context = createMockSupplyDataContext({
      mode: 'retrieve-as-tool',
      tableName: 'test_table',
      toolName: 'test_tool',
      toolDescription: 'A test tool',
      topK: 4,
      includeDocumentMetadata: true,
      options: {},
    });

    context.getInputConnectionData = jest.fn().mockResolvedValue(embeddings);

    // Mock createPoolFromCredentials to return our mock pool
    const { createPoolFromCredentials } = require('../../nodes/VectorStoreHologres/HologresVectorStore');
    (createPoolFromCredentials as jest.Mock).mockReturnValue(mockPool);

    const MockHologresVectorStore = require('../../nodes/VectorStoreHologres/HologresVectorStore').HologresVectorStore;
    MockHologresVectorStore.mockImplementation(() => mockStore);

    const result = await node.supplyData.call(context, 0);

    // Test closeFunction - this covers line 1023
    expect(result.closeFunction).toBeDefined();
    await result.closeFunction!();

    // Pool should be ended via closeFunction
    expect(mockPool.end).toHaveBeenCalled();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// supplyData() Tests - error cases
// ─────────────────────────────────────────────────────────────────────────────

describe('supplyData() - error cases', () => {
  it('should throw for invalid mode', async () => {
    const node = new VectorStoreHologres();
    const context = createMockSupplyDataContext({
      mode: 'invalid_mode',
      tableName: 'test_table',
    });

    await expect(node.supplyData.call(context, 0)).rejects.toThrow(
      'The operation mode "invalid_mode" is not supported in supplyData',
    );
  });
});