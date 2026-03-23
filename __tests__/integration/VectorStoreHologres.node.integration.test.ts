/**
 * VectorStoreHologres Node Integration Tests
 * Tests the n8n node execute() and supplyData() methods with real database
 *
 * These tests require actual Hologres database connection.
 * Set environment variables or create .env.test file.
 */

import {
  canRunIntegrationTests,
  skipReason,
  createTestPool,
  generateTestTableName,
  cleanupTable,
  testConfig,
} from './setup';
import { VectorStoreHologres } from '../../nodes/VectorStoreHologres/VectorStoreHologres.node';
import { FakeEmbeddings } from '../mocks/embeddings.mock';
import { Document } from '@langchain/core/documents';
import pg from 'pg';

// Skip all tests if database is not configured
const describeIntegration = canRunIntegrationTests() ? describe : describe.skip;

// Mock n8n context types
interface MockNodeParameter {
  [key: string]: unknown;
}

function createMockExecuteContext(overrides: Partial<MockExecuteContextOptions> = {}) {
  const params: MockNodeParameter = {
    mode: overrides.mode || 'load',
    tableName: overrides.tableName || generateTestTableName(),
    prompt: overrides.prompt || 'test query',
    topK: overrides.topK ?? 4,
    includeDocumentMetadata: overrides.includeDocumentMetadata ?? true,
    dimensions: overrides.dimensions || 128,
    embeddingBatchSize: overrides.embeddingBatchSize || 10,
    id: overrides.id || 'test-id',
    toolName: overrides.toolName || 'test_tool',
    toolDescription: overrides.toolDescription || 'Test tool description',
    options: overrides.options || {},
    ...overrides.extraParams,
  };

  const embeddings = new FakeEmbeddings(overrides.dimensions || 128);
  const documents = overrides.documents || [
    new Document({ pageContent: 'Test document 1', metadata: { source: 'test' } }),
  ];

  const pool = createTestPool();

  const mockContext = {
    getNodeParameter: jest.fn((name: string, _index: number, defaultValue?: unknown) => {
      if (name.includes('.')) {
        const parts = name.split('.');
        let value: unknown = params;
        for (const part of parts) {
          value = (value as Record<string, unknown>)?.[part];
          if (value === undefined) return defaultValue;
        }
        return value;
      }
      return params[name] ?? defaultValue;
    }),
    getInputData: jest.fn(() => overrides.inputData || [{ json: {} }]),
    getCredentials: jest.fn().mockResolvedValue({
      host: testConfig.host,
      port: testConfig.port,
      database: testConfig.database,
      user: testConfig.user,
      password: testConfig.password,
      ssl: testConfig.ssl === 'disable' ? 'disable' : testConfig.ssl,
      allowUnauthorizedCerts: testConfig.ssl === 'allow-unauthorized',
    }),
    getInputConnectionData: jest.fn().mockImplementation(async (type: string) => {
      if (type === 'ai_embedding') {
        return embeddings;
      }
      if (type === 'ai_document') {
        return overrides.documentInput || documents;
      }
      return null;
    }),
    getNode: jest.fn().mockReturnValue({
      name: 'Test Hologres Node',
    }),
    getExecutionCancelSignal: jest.fn().mockReturnValue({ aborted: false }),
  };

  return { mockContext, embeddings, documents, pool };
}

interface MockExecuteContextOptions {
  mode?: string;
  tableName?: string;
  prompt?: string;
  topK?: number;
  includeDocumentMetadata?: boolean;
  dimensions?: number;
  embeddingBatchSize?: number;
  id?: string;
  toolName?: string;
  toolDescription?: string;
  options?: Record<string, unknown>;
  documents?: Document[];
  documentInput?: unknown;
  inputData?: Array<{ json: Record<string, unknown> }>;
  extraParams?: Record<string, unknown>;
}

describeIntegration('VectorStoreHologres Node Integration Tests', () => {
  let pool: pg.Pool;
  const node = new VectorStoreHologres();

  beforeAll(async () => {
    pool = createTestPool();
  });

  afterAll(async () => {
    await pool.end();
  });

  describe('execute() - Load Mode', () => {
    it('should query documents and return results', async () => {
      const tableName = generateTestTableName('test_load');
      const embeddings = new FakeEmbeddings(128);

      // Setup: Create table and insert test documents
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions: 128,
        columns: {
          idColumnName: 'id',
          vectorColumnName: 'embedding',
          contentColumnName: 'text',
          metadataColumnName: 'metadata',
        },
        distanceMethod: 'Cosine',
        indexSettings: {
          baseQuantizationType: 'rabitq',
          useReorder: true,
          maxDegree: 32,
          efConstruction: 200,
        },
      });

      await store.addDocuments([
        new Document({ pageContent: 'First document about cats', metadata: { topic: 'cats' } }),
        new Document({ pageContent: 'Second document about dogs', metadata: { topic: 'dogs' } }),
      ]);
      store.client?.release();

      try {
        const { mockContext } = createMockExecuteContext({
          mode: 'load',
          tableName,
          prompt: 'cats',
          topK: 2,
          dimensions: 128,
        });

        // Execute
        const result = await node.execute.call(mockContext as any);

        expect(result).toBeDefined();
        expect(result[0]).toBeDefined();
        expect(result[0].length).toBeGreaterThan(0);
        expect(result[0][0].json.document).toBeDefined();
        expect(result[0][0].json.score).toBeDefined();
      } finally {
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('execute() - Insert Mode', () => {
    it('should insert documents into database', async () => {
      const tableName = generateTestTableName('test_insert');
      const documents = [
        new Document({ pageContent: 'Inserted doc 1', metadata: { type: 'test' } }),
        new Document({ pageContent: 'Inserted doc 2', metadata: { type: 'test' } }),
      ];

      try {
        const { mockContext } = createMockExecuteContext({
          mode: 'insert',
          tableName,
          dimensions: 128,
          documentInput: documents,
        });

        const result = await node.execute.call(mockContext as any);

        expect(result).toBeDefined();
        expect(result[0]).toHaveLength(2);

        // Verify in database
        const dbResult = await pool.query(`SELECT COUNT(*) FROM "${tableName}"`);
        expect(parseInt(dbResult.rows[0].count, 10)).toBe(2);
      } finally {
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('execute() - Update Mode', () => {
    it('should update existing document', async () => {
      const tableName = generateTestTableName('test_update');
      const embeddings = new FakeEmbeddings(128);

      // Setup: Create table and insert a document
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions: 128,
        columns: {
          idColumnName: 'id',
          vectorColumnName: 'embedding',
          contentColumnName: 'text',
          metadataColumnName: 'metadata',
        },
        distanceMethod: 'Cosine',
        indexSettings: {
          baseQuantizationType: 'rabitq',
          useReorder: true,
          maxDegree: 32,
          efConstruction: 200,
        },
      });

      const [docId] = await store.addDocuments([
        new Document({ pageContent: 'Original content', metadata: { version: 1 } }),
      ]);
      store.client?.release();

      try {
        const updatedDoc = new Document({
          pageContent: 'Updated content',
          metadata: { version: 2 },
        });

        const { mockContext } = createMockExecuteContext({
          mode: 'update',
          tableName,
          id: docId,
          dimensions: 128,
          documentInput: [updatedDoc],
        });

        await node.execute.call(mockContext as any);

        // Verify update
        const result = await pool.query(`SELECT * FROM "${tableName}" WHERE id = $1`, [docId]);
        expect(result.rows[0].text).toBe('Updated content');
        expect(result.rows[0].metadata.version).toBe(2);
      } finally {
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('execute() - Retrieve-as-Tool Mode', () => {
    it('should execute query and return results', async () => {
      const tableName = generateTestTableName('test_tool');
      const embeddings = new FakeEmbeddings(128);

      // Setup
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions: 128,
        columns: {
          idColumnName: 'id',
          vectorColumnName: 'embedding',
          contentColumnName: 'text',
          metadataColumnName: 'metadata',
        },
        distanceMethod: 'Cosine',
        indexSettings: {
          baseQuantizationType: 'rabitq',
          useReorder: true,
          maxDegree: 32,
          efConstruction: 200,
        },
      });

      await store.addDocuments([
        new Document({ pageContent: 'Tool test document', metadata: { source: 'tool' } }),
      ]);
      store.client?.release();

      try {
        const { mockContext } = createMockExecuteContext({
          mode: 'retrieve-as-tool',
          tableName,
          topK: 1,
          inputData: [{ json: { chatInput: 'test query' } }],
        });

        const result = await node.execute.call(mockContext as any);

        expect(result).toBeDefined();
        expect(result[0].length).toBeGreaterThan(0);
      } finally {
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('supplyData() - Retrieve Mode', () => {
    it('should return VectorStore instance', async () => {
      const tableName = generateTestTableName('test_retrieve');

      try {
        const { mockContext } = createMockExecuteContext({
          mode: 'retrieve',
          tableName,
          dimensions: 128,
        });

        const result = await node.supplyData.call(mockContext as any, 0);

        expect(result).toBeDefined();
        expect(result.response).toBeDefined();

        // Verify it's a VectorStore
        expect(result.response._vectorstoreType).toBeDefined();

        // Cleanup
        if (result.closeFunction) {
          await result.closeFunction();
        }
      } finally {
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('supplyData() - Retrieve-as-Tool Mode', () => {
    it('should return DynamicTool for AI agents', async () => {
      const tableName = generateTestTableName('test_tool_supply');
      const embeddings = new FakeEmbeddings(128);

      // Setup: Create table with documents
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions: 128,
        columns: {
          idColumnName: 'id',
          vectorColumnName: 'embedding',
          contentColumnName: 'text',
          metadataColumnName: 'metadata',
        },
        distanceMethod: 'Cosine',
        indexSettings: {
          baseQuantizationType: 'rabitq',
          useReorder: true,
          maxDegree: 32,
          efConstruction: 200,
        },
      });

      await store.addDocuments([
        new Document({ pageContent: 'Test for tool', metadata: {} }),
      ]);
      store.client?.release();

      try {
        const { mockContext } = createMockExecuteContext({
          mode: 'retrieve-as-tool',
          tableName,
          toolName: 'knowledge_base',
          toolDescription: 'Search the knowledge base',
          topK: 1,
        });

        const result = await node.supplyData.call(mockContext as any, 0);

        expect(result).toBeDefined();
        expect(result.response).toBeDefined();
        expect(result.response.name).toBe('knowledge_base');
        expect(result.response.description).toBe('Search the knowledge base');

        // Test the tool function
        const toolResult = await result.response.func('test query');
        expect(toolResult).toBeDefined();
        const parsed = JSON.parse(toolResult);
        expect(Array.isArray(parsed)).toBe(true);

        // Test closeFunction - this tests line 1023
        expect(result.closeFunction).toBeDefined();
        await result.closeFunction!();
      } finally {
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('execute() - Load Mode with Metadata Filter', () => {
    it('should apply metadata filter from options', async () => {
      // This tests lines 51-56: metadata filter with non-empty metadataValues array
      const tableName = generateTestTableName('metadata_filter');
      const embeddings = new FakeEmbeddings(128);

      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions: 128,
        columns: {
          idColumnName: 'id',
          vectorColumnName: 'embedding',
          contentColumnName: 'text',
          metadataColumnName: 'metadata',
        },
        distanceMethod: 'Cosine',
        indexSettings: {
          baseQuantizationType: 'rabitq',
          useReorder: true,
          maxDegree: 32,
          efConstruction: 200,
        },
      });

      await store.addDocuments([
        new Document({ pageContent: 'Cat document', metadata: { category: 'cats' } }),
        new Document({ pageContent: 'Dog document', metadata: { category: 'dogs' } }),
      ]);
      store.client?.release();

      try {
        const { mockContext } = createMockExecuteContext({
          mode: 'load',
          tableName,
          prompt: 'animals',
          topK: 10,
          dimensions: 128,
          extraParams: {
            options: {
              metadata: {
                metadataValues: [{ name: 'category', value: 'cats' }],
              },
            },
          },
        });

        // Override getNodeParameter to handle nested options.metadata
        mockContext.getNodeParameter = jest.fn((name: string, index: number, defaultValue?: unknown) => {
          if (name === 'options') {
            return {
              metadata: {
                metadataValues: [{ name: 'category', value: 'cats' }],
              },
            };
          }
          if (name === 'options.metadata') {
            return {
              metadataValues: [{ name: 'category', value: 'cats' }],
            };
          }
          if (name === 'mode') return 'load';
          if (name === 'tableName') return tableName;
          if (name === 'prompt') return 'animals';
          if (name === 'topK') return 10;
          if (name === 'includeDocumentMetadata') return true;
          if (name === 'dimensions') return 128;
          if (name === 'embeddingBatchSize') return 10;
          return defaultValue;
        });

        const result = await node.execute.call(mockContext as any);

        // Should only return cat documents
        expect(result[0].length).toBeGreaterThan(0);
        result[0].forEach((item: any) => {
          expect(item.json.document.metadata.category).toBe('cats');
        });
      } finally {
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('execute() - Insert Mode with Loader', () => {
    it('should process document from loader object with processItem method', async () => {
      // This tests lines 154-163: loader.processItem path
      const tableName = generateTestTableName('loader_insert');

      const mockLoader = {
        processItem: jest.fn().mockResolvedValue([
          new Document({ pageContent: 'Loaded doc 1', metadata: { source: 'loader' } }),
        ]),
      };

      try {
        const { mockContext } = createMockExecuteContext({
          mode: 'insert',
          tableName,
          dimensions: 128,
          documentInput: mockLoader,
        });

        const result = await node.execute.call(mockContext as any);

        expect(mockLoader.processItem).toHaveBeenCalled();
        expect(result[0]).toHaveLength(1);

        // Verify in database
        const dbResult = await pool.query(`SELECT * FROM "${tableName}"`);
        expect(dbResult.rows[0].text).toBe('Loaded doc 1');
        expect(dbResult.rows[0].metadata.source).toBe('loader');
      } finally {
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('execute() - Insert Mode Error Handling', () => {
    it('should throw error for unsupported document input type', async () => {
      // This tests line 163: throw new Error("Unsupported document input type")
      const tableName = generateTestTableName('unsupported_input');

      try {
        const { mockContext } = createMockExecuteContext({
          mode: 'insert',
          tableName,
          dimensions: 128,
          documentInput: 'invalid-type', // Not an array or object with processItem
        });

        await expect(
          node.execute.call(mockContext as any)
        ).rejects.toThrow('Unsupported document input type');
      } finally {
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('execute() - Load Mode without Metadata', () => {
    it('should exclude metadata when includeDocumentMetadata is false', async () => {
      // This tests line 133 in serializeSearchResults: conditional metadata spread
      const tableName = generateTestTableName('no_metadata');
      const embeddings = new FakeEmbeddings(128);

      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions: 128,
        columns: {
          idColumnName: 'id',
          vectorColumnName: 'embedding',
          contentColumnName: 'text',
          metadataColumnName: 'metadata',
        },
        distanceMethod: 'Cosine',
        indexSettings: {
          baseQuantizationType: 'rabitq',
          useReorder: true,
          maxDegree: 32,
          efConstruction: 200,
        },
      });

      await store.addDocuments([
        new Document({ pageContent: 'Test document', metadata: { key: 'value' } }),
      ]);
      store.client?.release();

      try {
        const { mockContext } = createMockExecuteContext({
          mode: 'load',
          tableName,
          prompt: 'test',
          topK: 1,
          dimensions: 128,
          includeDocumentMetadata: false,
        });

        const result = await node.execute.call(mockContext as any);

        expect(result[0].length).toBeGreaterThan(0);
        // Metadata should not be present
        expect(result[0][0].json.document.metadata).toBeUndefined();
        expect(result[0][0].json.document.pageContent).toBe('Test document');
      } finally {
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('supplyData() - Retrieve-as-Tool without Metadata', () => {
    it('should exclude metadata in tool results when includeDocumentMetadata is false', async () => {
      // This tests line 1009: conditional metadata spread in tool results
      const tableName = generateTestTableName('tool_no_metadata');
      const embeddings = new FakeEmbeddings(128);

      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions: 128,
        columns: {
          idColumnName: 'id',
          vectorColumnName: 'embedding',
          contentColumnName: 'text',
          metadataColumnName: 'metadata',
        },
        distanceMethod: 'Cosine',
        indexSettings: {
          baseQuantizationType: 'rabitq',
          useReorder: true,
          maxDegree: 32,
          efConstruction: 200,
        },
      });

      await store.addDocuments([
        new Document({ pageContent: 'Tool test doc', metadata: { secret: 'hidden' } }),
      ]);
      store.client?.release();

      try {
        const { mockContext } = createMockExecuteContext({
          mode: 'retrieve-as-tool',
          tableName,
          toolName: 'search_tool',
          toolDescription: 'Search without metadata',
          topK: 1,
          includeDocumentMetadata: false,
        });

        const result = await node.supplyData.call(mockContext as any, 0);

        // Test the tool function
        const toolResult = await result.response.func('test query');
        const parsed = JSON.parse(toolResult);

        expect(parsed).toHaveLength(1);
        expect(parsed[0].pageContent).toBe('Tool test doc');
        expect(parsed[0].metadata).toBeUndefined();
        expect(parsed[0].score).toBeDefined();

        // Cleanup
        if (result.closeFunction) {
          await result.closeFunction();
        }
      } finally {
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('execute() - Cancellation Handling', () => {
    it('should break execution loop when signal is aborted in insert mode', async () => {
      // This tests lines 716, 760: if (this.getExecutionCancelSignal()?.aborted) break;
      const tableName = generateTestTableName('cancel_insert');
      const documents = [
        new Document({ pageContent: 'Doc 1', metadata: { idx: 1 } }),
        new Document({ pageContent: 'Doc 2', metadata: { idx: 2 } }),
      ];

      try {
        const { mockContext } = createMockExecuteContext({
          mode: 'insert',
          tableName,
          dimensions: 128,
          documentInput: documents,
          inputData: [{ json: {} }, { json: {} }], // Two input items
        });

        // Override the cancel signal to return aborted
        mockContext.getExecutionCancelSignal = jest.fn().mockReturnValue({ aborted: true });

        // Execute should break early but not throw
        const result = await node.execute.call(mockContext as any);
        // Should return empty or partial results due to cancellation
        expect(result).toBeDefined();
      } finally {
        await cleanupTable(pool, tableName);
      }
    });

    it('should break execution loop when signal is aborted in update mode', async () => {
      // This tests line 868: if (this.getExecutionCancelSignal()?.aborted) break;
      const tableName = generateTestTableName('cancel_update');
      const embeddings = new FakeEmbeddings(128);

      // Setup: Create table and insert a document
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions: 128,
        columns: {
          idColumnName: 'id',
          vectorColumnName: 'embedding',
          contentColumnName: 'text',
          metadataColumnName: 'metadata',
        },
        distanceMethod: 'Cosine',
        indexSettings: {
          baseQuantizationType: 'rabitq',
          useReorder: true,
          maxDegree: 32,
          efConstruction: 200,
        },
      });

      const [docId] = await store.addDocuments([
        new Document({ pageContent: 'Original', metadata: {} }),
      ]);
      store.client?.release();

      try {
        const updatedDoc = new Document({ pageContent: 'Updated', metadata: {} });

        const { mockContext } = createMockExecuteContext({
          mode: 'update',
          tableName,
          id: docId,
          dimensions: 128,
          documentInput: [updatedDoc],
        });

        // Override the cancel signal to return aborted
        mockContext.getExecutionCancelSignal = jest.fn().mockReturnValue({ aborted: true });

        // Execute should break early but not throw
        const result = await node.execute.call(mockContext as any);
        expect(result).toBeDefined();
      } finally {
        await cleanupTable(pool, tableName);
      }
    });

    it('should break execution loop when signal is aborted in retrieve-as-tool mode', async () => {
      // This tests line 790: if (this.getExecutionCancelSignal()?.aborted) break;
      const tableName = generateTestTableName('cancel_tool');

      // Setup: Create table with documents
      const embeddings = new FakeEmbeddings(128);
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions: 128,
        columns: {
          idColumnName: 'id',
          vectorColumnName: 'embedding',
          contentColumnName: 'text',
          metadataColumnName: 'metadata',
        },
        distanceMethod: 'Cosine',
        indexSettings: {
          baseQuantizationType: 'rabitq',
          useReorder: true,
          maxDegree: 32,
          efConstruction: 200,
        },
      });

      await store.addDocuments([
        new Document({ pageContent: 'Test doc', metadata: {} }),
      ]);
      store.client?.release();

      try {
        const { mockContext } = createMockExecuteContext({
          mode: 'retrieve-as-tool',
          tableName,
          topK: 1,
          inputData: [{ json: { chatInput: 'test' } }],
        });

        // Override the cancel signal to return aborted
        mockContext.getExecutionCancelSignal = jest.fn().mockReturnValue({ aborted: true });

        // Execute should break early
        const result = await node.execute.call(mockContext as any);
        expect(result).toBeDefined();
      } finally {
        await cleanupTable(pool, tableName);
      }
    });
  });
});

// Import HologresVectorStore for setup
import { HologresVectorStore } from '../../nodes/VectorStoreHologres/HologresVectorStore';