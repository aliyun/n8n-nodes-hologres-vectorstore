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
      allowUnauthorizedCerts: true,
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
      } finally {
        await cleanupTable(pool, tableName);
      }
    });
  });
});

// Import HologresVectorStore for setup
import { HologresVectorStore } from '../../nodes/VectorStoreHologres/HologresVectorStore';