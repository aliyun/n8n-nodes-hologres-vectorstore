/**
 * Node Parameters Integration Tests
 * Tests various node parameter configurations for VectorStoreHologres
 *
 * These tests require actual Hologres database connection.
 * Set environment variables or create .env.test file.
 */

import {
  canRunIntegrationTests,
  createTestPool,
  generateTestTableName,
  cleanupTable,
  testConfig,
  defaultTestConfig,
} from './setup';
import { VectorStoreHologres } from '../../nodes/VectorStoreHologres/VectorStoreHologres.node';
import { HologresVectorStore } from '../../nodes/VectorStoreHologres/HologresVectorStore';
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
  // Build options object with searchFilterJson if provided
  const options = overrides.searchFilterJson
    ? { searchFilterJson: overrides.searchFilterJson, ...overrides.options }
    : overrides.options || {};

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
    options,
    ...overrides.extraParams,
  };

  const embeddings = new FakeEmbeddings(overrides.dimensions || 128);
  const documents = overrides.documents || [
    new Document({ pageContent: 'Test document 1', metadata: { source: 'test' } }),
  ];

  const pool = createTestPool();

  const mockContext = {
    getNodeParameter: jest.fn((name: string, _index: number, defaultValue?: unknown, options?: { ensureType?: string }) => {
      // Handle nested parameters like 'options.searchFilterJson'
      if (name.includes('.')) {
        const parts = name.split('.');
        let value: unknown = params;
        for (const part of parts) {
          value = (value as Record<string, unknown>)?.[part];
          if (value === undefined) return defaultValue;
        }
        // If ensureType is 'object' and value is a JSON string, parse it
        if (options?.ensureType === 'object' && typeof value === 'string') {
          try {
            return JSON.parse(value);
          } catch {
            return defaultValue;
          }
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
  searchFilterJson?: string;
  options?: Record<string, unknown>;
  documents?: Document[];
  documentInput?: unknown;
  inputData?: Array<{ json: Record<string, unknown> }>;
  extraParams?: Record<string, unknown>;
}

describeIntegration('Node Parameters Integration Tests', () => {
  let pool: pg.Pool;
  const node = new VectorStoreHologres();
  const dimensions = 128;

  beforeAll(async () => {
    pool = createTestPool();
  });

  afterAll(async () => {
    await pool.end();
  });

  describe('includeDocumentMetadata Parameter', () => {
    it('should exclude metadata when includeDocumentMetadata=false', async () => {
      const tableName = generateTestTableName('no_metadata');
      const embeddings = new FakeEmbeddings(dimensions);

      // Setup: Create table with documents
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      await store.addDocuments([
        new Document({ pageContent: 'Test doc', metadata: { key: 'value' } }),
      ]);
      store.client?.release();

      try {
        const { mockContext } = createMockExecuteContext({
          mode: 'load',
          tableName,
          prompt: 'test',
          topK: 1,
          includeDocumentMetadata: false,
          dimensions,
        });

        const result = await node.execute.call(mockContext as any);

        expect(result[0]).toBeDefined();
        // Metadata should be undefined when excluded (not included in response)
        expect(result[0][0].json.document.metadata).toBeUndefined();
      } finally {
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('topK Parameter', () => {
    it('should handle topK larger than available documents', async () => {
      const tableName = generateTestTableName('large_topk');
      const embeddings = new FakeEmbeddings(dimensions);

      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      await store.addDocuments([
        new Document({ pageContent: 'Doc 1' }),
        new Document({ pageContent: 'Doc 2' }),
      ]);
      store.client?.release();

      try {
        const { mockContext } = createMockExecuteContext({
          mode: 'load',
          tableName,
          prompt: 'test',
          topK: 100, // More than available
          dimensions,
        });

        const result = await node.execute.call(mockContext as any);

        // Should return only available documents
        expect(result[0].length).toBeLessThanOrEqual(2);
      } finally {
        await cleanupTable(pool, tableName);
      }
    });

    it('should respect topK=1', async () => {
      const tableName = generateTestTableName('topk_1');
      const embeddings = new FakeEmbeddings(dimensions);

      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      await store.addDocuments([
        new Document({ pageContent: 'Doc 1' }),
        new Document({ pageContent: 'Doc 2' }),
        new Document({ pageContent: 'Doc 3' }),
      ]);
      store.client?.release();

      try {
        const { mockContext } = createMockExecuteContext({
          mode: 'load',
          tableName,
          prompt: 'test',
          topK: 1,
          dimensions,
        });

        const result = await node.execute.call(mockContext as any);

        expect(result[0]).toHaveLength(1);
      } finally {
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('searchFilterJson Parameter', () => {
    it('should filter results by metadata via searchFilterJson', async () => {
      const tableName = generateTestTableName('filter_json');
      const embeddings = new FakeEmbeddings(dimensions);

      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      await store.addDocuments([
        new Document({ pageContent: 'Apple fruit', metadata: { category: 'fruit' } }),
        new Document({ pageContent: 'Carrot vegetable', metadata: { category: 'vegetable' } }),
      ]);
      store.client?.release();

      try {
        const { mockContext } = createMockExecuteContext({
          mode: 'load',
          tableName,
          prompt: 'test',
          topK: 10,
          searchFilterJson: '{"category": "fruit"}',
          dimensions,
        });

        const result = await node.execute.call(mockContext as any);

        // Should only return fruit category (filter applied)
        expect(result[0].length).toBeGreaterThan(0);
        const allCategories = result[0].map((item: any) => item.json.document.metadata?.category);
        expect(allCategories.every((cat: string) => cat === 'fruit')).toBe(true);
      } finally {
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('Tool Parameters (retrieve-as-tool mode)', () => {
    it('should use custom tool name and description', async () => {
      const tableName = generateTestTableName('tool_params');
      const embeddings = new FakeEmbeddings(dimensions);

      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      await store.addDocuments([
        new Document({ pageContent: 'Test for tool' }),
      ]);
      store.client?.release();

      try {
        const { mockContext } = createMockExecuteContext({
          mode: 'retrieve-as-tool',
          tableName,
          toolName: 'custom_knowledge_search',
          toolDescription: 'Search the custom knowledge base for relevant information',
          topK: 5,
        });

        const result = await node.supplyData.call(mockContext as any, 0);

        expect(result.response.name).toBe('custom_knowledge_search');
        expect(result.response.description).toBe('Search the custom knowledge base for relevant information');
      } finally {
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('Distance Method via Options', () => {
    it('should use distance method from options', async () => {
      const tableName = generateTestTableName('dist_option');
      const embeddings = new FakeEmbeddings(dimensions);

      try {
        const { mockContext } = createMockExecuteContext({
          mode: 'retrieve',
          tableName,
          dimensions,
          options: {
            distanceMethod: 'Euclidean',
          },
        });

        const result = await node.supplyData.call(mockContext as any, 0);

        expect(result.response).toBeDefined();
        expect(result.response._vectorstoreType()).toBe('hologres');

        // Cleanup
        if (result.closeFunction) {
          await result.closeFunction();
        }
      } finally {
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('Filter in retrieve mode', () => {
    it('should apply filter in supplyData for retrieve mode', async () => {
      const tableName = generateTestTableName('retrieve_filter');
      const embeddings = new FakeEmbeddings(dimensions);

      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      await store.addDocuments([
        new Document({ pageContent: 'Apple', metadata: { type: 'fruit' } }),
        new Document({ pageContent: 'Carrot', metadata: { type: 'vegetable' } }),
      ]);
      store.client?.release();

      try {
        const { mockContext } = createMockExecuteContext({
          mode: 'retrieve',
          tableName,
          dimensions,
          searchFilterJson: '{"type": "fruit"}',
        });

        const result = await node.supplyData.call(mockContext as any, 0);

        // The store should have the filter applied
        expect(result.response).toBeDefined();

        // Cleanup
        if (result.closeFunction) {
          await result.closeFunction();
        }
      } finally {
        await cleanupTable(pool, tableName);
      }
    });
  });
});