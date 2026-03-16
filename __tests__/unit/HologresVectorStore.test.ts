/**
 * Unit tests for HologresVectorStore
 * Priority P0: SQL injection protection tests
 * Priority P1: Core data operations tests
 */

import { jest, describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import pg from 'pg';
import { Document } from '@langchain/core/documents';
import {
  HologresVectorStore,
  quoteIdentifier,
  VALID_IDENTIFIER,
  createPoolFromCredentials,
  type HologresVectorStoreArgs,
  type ColumnOptions,
  type HGraphIndexSettings,
} from '../../nodes/VectorStoreHologres/HologresVectorStore';
import { FakeEmbeddings, createStaticVector } from '../mocks/embeddings.mock';

// ─────────────────────────────────────────────────────────────────────────────
// P0: SQL Injection Protection Tests (Critical Security Tests)
// ─────────────────────────────────────────────────────────────────────────────

describe('P0: SQL Injection Protection', () => {
  describe('VALID_IDENTIFIER regex', () => {
    it('should accept valid identifiers', () => {
      const validIdentifiers = [
        'id',
        '_id',
        'table_name',
        'column1',
        'myColumn',
        'CamelCase',
        '_private_field',
        'field_123',
      ];

      validIdentifiers.forEach((identifier) => {
        expect(VALID_IDENTIFIER.test(identifier)).toBe(true);
      });
    });

    it('should reject identifiers starting with digits', () => {
      expect(VALID_IDENTIFIER.test('1column')).toBe(false);
      expect(VALID_IDENTIFIER.test('123test')).toBe(false);
    });

    it('should reject identifiers with special characters', () => {
      const invalidIdentifiers = [
        'column-name',
        'column.name',
        'column name',
        "column'name",
        'column"name',
        'column;drop',
        'column--comment',
        'column/**/',
      ];

      invalidIdentifiers.forEach((identifier) => {
        expect(VALID_IDENTIFIER.test(identifier)).toBe(false);
      });
    });

    it('should reject identifiers with SQL injection attempts', () => {
      const injectionAttempts = [
        "id'; DROP TABLE users;--",
        'id OR 1=1',
        'id UNION SELECT',
        'id; DELETE FROM',
        'id)(()',
        '<script>alert(1)</script>',
      ];

      injectionAttempts.forEach((attempt) => {
        expect(VALID_IDENTIFIER.test(attempt)).toBe(false);
      });
    });

    it('should reject empty strings', () => {
      expect(VALID_IDENTIFIER.test('')).toBe(false);
    });
  });

  describe('quoteIdentifier()', () => {
    it('should wrap valid identifiers with double quotes', () => {
      expect(quoteIdentifier('id')).toBe('"id"');
      expect(quoteIdentifier('table_name')).toBe('"table_name"');
      expect(quoteIdentifier('_private')).toBe('"_private"');
    });

    it('should throw error for invalid identifiers', () => {
      expect(() => quoteIdentifier('invalid-name')).toThrow(
        'Invalid SQL identifier: "invalid-name"',
      );
      expect(() => quoteIdentifier('123invalid')).toThrow(
        'Invalid SQL identifier: "123invalid"',
      );
      expect(() => quoteIdentifier('invalid name')).toThrow(
        'Invalid SQL identifier: "invalid name"',
      );
    });

    it('should throw error for SQL injection attempts', () => {
      expect(() => quoteIdentifier("id'; DROP TABLE users;--")).toThrow();
      expect(() => quoteIdentifier('id OR 1=1')).toThrow();
      expect(() => quoteIdentifier('id UNION SELECT')).toThrow();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// P1: Connection Pool Tests
// ─────────────────────────────────────────────────────────────────────────────

describe('P1: createPoolFromCredentials', () => {
  it('should create pool with basic credentials', () => {
    const credentials = {
      host: 'localhost',
      port: 5432,
      database: 'test_db',
      user: 'test_user',
      password: 'test_password',
    };

    const pool = createPoolFromCredentials(credentials);

    expect(pg.Pool).toHaveBeenCalledWith(
      expect.objectContaining({
        host: 'localhost',
        port: 5432,
        database: 'test_db',
        user: 'test_user',
        password: 'test_password',
        application_name: 'n8n_hologres_vector_store',
      }),
    );
  });

  it('should set default max connections to 100', () => {
    const credentials = {
      host: 'localhost',
      port: 5432,
      database: 'test_db',
      user: 'test_user',
      password: 'test_password',
    };

    createPoolFromCredentials(credentials);

    expect(pg.Pool).toHaveBeenCalledWith(
      expect.objectContaining({
        max: 100,
      }),
    );
  });

  it('should use custom max connections from credentials', () => {
    const credentials = {
      host: 'localhost',
      port: 5432,
      database: 'test_db',
      user: 'test_user',
      password: 'test_password',
      maxConnections: 50,
    };

    createPoolFromCredentials(credentials);

    expect(pg.Pool).toHaveBeenCalledWith(
      expect.objectContaining({
        max: 50,
      }),
    );
  });

  it('should disable SSL by default', () => {
    const credentials = {
      host: 'localhost',
      port: 5432,
      database: 'test_db',
      user: 'test_user',
      password: 'test_password',
    };

    createPoolFromCredentials(credentials);

    expect(pg.Pool).toHaveBeenCalledWith(
      expect.objectContaining({
        ssl: false,
      }),
    );
  });

  it('should enable SSL when ssl is true', () => {
    const credentials = {
      host: 'localhost',
      port: 5432,
      database: 'test_db',
      user: 'test_user',
      password: 'test_password',
      ssl: 'enable',
    };

    createPoolFromCredentials(credentials);

    expect(pg.Pool).toHaveBeenCalledWith(
      expect.objectContaining({
        ssl: true,
      }),
    );
  });

  it('should allow unauthorized certs when allowUnauthorizedCerts is true', () => {
    const credentials = {
      host: 'localhost',
      port: 5432,
      database: 'test_db',
      user: 'test_user',
      password: 'test_password',
      allowUnauthorizedCerts: true,
    };

    createPoolFromCredentials(credentials);

    expect(pg.Pool).toHaveBeenCalledWith(
      expect.objectContaining({
        ssl: { rejectUnauthorized: false },
      }),
    );
  });

  it('should disable SSL when ssl is "disable"', () => {
    const credentials = {
      host: 'localhost',
      port: 5432,
      database: 'test_db',
      user: 'test_user',
      password: 'test_password',
      ssl: 'disable',
    };

    createPoolFromCredentials(credentials);

    expect(pg.Pool).toHaveBeenCalledWith(
      expect.objectContaining({
        ssl: false,
      }),
    );
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// P1: HologresVectorStore Core Tests
// ─────────────────────────────────────────────────────────────────────────────

describe('P1: HologresVectorStore', () => {
  let embeddings: FakeEmbeddings;
  let mockPool: pg.Pool;
  let mockClient: pg.PoolClient;
  let defaultArgs: HologresVectorStoreArgs;

  beforeEach(() => {
    jest.clearAllMocks();

    embeddings = new FakeEmbeddings(4);

    // Get mock instances from the mocked pg module
    mockPool = new pg.Pool();
    mockClient = {
      query: jest.fn().mockResolvedValue({ rows: [], rowCount: 0 }),
      release: jest.fn(),
    } as unknown as pg.PoolClient;
    (mockPool.connect as jest.Mock).mockResolvedValue(mockClient);

    defaultArgs = {
      pool: mockPool,
      tableName: 'test_vectors',
      dimensions: 4,
      distanceMethod: 'Cosine',
      columns: {
        idColumnName: 'id',
        vectorColumnName: 'embedding',
        contentColumnName: 'text',
        metadataColumnName: 'metadata',
      },
      indexSettings: {
        baseQuantizationType: 'rabitq',
        useReorder: true,
        maxDegree: 64,
        efConstruction: 400,
      },
    };
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('constructor', () => {
    it('should create instance with provided arguments', () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);

      expect(store.pool).toBe(mockPool);
      expect(store.tableName).toBe('test_vectors');
      expect(store.dimensions).toBe(4);
      expect(store.distanceMethod).toBe('Cosine');
      expect(store.columns).toEqual(defaultArgs.columns);
      expect(store.indexSettings).toEqual(defaultArgs.indexSettings);
    });

    it('should store filter if provided', () => {
      const filter = { category: 'test' };
      const store = new HologresVectorStore(embeddings, {
        ...defaultArgs,
        filter,
      });

      expect(store.filter).toEqual(filter);
    });
  });

  describe('_vectorstoreType()', () => {
    it('should return "hologres"', () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);

      expect(store._vectorstoreType()).toBe('hologres');
    });
  });

  describe('_initializeClient()', () => {
    it('should connect to pool and store client', async () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);

      await store._initializeClient();

      expect(mockPool.connect).toHaveBeenCalled();
      expect(store.client).toBe(mockClient);
    });
  });

  describe('addVectors()', () => {
    it('should generate UUIDs when ids not provided', async () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);
      const vectors = [createStaticVector(4)];
      const documents = [new Document({ pageContent: 'test' })];

      const ids = await store.addVectors(vectors, documents);

      expect(ids).toHaveLength(1);
      expect(ids[0]).toBe('test-uuid-1234-5678');
    });

    it('should use provided ids', async () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);
      const vectors = [createStaticVector(4)];
      const documents = [new Document({ pageContent: 'test' })];
      const providedIds = ['custom-id-1'];

      const ids = await store.addVectors(vectors, documents, { ids: providedIds });

      expect(ids).toEqual(['custom-id-1']);
    });

    it('should build correct INSERT SQL', async () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);
      const vectors = [[0.1, 0.2, 0.3, 0.4]];
      const documents = [new Document({ pageContent: 'test content', metadata: { key: 'value' } })];
      const ids = ['doc-1'];

      await store.addVectors(vectors, documents, { ids });

      expect(mockPool.query).toHaveBeenCalledWith(
        expect.stringContaining('INSERT INTO "test_vectors"'),
        expect.arrayContaining(['doc-1', 'test content', '{0.1,0.2,0.3,0.4}', '{"key":"value"}']),
      );
    });

    it('should handle multiple vectors', async () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);
      const vectors = [createStaticVector(4), createStaticVector(4)];
      const documents = [
        new Document({ pageContent: 'doc1' }),
        new Document({ pageContent: 'doc2' }),
      ];

      await store.addVectors(vectors, documents);

      expect(mockPool.query).toHaveBeenCalledTimes(2);
    });

    it('should quote custom column names in SQL', async () => {
      const store = new HologresVectorStore(embeddings, {
        ...defaultArgs,
        columns: {
          idColumnName: 'doc_id',
          vectorColumnName: 'vec',
          contentColumnName: 'content',
          metadataColumnName: 'meta',
        },
      });
      const vectors = [[0.1, 0.2, 0.3, 0.4]];
      const documents = [new Document({ pageContent: 'test' })];

      await store.addVectors(vectors, documents);

      const call = (mockPool.query as jest.Mock).mock.calls[0];
      expect(call[0]).toContain('"doc_id"');
      expect(call[0]).toContain('"vec"');
      expect(call[0]).toContain('"content"');
      expect(call[0]).toContain('"meta"');
    });
  });

  describe('addDocuments()', () => {
    it('should embed documents and call addVectors', async () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);
      const documents = [new Document({ pageContent: 'test document' })];

      const embedSpy = jest.spyOn(embeddings, 'embedDocuments');
      const addVectorsSpy = jest.spyOn(store, 'addVectors');

      await store.addDocuments(documents);

      expect(embedSpy).toHaveBeenCalledWith(['test document']);
      expect(addVectorsSpy).toHaveBeenCalled();
    });
  });

  describe('similaritySearchVectorWithScore()', () => {
    it('should use cosine distance by default', async () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);
      (mockPool.query as jest.Mock).mockResolvedValueOnce({ rows: [] });

      await store.similaritySearchVectorWithScore(createStaticVector(4), 5);

      const call = (mockPool.query as jest.Mock).mock.calls[0];
      expect(call[0]).toContain('approx_cosine_distance');
      expect(call[0]).toContain('DESC');
    });

    it('should use inner product distance when configured', async () => {
      const store = new HologresVectorStore(embeddings, {
        ...defaultArgs,
        distanceMethod: 'InnerProduct',
      });
      (mockPool.query as jest.Mock).mockResolvedValueOnce({ rows: [] });

      await store.similaritySearchVectorWithScore(createStaticVector(4), 5);

      const call = (mockPool.query as jest.Mock).mock.calls[0];
      expect(call[0]).toContain('approx_inner_product_distance');
      expect(call[0]).toContain('DESC');
    });

    it('should use euclidean distance when configured', async () => {
      const store = new HologresVectorStore(embeddings, {
        ...defaultArgs,
        distanceMethod: 'Euclidean',
      });
      (mockPool.query as jest.Mock).mockResolvedValueOnce({ rows: [] });

      await store.similaritySearchVectorWithScore(createStaticVector(4), 5);

      const call = (mockPool.query as jest.Mock).mock.calls[0];
      expect(call[0]).toContain('approx_euclidean_distance');
      expect(call[0]).toContain('ASC');
    });

    it('should fallback to cosine distance for unknown distance method', async () => {
      // Force an unknown distance method to test the default case
      const store = new HologresVectorStore(embeddings, {
        ...defaultArgs,
        distanceMethod: 'Unknown' as unknown as 'Cosine',
      });
      (mockPool.query as jest.Mock).mockResolvedValueOnce({ rows: [] });

      await store.similaritySearchVectorWithScore(createStaticVector(4), 5);

      const call = (mockPool.query as jest.Mock).mock.calls[0];
      expect(call[0]).toContain('approx_cosine_distance');
      expect(call[0]).toContain('DESC');
    });

    it('should return documents with scores', async () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);
      (mockPool.query as jest.Mock).mockResolvedValueOnce({
        rows: [
          {
            id: 'doc-1',
            text: 'content 1',
            metadata: { key: 'value' },
            _distance: 0.95,
          },
        ],
      });

      const results = await store.similaritySearchVectorWithScore(createStaticVector(4), 5);

      expect(results).toHaveLength(1);
      expect(results[0][0].pageContent).toBe('content 1');
      expect(results[0][0].metadata).toEqual({ key: 'value' });
      expect(results[0][1]).toBe(0.95);
    });

    it('should apply metadata filter', async () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);
      (mockPool.query as jest.Mock).mockResolvedValueOnce({ rows: [] });

      await store.similaritySearchVectorWithScore(createStaticVector(4), 5, { category: 'test' });

      const call = (mockPool.query as jest.Mock).mock.calls[0];
      expect(call[0]).toContain('WHERE');
      expect(call[0]).toContain('"metadata"');
      expect(call[1]).toContain('test');
    });

    it('should reject invalid metadata filter keys', async () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);

      await expect(
        store.similaritySearchVectorWithScore(createStaticVector(4), 5, { 'invalid-key': 'value' }),
      ).rejects.toThrow('Invalid metadata filter key');
    });

    it('should return empty array when no results', async () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);
      (mockPool.query as jest.Mock).mockResolvedValueOnce({ rows: [] });

      const results = await store.similaritySearchVectorWithScore(createStaticVector(4), 5);

      expect(results).toEqual([]);
    });

    it('should skip rows with null distance', async () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);
      (mockPool.query as jest.Mock).mockResolvedValueOnce({
        rows: [
          { id: 'doc-1', text: 'content 1', metadata: {}, _distance: null },
          { id: 'doc-2', text: 'content 2', metadata: {}, _distance: 0.5 },
        ],
      });

      const results = await store.similaritySearchVectorWithScore(createStaticVector(4), 5);

      expect(results).toHaveLength(1);
      expect(results[0][0].id).toBe('doc-2');
    });

    it('should handle null metadata with fallback to empty object', async () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);
      (mockPool.query as jest.Mock).mockResolvedValueOnce({
        rows: [
          {
            id: 'doc-1',
            text: 'content 1',
            metadata: null,
            _distance: 0.5,
          },
        ],
      });

      const results = await store.similaritySearchVectorWithScore(createStaticVector(4), 5);

      expect(results).toHaveLength(1);
      expect(results[0][0].metadata).toEqual({});
    });
  });

  describe('delete()', () => {
    it('should build correct DELETE SQL', async () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);

      await store.delete({ ids: ['id-1', 'id-2'] });

      const call = (mockPool.query as jest.Mock).mock.calls[0];
      expect(call[0]).toContain('DELETE FROM "test_vectors"');
      expect(call[0]).toContain('"id" = ANY($1::text[])');
      expect(call[1]).toEqual([['id-1', 'id-2']]);
    });
  });

  describe('update()', () => {
    it('should re-embed content and update all fields', async () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);
      const document = new Document({
        pageContent: 'updated content',
        metadata: { updated: true },
      });

      // Spy on embeddings.embedDocuments
      const embedSpy = jest.spyOn(embeddings, 'embedDocuments');

      await store.update({ id: 'doc-1', document });

      // Should call embedDocuments for re-embedding
      expect(embedSpy).toHaveBeenCalledWith(['updated content']);

      // Should build UPDATE SQL
      const call = (mockPool.query as jest.Mock).mock.calls[0];
      expect(call[0]).toContain('UPDATE "test_vectors"');
      expect(call[0]).toContain('SET "text" = $1');
      expect(call[0]).toContain('"embedding" = $2::float4[]');
      expect(call[0]).toContain('"metadata" = $3::jsonb');
      expect(call[0]).toContain('WHERE "id" = $4');
    });
  });

  describe('ensureTableInDatabase()', () => {
    it('should create table with correct structure', async () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);

      await store.ensureTableInDatabase();

      const call = (mockPool.query as jest.Mock).mock.calls[0];
      expect(call[0]).toContain('CREATE TABLE IF NOT EXISTS "test_vectors"');
      expect(call[0]).toContain('"id" text NOT NULL PRIMARY KEY');
      expect(call[0]).toContain('"text" text');
      expect(call[0]).toContain('"metadata" jsonb');
      expect(call[0]).toContain('"embedding" float4[]');
    });

    it('should include dimension constraint', async () => {
      const store = new HologresVectorStore(embeddings, {
        ...defaultArgs,
        dimensions: 768,
      });

      await store.ensureTableInDatabase();

      const call = (mockPool.query as jest.Mock).mock.calls[0];
      expect(call[0]).toContain('array_length("embedding", 1) = 768');
    });
  });

  describe('ensureVectorIndex()', () => {
    it('should set HGraph index with correct parameters', async () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);

      await store.ensureVectorIndex();

      const call = (mockPool.query as jest.Mock).mock.calls[0];
      expect(call[0]).toContain('ALTER TABLE "test_vectors" SET (vectors =');
      expect(call[0]).toContain('HGraph');
      expect(call[0]).toContain('"distance_method":"Cosine"');
    });

    it('should include builder_params', async () => {
      const store = new HologresVectorStore(embeddings, defaultArgs);

      await store.ensureVectorIndex();

      const call = (mockPool.query as jest.Mock).mock.calls[0];
      expect(call[0]).toContain('base_quantization_type');
      expect(call[0]).toContain('max_degree');
      expect(call[0]).toContain('ef_construction');
    });

    it('should include reorder params when useReorder is true', async () => {
      const store = new HologresVectorStore(embeddings, {
        ...defaultArgs,
        indexSettings: {
          baseQuantizationType: 'rabitq',
          useReorder: true,
          preciseQuantizationType: 'fp32',
          preciseIoType: 'block_memory_io',
          maxDegree: 64,
          efConstruction: 400,
        },
      });

      await store.ensureVectorIndex();

      const call = (mockPool.query as jest.Mock).mock.calls[0];
      expect(call[0]).toContain('use_reorder');
      expect(call[0]).toContain('precise_quantization_type');
      expect(call[0]).toContain('precise_io_type');
    });

    it('should not include reorder params when useReorder is false', async () => {
      const store = new HologresVectorStore(embeddings, {
        ...defaultArgs,
        indexSettings: {
          baseQuantizationType: 'rabitq',
          useReorder: false,
          maxDegree: 64,
          efConstruction: 400,
        },
      });

      await store.ensureVectorIndex();

      const call = (mockPool.query as jest.Mock).mock.calls[0];
      expect(call[0]).not.toContain('use_reorder');
      expect(call[0]).not.toContain('precise_quantization_type');
    });
  });

  describe('static initialize()', () => {
    it('should create client, table, and index', async () => {
      const store = await HologresVectorStore.initialize(embeddings, defaultArgs);

      expect(mockPool.connect).toHaveBeenCalled();
      expect(mockPool.query).toHaveBeenCalledTimes(2); // table + index
      expect(store).toBeInstanceOf(HologresVectorStore);
    });
  });

  describe('static fromDocuments()', () => {
    it('should initialize and add documents', async () => {
      const documents = [new Document({ pageContent: 'test' })];

      const store = await HologresVectorStore.fromDocuments(documents, embeddings, defaultArgs);

      expect(store).toBeInstanceOf(HologresVectorStore);
      expect(mockPool.query).toHaveBeenCalled();
    });
  });
});