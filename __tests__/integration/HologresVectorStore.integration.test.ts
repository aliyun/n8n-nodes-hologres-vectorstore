/**
 * HologresVectorStore Integration Tests
 * Tests real database operations against a Hologres instance
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
} from './setup';
import { HologresVectorStore } from '../../nodes/VectorStoreHologres/HologresVectorStore';
import { FakeEmbeddings } from '../mocks/embeddings.mock';
import { Document } from '@langchain/core/documents';
import pg from 'pg';

// Skip all tests if database is not configured
const describeIntegration = canRunIntegrationTests() ? describe : describe.skip;

describeIntegration('HologresVectorStore Integration Tests', () => {
  let pool: pg.Pool;
  let tableName: string;
  let embeddings: FakeEmbeddings;
  const dimensions = 128; // Use smaller dimensions for faster tests

  // Default test configuration
  const defaultConfig = {
    columns: {
      idColumnName: 'id',
      vectorColumnName: 'embedding',
      contentColumnName: 'text',
      metadataColumnName: 'metadata',
    },
    distanceMethod: 'Cosine' as const,
    indexSettings: {
      baseQuantizationType: 'rabitq',
      useReorder: true,
      preciseQuantizationType: 'fp32',
      preciseIoType: 'block_memory_io',
      maxDegree: 32,
      efConstruction: 200,
    },
  };

  beforeAll(async () => {
    pool = createTestPool();
    tableName = generateTestTableName();
    embeddings = new FakeEmbeddings(dimensions);
  });

  afterAll(async () => {
    await cleanupTable(pool, tableName);
    await pool.end();
  });

  describe('Table and Index Creation', () => {
    it('should create table with correct schema', async () => {
      const store = new HologresVectorStore(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultConfig,
      });

      await store.ensureTableInDatabase();

      // Verify table exists
      const result = await pool.query(
        `SELECT column_name, data_type
         FROM information_schema.columns
         WHERE table_name = $1`,
        [tableName]
      );

      expect(result.rows).toHaveLength(4);
      const columnNames = result.rows.map((r) => r.column_name);
      expect(columnNames).toContain('id');
      expect(columnNames).toContain('text');
      expect(columnNames).toContain('embedding');
      expect(columnNames).toContain('metadata');
    });

    it('should set HGraph vector index on table', async () => {
      const store = new HologresVectorStore(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultConfig,
      });

      await store.ensureTableInDatabase();
      await store.ensureVectorIndex();

      // Verify index is set (check table options)
      const result = await pool.query(
        `SELECT reloptions FROM pg_class WHERE relname = $1`,
        [tableName]
      );

      expect(result.rows[0].reloptions).toBeDefined();
      // HGraph index should be mentioned in options
      const options = result.rows[0].reloptions.join(',');
      expect(options).toContain('vectors');
    });
  });

  describe('Document Operations', () => {
    let store: HologresVectorStore;

    beforeEach(async () => {
      store = new HologresVectorStore(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultConfig,
      });
      await store._initializeClient();
    });

    it('should insert documents with auto-generated IDs', async () => {
      const docs = [
        new Document({ pageContent: 'Test document 1', metadata: { source: 'test' } }),
        new Document({ pageContent: 'Test document 2', metadata: { source: 'test' } }),
      ];

      const ids = await store.addDocuments(docs);

      expect(ids).toHaveLength(2);
      expect(ids[0]).toBeDefined();
      expect(ids[1]).toBeDefined();

      // Verify documents are in database
      const result = await pool.query(`SELECT * FROM "${tableName}"`);
      expect(result.rows).toHaveLength(2);
    });

    it('should insert documents with custom IDs', async () => {
      const customIds = ['custom-id-1', 'custom-id-2'];
      const docs = [
        new Document({ pageContent: 'Custom doc 1', metadata: { type: 'custom' } }),
        new Document({ pageContent: 'Custom doc 2', metadata: { type: 'custom' } }),
      ];

      const ids = await store.addDocuments(docs, { ids: customIds });

      expect(ids).toEqual(customIds);

      // Verify custom IDs
      const result = await pool.query(
        `SELECT id FROM "${tableName}" WHERE id = ANY($1)`,
        [customIds]
      );
      expect(result.rows).toHaveLength(2);
    });

    it('should update existing document', async () => {
      // First insert a document
      const doc = new Document({ pageContent: 'Original content', metadata: { version: 1 } });
      const [id] = await store.addDocuments([doc]);

      // Update the document
      const updatedDoc = new Document({
        pageContent: 'Updated content',
        metadata: { version: 2 },
      });

      await store.update({ id, document: updatedDoc });

      // Verify update
      const result = await pool.query(`SELECT * FROM "${tableName}" WHERE id = $1`, [id]);
      expect(result.rows[0].text).toBe('Updated content');
      expect(result.rows[0].metadata.version).toBe(2);
    });

    it('should delete documents by IDs', async () => {
      // Insert documents
      const docs = [
        new Document({ pageContent: 'To delete 1' }),
        new Document({ pageContent: 'To keep' }),
      ];
      const ids = await store.addDocuments(docs);

      // Delete first document
      await store.delete({ ids: [ids[0]] });

      // Verify deletion
      const result = await pool.query(`SELECT id FROM "${tableName}"`);
      expect(result.rows).toHaveLength(1);
      expect(result.rows[0].id).toBe(ids[1]);
    });
  });

  describe('Similarity Search', () => {
    let store: HologresVectorStore;
    let insertedIds: string[];

    beforeAll(async () => {
      store = new HologresVectorStore(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultConfig,
      });
      await store._initializeClient();

      // Insert test documents
      const docs = [
        new Document({ pageContent: 'The quick brown fox', metadata: { category: 'animal' } }),
        new Document({ pageContent: 'A lazy dog sleeps', metadata: { category: 'animal' } }),
        new Document({ pageContent: 'The blue sky', metadata: { category: 'nature' } }),
        new Document({ pageContent: 'Green grass grows', metadata: { category: 'nature' } }),
      ];
      insertedIds = await store.addDocuments(docs);
    });

    it('should return similar documents with scores', async () => {
      const queryVector = await embeddings.embedQuery('quick fox animal');
      const results = await store.similaritySearchVectorWithScore(queryVector, 2);

      expect(results).toHaveLength(2);
      expect(results[0][0].pageContent).toBeDefined();
      expect(typeof results[0][1]).toBe('number');
    });

    it('should respect k parameter for result count', async () => {
      const queryVector = await embeddings.embedQuery('nature');
      const results = await store.similaritySearchVectorWithScore(queryVector, 1);

      expect(results).toHaveLength(1);
    });

    it('should filter by metadata', async () => {
      const queryVector = await embeddings.embedQuery('something');
      const results = await store.similaritySearchVectorWithScore(queryVector, 4, {
        category: 'animal',
      });

      // Should only return animal category documents
      expect(results.length).toBeGreaterThan(0);
      results.forEach(([doc]) => {
        expect(doc.metadata.category).toBe('animal');
      });
    });
  });

  describe('Distance Methods', () => {
    const distanceMethods: Array<'Cosine' | 'InnerProduct' | 'Euclidean'> = [
      'Cosine',
      'InnerProduct',
      'Euclidean',
    ];

    distanceMethods.forEach((method) => {
      it(`should work with ${method} distance`, async () => {
        const testTable = generateTestTableName('test_dist');
        const store = new HologresVectorStore(embeddings, {
          pool,
          tableName: testTable,
          dimensions,
          ...defaultConfig,
          distanceMethod: method,
        });

        try {
          await store._initializeClient();
          await store.ensureTableInDatabase();
          await store.ensureVectorIndex();

          // Insert and search
          const doc = new Document({ pageContent: 'Test for distance' });
          await store.addDocuments([doc]);

          const queryVector = await embeddings.embedQuery('Test for distance');
          const results = await store.similaritySearchVectorWithScore(queryVector, 1);

          expect(results).toHaveLength(1);
        } finally {
          await cleanupTable(pool, testTable);
        }
      });
    });
  });

  describe('Static Methods', () => {
    it('should initialize via static method', async () => {
      const testTable = generateTestTableName('test_static');
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName: testTable,
        dimensions,
        ...defaultConfig,
      });

      try {
        expect(store).toBeInstanceOf(HologresVectorStore);
        expect(store.client).toBeDefined();

        // Verify table was created
        const result = await pool.query(
          `SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)`,
          [testTable]
        );
        expect(result.rows[0].exists).toBe(true);
      } finally {
        store.client?.release();
        await cleanupTable(pool, testTable);
      }
    });

    it('should create from documents', async () => {
      const testTable = generateTestTableName('test_from_docs');
      const docs = [
        new Document({ pageContent: 'Doc 1' }),
        new Document({ pageContent: 'Doc 2' }),
      ];

      const store = await HologresVectorStore.fromDocuments(docs, embeddings, {
        pool,
        tableName: testTable,
        dimensions,
        ...defaultConfig,
      });

      try {
        // Verify documents were inserted
        const result = await pool.query(`SELECT COUNT(*) FROM "${testTable}"`);
        expect(parseInt(result.rows[0].count, 10)).toBe(2);
      } finally {
        store.client?.release();
        await cleanupTable(pool, testTable);
      }
    });
  });
});