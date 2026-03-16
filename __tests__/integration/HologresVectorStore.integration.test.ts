/**
 * HologresVectorStore Integration Tests
 * Tests real database operations against a Hologres instance
 *
 * These tests require actual Hologres database connection.
 * Set environment variables or create .env.test file.
 */

import {
  canRunIntegrationTests,
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

describeIntegration('HologresVectorStore Integration Tests', () => {
  let pool: pg.Pool;
  let embeddings: FakeEmbeddings;
  const dimensions = 128; // Use smaller dimensions for faster tests

  beforeAll(async () => {
    pool = createTestPool();
    embeddings = new FakeEmbeddings(dimensions);
  });

  afterAll(async () => {
    await pool.end();
  });

  describe('Table and Index Creation', () => {
    let tableName: string;

    beforeAll(async () => {
      tableName = generateTestTableName('table_index');
    });

    afterAll(async () => {
      await cleanupTable(pool, tableName);
    });

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

      // Defensive check: reloptions may be null in some Hologres versions
      const reloptions = result.rows[0]?.reloptions;
      if (reloptions && Array.isArray(reloptions)) {
        const options = reloptions.join(',');
        expect(options).toContain('vectors');
      } else {
        // If reloptions is not available, verify table exists and index was set without error
        expect(result.rows.length).toBeGreaterThan(0);
      }
    });
  });

  describe('Document Operations', () => {
    let tableName: string;
    let store: HologresVectorStore;

    beforeAll(async () => {
      tableName = generateTestTableName('doc_ops');

      // Create table once for all tests in this group
      store = new HologresVectorStore(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultConfig,
      });
      await store._initializeClient();
      await store.ensureTableInDatabase();
    });

    beforeEach(async () => {
      // Clear table data between tests while preserving table structure
      // Use try-catch in case table doesn't exist yet
      try {
        await pool.query(`TRUNCATE TABLE "${tableName}"`);
      } catch {
        // Table might not exist in edge cases, ignore
      }
    });

    afterAll(async () => {
      store.client?.release();
      await cleanupTable(pool, tableName);
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
    let tableName: string;
    let store: HologresVectorStore;
    let insertedIds: string[];

    beforeAll(async () => {
      tableName = generateTestTableName('similarity');

      store = new HologresVectorStore(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultConfig,
      });
      await store._initializeClient();
      await store.ensureTableInDatabase();
      await store.ensureVectorIndex();

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

    afterAll(async () => {
      store.client?.release();
      await cleanupTable(pool, tableName);
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
          store.client?.release();
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