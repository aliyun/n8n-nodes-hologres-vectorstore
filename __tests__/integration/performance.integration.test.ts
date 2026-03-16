/**
 * Performance Integration Tests
 * Tests bulk operations, batch processing, and concurrent operations
 *
 * These tests require actual Hologres database connection.
 * Set environment variables or create .env.test file.
 *
 * Note: These tests have extended timeouts (30-120 seconds)
 */

import {
  canRunIntegrationTests,
  createTestPool,
  generateTestTableName,
  cleanupTable,
  defaultTestConfig,
  measureTime,
  generateBulkDocuments,
} from './setup';
import { HologresVectorStore } from '../../nodes/VectorStoreHologres/HologresVectorStore';
import { FakeEmbeddings } from '../mocks/embeddings.mock';
import { Document } from '@langchain/core/documents';
import pg from 'pg';

// Skip all tests if database is not configured
const describeIntegration = canRunIntegrationTests() ? describe : describe.skip;

describeIntegration('Performance Integration Tests', () => {
  let pool: pg.Pool;
  const dimensions = 128;

  beforeAll(async () => {
    pool = createTestPool();
  });

  afterAll(async () => {
    await pool.end();
  });

  describe('Bulk Insert Operations', () => {
    it(
      'should insert 100 documents efficiently',
      async () => {
        const tableName = generateTestTableName('bulk_100');
        const embeddings = new FakeEmbeddings(dimensions);
        const store = await HologresVectorStore.initialize(embeddings, {
          pool,
          tableName,
          dimensions,
          ...defaultTestConfig,
        });

        try {
          const docs = generateBulkDocuments(100, 'Bulk doc');
          const ids = await store.addDocuments(docs);

          expect(ids).toHaveLength(100);

          // Verify count
          const result = await pool.query(`SELECT COUNT(*) FROM "${tableName}"`);
          expect(parseInt(result.rows[0].count, 10)).toBe(100);
        } finally {
          store.client?.release();
          await cleanupTable(pool, tableName);
        }
      },
      30000
    );

    it(
      'should insert 500 documents efficiently',
      async () => {
        const tableName = generateTestTableName('bulk_500');
        const embeddings = new FakeEmbeddings(dimensions);
        const store = await HologresVectorStore.initialize(embeddings, {
          pool,
          tableName,
          dimensions,
          ...defaultTestConfig,
        });

        try {
          const docs = generateBulkDocuments(500, 'Bulk doc');
          const ids = await store.addDocuments(docs);

          expect(ids).toHaveLength(500);

          // Verify count
          const result = await pool.query(`SELECT COUNT(*) FROM "${tableName}"`);
          expect(parseInt(result.rows[0].count, 10)).toBe(500);
        } finally {
          store.client?.release();
          await cleanupTable(pool, tableName);
        }
      },
      60000
    );
  });

  describe('Search Performance', () => {
    it(
      'should search efficiently in 1000 documents',
      async () => {
        const tableName = generateTestTableName('search_1000');
        const embeddings = new FakeEmbeddings(dimensions);
        const store = await HologresVectorStore.initialize(embeddings, {
          pool,
          tableName,
          dimensions,
          ...defaultTestConfig,
        });

        try {
          // Insert 1000 documents
          const docs = generateBulkDocuments(1000, 'Search test doc');
          await store.addDocuments(docs);

          // Measure search time
          const queryVector = await embeddings.embedQuery('Search test');
          const { result, durationMs } = await measureTime(() =>
            store.similaritySearchVectorWithScore(queryVector, 10)
          );

          expect(result).toHaveLength(10);
          console.log(`Search 1000 docs took ${durationMs}ms`);

          // Search should be reasonably fast (less than 5 seconds)
          expect(durationMs).toBeLessThan(5000);
        } finally {
          store.client?.release();
          await cleanupTable(pool, tableName);
        }
      },
      120000
    );
  });

  describe('Large Metadata Operations', () => {
    it(
      'should handle documents with large metadata objects',
      async () => {
        const tableName = generateTestTableName('large_meta');
        const embeddings = new FakeEmbeddings(dimensions);
        const store = await HologresVectorStore.initialize(embeddings, {
          pool,
          tableName,
          dimensions,
          ...defaultTestConfig,
        });

        try {
          // Create documents with large metadata (100 keys each)
          const docs = Array.from({ length: 10 }, (_, i) => {
            const largeMetadata = Object.fromEntries(
              Array.from({ length: 100 }, (_, j) => [`key_${j}`, `value_${i}_${j}`])
            );
            return new Document({
              pageContent: `Document ${i} with large metadata`,
              metadata: largeMetadata,
            });
          });

          const ids = await store.addDocuments(docs);
          expect(ids).toHaveLength(10);

          // Verify metadata is preserved
          const result = await pool.query(`SELECT metadata FROM "${tableName}" LIMIT 1`);
          expect(Object.keys(result.rows[0].metadata)).toHaveLength(100);
        } finally {
          store.client?.release();
          await cleanupTable(pool, tableName);
        }
      },
      30000
    );
  });

  describe('Concurrent Operations', () => {
    it(
      'should handle concurrent reads and writes',
      async () => {
        const tableName = generateTestTableName('concurrent');
        const embeddings = new FakeEmbeddings(dimensions);
        const store = await HologresVectorStore.initialize(embeddings, {
          pool,
          tableName,
          dimensions,
          ...defaultTestConfig,
        });

        try {
          // Initial data
          const initialDocs = generateBulkDocuments(10, 'Initial doc');
          await store.addDocuments(initialDocs);

          // Concurrent operations: insert and search simultaneously
          const insertPromise = store.addDocuments(generateBulkDocuments(5, 'Concurrent insert'));
          const searchPromise = embeddings.embedQuery('test').then((vec) =>
            store.similaritySearchVectorWithScore(vec, 5)
          );

          const [insertResult, searchResult] = await Promise.all([insertPromise, searchPromise]);

          expect(insertResult).toHaveLength(5);
          expect(searchResult.length).toBeGreaterThan(0);
        } finally {
          store.client?.release();
          await cleanupTable(pool, tableName);
        }
      },
      30000
    );

    it(
      'should handle multiple concurrent searches',
      async () => {
        const tableName = generateTestTableName('concurrent_search');
        const embeddings = new FakeEmbeddings(dimensions);
        const store = await HologresVectorStore.initialize(embeddings, {
          pool,
          tableName,
          dimensions,
          ...defaultTestConfig,
        });

        try {
          // Insert documents
          await store.addDocuments(generateBulkDocuments(50, 'Search target'));

          // Multiple concurrent searches
          const searchPromises = Array.from({ length: 5 }, (_, i) =>
            embeddings.embedQuery(`query ${i}`).then((vec) =>
              store.similaritySearchVectorWithScore(vec, 5)
            )
          );

          const results = await Promise.all(searchPromises);

          results.forEach((result) => {
            expect(result.length).toBeGreaterThan(0);
          });
        } finally {
          store.client?.release();
          await cleanupTable(pool, tableName);
        }
      },
      30000
    );
  });
});