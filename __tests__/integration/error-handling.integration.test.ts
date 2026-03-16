/**
 * Error Handling Integration Tests
 * Tests SQL injection protection, invalid parameters, and error conditions
 *
 * These tests require actual Hologres database connection.
 * Set environment variables or create .env.test file.
 */

import {
  canRunIntegrationTests,
  createTestPool,
  generateTestTableName,
  cleanupTable,
  defaultTestConfig,
} from './setup';
import { HologresVectorStore, quoteIdentifier, VALID_IDENTIFIER } from '../../nodes/VectorStoreHologres/HologresVectorStore';
import { FakeEmbeddings } from '../mocks/embeddings.mock';
import { Document } from '@langchain/core/documents';
import pg from 'pg';

// Skip all tests if database is not configured
const describeIntegration = canRunIntegrationTests() ? describe : describe.skip;

describeIntegration('Error Handling Integration Tests', () => {
  let pool: pg.Pool;
  let embeddings: FakeEmbeddings;
  const dimensions = 128;

  beforeAll(async () => {
    pool = createTestPool();
    embeddings = new FakeEmbeddings(dimensions);
  });

  afterAll(async () => {
    await pool.end();
  });

  describe('SQL Injection Protection', () => {
    const maliciousTableNames = [
      'users; DROP TABLE users;--',
      'table`name`with`backticks',
      "table'name'with'quotes",
      'table"name"with"double',
      'table with spaces',
      'table;DELETE FROM users',
      'table UNION SELECT * FROM users',
      '../../../etc/passwd',
    ];

    it.each(maliciousTableNames)('should reject malicious table name: %s', async (maliciousName) => {
      expect(() => quoteIdentifier(maliciousName)).toThrow(
        /Invalid SQL identifier/
      );
    });

    const maliciousColumnNames = [
      "id; DROP TABLE users;--",
      "column'name",
      'column"name',
      'col;umn',
      'col->name',
    ];

    it.each(maliciousColumnNames)('should reject malicious column name: %s', async (maliciousName) => {
      expect(() => quoteIdentifier(maliciousName)).toThrow(
        /Invalid SQL identifier/
      );
    });

    it('should reject metadata filter with malicious key', async () => {
      const tableName = generateTestTableName('filter_test');
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      try {
        const queryVector = await embeddings.embedQuery('test');

        // Malicious filter key should be rejected
        await expect(
          store.similaritySearchVectorWithScore(queryVector, 1, {
            "key'; DROP TABLE users;--": 'value',
          })
        ).rejects.toThrow(/Invalid metadata filter key/);
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });

    it('should accept valid identifiers', () => {
      const validNames = [
        'users',
        'Users',
        'users_table',
        'UsersTable',
        'users123',
        '_users',
        'users_',
      ];

      validNames.forEach((name) => {
        expect(VALID_IDENTIFIER.test(name)).toBe(true);
      });
    });
  });

  describe('Database Error Handling', () => {
    it('should handle duplicate ID insertion', async () => {
      const tableName = generateTestTableName('duplicate_id');
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      try {
        const doc = new Document({ pageContent: 'Test document', metadata: { id: 1 } });
        const customId = 'duplicate-test-id';

        // First insert should succeed
        await store.addDocuments([doc], { ids: [customId] });

        // Second insert with same ID should fail
        await expect(
          store.addDocuments([doc], { ids: [customId] })
        ).rejects.toThrow();

        // Verify only one document exists
        const result = await pool.query(`SELECT COUNT(*) FROM "${tableName}"`);
        expect(parseInt(result.rows[0].count, 10)).toBe(1);
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });

    it('should handle wrong dimension vectors', async () => {
      const tableName = generateTestTableName('wrong_dim');
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      try {
        // Create embeddings with wrong dimensions
        const wrongEmbeddings = new FakeEmbeddings(256); // Different from store dimensions

        const doc = new Document({ pageContent: 'Test document' });

        // This should fail because the vector dimensions don't match the CHECK constraint
        await expect(
          store.addDocuments([doc])
        ).resolves.toBeDefined(); // addDocuments uses store's embeddings, not wrongEmbeddings

        // But if we try to insert a vector directly with wrong dimensions via raw SQL
        const wrongVector = Array(64).fill(0.5); // Wrong dimensions
        const embeddingString = `{${wrongVector.join(',')}}`;

        await expect(
          pool.query(
            `INSERT INTO "${tableName}"(id, text, embedding, metadata) VALUES ($1, $2, $3::float4[], $4::jsonb)`,
            ['wrong-dim-id', 'test', embeddingString, '{}']
          )
        ).rejects.toThrow();
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });

    it('should handle querying non-existent table', async () => {
      const tableName = generateTestTableName('non_existent');
      const store = new HologresVectorStore(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      // Don't initialize - table doesn't exist

      const queryVector = await embeddings.embedQuery('test');
      await expect(
        store.similaritySearchVectorWithScore(queryVector, 1)
      ).rejects.toThrow();
    });

    it('should handle update of non-existent document', async () => {
      const tableName = generateTestTableName('update_nonexistent');
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      try {
        const updatedDoc = new Document({
          pageContent: 'Updated content',
          metadata: { updated: true },
        });

        // Update non-existent document should succeed but not change anything
        await store.update({ id: 'non-existent-id', document: updatedDoc });

        // Verify table is still empty
        const result = await pool.query(`SELECT COUNT(*) FROM "${tableName}"`);
        expect(parseInt(result.rows[0].count, 10)).toBe(0);
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });

    it('should handle delete of non-existent IDs', async () => {
      const tableName = generateTestTableName('delete_nonexistent');
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      try {
        // Delete non-existent IDs should not throw
        await expect(
          store.delete({ ids: ['non-existent-1', 'non-existent-2'] })
        ).resolves.not.toThrow();
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('Connection Error Handling', () => {
    it('should handle invalid credentials gracefully', async () => {
      const invalidPool = new pg.Pool({
        host: 'invalid-host-that-does-not-exist',
        port: 5432,
        database: 'invalid_db',
        user: 'invalid_user',
        password: 'invalid_password',
        connectionTimeoutMillis: 5000, // Short timeout for tests
      });

      const tableName = generateTestTableName('invalid_conn');
      const store = new HologresVectorStore(embeddings, {
        pool: invalidPool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      await expect(store.ensureTableInDatabase()).rejects.toThrow();

      await invalidPool.end();
    });
  });
});