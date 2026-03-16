/**
 * Edge Cases Integration Tests
 * Tests boundary conditions, special characters, and unusual data patterns
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
  generateEdgeCaseDocuments,
} from './setup';
import { HologresVectorStore } from '../../nodes/VectorStoreHologres/HologresVectorStore';
import { FakeEmbeddings } from '../mocks/embeddings.mock';
import { Document } from '@langchain/core/documents';
import pg from 'pg';

// Skip all tests if database is not configured
const describeIntegration = canRunIntegrationTests() ? describe : describe.skip;

describeIntegration('Edge Cases Integration Tests', () => {
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

  describe('Empty Table Operations', () => {
    it('should return empty array when searching empty table', async () => {
      const tableName = generateTestTableName('empty_table');
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      try {
        const queryVector = await embeddings.embedQuery('test query');
        const results = await store.similaritySearchVectorWithScore(queryVector, 5);

        expect(results).toEqual([]);
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });

    it('should handle delete all documents leaving empty table', async () => {
      const tableName = generateTestTableName('delete_all');
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      try {
        // Insert some documents
        const docs = [
          new Document({ pageContent: 'Doc 1' }),
          new Document({ pageContent: 'Doc 2' }),
        ];
        const ids = await store.addDocuments(docs);

        // Delete all
        await store.delete({ ids });

        // Search should return empty
        const queryVector = await embeddings.embedQuery('test');
        const results = await store.similaritySearchVectorWithScore(queryVector, 5);
        expect(results).toEqual([]);
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('Metadata Handling', () => {
    it('should handle null metadata (defaults to empty object)', async () => {
      const tableName = generateTestTableName('null_metadata');
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      try {
        const doc = new Document({ pageContent: 'Doc without explicit metadata' });
        const [id] = await store.addDocuments([doc]);

        // Verify in database
        const result = await pool.query(`SELECT * FROM "${tableName}" WHERE id = $1`, [id]);
        expect(result.rows[0].metadata).toEqual({});
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });

    it('should preserve complex nested metadata', async () => {
      const tableName = generateTestTableName('nested_metadata');
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      try {
        const complexMetadata = {
          string: 'value',
          number: 42,
          boolean: true,
          null: null,
          nested: {
            level1: {
              level2: {
                array: [1, 2, 3],
                object: { a: 'b' },
              },
            },
          },
          array: [1, 'two', { three: 3 }],
        };

        const doc = new Document({
          pageContent: 'Document with complex metadata',
          metadata: complexMetadata,
        });

        const [id] = await store.addDocuments([doc]);

        // Verify metadata is preserved
        const result = await pool.query(`SELECT metadata FROM "${tableName}" WHERE id = $1`, [id]);
        expect(result.rows[0].metadata).toEqual(complexMetadata);
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });

    it('should handle metadata with 100 keys', async () => {
      const tableName = generateTestTableName('large_metadata');
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      try {
        const largeMetadata = Object.fromEntries(
          Array.from({ length: 100 }, (_, i) => [`key_${i}`, `value_${i}`])
        );

        const doc = new Document({
          pageContent: 'Document with large metadata',
          metadata: largeMetadata,
        });

        const [id] = await store.addDocuments([doc]);

        // Verify metadata is preserved
        const result = await pool.query(`SELECT metadata FROM "${tableName}" WHERE id = $1`, [id]);
        expect(result.rows[0].metadata).toEqual(largeMetadata);
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('Unicode and Special Characters', () => {
    it('should handle Chinese characters', async () => {
      const tableName = generateTestTableName('chinese');
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      try {
        const doc = new Document({
          pageContent: '这是中文测试内容，包含特殊字符：你好世界！',
          metadata: { language: 'zh-CN' },
        });

        const [id] = await store.addDocuments([doc]);

        // Verify content is preserved
        const result = await pool.query(`SELECT text FROM "${tableName}" WHERE id = $1`, [id]);
        expect(result.rows[0].text).toBe('这是中文测试内容，包含特殊字符：你好世界！');
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });

    it('should handle Japanese characters', async () => {
      const tableName = generateTestTableName('japanese');
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      try {
        const doc = new Document({
          pageContent: '日本語テストコンテンツ：こんにちは世界！',
          metadata: { language: 'ja-JP' },
        });

        const [id] = await store.addDocuments([doc]);

        const result = await pool.query(`SELECT text FROM "${tableName}" WHERE id = $1`, [id]);
        expect(result.rows[0].text).toBe('日本語テストコンテンツ：こんにちは世界！');
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });

    it('should handle emojis', async () => {
      const tableName = generateTestTableName('emoji');
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      try {
        const doc = new Document({
          pageContent: 'Emoji test: 🚀 🎉 ✅ ❤️ 🔥 🌟 👍 🎵 📱 💻',
          metadata: { type: 'emoji' },
        });

        const [id] = await store.addDocuments([doc]);

        const result = await pool.query(`SELECT text FROM "${tableName}" WHERE id = $1`, [id]);
        expect(result.rows[0].text).toBe('Emoji test: 🚀 🎉 ✅ ❤️ 🔥 🌟 👍 🎵 📱 💻');
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });

    it('should safely handle SQL special characters in content', async () => {
      const tableName = generateTestTableName('sql_chars');
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      try {
        const doc = new Document({
          pageContent: "Content with 'single quotes', \"double quotes\", semicolon; and backslash\\",
          metadata: { has_special_chars: true },
        });

        const [id] = await store.addDocuments([doc]);

        const result = await pool.query(`SELECT text FROM "${tableName}" WHERE id = $1`, [id]);
        expect(result.rows[0].text).toBe("Content with 'single quotes', \"double quotes\", semicolon; and backslash\\");
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('Content Size Boundaries', () => {
    it('should handle empty content string', async () => {
      const tableName = generateTestTableName('empty_content');
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      try {
        const doc = new Document({
          pageContent: '',
          metadata: { type: 'empty' },
        });

        const [id] = await store.addDocuments([doc]);

        const result = await pool.query(`SELECT text FROM "${tableName}" WHERE id = $1`, [id]);
        expect(result.rows[0].text).toBe('');
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });

    it('should handle 100KB content', async () => {
      const tableName = generateTestTableName('large_content');
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      try {
        const largeContent = 'x'.repeat(100000); // 100KB
        const doc = new Document({
          pageContent: largeContent,
          metadata: { size: '100kb' },
        });

        const [id] = await store.addDocuments([doc]);

        const result = await pool.query(`SELECT text FROM "${tableName}" WHERE id = $1`, [id]);
        expect(result.rows[0].text).toBe(largeContent);
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });

    it('should handle content with newlines and tabs', async () => {
      const tableName = generateTestTableName('whitespace');
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions,
        ...defaultTestConfig,
      });

      try {
        const doc = new Document({
          pageContent: 'Line 1\nLine 2\r\nLine 3\tTabbed\tContent',
          metadata: { has_whitespace: true },
        });

        const [id] = await store.addDocuments([doc]);

        const result = await pool.query(`SELECT text FROM "${tableName}" WHERE id = $1`, [id]);
        expect(result.rows[0].text).toBe('Line 1\nLine 2\r\nLine 3\tTabbed\tContent');
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });
  });
});