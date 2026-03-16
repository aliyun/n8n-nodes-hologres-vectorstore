/**
 * Integration Test Setup
 * Configures environment for integration tests with real Hologres database
 *
 * Set environment variables in .env.test or directly:
 * - HOLOGRES_HOST
 * - HOLOGRES_PORT (default: 80)
 * - HOLOGRES_DATABASE
 * - HOLOGRES_USER
 * - HOLOGRES_PASSWORD
 */

import pg from 'pg';
import * as dotenv from 'dotenv';
import * as path from 'path';

// Load .env.test if it exists
dotenv.config({ path: path.resolve(__dirname, '../../.env.test') });

// Test configuration from environment
export const testConfig = {
  host: process.env.HOLOGRES_HOST || '',
  port: parseInt(process.env.HOLOGRES_PORT || '80', 10),
  database: process.env.HOLOGRES_DATABASE || '',
  user: process.env.HOLOGRES_USER || '',
  password: process.env.HOLOGRES_PASSWORD || '',
  ssl: process.env.HOLOGRES_SSL || 'disable',
};

/**
 * Check if integration tests can run (database connection available)
 */
export function canRunIntegrationTests(): boolean {
  return !!(testConfig.host && testConfig.database && testConfig.user);
}

/**
 * Skip reason when database is not available
 */
export const skipReason = 'HOLOGRES_* environment variables not set. Skip integration tests.';

/**
 * Create a test connection pool
 */
export function createTestPool(): pg.Pool {
  // SSL configuration matching main code logic
  let ssl: boolean | { rejectUnauthorized: boolean } = false;
  if (testConfig.ssl === 'require') {
    ssl = true;
  } else if (testConfig.ssl === 'allow-unauthorized') {
    ssl = { rejectUnauthorized: false };
  }
  // 'disable' or other values -> ssl = false (default)

  return new pg.Pool({
    host: testConfig.host,
    port: testConfig.port,
    database: testConfig.database,
    user: testConfig.user,
    password: testConfig.password,
    max: 5,
    ssl,
  });
}

/**
 * Generate a unique test table name to avoid collisions
 */
export function generateTestTableName(prefix = 'test_vectors'): string {
  const timestamp = Date.now();
  const randomId = Math.random().toString(36).substring(2, 8);
  return `${prefix}_${timestamp}_${randomId}`;
}

/**
 * Clean up test table
 */
export async function cleanupTable(pool: pg.Pool, tableName: string): Promise<void> {
  try {
    await pool.query(`DROP TABLE IF EXISTS ${quoteIdentifier(tableName)}`);
  } catch (error) {
    console.warn(`Failed to cleanup table ${tableName}:`, error);
  }
}

/**
 * Quote SQL identifier safely
 */
function quoteIdentifier(name: string): string {
  return `"${name.replace(/"/g, '""')}"`;
}

/**
 * Global setup for integration test suite
 * Runs once before all tests
 */
export async function globalSetup(): Promise<void> {
  if (!canRunIntegrationTests()) {
    console.log('Integration tests: Database not configured, tests will be skipped');
    return;
  }

  const pool = createTestPool();
  try {
    // Verify connection
    await pool.query('SELECT 1');
    console.log('Integration tests: Database connection verified');
  } catch (error) {
    console.error('Integration tests: Failed to connect to database:', error);
    throw error;
  } finally {
    await pool.end();
  }
}

/**
 * Global teardown for integration test suite
 * Runs once after all tests
 */
export async function globalTeardown(): Promise<void> {
  // No global cleanup needed - each test cleans up its own tables
}

// ─── Enhanced Test Utilities ──────────────────────────────────────────────────

import { HologresVectorStore, HologresVectorStoreArgs, DistanceMethod, ColumnOptions, HGraphIndexSettings } from '../../nodes/VectorStoreHologres/HologresVectorStore';
import type { EmbeddingsInterface } from '@langchain/core/embeddings';

/**
 * Default test configuration for HologresVectorStore
 */
export const defaultTestConfig = {
  columns: {
    idColumnName: 'id',
    vectorColumnName: 'embedding',
    contentColumnName: 'text',
    metadataColumnName: 'metadata',
  } as ColumnOptions,
  distanceMethod: 'Cosine' as DistanceMethod,
  indexSettings: {
    baseQuantizationType: 'rabitq',
    useReorder: true,
    preciseQuantizationType: 'fp32',
    preciseIoType: 'block_memory_io',
    maxDegree: 32,
    efConstruction: 200,
  } as HGraphIndexSettings,
};

/**
 * Create a test HologresVectorStore with automatic cleanup
 *
 * @param embeddings - Embeddings instance to use
 * @param options - Partial store options (tableName and dimensions are required)
 * @returns Object containing store and cleanup function
 */
export async function createTestStore(
  embeddings: EmbeddingsInterface,
  options: {
    pool: pg.Pool;
    tableName: string;
    dimensions: number;
    distanceMethod?: DistanceMethod;
    columns?: Partial<ColumnOptions>;
    indexSettings?: Partial<HGraphIndexSettings>;
    filter?: Record<string, unknown>;
  }
): Promise<{
  store: HologresVectorStore;
  cleanup: () => Promise<void>;
}> {
  const store = await HologresVectorStore.initialize(embeddings, {
    pool: options.pool,
    tableName: options.tableName,
    dimensions: options.dimensions,
    columns: {
      ...defaultTestConfig.columns,
      ...options.columns,
    },
    distanceMethod: options.distanceMethod ?? defaultTestConfig.distanceMethod,
    indexSettings: {
      ...defaultTestConfig.indexSettings,
      ...options.indexSettings,
    },
    filter: options.filter,
  });

  return {
    store,
    cleanup: async () => {
      store.client?.release();
      void store.pool.end();
      await cleanupTable(options.pool, options.tableName);
    },
  };
}

/**
 * Measure execution time of an async function
 *
 * @param fn - Async function to measure
 * @returns Object containing result and duration in milliseconds
 */
export async function measureTime<T>(
  fn: () => Promise<T>
): Promise<{ result: T; durationMs: number }> {
  const start = Date.now();
  const result = await fn();
  const durationMs = Date.now() - start;
  return { result, durationMs };
}

/**
 * Generate edge case test documents for boundary testing
 *
 * @param dimensions - Vector dimensions to use for embeddings
 * @returns Array of Documents with various edge cases
 */
export function generateEdgeCaseDocuments(dimensions = 128): Array<{
  name: string;
  document: import('@langchain/core/documents').Document;
}> {
  const { Document } = require('@langchain/core/documents');

  return [
    {
      name: 'empty-content',
      document: new Document({ pageContent: '', metadata: { type: 'empty' } }),
    },
    {
      name: 'null-metadata',
      document: new Document({ pageContent: 'Document with null metadata' }),
    },
    {
      name: 'empty-metadata',
      document: new Document({ pageContent: 'Document with empty metadata', metadata: {} }),
    },
    {
      name: 'complex-nested-metadata',
      document: new Document({
        pageContent: 'Document with nested metadata',
        metadata: {
          level1: {
            level2: {
              level3: 'deep value',
              array: [1, 2, 3],
            },
          },
          mixed: { a: 1, b: 'string', c: true },
        },
      }),
    },
    {
      name: 'unicode-chinese',
      document: new Document({
        pageContent: '这是中文测试内容，包含特殊字符：你好世界！',
        metadata: { language: 'zh-CN', type: 'unicode' },
      }),
    },
    {
      name: 'unicode-japanese',
      document: new Document({
        pageContent: '日本語テストコンテンツ：こんにちは世界！',
        metadata: { language: 'ja-JP', type: 'unicode' },
      }),
    },
    {
      name: 'unicode-emoji',
      document: new Document({
        pageContent: 'Emoji test: 🚀 🎉 ✅ ❤️ 🔥 🌟',
        metadata: { type: 'emoji' },
      }),
    },
    {
      name: 'sql-special-chars',
      document: new Document({
        pageContent: "Content with SQL chars: 'single quotes', \"double quotes\", semicolon; backslash\\",
        metadata: { has_special_chars: true },
      }),
    },
    {
      name: 'large-metadata',
      document: new Document({
        pageContent: 'Document with large metadata object',
        metadata: Object.fromEntries(
          Array.from({ length: 100 }, (_, i) => [`key_${i}`, `value_${i}`])
        ),
      }),
    },
    {
      name: 'very-long-content',
      document: new Document({
        pageContent: 'x'.repeat(100000), // 100KB content
        metadata: { size: '100kb' },
      }),
    },
  ];
}

/**
 * Generate a batch of documents for bulk testing
 *
 * @param count - Number of documents to generate
 * @param prefix - Content prefix for each document
 * @returns Array of Documents
 */
export function generateBulkDocuments(
  count: number,
  prefix = 'Bulk document'
): import('@langchain/core/documents').Document[] {
  const { Document } = require('@langchain/core/documents');
  return Array.from({ length: count }, (_, i) =>
    new Document({
      pageContent: `${prefix} ${i + 1}`,
      metadata: { index: i, batch: true },
    })
  );
}