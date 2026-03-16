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
  return new pg.Pool({
    host: testConfig.host,
    port: testConfig.port,
    database: testConfig.database,
    user: testConfig.user,
    password: testConfig.password,
    max: 5,
    ssl: { rejectUnauthorized: false },
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