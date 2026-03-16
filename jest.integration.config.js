/** @type {import('ts-jest').JestConfigWithTsJest} */
/**
 * Jest configuration for integration tests
 * Uses real database connections (no mocks)
 */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/__tests__/integration'],
  testMatch: ['**/*.integration.test.ts'],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
  // No global setup file - integration tests handle their own setup
  setupFilesAfterEnv: [],
  moduleNameMapper: {
    '^(\\.{1,2}/.*)\\.js$': '$1',
  },
  transform: {
    '^.+\\.tsx?$': [
      'ts-jest',
      {
        useESM: false,
      },
    ],
  },
  // Longer timeout for database operations
  testTimeout: 30000,
  // Run tests sequentially to avoid database conflicts
  maxConcurrency: 1,
};