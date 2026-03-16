/**
 * Mock for HologresVectorStore
 * Provides mock implementations for testing VectorStoreHologres node
 */

import { jest } from '@jest/globals';
import { Document } from '@langchain/core/documents';

export interface MockHologresVectorStore {
  _initializeClient: jest.Mock;
  similaritySearchVectorWithScore: jest.Mock;
  addDocuments: jest.Mock;
  update: jest.Mock;
  client: { release: jest.Mock };
  pool: { end: jest.Mock };
}

/**
 * Create a mock HologresVectorStore instance
 */
export function createMockHologresVectorStore(): MockHologresVectorStore {
  return {
    _initializeClient: jest.fn().mockResolvedValue(undefined),
    similaritySearchVectorWithScore: jest.fn().mockResolvedValue([
      [new Document({ pageContent: 'test doc 1', metadata: { source: 'test' } }), 0.95],
      [new Document({ pageContent: 'test doc 2', metadata: { source: 'test' } }), 0.85],
    ]),
    addDocuments: jest.fn().mockResolvedValue(['id-1', 'id-2']),
    update: jest.fn().mockResolvedValue(undefined),
    client: { release: jest.fn() },
    pool: { end: jest.fn().mockResolvedValue(undefined) },
  };
}

/**
 * Create a mock for HologresVectorStore.initialize static method
 */
export function createMockInitialize(
  mockStore: MockHologresVectorStore,
): jest.Mock {
  return jest.fn().mockResolvedValue(mockStore);
}