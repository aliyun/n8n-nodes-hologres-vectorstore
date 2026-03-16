/**
 * Mock for pg module
 * Provides mock implementations for Pool and PoolClient
 */

import { jest } from '@jest/globals';

export interface MockQueryResult {
  rows: Record<string, unknown>[];
  rowCount?: number;
  command?: string;
}

export function createMockPoolClient() {
  return {
    query: jest.fn().mockResolvedValue({ rows: [], rowCount: 0 }),
    release: jest.fn(),
  };
}

export function createMockPool() {
  const mockClient = createMockPoolClient();

  return {
    connect: jest.fn().mockResolvedValue(mockClient),
    query: jest.fn().mockResolvedValue({ rows: [], rowCount: 0 }),
    end: jest.fn().mockResolvedValue(undefined),
    on: jest.fn(),
    _mockClient: mockClient, // Expose for test access
  };
}

export type MockPool = ReturnType<typeof createMockPool>;
export type MockPoolClient = ReturnType<typeof createMockPoolClient>;