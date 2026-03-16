/**
 * Global Jest setup file
 * Configures mocks and test environment before each test suite
 */

// Mock the pg module globally
jest.mock('pg', () => {
  const mockClient = {
    query: jest.fn(),
    release: jest.fn(),
  };

  const mockPool = {
    connect: jest.fn().mockResolvedValue(mockClient),
    query: jest.fn(),
    end: jest.fn().mockResolvedValue(undefined),
    on: jest.fn(),
  };

  return {
    Pool: jest.fn(() => mockPool),
    PoolClient: jest.fn(() => mockClient),
  };
});

// Mock crypto.randomUUID for consistent testing
jest.mock('node:crypto', () => ({
  randomUUID: jest.fn(() => 'test-uuid-1234-5678'),
}));