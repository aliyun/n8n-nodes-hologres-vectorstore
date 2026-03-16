/**
 * Mock for n8n execution context (IExecuteFunctions / ISupplyDataFunctions)
 * Provides mock implementations for testing n8n node logic
 */

import { jest } from '@jest/globals';
import type {
  IExecuteFunctions,
  ISupplyDataFunctions,
  INodeExecutionData,
  INodeTypeDescription,
} from 'n8n-workflow';

export interface MockNodeParameterOptions {
  [key: string]: unknown;
}

/**
 * Create a mock execution context for n8n node testing
 */
export function createMockExecuteContext(
  params: MockNodeParameterOptions = {},
  inputData: INodeExecutionData[] = [],
): jest.Mocked<IExecuteFunctions> {
  const mockNode = {
    name: 'TestNode',
    type: 'test.test',
    typeVersion: 1,
    position: [0, 0] as [number, number],
    parameters: params,
  };

  const mockContext = {
    getNode: jest.fn().mockReturnValue(mockNode),
    getNodeParameter: jest.fn((name: string, _index: number, fallback?: unknown) => {
      if (name in params) {
        return params[name];
      }
      return fallback;
    }),
    getInputData: jest.fn().mockReturnValue(inputData),
    getCredentials: jest.fn().mockResolvedValue({
      host: 'localhost',
      port: 5432,
      database: 'test_db',
      user: 'test_user',
      password: 'test_password',
    }),
    getInputConnectionData: jest.fn().mockResolvedValue(null),
    getExecutionCancelSignal: jest.fn().mockReturnValue({ aborted: false }),
    continueOnFail: jest.fn().mockReturnValue(false),
    logger: {
      error: jest.fn(),
      warn: jest.fn(),
      info: jest.fn(),
      debug: jest.fn(),
    },
  } as unknown as jest.Mocked<IExecuteFunctions>;

  return mockContext;
}

/**
 * Create a mock supply data context for AI node testing
 */
export function createMockSupplyDataContext(
  params: MockNodeParameterOptions = {},
): jest.Mocked<ISupplyDataFunctions> {
  const mockNode = {
    name: 'TestNode',
    type: 'test.test',
    typeVersion: 1,
    position: [0, 0] as [number, number],
    parameters: params,
  };

  return {
    getNode: jest.fn().mockReturnValue(mockNode),
    getNodeParameter: jest.fn((name: string, _index: number, fallback?: unknown) => {
      if (name in params) {
        return params[name];
      }
      return fallback;
    }),
    getCredentials: jest.fn().mockResolvedValue({
      host: 'localhost',
      port: 5432,
      database: 'test_db',
      user: 'test_user',
      password: 'test_password',
    }),
    getInputConnectionData: jest.fn().mockResolvedValue(null),
  } as unknown as jest.Mocked<ISupplyDataFunctions>;
}

/**
 * Create mock input data items
 */
export function createMockInputData(
  items: Array<Record<string, unknown>> = [{}],
): INodeExecutionData[] {
  return items.map((json, index) => ({
    json,
    pairedItem: { item: index },
  }));
}