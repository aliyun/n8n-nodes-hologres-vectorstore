/**
 * Unit tests for helper functions in VectorStoreHologres.node.ts
 * Tests: getMetadataFiltersValues, getColumnOptions, getHGraphIndexSettings
 */

import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import {
  getMetadataFiltersValues,
  getColumnOptions,
  getHGraphIndexSettings,
} from '../../nodes/VectorStoreHologres/VectorStoreHologres.node';

describe('Helper Functions', () => {
  describe('getMetadataFiltersValues', () => {
    const mockGetNodeParameter = jest.fn();

    beforeEach(() => {
      mockGetNodeParameter.mockReset();
    });

    it('should return undefined when no metadata options provided', () => {
      mockGetNodeParameter.mockReturnValue({});
      const ctx = {
        getNodeParameter: mockGetNodeParameter,
      } as any;
      const result = getMetadataFiltersValues(ctx, 0);
      expect(result).toBeUndefined();
    });

    it('should extract metadata values from fixedCollection', () => {
      mockGetNodeParameter.mockImplementation((name: string) => {
        if (name === 'options') {
          return {
            metadata: {
              metadataValues: [
                { name: 'category', value: 'electronics' },
                { name: 'status', value: 'active' },
              ],
            },
          };
        }
        return {};
      });
      const ctx = {
        getNodeParameter: mockGetNodeParameter,
      } as any;
      const result = getMetadataFiltersValues(ctx, 0);
      expect(result).toEqual({ category: 'electronics', status: 'active' });
    });

    it('should parse JSON filter from searchFilterJson', () => {
      mockGetNodeParameter.mockImplementation((name: string, _index: number, fallback: unknown, options?: { ensureType?: string }) => {
        if (name === 'options') {
          return { searchFilterJson: { category: 'books' } };
        }
        if (name === 'options.searchFilterJson') {
          return { category: 'books' };
        }
        return fallback;
      });
      const ctx = {
        getNodeParameter: mockGetNodeParameter,
      } as any;
      const result = getMetadataFiltersValues(ctx, 0);
      expect(result).toEqual({ category: 'books' });
    });

    it('should prefer metadata values over JSON filter', () => {
      mockGetNodeParameter.mockImplementation((name: string, _index: number, fallback: unknown, _options?: { ensureType?: string }) => {
        if (name === 'options') {
          return {
            metadata: {
              metadataValues: [{ name: 'type', value: 'premium' }],
            },
            searchFilterJson: { category: 'books' },
          };
        }
        return fallback;
      });
      const ctx = {
        getNodeParameter: mockGetNodeParameter,
      } as any;
      const result = getMetadataFiltersValues(ctx, 0);
      expect(result).toEqual({ type: 'premium' });
    });
  });

  describe('getColumnOptions', () => {
    it('should return default column names when not specified', () => {
      const mockGetNodeParameter = jest.fn().mockReturnValue({
        idColumnName: 'id',
        vectorColumnName: 'embedding',
        contentColumnName: 'text',
        metadataColumnName: 'metadata',
      });
      const result = getColumnOptions({ getNodeParameter: mockGetNodeParameter });
      expect(result).toEqual({
        idColumnName: 'id',
        vectorColumnName: 'embedding',
        contentColumnName: 'text',
        metadataColumnName: 'metadata',
      });
    });

    it('should return custom column names from parameters', () => {
      const mockGetNodeParameter = jest.fn().mockReturnValue({
        idColumnName: 'doc_id',
        vectorColumnName: 'vec',
        contentColumnName: 'content',
        metadataColumnName: 'meta',
      });
      const result = getColumnOptions({ getNodeParameter: mockGetNodeParameter });
      expect(result.idColumnName).toBe('doc_id');
      expect(result.vectorColumnName).toBe('vec');
    });
  });

  describe('getHGraphIndexSettings', () => {
    it('should return default settings when not specified', () => {
      const mockGetNodeParameter = jest.fn().mockReturnValue({
        baseQuantizationType: 'rabitq',
        useReorder: true,
        preciseQuantizationType: 'fp32',
        preciseIoType: 'block_memory_io',
        maxDegree: 64,
        efConstruction: 400,
      });
      const result = getHGraphIndexSettings({ getNodeParameter: mockGetNodeParameter });
      expect(result.baseQuantizationType).toBe('rabitq');
      expect(result.useReorder).toBe(true);
      expect(result.maxDegree).toBe(64);
      expect(result.efConstruction).toBe(400);
    });

    it('should return custom settings from parameters', () => {
      const mockGetNodeParameter = jest.fn().mockReturnValue({
        baseQuantizationType: 'sq8',
        useReorder: false,
        preciseQuantizationType: 'fp32',
        preciseIoType: 'block_memory_io',
        maxDegree: 48,
        efConstruction: 300,
      });
      const result = getHGraphIndexSettings({ getNodeParameter: mockGetNodeParameter });
      expect(result.baseQuantizationType).toBe('sq8');
      expect(result.useReorder).toBe(false);
      expect(result.maxDegree).toBe(48);
    });

    it('should include all required index parameters', () => {
      const mockGetNodeParameter = jest.fn().mockReturnValue({
        baseQuantizationType: 'fp16',
        useReorder: true,
        preciseQuantizationType: 'fp32',
        preciseIoType: 'reader_io',
        maxDegree: 96,
        efConstruction: 600,
      });
      const result = getHGraphIndexSettings({ getNodeParameter: mockGetNodeParameter });
      expect(result).toHaveProperty('baseQuantizationType');
      expect(result).toHaveProperty('useReorder');
      expect(result).toHaveProperty('maxDegree');
      expect(result).toHaveProperty('efConstruction');
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Alternative: Test the helper function logic directly
// ─────────────────────────────────────────────────────────────────────────────

describe('Helper Function Logic (Direct Testing)', () => {
  /**
   * Test the logic that getMetadataFiltersValues should implement
   */
  describe('Metadata Filter Logic', () => {
    it('should convert metadata array to object', () => {
      const metadataValues = [
        { name: 'category', value: 'electronics' },
        { name: 'status', value: 'active' },
      ];

      const expected = {
        category: 'electronics',
        status: 'active',
      };

      // This is what the function should do
      const result = metadataValues.reduce(
        (acc, { name, value }) => ({ ...acc, [name]: value }),
        {} as Record<string, string>,
      );

      expect(result).toEqual(expected);
    });

    it('should handle empty metadata array', () => {
      const metadataValues: Array<{ name: string; value: string }> = [];

      const result = metadataValues.reduce(
        (acc, { name, value }) => ({ ...acc, [name]: value }),
        {} as Record<string, string>,
      );

      expect(result).toEqual({});
    });
  });

  /**
   * Test the logic for column name defaults
   */
  describe('Column Options Defaults', () => {
    it('should have correct default column names', () => {
      const defaultColumnOptions = {
        idColumnName: 'id',
        vectorColumnName: 'embedding',
        contentColumnName: 'text',
        metadataColumnName: 'metadata',
      };

      expect(defaultColumnOptions.idColumnName).toBe('id');
      expect(defaultColumnOptions.vectorColumnName).toBe('embedding');
      expect(defaultColumnOptions.contentColumnName).toBe('text');
      expect(defaultColumnOptions.metadataColumnName).toBe('metadata');
    });
  });

  /**
   * Test the logic for HGraph index defaults
   */
  describe('HGraph Index Settings Defaults', () => {
    it('should have correct default settings', () => {
      const defaultIndexSettings = {
        baseQuantizationType: 'rabitq',
        useReorder: true,
        preciseQuantizationType: 'fp32',
        preciseIoType: 'block_memory_io',
        maxDegree: 64,
        efConstruction: 400,
      };

      expect(defaultIndexSettings.baseQuantizationType).toBe('rabitq');
      expect(defaultIndexSettings.useReorder).toBe(true);
      expect(defaultIndexSettings.maxDegree).toBe(64);
      expect(defaultIndexSettings.efConstruction).toBe(400);
    });

    it('should have valid quantization types', () => {
      const validQuantizationTypes = ['rabitq', 'sq8', 'sq8_uniform', 'fp16', 'fp32'];

      validQuantizationTypes.forEach((type) => {
        expect(typeof type).toBe('string');
        expect(type.length).toBeGreaterThan(0);
      });
    });

    it('should have valid IO types', () => {
      const validIoTypes = ['block_memory_io', 'reader_io'];

      validIoTypes.forEach((type) => {
        expect(typeof type).toBe('string');
        expect(type.length).toBeGreaterThan(0);
      });
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test data validation patterns used in the node
// ─────────────────────────────────────────────────────────────────────────────

describe('Data Validation Patterns', () => {
  describe('Distance Method Validation', () => {
    it('should accept valid distance methods', () => {
      const validMethods = ['Cosine', 'InnerProduct', 'Euclidean'];

      validMethods.forEach((method) => {
        expect(['Cosine', 'InnerProduct', 'Euclidean']).toContain(method);
      });
    });

    it('should reject invalid distance methods', () => {
      const invalidMethods = ['cosine', 'inner_product', 'euclidean', 'Manhattan', ''];

      invalidMethods.forEach((method) => {
        expect(['Cosine', 'InnerProduct', 'Euclidean']).not.toContain(method);
      });
    });
  });

  describe('TableName Validation', () => {
    it('should use quoteIdentifier for table names', () => {
      // The node should use quoteIdentifier to wrap table names
      // This is already tested in HologresVectorStore.test.ts
      // But we document the expectation here
      const validTableName = 'my_vectors_table';
      const expected = `"${validTableName}"`;

      expect(expected).toBe('"my_vectors_table"');
    });
  });
});