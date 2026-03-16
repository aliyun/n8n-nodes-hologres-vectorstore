/**
 * Configuration Variants Integration Tests
 * Tests different dimensions, quantization types, and custom column names
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
import { HologresVectorStore } from '../../nodes/VectorStoreHologres/HologresVectorStore';
import { FakeEmbeddings } from '../mocks/embeddings.mock';
import { Document } from '@langchain/core/documents';
import pg from 'pg';

// Skip all tests if database is not configured
const describeIntegration = canRunIntegrationTests() ? describe : describe.skip;

describeIntegration('Configuration Variants Integration Tests', () => {
  let pool: pg.Pool;
  const defaultDimensions = 128;

  beforeAll(async () => {
    pool = createTestPool();
  });

  afterAll(async () => {
    await pool.end();
  });

  describe('Different Dimensions', () => {
    const dimensions = [64, 256, 768, 1536];

    it.each(dimensions)('should work with %d dimensions', async (dim) => {
      const tableName = generateTestTableName(`dim_${dim}`);
      const embeddings = new FakeEmbeddings(dim);
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions: dim,
        ...defaultTestConfig,
      });

      try {
        const doc = new Document({ pageContent: `Test with ${dim} dimensions` });
        const [id] = await store.addDocuments([doc]);

        // Verify vector dimensions in database
        const result = await pool.query(
          `SELECT array_length(embedding, 1) as dim FROM "${tableName}" WHERE id = $1`,
          [id]
        );
        expect(result.rows[0].dim).toBe(dim);

        // Search should work
        const queryVector = await embeddings.embedQuery('test');
        const results = await store.similaritySearchVectorWithScore(queryVector, 1);
        expect(results).toHaveLength(1);
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    }, 30000); // Extended timeout for larger dimensions
  });

  describe('Different Quantization Types', () => {
    const quantizationTypes = [
      { baseQuantizationType: 'rabitq', preciseQuantizationType: 'fp32' },
      { baseQuantizationType: 'rabitq', preciseQuantizationType: 'fp16' },
      { baseQuantizationType: 'sq8', preciseQuantizationType: 'fp32' },
    ];

    it.each(quantizationTypes)(
      'should work with $baseQuantizationType / $preciseQuantizationType',
      async ({ baseQuantizationType, preciseQuantizationType }) => {
        const tableName = generateTestTableName(`quant_${baseQuantizationType}`);
        const embeddings = new FakeEmbeddings(defaultDimensions);
        const store = await HologresVectorStore.initialize(embeddings, {
          pool,
          tableName,
          dimensions: defaultDimensions,
          ...defaultTestConfig,
          indexSettings: {
            ...defaultTestConfig.indexSettings,
            baseQuantizationType,
            preciseQuantizationType,
          },
        });

        try {
          const doc = new Document({ pageContent: 'Quantization test' });
          await store.addDocuments([doc]);

          const queryVector = await embeddings.embedQuery('test');
          const results = await store.similaritySearchVectorWithScore(queryVector, 1);
          expect(results).toHaveLength(1);
        } finally {
          store.client?.release();
          await cleanupTable(pool, tableName);
        }
      }
    );
  });

  describe('Custom Column Names', () => {
    it('should work with custom column names', async () => {
      const tableName = generateTestTableName('custom_cols');
      const embeddings = new FakeEmbeddings(defaultDimensions);
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions: defaultDimensions,
        columns: {
          idColumnName: 'doc_id',
          vectorColumnName: 'vec',
          contentColumnName: 'content',
          metadataColumnName: 'meta',
        },
        distanceMethod: 'Cosine',
        indexSettings: defaultTestConfig.indexSettings,
      });

      try {
        const doc = new Document({ pageContent: 'Custom columns test' });
        const [id] = await store.addDocuments([doc]);

        // Verify custom column names exist
        const result = await pool.query(
          `SELECT column_name FROM information_schema.columns WHERE table_name = $1`,
          [tableName]
        );
        const columnNames = result.rows.map((r) => r.column_name);
        expect(columnNames).toContain('doc_id');
        expect(columnNames).toContain('vec');
        expect(columnNames).toContain('content');
        expect(columnNames).toContain('meta');
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });
  });

  describe('Index Settings Variants', () => {
    it('should work with useReorder=false (requires sq8 quantization)', async () => {
      const tableName = generateTestTableName('no_reorder');
      const embeddings = new FakeEmbeddings(defaultDimensions);
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions: defaultDimensions,
        ...defaultTestConfig,
        indexSettings: {
          // useReorder=false requires a quantization type that supports it
          // sq8 can work without reorder, but rabitq requires useReorder=true
          baseQuantizationType: 'sq8',
          useReorder: false,
          maxDegree: 32,
          efConstruction: 200,
        },
      });

      try {
        const doc = new Document({ pageContent: 'No reorder test' });
        await store.addDocuments([doc]);

        const queryVector = await embeddings.embedQuery('test');
        const results = await store.similaritySearchVectorWithScore(queryVector, 1);
        expect(results).toHaveLength(1);
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });

    it('should work with preciseIoType=reader_io', async () => {
      const tableName = generateTestTableName('reader_io');
      const embeddings = new FakeEmbeddings(defaultDimensions);
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions: defaultDimensions,
        ...defaultTestConfig,
        indexSettings: {
          baseQuantizationType: 'rabitq',
          useReorder: true,
          preciseQuantizationType: 'fp32',
          preciseIoType: 'reader_io',
          maxDegree: 32,
          efConstruction: 200,
        },
      });

      try {
        const doc = new Document({ pageContent: 'Reader IO test' });
        await store.addDocuments([doc]);

        const queryVector = await embeddings.embedQuery('test');
        const results = await store.similaritySearchVectorWithScore(queryVector, 1);
        expect(results).toHaveLength(1);
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });

    it('should work with different maxDegree values', async () => {
      const maxDegrees = [16, 32, 64];

      for (const maxDegree of maxDegrees) {
        const tableName = generateTestTableName(`degree_${maxDegree}`);
        const embeddings = new FakeEmbeddings(defaultDimensions);
        const store = await HologresVectorStore.initialize(embeddings, {
          pool,
          tableName,
          dimensions: defaultDimensions,
          ...defaultTestConfig,
          indexSettings: {
            ...defaultTestConfig.indexSettings,
            maxDegree,
          },
        });

        try {
          const doc = new Document({ pageContent: `MaxDegree ${maxDegree} test` });
          await store.addDocuments([doc]);

          const queryVector = await embeddings.embedQuery('test');
          const results = await store.similaritySearchVectorWithScore(queryVector, 1);
          expect(results).toHaveLength(1);
        } finally {
          store.client?.release();
          await cleanupTable(pool, tableName);
        }
      }
    });

    it('should work with different efConstruction values', async () => {
      const efConstructions = [100, 200, 400];

      for (const efConstruction of efConstructions) {
        const tableName = generateTestTableName(`ef_${efConstruction}`);
        const embeddings = new FakeEmbeddings(defaultDimensions);
        const store = await HologresVectorStore.initialize(embeddings, {
          pool,
          tableName,
          dimensions: defaultDimensions,
          ...defaultTestConfig,
          indexSettings: {
            ...defaultTestConfig.indexSettings,
            efConstruction,
          },
        });

        try {
          const doc = new Document({ pageContent: `EfConstruction ${efConstruction} test` });
          await store.addDocuments([doc]);

          const queryVector = await embeddings.embedQuery('test');
          const results = await store.similaritySearchVectorWithScore(queryVector, 1);
          expect(results).toHaveLength(1);
        } finally {
          store.client?.release();
          await cleanupTable(pool, tableName);
        }
      }
    });
  });

  describe('Distance Methods', () => {
    const distanceMethods: Array<'Cosine' | 'InnerProduct' | 'Euclidean'> = [
      'Cosine',
      'InnerProduct',
      'Euclidean',
    ];

    it.each(distanceMethods)('should work with %s distance method', async (method) => {
      const tableName = generateTestTableName(`dist_${method.toLowerCase()}`);
      const embeddings = new FakeEmbeddings(defaultDimensions);
      const store = await HologresVectorStore.initialize(embeddings, {
        pool,
        tableName,
        dimensions: defaultDimensions,
        ...defaultTestConfig,
        distanceMethod: method,
      });

      try {
        const docs = [
          new Document({ pageContent: 'First document' }),
          new Document({ pageContent: 'Second document' }),
        ];
        await store.addDocuments(docs);

        const queryVector = await embeddings.embedQuery('test');
        const results = await store.similaritySearchVectorWithScore(queryVector, 2);
        expect(results).toHaveLength(2);

        // Scores should be numbers
        results.forEach(([_, score]) => {
          expect(typeof score).toBe('number');
        });
      } finally {
        store.client?.release();
        await cleanupTable(pool, tableName);
      }
    });
  });
});