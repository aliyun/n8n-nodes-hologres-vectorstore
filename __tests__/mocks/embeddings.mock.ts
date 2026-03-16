/**
 * Fake Embeddings implementation for testing
 * Provides deterministic embedding vectors without calling external APIs
 */

import { EmbeddingsInterface } from '@langchain/core/embeddings';

/**
 * FakeEmbeddings class that returns deterministic vectors for testing
 * Each document/query is embedded to a vector based on its content hash
 */
export class FakeEmbeddings implements EmbeddingsInterface {
  private dimensions: number;

  constructor(dimensions = 1536) {
    this.dimensions = dimensions;
  }

  /**
   * Generate a deterministic vector from text content
   * Uses simple hash to create reproducible test vectors
   */
  private textToVector(text: string): number[] {
    const vector: number[] = [];
    let hash = 0;

    for (let i = 0; i < text.length; i++) {
      hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
    }

    for (let i = 0; i < this.dimensions; i++) {
      // Generate values between -1 and 1
      vector.push(Math.sin(hash + i) * 0.5);
    }

    return vector;
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    return documents.map((doc) => this.textToVector(doc));
  }

  async embedQuery(query: string): Promise<number[]> {
    return this.textToVector(query);
  }
}

/**
 * Simple static embeddings for quick test assertions
 */
export function createStaticVector(dimensions = 1536, value = 0.5): number[] {
  return Array(dimensions).fill(value);
}

/**
 * Create multiple static vectors for batch testing
 */
export function createStaticVectors(count: number, dimensions = 1536, value = 0.5): number[][] {
  return Array(count)
    .fill(null)
    .map(() => createStaticVector(dimensions, value));
}