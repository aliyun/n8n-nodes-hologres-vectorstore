import { VectorStore } from "@langchain/core/vectorstores";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import { Document } from "@langchain/core/documents";
import pg from "pg";
import crypto from "node:crypto";

// ─── Identifier Safety ────────────────────────────────────────────────────────

export const VALID_IDENTIFIER = /^[a-zA-Z_][a-zA-Z0-9_]*$/;

/**
 * Validates and double-quote-wraps a SQL identifier (table / column name).
 * Only alphanumeric characters and underscores are allowed.
 */
export function quoteIdentifier(name: string): string {
  if (!VALID_IDENTIFIER.test(name)) {
    throw new Error(
      `Invalid SQL identifier: "${name}". Only letters, digits, and underscores are allowed.`,
    );
  }
  return `"${name}"`;
}

// ─── Types ───────────────────────────────────────────────────────────────────

export type DistanceMethod = "Cosine" | "InnerProduct" | "Euclidean";

export type ColumnOptions = {
  idColumnName: string;
  vectorColumnName: string;
  contentColumnName: string;
  metadataColumnName: string;
};

export type HGraphIndexSettings = {
  baseQuantizationType: string;
  useReorder: boolean;
  preciseQuantizationType?: string;
  preciseIoType?: string;
  maxDegree: number;
  efConstruction: number;
};

export interface HologresVectorStoreArgs {
  pool: pg.Pool;
  tableName: string;
  dimensions: number;
  distanceMethod: DistanceMethod;
  columns: ColumnOptions;
  indexSettings: HGraphIndexSettings;
  filter?: Record<string, unknown>;
}

// ─── Helper Functions ────────────────────────────────────────────────────────

export function createPoolFromCredentials(
  credentials: Record<string, unknown>,
): pg.Pool {
  let ssl: boolean | { rejectUnauthorized: boolean } = false;
  if (credentials.allowUnauthorizedCerts === true) {
    ssl = { rejectUnauthorized: false };
  } else if (credentials.ssl && credentials.ssl !== "disable") {
    ssl = true;
  }

  return new pg.Pool({
    host: credentials.host as string,
    port: credentials.port as number,
    database: credentials.database as string,
    user: credentials.user as string,
    password: credentials.password as string,
    max: (credentials.maxConnections as number) ?? 100,
    ssl,
    application_name: "n8n_hologres_vector_store",
  });
}

// ─── HologresVectorStore Class ───────────────────────────────────────────────

export class HologresVectorStore extends VectorStore {
  declare FilterType: Record<string, unknown>;

  pool: pg.Pool;

  client?: pg.PoolClient;

  tableName: string;

  dimensions: number;

  distanceMethod: DistanceMethod;

  columns: ColumnOptions;

  indexSettings: HGraphIndexSettings;

  filter?: Record<string, unknown>;

  _vectorstoreType(): string {
    return "hologres";
  }

  constructor(embeddings: EmbeddingsInterface, args: HologresVectorStoreArgs) {
    super(embeddings, args);
    this.pool = args.pool;
    this.tableName = args.tableName;
    this.dimensions = args.dimensions;
    this.distanceMethod = args.distanceMethod;
    this.columns = args.columns;
    this.indexSettings = args.indexSettings;
    this.filter = args.filter;
  }

  async _initializeClient(): Promise<void> {
    this.client = await this.pool.connect();
  }

  /**
   * Creates the table if it does not exist with the Hologres-specific
   * float4[] + CHECK constraint for vector columns.
   */
  async ensureTableInDatabase(): Promise<void> {
    const {
      idColumnName,
      contentColumnName,
      metadataColumnName,
      vectorColumnName,
    } = this.columns;
    const qTable = quoteIdentifier(this.tableName);
    const qId = quoteIdentifier(idColumnName);
    const qContent = quoteIdentifier(contentColumnName);
    const qMetadata = quoteIdentifier(metadataColumnName);
    const qVector = quoteIdentifier(vectorColumnName);
    const tableQuery = `
			CREATE TABLE IF NOT EXISTS ${qTable} (
				${qId} text NOT NULL PRIMARY KEY,
				${qContent} text,
				${qMetadata} jsonb,
				${qVector} float4[] CHECK (
					array_ndims(${qVector}) = 1
					AND array_length(${qVector}, 1) = ${Number(this.dimensions)}
				)
			);
		`;
    await this.pool.query(tableQuery);
  }

  /**
   * Sets the HGraph vector index on the table via ALTER TABLE SET.
   */
  async ensureVectorIndex(): Promise<void> {
    const { vectorColumnName } = this.columns;
    const builderParams: Record<string, unknown> = {
      base_quantization_type: this.indexSettings.baseQuantizationType,
      max_degree: this.indexSettings.maxDegree,
      ef_construction: this.indexSettings.efConstruction,
    };

    if (this.indexSettings.useReorder) {
      builderParams.use_reorder = true;
      if (this.indexSettings.preciseQuantizationType) {
        builderParams.precise_quantization_type =
          this.indexSettings.preciseQuantizationType;
      }
      if (this.indexSettings.preciseIoType) {
        builderParams.precise_io_type = this.indexSettings.preciseIoType;
      }
    }

    const vectorsConfig = JSON.stringify({
      [vectorColumnName]: {
        algorithm: "HGraph",
        distance_method: this.distanceMethod,
        builder_params: builderParams,
      },
    });

    const qTable = quoteIdentifier(this.tableName);
    const alterQuery = `ALTER TABLE ${qTable} SET (vectors = '${vectorsConfig}');`;
    await this.pool.query(alterQuery);
  }

  static async initialize(
    embeddings: EmbeddingsInterface,
    args: HologresVectorStoreArgs,
  ): Promise<HologresVectorStore> {
    const store = new HologresVectorStore(embeddings, args);
    await store._initializeClient();
    await store.ensureTableInDatabase();
    await store.ensureVectorIndex();
    return store;
  }

  async addDocuments(
    documents: Document[],
    options?: { ids?: string[] },
  ): Promise<string[]> {
    const texts = documents.map(({ pageContent }) => pageContent);
    const vectors = await this.embeddings.embedDocuments(texts);
    return await this.addVectors(vectors, documents, options);
  }

  async addVectors(
    vectors: number[][],
    documents: Document[],
    options?: { ids?: string[] },
  ): Promise<string[]> {
    const ids = options?.ids ?? vectors.map(() => crypto.randomUUID());
    const {
      idColumnName,
      contentColumnName,
      vectorColumnName,
      metadataColumnName,
    } = this.columns;
    const qTable = quoteIdentifier(this.tableName);
    const qId = quoteIdentifier(idColumnName);
    const qContent = quoteIdentifier(contentColumnName);
    const qVector = quoteIdentifier(vectorColumnName);
    const qMetadata = quoteIdentifier(metadataColumnName);

    for (let i = 0; i < vectors.length; i++) {
      const embeddingString = `{${vectors[i].join(",")}}`;
      const queryText = `
				INSERT INTO ${qTable}(
					${qId}, ${qContent}, ${qVector}, ${qMetadata}
				)
				VALUES ($1, $2, $3::float4[], $4::jsonb)
			`;
      await this.pool.query(queryText, [
        ids[i],
        documents[i].pageContent,
        embeddingString,
        JSON.stringify(documents[i].metadata),
      ]);
    }
    return ids;
  }

  /**
   * Returns the Hologres approx distance function name and ORDER BY direction
   * for the configured distance method.
   */
  private getDistanceFunctionAndOrder(): { func: string; order: string } {
    switch (this.distanceMethod) {
      case "Cosine":
        return { func: "approx_cosine_distance", order: "DESC" };
      case "InnerProduct":
        return { func: "approx_inner_product_distance", order: "DESC" };
      case "Euclidean":
        return { func: "approx_euclidean_distance", order: "ASC" };
      default:
        return { func: "approx_cosine_distance", order: "DESC" };
    }
  }

  /**
   * Build WHERE clause from a metadata filter object.
   */
  private buildFilterClauses(
    filter: Record<string, unknown>,
    paramOffset = 0,
  ): { whereClauses: string[]; parameters: unknown[]; paramCount: number } {
    const whereClauses: string[] = [];
    const parameters: unknown[] = [];
    let paramCount = paramOffset;
    const qMetadata = quoteIdentifier(this.columns.metadataColumnName);

    for (const [key, value] of Object.entries(filter)) {
      if (!VALID_IDENTIFIER.test(key)) {
        throw new Error(
          `Invalid metadata filter key: "${key}". Only letters, digits, and underscores are allowed.`,
        );
      }
      paramCount += 1;
      whereClauses.push(`${qMetadata}->>'${key}' = $${paramCount}`);
      parameters.push(value);
    }

    return { whereClauses, parameters, paramCount };
  }

  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: Record<string, unknown>,
  ): Promise<[Document, number][]> {
    const { func, order } = this.getDistanceFunctionAndOrder();
    const { vectorColumnName } = this.columns;
    const embeddingString = `{${query.join(",")}}`;

    const mergedFilter = { ...this.filter, ...filter };
    const baseParams: unknown[] = [embeddingString, k];
    let paramOffset = 2;

    let whereClause = "";
    if (Object.keys(mergedFilter).length > 0) {
      const { whereClauses, parameters } = this.buildFilterClauses(
        mergedFilter,
        paramOffset,
      );
      baseParams.push(...parameters);
      whereClause = `WHERE ${whereClauses.join(" AND ")}`;
    }

    const qVector = quoteIdentifier(vectorColumnName);
    const qTable = quoteIdentifier(this.tableName);
    const queryString = `
			SELECT *, ${func}(${qVector}, $1::float4[]) AS "_distance"
			FROM ${qTable}
			${whereClause}
			ORDER BY "_distance" ${order}
			LIMIT $2
		`;

    const result = await this.pool.query(queryString, baseParams);
    const results: [Document, number][] = [];

    for (const row of result.rows) {
      if (
        row._distance != null &&
        row[this.columns.contentColumnName] != null
      ) {
        const doc = new Document({
          pageContent: row[this.columns.contentColumnName] as string,
          metadata:
            (row[this.columns.metadataColumnName] as Record<string, unknown>) ??
            {},
          id: row[this.columns.idColumnName] as string,
        });
        results.push([doc, row._distance as number]);
      }
    }

    return results;
  }

  async delete(params: { ids: string[] }): Promise<void> {
    const { idColumnName } = this.columns;
    const qTable = quoteIdentifier(this.tableName);
    const qId = quoteIdentifier(idColumnName);
    const queryString = `
			DELETE FROM ${qTable}
			WHERE ${qId} = ANY($1::text[])
		`;
    await this.pool.query(queryString, [params.ids]);
  }

  /**
   * Update a document by ID with new content and metadata.
   * This will re-embed the content and update the vector.
   */
  async update(params: { id: string; document: Document }): Promise<void> {
    const {
      idColumnName,
      contentColumnName,
      vectorColumnName,
      metadataColumnName,
    } = this.columns;
    const qTable = quoteIdentifier(this.tableName);
    const qId = quoteIdentifier(idColumnName);
    const qContent = quoteIdentifier(contentColumnName);
    const qVector = quoteIdentifier(vectorColumnName);
    const qMetadata = quoteIdentifier(metadataColumnName);

    // Re-embed the content
    const [vector] = await this.embeddings.embedDocuments([
      params.document.pageContent,
    ]);
    const embeddingString = `{${vector.join(",")}}`;

    const queryString = `
			UPDATE ${qTable}
			SET ${qContent} = $1,
			    ${qVector} = $2::float4[],
			    ${qMetadata} = $3::jsonb
			WHERE ${qId} = $4
		`;
    await this.pool.query(queryString, [
      params.document.pageContent,
      embeddingString,
      JSON.stringify(params.document.metadata),
      params.id,
    ]);
  }

  static async fromDocuments(
    docs: Document[],
    embeddings: EmbeddingsInterface,
    args: HologresVectorStoreArgs,
  ): Promise<HologresVectorStore> {
    const instance = await HologresVectorStore.initialize(embeddings, args);
    await instance.addDocuments(docs);
    return instance;
  }
}
