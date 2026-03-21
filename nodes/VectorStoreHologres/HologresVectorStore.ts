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

/**
 * Converts a numeric vector to PostgreSQL array literal format.
 * Example: [1.0, 2.0, 3.0] -> "{1,2,3}"
 */
export function vectorToPostgresArray(vector: number[]): string {
  return `{${vector.join(",")}}`;
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

/** Typed credentials for Hologres connection */
export interface HologresCredentials {
  host: string;
  port: number;
  database: string;
  user: string;
  password: string;
  maxConnections?: number;
  ssl?: "disable" | "allow" | "require";
  allowUnauthorizedCerts?: boolean;
}

export interface HologresVectorStoreArgs {
  pool: pg.Pool;
  tableName: string;
  dimensions: number;
  distanceMethod: DistanceMethod;
  columns: ColumnOptions;
  indexSettings: HGraphIndexSettings;
  filter?: Record<string, unknown>;
}

// ─── Default Values ───────────────────────────────────────────────────────────

export const DEFAULT_COLUMN_OPTIONS: ColumnOptions = {
  idColumnName: "id",
  vectorColumnName: "embedding",
  contentColumnName: "text",
  metadataColumnName: "metadata",
};

export const DEFAULT_HGRAPH_INDEX_SETTINGS: HGraphIndexSettings = {
  baseQuantizationType: "rabitq",
  useReorder: true,
  preciseQuantizationType: "fp32",
  preciseIoType: "block_memory_io",
  maxDegree: 64,
  efConstruction: 400,
};

/** Minimal index settings for read-only operations (load, update, retrieve) */
export const MINIMAL_HGRAPH_INDEX_SETTINGS: Omit<
  HGraphIndexSettings,
  "preciseQuantizationType" | "preciseIoType"
> = {
  baseQuantizationType: "rabitq",
  useReorder: true,
  maxDegree: 64,
  efConstruction: 400,
};

// ─── Distance Function Mapping ────────────────────────────────────────────────

const DISTANCE_FUNCTION_MAP: Record<
  DistanceMethod,
  { func: string; order: string }
> = {
  Cosine: { func: "approx_cosine_distance", order: "DESC" },
  InnerProduct: { func: "approx_inner_product_distance", order: "DESC" },
  Euclidean: { func: "approx_euclidean_distance", order: "ASC" },
};

// ─── Helper Functions ────────────────────────────────────────────────────────

export function createPoolFromCredentials(
  credentials: HologresCredentials,
): pg.Pool {
  let ssl: boolean | { rejectUnauthorized: boolean } = false;
  if (credentials.allowUnauthorizedCerts === true) {
    ssl = { rejectUnauthorized: false };
  } else if (credentials.ssl && credentials.ssl !== "disable") {
    ssl = true;
  }

  return new pg.Pool({
    host: credentials.host,
    port: credentials.port,
    database: credentials.database,
    user: credentials.user,
    password: credentials.password,
    max: credentials.maxConnections ?? 100,
    ssl,
    application_name: "n8n_hologres_vector_store",
  });
}

// ─── HologresVectorStore Class ───────────────────────────────────────────────

export class HologresVectorStore extends VectorStore {
  declare FilterType: Record<string, unknown>;

  private _pool: pg.Pool;

  private _client?: pg.PoolClient;

  tableName: string;

  dimensions: number;

  distanceMethod: DistanceMethod;

  columns: ColumnOptions;

  indexSettings: HGraphIndexSettings;

  filter?: Record<string, unknown>;

  /** Expose pool for backward compatibility (deprecated: use close() instead) */
  get pool(): pg.Pool {
    return this._pool;
  }

  /** Expose client for backward compatibility (deprecated: use close() instead) */
  get client(): pg.PoolClient | undefined {
    return this._client;
  }

  _vectorstoreType(): string {
    return "hologres";
  }

  constructor(embeddings: EmbeddingsInterface, args: HologresVectorStoreArgs) {
    super(embeddings, args);
    this._pool = args.pool;
    this.tableName = args.tableName;
    this.dimensions = args.dimensions;
    this.distanceMethod = args.distanceMethod;
    this.columns = args.columns;
    this.indexSettings = args.indexSettings;
    this.filter = args.filter;
  }

  async _initializeClient(): Promise<void> {
    this._client = await this._pool.connect();
  }

  /** Release client and close pool */
  async close(): Promise<void> {
    this._client?.release();
    await this._pool.end();
  }

  /** Get quoted identifiers for all columns and table */
  getQuotedIdentifiers(): {
    table: string;
    id: string;
    content: string;
    metadata: string;
    vector: string;
  } {
    const {
      idColumnName,
      contentColumnName,
      metadataColumnName,
      vectorColumnName,
    } = this.columns;
    return {
      table: quoteIdentifier(this.tableName),
      id: quoteIdentifier(idColumnName),
      content: quoteIdentifier(contentColumnName),
      metadata: quoteIdentifier(metadataColumnName),
      vector: quoteIdentifier(vectorColumnName),
    };
  }

  /**
   * Creates the table if it does not exist with the Hologres-specific
   * float4[] + CHECK constraint for vector columns.
   */
  async ensureTableInDatabase(): Promise<void> {
    const { table, id, content, metadata, vector } =
      this.getQuotedIdentifiers();
    const tableQuery = `
			CREATE TABLE IF NOT EXISTS ${table} (
				${id} text NOT NULL PRIMARY KEY,
				${content} text,
				${metadata} jsonb,
				${vector} float4[] CHECK (
					array_ndims(${vector}) = 1
					AND array_length(${vector}, 1) = ${Number(this.dimensions)}
				)
			);
		`;
    await this._pool.query(tableQuery);
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
      builderParams.precise_quantization_type =
        this.indexSettings.preciseQuantizationType ?? "fp32";
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

    const { table } = this.getQuotedIdentifiers();
    const alterQuery = `ALTER TABLE ${table} SET (vectors = '${vectorsConfig}');`;
    await this._pool.query(alterQuery);
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
    if (vectors.length === 0) return [];

    const ids = options?.ids ?? vectors.map(() => crypto.randomUUID());
    const { table, id, content, vector, metadata } =
      this.getQuotedIdentifiers();

    // Use multi-row INSERT for efficiency
    const valuePlaceholders = vectors
      .map(
        (_, i) =>
          `($${i * 4 + 1}, $${i * 4 + 2}, $${i * 4 + 3}::float4[], $${i * 4 + 4}::jsonb)`,
      )
      .join(", ");

    const queryText = `
			INSERT INTO ${table}(${id}, ${content}, ${vector}, ${metadata})
			VALUES ${valuePlaceholders}
		`;

    const params = vectors.flatMap((v, i) => [
      ids[i],
      documents[i].pageContent,
      vectorToPostgresArray(v),
      JSON.stringify(documents[i].metadata),
    ]);

    await this._pool.query(queryText, params);
    return ids;
  }

  /**
   * Returns the Hologres approx distance function name and ORDER BY direction
   * for the configured distance method.
   */
  private getDistanceFunctionAndOrder(): { func: string; order: string } {
    return (
      DISTANCE_FUNCTION_MAP[this.distanceMethod] ?? DISTANCE_FUNCTION_MAP.Cosine
    );
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
    const { table, vector } = this.getQuotedIdentifiers();
    const embeddingString = vectorToPostgresArray(query);

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

    const queryString = `
			SELECT *, ${func}(${vector}, $1::float4[]) AS "_distance"
			FROM ${table}
			${whereClause}
			ORDER BY "_distance" ${order}
			LIMIT $2
		`;

    const result = await this._pool.query(queryString, baseParams);
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
    const { table, id } = this.getQuotedIdentifiers();
    const queryString = `
			DELETE FROM ${table}
			WHERE ${id} = ANY($1::text[])
		`;
    await this._pool.query(queryString, [params.ids]);
  }

  /**
   * Update a document by ID with new content and metadata.
   * This will re-embed the content and update the vector.
   */
  async update(params: { id: string; document: Document }): Promise<void> {
    const { table, id, content, vector, metadata } =
      this.getQuotedIdentifiers();

    // Re-embed the content
    const [vec] = await this.embeddings.embedDocuments([
      params.document.pageContent,
    ]);
    const embeddingString = vectorToPostgresArray(vec);

    const queryString = `
			UPDATE ${table}
			SET ${content} = $1,
			    ${vector} = $2::float4[],
			    ${metadata} = $3::jsonb
			WHERE ${id} = $4
		`;
    await this._pool.query(queryString, [
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
