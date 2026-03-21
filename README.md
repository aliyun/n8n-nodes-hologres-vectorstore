# n8n-nodes-hologres-vectorstore

A [n8n](https://n8n.io/) community node for using [Hologres](https://www.alibabacloud.com/product/hologres) as a vector database with HGraph vector index support.

## Features

- **Vector Storage**: Store documents and embedding vectors in Hologres
- **HGraph Index Support**: Leverage Hologres' HGraph vector index for high-performance similarity search
- **Multiple Operation Modes**:
  - **Get Many**: Retrieve top-ranked documents for a given query
  - **Insert Documents**: Insert documents into the vector store
  - **Update Documents**: Update existing documents by ID
  - **Retrieve Documents**: Retrieve documents for use with other AI nodes
  - **Retrieve Documents (As Tool)**: Use as a retrieval tool for AI Agents
- **Flexible Configuration**: Customize table names, column names, distance methods, and more
- **Metadata Filtering**: Filter documents based on metadata

## Prerequisites

- n8n >= 1.0.0
- Hologres instance (version supporting HGraph vector index)
- Node.js >= 18

## Installation

### Community Node Installation (Recommended)

1. In n8n, go to **Settings** > **Community Nodes**
2. Click **Install**
3. Enter the package name `n8n-nodes-hologres-vectorstore`
4. Agree to the security prompt and wait for installation to complete

### Manual Installation

```bash
npm install n8n-nodes-hologres-vectorstore
```

## Configuration

### 1. Create Hologres Credentials

Configure Hologres connection information in n8n:

| Parameter | Description | Default |
|-----------|-------------|---------|
| Host | Hologres instance address | localhost |
| Port | Port number | 80 |
| Database | Database name | postgres |
| User | Username | - |
| Password | Password | - |
| Maximum Number of Connections | Maximum number of connections | 100 |
| SSL | SSL connection options | disable |
| Allow Unauthorized Certificates | Allow unauthorized certificates | false |

### 2. Node Configuration

#### General Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| Table Name | Table name for storing vectors | n8n_hologres_vectors |
| Dimensions | Vector dimensions (must match your embedding model output) | 1536 |
| Distance Method | Distance calculation method | Cosine |
| Embedding Batch Size | Number of documents to embed in a single batch | 10 |

#### Distance Methods

- **Cosine**: Cosine similarity (recommended for semantic search)
- **Inner Product**: Inner product
- **Euclidean**: Euclidean distance

#### Column Names Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| ID Column Name | ID column name | id |
| Vector Column Name | Vector column name | embedding |
| Content Column Name | Content column name | text |
| Metadata Column Name | Metadata column name | metadata |

#### HGraph Index Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| Base Quantization Type | Base quantization type | rabitq |
| Use Reorder | Whether to use reordering | true |
| Precise Quantization Type | High-precision quantization type | fp32 |
| Precise IO Type | High-precision index storage medium | block_memory_io |
| Max Degree | Maximum connections per vertex | 64 |
| EF Construction | Search depth during index construction | 400 |

## Usage Examples

### Insert Documents

1. Select **Insert Documents** mode
2. Connect an **Embedding** node (e.g., OpenAI Embeddings)
3. Connect a **Document** node (providing documents to store)
4. Configure table name and vector dimensions
5. Optional: Adjust **Embedding Batch Size** if your embedding model has batch size limits
6. Run the workflow

> **Note:** Documents are processed in batches according to the Embedding Batch Size setting. This helps prevent timeout issues with large document sets or embedding models with strict batch limits.

### Update Documents

1. Select **Update Documents** mode
2. Connect an **Embedding** node (for re-embedding the updated content)
3. Connect a **Document** node (providing the updated document)
4. Enter the **ID** of the document to update
5. Configure table name and column names (if different from defaults)
6. Run the workflow

> **Note:** The update operation will re-embed the document content and update both the vector and metadata in the database.

### Retrieve Documents

1. Select **Get Many** or **Retrieve Documents** mode
2. Connect an **Embedding** node
3. Enter a search prompt
4. Set the number of results to return (Limit)
5. Optional: Configure metadata filters

### Use as AI Agent Tool

1. Select **Retrieve Documents (As Tool)** mode
2. Configure the tool name and description
3. Connect to an AI Agent node

#### Execute Mode (Direct Query)

When used in execute mode (with a Main input connection), the node expects the input item to contain either:
- `chatInput` field - The query string
- `query` field - Alternative query field

This allows direct querying of the vector store without an AI Agent.

## Development

```bash
# Install dependencies
npm install

# Development mode (hot reload)
npm run dev

# Code formatting
npm run format

# Linting
npm run lint

# Build
npm run build
```

## Publishing

```bash
# Update version (automatically runs lint + build twice)
# For patch releases (bug fixes)
npm version patch

# For minor releases (new features, backward compatible)
npm version minor

# For major releases (breaking changes)
npm version major

# Login
npm login

# Publish
npm publish --dry-run  # Test publish first
npm publish

# Push tags
git push origin --tags
```

> **Note:** `npm version` will automatically run `npm run lint && npm run build` before updating the version, then run `npm run build` again after to ensure the build artifacts contain the new version number.

## Testing

This project includes both unit tests and integration tests.

### Unit Tests

Unit tests use mocked database connections and can run without a real Hologres instance.

```bash
# Run unit tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage report
npm run test:coverage
```

### Integration Tests

Integration tests require a real Hologres database connection to test actual database operations.

#### Setup

1. Copy the example environment file:
   ```bash
   cp .env.test.example .env.test
   ```

2. Edit `.env.test` with your Hologres connection details:
   ```bash
   HOLOGRES_HOST=your-instance.hologres.aliyuncs.com
   HOLOGRES_PORT=80
   HOLOGRES_DATABASE=your_database
   HOLOGRES_USER=your_user
   HOLOGRES_PASSWORD=your_password
   ```

#### Run Integration Tests

```bash
# Run integration tests only
npm run test:integration

# Run all tests (unit + integration)
npm run test:all
```

> **Note:** Integration tests will be automatically skipped if the database connection is not configured. This allows CI/CD pipelines to run unit tests without requiring a database.

### Test Structure

```
__tests__/
├── setup.ts              # Global mock setup for unit tests
├── mocks/
│   ├── pg.mock.ts        # PostgreSQL client mocks
│   ├── embeddings.mock.ts # Fake embeddings for testing
│   ├── hologres-store.mock.ts # HologresVectorStore mocks
│   └── n8n-context.mock.ts # n8n execution context mocks
├── unit/
│   ├── HologresVectorStore.test.ts  # Core vector store tests
│   └── helpers.test.ts              # Helper function tests
└── integration/
    ├── setup.ts           # Integration test configuration
    ├── HologresVectorStore.integration.test.ts  # Core DB operations
    ├── VectorStoreHologres.node.integration.test.ts  # Node tests
    ├── configurations.integration.test.ts  # Configuration tests
    ├── edge-cases.integration.test.ts  # Edge case handling
    ├── error-handling.integration.test.ts  # Error handling tests
    ├── node-parameters.integration.test.ts  # Node parameter tests
    └── performance.integration.test.ts  # Performance tests
```

### Coverage

Current test coverage:

| Metric | Coverage |
|--------|----------|
| Statements | 100% |
| Branches | 96% |
| Functions | 100% |
| Lines | 100% |

## Tech Stack

- [n8n-workflow](https://www.npmjs.com/package/n8n-workflow) - n8n workflow SDK
- [@langchain/core](https://www.npmjs.com/package/@langchain/core) - LangChain core library
- [pg](https://www.npmjs.com/package/pg) - PostgreSQL client

## License

Apache-2.0

## Related Links

- [Hologres Documentation](https://www.alibabacloud.com/help/en/hologres/)
- [n8n Community Nodes Documentation](https://docs.n8n.io/integrations/community-nodes/)
- [LangChain Vector Stores](https://js.langchain.com/docs/modules/data_connection/vectorstores/)
