# n8n-nodes-hologres-vectorstore

A [n8n](https://n8n.io/) community node for using [Hologres](https://www.alibabacloud.com/product/hologres) as a vector database with HGraph vector index support.

## Features

- **Vector Storage**: Store documents and embedding vectors in Hologres
- **HGraph Index Support**: Leverage Hologres' HGraph vector index for high-performance similarity search
- **Multiple Operation Modes**:
  - **Get Many**: Retrieve top-ranked documents for a given query
  - **Insert Documents**: Insert documents into the vector store
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
5. Run the workflow

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

## Development

```bash
# Install dependencies
npm install

# Development mode (hot reload)
npm run dev

# Build
npm run build

# Code formatting
npm run format

# Linting
npm run lint
```

## Tech Stack

- [n8n-workflow](https://www.npmjs.com/package/n8n-workflow) - n8n workflow SDK
- [@langchain/core](https://www.npmjs.com/package/@langchain/core) - LangChain core library
- [pg](https://www.npmjs.com/package/pg) - PostgreSQL client

## License

MIT

## Related Links

- [Hologres Documentation](https://www.alibabacloud.com/help/en/hologres/)
- [n8n Community Nodes Documentation](https://docs.n8n.io/integrations/community-nodes/)
- [LangChain Vector Stores](https://js.langchain.com/docs/modules/data_connection/vectorstores/)
