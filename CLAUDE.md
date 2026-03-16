# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an n8n community node that provides Hologres vector database integration with HGraph index support. It extends LangChain's `VectorStore` class and implements n8n's `INodeType` interface.

## Commands

```bash
npm install              # Install dependencies
npm run build            # Compile TypeScript + copy SVG icons to dist/
npm run dev              # Watch mode for development
npm run format           # Format code with Prettier
npm run lint             # Run ESLint
npm run lintfix          # Auto-fix lint issues

# Testing
npm test                 # Run unit tests only
npm run test:watch       # Watch mode for unit tests
npm run test:coverage    # Generate coverage report
npm run test:integration # Run integration tests (requires database)
npm run test:all         # Run all tests

# Publishing
npm publish --dry-run    # Verify what will be published
npm login && npm publish # Publish to npm
```

## Architecture

### Core Files

- `nodes/VectorStoreHologres/VectorStoreHologres.node.ts` - n8n node definition with UI properties, `execute()` for direct operations, and `supplyData()` for AI chain integration
- `nodes/VectorStoreHologres/HologresVectorStore.ts` - LangChain VectorStore implementation with HGraph index support
- `credentials/HologresApi.credentials.ts` - Hologres connection credentials definition

### Operation Modes

The node supports 5 modes with different input/output configurations:

| Mode | Purpose | Outputs |
|------|---------|---------|
| `load` | Query documents directly | Main |
| `insert` | Insert documents with embeddings | Main |
| `update` | Update document by ID | Main |
| `retrieve` | Return VectorStore for AI chains | AiVectorStore |
| `retrieve-as-tool` | Return DynamicTool for AI Agents | AiTool |

### Key Patterns

**Dynamic I/O**: Inputs/outputs are defined via expressions that evaluate `parameters.mode`:
```typescript
inputs: `={{ ((parameters) => { ... })($parameter) }}`
```

**Execute vs SupplyData**:
- `execute()` - Called for `load`, `insert`, `update`, `retrieve-as-tool` modes when used with Main input
- `supplyData()` - Called for `retrieve` and `retrieve-as-tool` modes when connected to AI nodes

**SQL Identifier Safety**: All table/column names are validated with `VALID_IDENTIFIER` regex and wrapped with `quoteIdentifier()` to prevent SQL injection.

**Connection Management**: Always release client and end pool in `finally` blocks:
```typescript
try {
  // operations
} finally {
  store.client?.release();
  void store.pool.end();
}
```

### n8n Integration

- Node metadata is defined in `package.json` under the `n8n` key
- SVG icons must be copied to `dist/` via gulp task
- The `files: ["dist"]` field ensures only compiled code is published

## Testing

### Unit Tests

Unit tests use mocked database connections (`__tests__/mocks/`) and run without a real Hologres instance. The `pg` module is globally mocked in `__tests__/setup.ts`.

```bash
npm test                 # Run unit tests
npm run test:watch       # Watch mode
npm run test:coverage    # With coverage
```

### Integration Tests

Integration tests require a real Hologres database. Configure via `.env.test`:

```bash
HOLOGRES_HOST=your-instance.hologres.aliyuncs.com
HOLOGRES_PORT=80
HOLOGRES_DATABASE=your_database
HOLOGRES_USER=your_user
HOLOGRES_PASSWORD=your_password
```

Integration tests:
- Use unique table names per test via `generateTestTableName()`
- Clean up tables after completion
- Skip automatically if database is not configured
- Have 120s timeout for HGraph index builds

```bash
npm run test:integration # Run integration tests
npm run test:all         # Run all tests
```

## Publishing

1. Update version in `package.json` and `package-lock.json` (both top-level and `packages[""]`)
2. Run `npm run build`
3. Run `npm publish --dry-run` to verify
4. Run `npm login` and `npm publish`