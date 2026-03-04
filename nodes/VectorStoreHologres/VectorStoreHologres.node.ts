import type {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	INodeProperties,
	ISupplyDataFunctions,
	SupplyData,
} from 'n8n-workflow';
import { NodeConnectionTypes, NodeOperationError } from 'n8n-workflow';
import type { Embeddings } from '@langchain/core/embeddings';
import type { Document } from '@langchain/core/documents';
import { DynamicTool } from '@langchain/core/tools';
import {
	HologresVectorStore,
	createPoolFromCredentials,
	type DistanceMethod,
	type ColumnOptions,
	type HGraphIndexSettings,
	type HologresVectorStoreArgs,
} from './HologresVectorStore';

// ─── Inline Helpers (from @n8n/ai-utilities, inlined for community node) ────

function getMetadataFiltersValues(
	ctx: IExecuteFunctions | ISupplyDataFunctions,
	itemIndex: number,
): Record<string, string> | undefined {
	const options = ctx.getNodeParameter('options', itemIndex, {}) as Record<string, unknown>;

	if (options.metadata) {
		const { metadataValues: metadata } = options.metadata as {
			metadataValues: Array<{ name: string; value: string }>;
		};
		if (metadata.length > 0) {
			return metadata.reduce(
				(acc, { name, value }) => ({ ...acc, [name]: value }),
				{} as Record<string, string>,
			);
		}
	}

	if (options.searchFilterJson) {
		return ctx.getNodeParameter('options.searchFilterJson', itemIndex, '', {
			ensureType: 'object',
		}) as Record<string, string>;
	}

	return undefined;
}

function getColumnOptions(context: {
	getNodeParameter: (name: string, index: number, fallback: unknown) => unknown;
}): ColumnOptions {
	return context.getNodeParameter('options.columnNames.values', 0, {
		idColumnName: 'id',
		vectorColumnName: 'embedding',
		contentColumnName: 'text',
		metadataColumnName: 'metadata',
	}) as ColumnOptions;
}

function getHGraphIndexSettings(context: {
	getNodeParameter: (name: string, index: number, fallback: unknown) => unknown;
}): HGraphIndexSettings {
	return context.getNodeParameter('options.hgraphIndex.values', 0, {
		baseQuantizationType: 'rabitq',
		useReorder: true,
		preciseQuantizationType: 'fp32',
		preciseIoType: 'block_memory_io',
		maxDegree: 64,
		efConstruction: 400,
	}) as HGraphIndexSettings;
}

/**
 * Process document input from the AI Document connection.
 * Handles both raw Document arrays and loader objects (N8nJsonLoader/N8nBinaryLoader)
 * using duck typing to avoid importing internal n8n packages.
 */
async function processDocumentInput(
	documentInput: unknown,
	inputItem: INodeExecutionData,
	itemIndex: number,
): Promise<{ processedDocuments: Document[]; serializedDocuments: INodeExecutionData[] }> {
	let processedDocuments: Document[];

	if (Array.isArray(documentInput)) {
		processedDocuments = documentInput;
	} else if (
		documentInput &&
		typeof (documentInput as { processItem?: Function }).processItem === 'function'
	) {
		processedDocuments = await (documentInput as { processItem: Function }).processItem(
			inputItem,
			itemIndex,
		);
	} else {
		throw new Error('Unsupported document input type');
	}

	const serializedDocuments: INodeExecutionData[] = processedDocuments.map(
		({ metadata, pageContent }) => ({
			json: { metadata, pageContent },
			pairedItem: { item: itemIndex },
		}),
	);

	return { processedDocuments, serializedDocuments };
}

// ─── Node Field Definitions ──────────────────────────────────────────────────

const metadataFilterField: INodeProperties = {
	displayName: 'Metadata Filter',
	name: 'metadata',
	type: 'fixedCollection',
	description: 'Metadata to filter the results by',
	default: {},
	placeholder: 'Add Metadata Filter',
	typeOptions: { multipleValues: true },
	options: [
		{
			name: 'metadataValues',
			displayName: 'Metadata',
			values: [
				{
					displayName: 'Name',
					name: 'name',
					type: 'string',
					default: '',
				},
				{
					displayName: 'Value',
					name: 'value',
					type: 'string',
					default: '',
				},
			],
		},
	],
};

const sharedFields: INodeProperties[] = [
	{
		displayName: 'Table Name',
		name: 'tableName',
		type: 'string',
		default: 'n8n_hologres_vectors',
		description:
			'The table name to store the vectors in. If the table does not exist, it will be created.',
	},
];

const dimensionsField: INodeProperties = {
	displayName: 'Dimensions',
	name: 'dimensions',
	type: 'number',
	default: 1536,
	required: true,
	description:
		'The number of dimensions of the embedding vectors. Must match the output of your embedding model.',
};

const distanceMethodField: INodeProperties = {
	displayName: 'Distance Method',
	name: 'distanceMethod',
	type: 'options',
	default: 'Cosine',
	description: 'The distance calculation method for vector search',
	options: [
		{ name: 'Cosine', value: 'Cosine' },
		{ name: 'Inner Product', value: 'InnerProduct' },
		{ name: 'Euclidean', value: 'Euclidean' },
	],
};

const columnNamesField: INodeProperties = {
	displayName: 'Column Names',
	name: 'columnNames',
	type: 'fixedCollection',
	description: 'The names of the columns in the Hologres table',
	default: {
		values: {
			idColumnName: 'id',
			vectorColumnName: 'embedding',
			contentColumnName: 'text',
			metadataColumnName: 'metadata',
		},
	},
	typeOptions: {},
	placeholder: 'Set Column Names',
	options: [
		{
			name: 'values',
			displayName: 'Column Name Settings',
			values: [
				{
					displayName: 'ID Column Name',
					name: 'idColumnName',
					type: 'string',
					default: 'id',
					required: true,
				},
				{
					displayName: 'Vector Column Name',
					name: 'vectorColumnName',
					type: 'string',
					default: 'embedding',
					required: true,
				},
				{
					displayName: 'Content Column Name',
					name: 'contentColumnName',
					type: 'string',
					default: 'text',
					required: true,
				},
				{
					displayName: 'Metadata Column Name',
					name: 'metadataColumnName',
					type: 'string',
					default: 'metadata',
					required: true,
				},
			],
		},
	],
};

const hgraphIndexField: INodeProperties = {
	displayName: 'HGraph Index Settings',
	name: 'hgraphIndex',
	type: 'fixedCollection',
	description: 'Configuration for the Hologres HGraph vector index',
	default: {
		values: {
			baseQuantizationType: 'rabitq',
			useReorder: true,
			preciseQuantizationType: 'fp32',
			preciseIoType: 'block_memory_io',
			maxDegree: 64,
			efConstruction: 400,
		},
	},
	placeholder: 'Configure HGraph Index',
	options: [
		{
			name: 'values',
			displayName: 'HGraph Index Parameters',
			values: [
				{
					displayName: 'Base Quantization Type',
					name: 'baseQuantizationType',
					type: 'options',
					default: 'rabitq',
					required: true,
					description: 'Low-precision index quantization method',
					options: [
						{ name: 'RaBitQ', value: 'rabitq' },
						{ name: 'SQ8', value: 'sq8' },
						{ name: 'SQ8 Uniform', value: 'sq8_uniform' },
						{ name: 'FP16', value: 'fp16' },
						{ name: 'FP32', value: 'fp32' },
					],
				},
				{
					displayName: 'Use Reorder',
					name: 'useReorder',
					type: 'boolean',
					default: true,
					description:
						'Whether to use high-precision index for reordering. When enabled, allows configuring precise quantization type and IO type.',
				},
				{
					displayName: 'Precise Quantization Type',
					name: 'preciseQuantizationType',
					type: 'options',
					default: 'fp32',
					description:
						'High-precision index quantization method. Should be higher precision than the base quantization type.',
					displayOptions: { show: { useReorder: [true] } },
					options: [
						{ name: 'FP32', value: 'fp32' },
						{ name: 'FP16', value: 'fp16' },
						{ name: 'SQ8', value: 'sq8' },
						{ name: 'SQ8 Uniform', value: 'sq8_uniform' },
					],
				},
				{
					displayName: 'Precise IO Type',
					name: 'preciseIoType',
					type: 'options',
					default: 'block_memory_io',
					description: 'Storage medium for the high-precision index',
					displayOptions: { show: { useReorder: [true] } },
					options: [
						{
							name: 'Block Memory IO',
							value: 'block_memory_io',
							description: 'Both low and high precision indexes stored in memory',
						},
						{
							name: 'Reader IO',
							value: 'reader_io',
							description: 'Low precision in memory, high precision on disk',
						},
					],
				},
				{
					displayName: 'Max Degree',
					name: 'maxDegree',
					type: 'number',
					default: 64,
					description:
						'Max connections per vertex during index construction. Higher values improve search quality but increase build cost. Not recommended to exceed 96.',
				},
				{
					displayName: 'EF Construction',
					name: 'efConstruction',
					type: 'number',
					default: 400,
					description:
						'Search depth during index construction. Higher values improve accuracy but increase build time. Not recommended to exceed 600.',
				},
			],
		},
	],
};

// ─── Node Class ──────────────────────────────────────────────────────────────

export class VectorStoreHologres implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Hologres Vector Store',
		name: 'vectorStoreHologres',
		icon: 'file:hologres.svg',
		group: ['transform'],
		version: [1, 1.1],
		description: 'Work with your data in Hologres with HGraph vector index',
		defaults: {
			name: 'Hologres Vector Store',
		},
		codex: {
			categories: ['AI'],
			subcategories: {
				AI: ['Vector Stores', 'Tools', 'Root Nodes'],
				'Vector Stores': ['Other Vector Stores'],
				Tools: ['Other Tools'],
			},
			resources: {
				primaryDocumentation: [
					{
						url: 'https://www.alibabacloud.com/help/en/hologres/',
					},
				],
			},
		},
		credentials: [
			{
				name: 'hologresApi',
				required: true,
			},
		],
		inputs: `={{
			((parameters) => {
				const mode = parameters?.mode;
				const inputs = [{ displayName: "Embedding", type: "${NodeConnectionTypes.AiEmbedding}", required: true, maxConnections: 1}]

				if (mode === 'retrieve-as-tool') {
					return inputs;
				}

				if (['insert', 'load'].includes(mode)) {
					inputs.push({ displayName: "", type: "${NodeConnectionTypes.Main}"})
				}

				if (['insert'].includes(mode)) {
					inputs.push({ displayName: "Document", type: "${NodeConnectionTypes.AiDocument}", required: true, maxConnections: 1})
				}
				return inputs
			})($parameter)
		}}`,
		outputs: `={{
			((parameters) => {
				const mode = parameters?.mode ?? 'retrieve';

				if (mode === 'retrieve-as-tool') {
					return [{ displayName: "Tool", type: "${NodeConnectionTypes.AiTool}"}]
				}

				if (mode === 'retrieve') {
					return [{ displayName: "Vector Store", type: "${NodeConnectionTypes.AiVectorStore}"}]
				}
				return [{ displayName: "", type: "${NodeConnectionTypes.Main}"}]
			})($parameter)
		}}`,
		properties: [
			// ── Mode Selector ──
			{
				displayName: 'Operation Mode',
				name: 'mode',
				type: 'options',
				noDataExpression: true,
				default: 'retrieve',
				options: [
					{
						name: 'Get Many',
						value: 'load',
						description: 'Get many ranked documents from vector store for query',
						action: 'Get ranked documents from vector store',
					},
					{
						name: 'Insert Documents',
						value: 'insert',
						description: 'Insert documents into vector store',
						action: 'Add documents to vector store',
					},
					{
						name: 'Retrieve Documents (As Vector Store for Chain/Tool)',
						value: 'retrieve',
						description: 'Retrieve documents from vector store to be used as vector store with AI nodes',
						action: 'Retrieve documents for Chain/Tool as Vector Store',
						outputConnectionType: NodeConnectionTypes.AiVectorStore,
					},
					{
						name: 'Retrieve Documents (As Tool for AI Agent)',
						value: 'retrieve-as-tool',
						description: 'Retrieve documents from vector store to be used as tool with AI nodes',
						action: 'Retrieve documents for AI Agent as Tool',
						outputConnectionType: NodeConnectionTypes.AiTool,
					},
				],
			},
			// ── Retrieve-as-tool fields ──
			{
				displayName: 'Name',
				name: 'toolName',
				type: 'string',
				default: '',
				required: true,
				description: 'Name of the vector store tool',
				placeholder: 'e.g. company_knowledge_base',
				displayOptions: { show: { mode: ['retrieve-as-tool'] } },
			},
			{
				displayName: 'Description',
				name: 'toolDescription',
				type: 'string',
				default: '',
				required: true,
				typeOptions: { rows: 2 },
				description:
					'Explain to the LLM what this tool does. A good description will help the AI produce better results.',
				placeholder: 'e.g. Search the Hologres vector store for relevant documents',
				displayOptions: { show: { mode: ['retrieve-as-tool'] } },
			},
			// ── Shared fields ──
			...sharedFields,
			// ── Insert-specific fields ──
			{
				...dimensionsField,
				displayOptions: { show: { mode: ['insert'] } },
			},
			{
				displayName: 'Embedding Batch Size',
				name: 'embeddingBatchSize',
				type: 'number',
				default: 10,
				description: 'Number of documents to embed in a single batch. Reduce this if your embedding model has batch size limits.',
				displayOptions: { show: { mode: ['insert'] } },
			},
			{
				displayName: 'Options',
				name: 'options',
				type: 'collection',
				placeholder: 'Add Option',
				default: {},
				displayOptions: { show: { mode: ['insert'] } },
				options: [distanceMethodField, columnNamesField, hgraphIndexField],
			},
			// ── Load-specific fields ──
			{
				displayName: 'Prompt',
				name: 'prompt',
				type: 'string',
				default: '',
				required: true,
				description:
					'Search prompt to retrieve matching documents from the vector store using similarity-based ranking',
				displayOptions: { show: { mode: ['load'] } },
			},
			{
				displayName: 'Limit',
				name: 'topK',
				type: 'number',
				default: 4,
				description: 'Number of top results to fetch from vector store',
				displayOptions: { show: { mode: ['load', 'retrieve-as-tool'] } },
			},
			{
				displayName: 'Include Metadata',
				name: 'includeDocumentMetadata',
				type: 'boolean',
				default: true,
				description: 'Whether or not to include document metadata',
				displayOptions: { show: { mode: ['load', 'retrieve-as-tool'] } },
			},
			// ── Load/Retrieve options ──
			{
				displayName: 'Options',
				name: 'options',
				type: 'collection',
				placeholder: 'Add Option',
				default: {},
				displayOptions: { show: { mode: ['load', 'retrieve-as-tool'] } },
				options: [distanceMethodField, columnNamesField, metadataFilterField],
			},
			{
				displayName: 'Options',
				name: 'options',
				type: 'collection',
				placeholder: 'Add Option',
				default: {},
				displayOptions: { show: { mode: ['retrieve'] } },
				options: [distanceMethodField, columnNamesField, metadataFilterField],
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const mode = this.getNodeParameter('mode', 0) as string;
		const embeddings = (await this.getInputConnectionData(
			NodeConnectionTypes.AiEmbedding,
			0,
		)) as Embeddings;

		// ── Load Mode ──
		if (mode === 'load') {
			const items = this.getInputData(0);
			const resultData: INodeExecutionData[] = [];

			for (let itemIndex = 0; itemIndex < items.length; itemIndex++) {
				const tableName = this.getNodeParameter('tableName', itemIndex, '') as string;
				const credentials = await this.getCredentials('hologresApi');
				const pool = createPoolFromCredentials(credentials);
				const columns = getColumnOptions(this);
				const distanceMethod = this.getNodeParameter(
					'options.distanceMethod',
					0,
					'Cosine',
				) as DistanceMethod;

				const config: HologresVectorStoreArgs = {
					pool,
					tableName,
					dimensions: 0,
					distanceMethod,
					columns,
					indexSettings: {
						baseQuantizationType: 'rabitq',
						useReorder: true,
						maxDegree: 64,
						efConstruction: 400,
					},
				};

				const store = new HologresVectorStore(embeddings, config);
				await store._initializeClient();

				try {
					const prompt = this.getNodeParameter('prompt', itemIndex) as string;
					const topK = this.getNodeParameter('topK', itemIndex, 4) as number;
					const includeDocumentMetadata = this.getNodeParameter(
						'includeDocumentMetadata',
						itemIndex,
						true,
					) as boolean;

					const filter = getMetadataFiltersValues(this, itemIndex);
					const embeddedPrompt = await embeddings.embedQuery(prompt);
					const docs = await store.similaritySearchVectorWithScore(
						embeddedPrompt,
						topK,
						filter,
					);

					const serializedDocs = docs.map(([doc, score]) => {
						const document = {
							pageContent: doc.pageContent,
							...(includeDocumentMetadata ? { metadata: doc.metadata } : {}),
						};
						return { json: { document, score }, pairedItem: { item: itemIndex } };
					});

					resultData.push(...serializedDocs);
				} finally {
					store.client?.release();
					void store.pool.end();
				}
			}

			return [resultData];
		}

		// ── Insert Mode ──
		if (mode === 'insert') {
			const items = this.getInputData();
			const documentInput = await this.getInputConnectionData(
				NodeConnectionTypes.AiDocument,
				0,
			);
			const resultData: INodeExecutionData[] = [];

			for (let itemIndex = 0; itemIndex < items.length; itemIndex++) {
				if (this.getExecutionCancelSignal()?.aborted) break;

				const { processedDocuments, serializedDocuments } = await processDocumentInput(
					documentInput,
					items[itemIndex],
					itemIndex,
				);
				resultData.push(...serializedDocuments);

				const tableName = this.getNodeParameter('tableName', itemIndex, '') as string;
				const dimensions = this.getNodeParameter('dimensions', itemIndex, 1536) as number;
				const embeddingBatchSize = this.getNodeParameter('embeddingBatchSize', itemIndex, 10) as number;
				const credentials = await this.getCredentials('hologresApi');
				const pool = createPoolFromCredentials(credentials);
				const columns = getColumnOptions(this);
				const distanceMethod = this.getNodeParameter(
					'options.distanceMethod',
					0,
					'Cosine',
				) as DistanceMethod;
				const indexSettings = getHGraphIndexSettings(this);

				const config: HologresVectorStoreArgs = {
					pool,
					tableName,
					dimensions,
					distanceMethod,
					columns,
					indexSettings,
				};

				// Process documents in batches
				const vectorStore = await HologresVectorStore.initialize(embeddings, config);
				try {
					for (let i = 0; i < processedDocuments.length; i += embeddingBatchSize) {
						if (this.getExecutionCancelSignal()?.aborted) break;
						const batch = processedDocuments.slice(i, i + embeddingBatchSize);
						await vectorStore.addDocuments(batch);
					}
				} finally {
					vectorStore.client?.release();
					await vectorStore.pool.end();
				}
			}

			return [resultData];
		}

		// ── Retrieve-as-Tool Mode (execute) ──
		if (mode === 'retrieve-as-tool') {
			const items = this.getInputData(0);
			const resultData: INodeExecutionData[] = [];

			for (let itemIndex = 0; itemIndex < items.length; itemIndex++) {
				if (this.getExecutionCancelSignal()?.aborted) break;

				const tableName = this.getNodeParameter('tableName', itemIndex, '') as string;
				const topK = this.getNodeParameter('topK', itemIndex, 4) as number;
				const includeDocumentMetadata = this.getNodeParameter(
					'includeDocumentMetadata',
					itemIndex,
					true,
				) as boolean;
				const filter = getMetadataFiltersValues(this, itemIndex);

				const credentials = await this.getCredentials('hologresApi');
				const pool = createPoolFromCredentials(credentials);
				const columns = getColumnOptions(this);
				const distanceMethod = this.getNodeParameter(
					'options.distanceMethod',
					0,
					'Cosine',
				) as DistanceMethod;

				const config: HologresVectorStoreArgs = {
					pool,
					tableName,
					dimensions: 0,
					distanceMethod,
					columns,
					indexSettings: {
						baseQuantizationType: 'rabitq',
						useReorder: true,
						maxDegree: 64,
						efConstruction: 400,
					},
				};

				const store = new HologresVectorStore(embeddings, config);
				await store._initializeClient();

				try {
					// Get query from input item
					const query = items[itemIndex].json?.chatInput as string ?? items[itemIndex].json?.query as string ?? '';
					if (!query) {
						throw new NodeOperationError(this.getNode(), 'No query found in input item. Expected "chatInput" or "query" field.');
					}

					const embeddedPrompt = await embeddings.embedQuery(query);
					const docs = await store.similaritySearchVectorWithScore(
						embeddedPrompt,
						topK,
						filter,
					);

					const serializedDocs = docs.map(([doc, score]) => {
						const document = {
							pageContent: doc.pageContent,
							...(includeDocumentMetadata ? { metadata: doc.metadata } : {}),
						};
						return { json: { document, score }, pairedItem: { item: itemIndex } };
					});

					resultData.push(...serializedDocs);
				} finally {
					store.client?.release();
					void store.pool.end();
				}
			}

			return [resultData];
		}

		throw new NodeOperationError(
			this.getNode(),
			`The operation mode "${mode}" is not supported in execute. Use "load", "insert", or "retrieve-as-tool".`,
		);
	}

	async supplyData(
		this: ISupplyDataFunctions,
		itemIndex: number,
	): Promise<SupplyData> {
		const mode = this.getNodeParameter('mode', 0) as string;
		const embeddings = (await this.getInputConnectionData(
			NodeConnectionTypes.AiEmbedding,
			0,
		)) as Embeddings;

		// ── Retrieve Mode ──
		if (mode === 'retrieve') {
			const filter = getMetadataFiltersValues(this, itemIndex);
			const tableName = this.getNodeParameter('tableName', itemIndex, '', {
				extractValue: true,
			}) as string;
			const credentials = await this.getCredentials('hologresApi');
			const pool = createPoolFromCredentials(credentials);
			const columns = getColumnOptions(this);
			const distanceMethod = this.getNodeParameter(
				'options.distanceMethod',
				0,
				'Cosine',
			) as DistanceMethod;

			const config: HologresVectorStoreArgs = {
				pool,
				tableName,
				dimensions: 0,
				distanceMethod,
				columns,
				indexSettings: {
					baseQuantizationType: 'rabitq',
					useReorder: true,
					maxDegree: 64,
					efConstruction: 400,
				},
				filter,
			};

			const store = new HologresVectorStore(embeddings, config);
			await store._initializeClient();

			return {
				response: store,
				closeFunction: async () => {
					store.client?.release();
					void store.pool.end();
				},
			};
		}

		// ── Retrieve-as-Tool Mode ──
		if (mode === 'retrieve-as-tool') {
			const toolDescription = this.getNodeParameter('toolDescription', itemIndex) as string;
			const toolName = this.getNodeParameter('toolName', itemIndex) as string;

			const vectorStoreTool = new DynamicTool({
				name: toolName,
				description: toolDescription,
				func: async (query: string) => {
					const topK = this.getNodeParameter('topK', itemIndex, 4) as number;
					const includeDocumentMetadata = this.getNodeParameter(
						'includeDocumentMetadata',
						itemIndex,
						true,
					) as boolean;
					const filter = getMetadataFiltersValues(this, itemIndex);

					const tableName = this.getNodeParameter('tableName', itemIndex, '', {
						extractValue: true,
					}) as string;
					const credentials = await this.getCredentials('hologresApi');
					const pool = createPoolFromCredentials(credentials);
					const columns = getColumnOptions(this);
					const distanceMethod = this.getNodeParameter(
						'options.distanceMethod',
						0,
						'Cosine',
					) as DistanceMethod;

					const config: HologresVectorStoreArgs = {
						pool,
						tableName,
						dimensions: 0,
						distanceMethod,
						columns,
						indexSettings: {
							baseQuantizationType: 'rabitq',
							useReorder: true,
							maxDegree: 64,
							efConstruction: 400,
						},
					};

					const store = new HologresVectorStore(embeddings, config);
					await store._initializeClient();

					try {
						const embeddedPrompt = await embeddings.embedQuery(query);
						const docs = await store.similaritySearchVectorWithScore(
							embeddedPrompt,
							topK,
							filter,
						);

						const results = docs.map(([doc, score]) => ({
							pageContent: doc.pageContent,
							...(includeDocumentMetadata ? { metadata: doc.metadata } : {}),
							score,
						}));

						return JSON.stringify(results);
					} finally {
						store.client?.release();
						void store.pool.end();
					}
				},
			});

			return {
				response: vectorStoreTool,
			};
		}

		throw new NodeOperationError(
			this.getNode(),
			`The operation mode "${mode}" is not supported in supplyData. Use "retrieve" or "retrieve-as-tool".`,
		);
	}
}
