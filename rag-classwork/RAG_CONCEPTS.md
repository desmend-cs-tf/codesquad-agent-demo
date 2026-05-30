# Retrieval-Augmented Generation (RAG) Concepts

## The Problem: Knowledge Cutoff and Hallucination

Large Language Models (LLMs) like Claude and Gemini have two critical limitations:

1. **Knowledge Cutoff:** Training data ends at a specific date. Claude Haiku's knowledge cutoff is April 2024. It cannot answer questions about events, articles, or technologies released after that date.

2. **Hallucination:** LLMs sometimes generate plausible-sounding but incorrect information, especially about specific facts, URLs, or recent events. Without external verification, you can't trust the accuracy.

Example:
```
User: "What's the latest GPU announced in May 2026?"
Claude (without RAG): "I don't know; my training data ends April 2024."
Claude (with RAG): "Retrieves recent tech articles, finds GPU announcement, answers accurately with sources."
```

## The RAG Solution

RAG (Retrieval-Augmented Generation) solves both problems:

1. **Retrieve:** Before generating an answer, search a database of documents for context relevant to the user's query
2. **Augment:** Include the retrieved documents in the prompt to the LLM
3. **Generate:** The LLM answers based on the provided context, not just its training data

This enables:
- Current information (retrieve today's articles, news, docs)
- Accurate citations (LLM can reference specific documents)
- Reduced hallucination (LLM grounds answers in retrieved facts)

## RAG Architecture: The Pipeline

```
User Query
    |
    v
Embedding Model (Gemini embedding-001)
    |
    v (converts query to vector)
Vector Database (Pinecone, Supabase, Weaviate)
    |
    v (similarity search: find top 3 matching documents)
Retrieved Documents (context)
    |
    v (augment prompt with context)
LLM Prompt: "Here are relevant documents: [docs]. User asks: [query]. Answer based on docs."
    |
    v
LLM Response (grounded, cited, accurate)
```

## Key Concepts

### Vector Embeddings

Embeddings convert text into high-dimensional vectors (lists of numbers). Words/documents with similar meaning have vectors that are close together in space.

Example:
```
"Python is a programming language" → [0.23, -0.51, 0.89, ..., 0.12] (768 dimensions)
"Python is used for AI and data science" → [0.24, -0.50, 0.91, ..., 0.11] (similar vector)
"The Eiffel Tower is in Paris" → [0.81, 0.22, -0.34, ..., -0.99] (different vector)
```

Distance between similar vectors is small; distance between unrelated vectors is large.

**Why Embeddings Matter:** Instead of keyword matching (which misses synonyms and context), embeddings enable semantic search. "What's a good language for AI?" will find Python articles because the meaning is similar, not because keywords match.

### Vector Databases

Vector DBs are optimized for storing and querying vectors:

- **Pinecone:** Managed service, free tier, simple API, scales automatically
- **Supabase:** Open-source PostgreSQL with pgvector extension, flexible, self-hosted option
- **Weaviate:** Open-source, rich filtering, GraphQL API
- **Milvus:** Open-source, high performance, complex setup

All support similarity search: given a query vector, return the K most similar vectors (top 3, top 10, etc.).

### Chunking

Documents are often long. Passing entire books to the embedding model is expensive and wastes context. Solution: split documents into chunks (500-1000 tokens each).

Example:
```
Document: "Python is a programming language... [1000 words total]"
↓ Chunk into 500-token pieces
Chunk 1: "Python is a programming language..." (500 tokens)
Chunk 2: "Advanced topics include..." (500 tokens)
↓ Embed each chunk independently
Chunk 1 Vector: [0.23, -0.51, ...]
Chunk 2 Vector: [0.18, -0.49, ...]
↓ Store both in vector DB
When user queries "basic Python syntax", only Chunk 1 retrieves (more relevant)
```

## RAG vs Alternatives

| Approach | Pros | Cons | Use Case |
|---|---|---|---|
| **RAG** | Current info, scalable, citable, reduces hallucination | Requires document store and embeddings API | Real-time Q&A, customer support, knowledge bases |
| **Fine-tuning** | Model deeply learns your data | Expensive, slow training, knowledge cutoff still applies | Domain-specific tasks with specialized terminology |
| **In-Context Learning** | Simple, no external DB | Limited by context window (4K-128K tokens), expensive for large docs | Small knowledge bases, demo prompts |
| **No Context** | Cheapest, fast | Hallucination prone, outdated knowledge | Casual conversation, brainstorming |

For CodeSquad agents: RAG is ideal because you want to reference current tech articles and avoid hallucinating article URLs.

## RAG with Gemini Embeddings

This classwork uses **Gemini embedding-001** for vector generation:

- **Cost:** Free tier included with Google AI (50 requests/minute free)
- **Dimensions:** 768-dimensional vectors
- **API:** Simple `GoogleGenerativeAIEmbeddings` from LangChain
- **Integration:** Uses same Google API key as your Gemini agent

Setup is minimal since you already have `GOOGLE_API_KEY`:

```javascript
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";

const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "embedding-001",
  apiKey: process.env.GOOGLE_API_KEY,
});
```

## RAG in LangChain

LangChain abstracts RAG complexity:

```javascript
// 1. Load documents
const documents = ["Article 1...", "Article 2...", ...];

// 2. Create embeddings
const embeddings = new GoogleGenerativeAIEmbeddings({ model: "embedding-001" });

// 3. Create vector store
const vectorStore = await Pinecone.fromDocuments(documents, embeddings);

// 4. Create retrieval tool
const retriever = vectorStore.asRetriever();
const retrievalTool = tool(
  async ({ query }) => {
    const docs = await retriever.getRelevantDocuments(query);
    return docs.map(doc => doc.pageContent).join("\n---\n");
  },
  { name: "retrieve_docs", ... }
);

// 5. Add to agent
const agent = createReactAgent({
  llm: model,
  tools: [retrievalTool, ...otherTools]
});
```

When agent receives query, it can call `retrieve_docs` to get context, then answer accurately.

## Evaluation: How Do You Know RAG Works?

RAGAS (RAG Assessment) metrics:
- **Faithfulness:** Is the LLM's answer grounded in retrieved documents? (not hallucinated)
- **Answer Relevance:** Does the answer address the user's query?
- **Context Precision:** Are retrieved documents actually relevant?
- **Context Recall:** Did retrieval find all relevant documents?

Simple manual check: Ask a question, verify retrieved docs are relevant, verify answer cites specific doc content.

## Summary

RAG = Retrieval + Augmentation + Generation

- Retrieval: Find relevant documents via vector similarity search
- Augmentation: Include documents in LLM prompt
- Generation: LLM answers grounded in context

This solves knowledge cutoff, enables current information, reduces hallucination, and enables citations.
