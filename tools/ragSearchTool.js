// tools/ragSearchTool.js
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// THE RAG SEARCH TOOL
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Purpose: Search all student Q&As in MongoDB by MEANING
// When:    Called by the agent EVERY TIME a question is asked
// What it does:
//   1. Converts the question to a vector (same model as ingestTool)
//   2. Runs MongoDB $vectorSearch to find the closest matching chunks
//   3. Returns the matching chunks to Claude so it can answer
//
// This is the RETRIEVAL part of RAG:
//   R = Retrieval  ← this tool
//   A = Augmented  ← we add retrieved chunks to the prompt
//   G = Generation ← Claude generates the answer from those chunks
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// dotenv MUST be first
import "dotenv/config";

import { tool }             from "@langchain/core/tools";
import { z }                from "zod";
import { Chunk, connectDB } from "../models/Document.js";
import { pipeline }         from "@xenova/transformers";

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// LOAD THE SAME EMBEDDING MODEL AS ingestTool
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// WHY THE SAME MODEL?
//   At ingest time:  "CTO of a dope startup" → [0.23, -0.81, 0.14, ...]
//   At search time:  "dream job"             → [0.22, -0.79, 0.16, ...]
//   These vectors are CLOSE in the 384-dimensional space
//   MongoDB $vectorSearch finds that closeness
//
//   If we used a DIFFERENT model for search:
//   "dream job" might produce completely different numbers
//   and the search would return garbage results
//
// Xenova caches the model after first download
// so both tools share the same cached copy
const embed = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2"
);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HELPER: getEmbedding(text)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Identical to the one in ingestTool — converts text to 384 numbers
// We need this here to convert the SEARCH QUESTION to a vector
// before running $vectorSearch
async function getEmbedding(text) {
  const output = await embed(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// THE SEARCH TOOL DEFINITION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export const ragSearchTool = tool(

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // THE FUNCTION THAT RUNS
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // Claude calls this with:
  //   question: "What is Desmend's dream job?"
  //   limit:    4 (optional — how many chunks to return)
  async ({ question, limit = 4 }) => {

    // ── STEP 1: Connect to MongoDB ──────────────────────────────
    await connectDB();
    console.log(`\n🔍 Searching: "${question}"`);

    // ── STEP 2: Convert question to vector ─────────────────────
    // "What is Desmend's dream job?"
    // → [0.22, -0.79, 0.16, ..., 384 numbers]
    // This vector represents the MEANING of the question
    // We'll use it to find chunks with similar meanings in MongoDB
    const queryVector = await getEmbedding(question);

    // ── STEP 3: Run MongoDB $vectorSearch ──────────────────────
    // Chunk.aggregate() runs an aggregation pipeline in MongoDB
    // An aggregation pipeline is a series of stages that transform data
    // We use two stages:
    //   Stage 1: $vectorSearch — finds similar chunks by vector distance
    //   Stage 2: $project — selects which fields to include in results
    const results = await Chunk.aggregate([

      // ── STAGE 1: $vectorSearch ────────────────────────────────
      // This is the MongoDB Atlas Vector Search aggregation stage
      // It takes a query vector and finds the closest stored vectors
      //
      // How it works:
      //   1. Takes queryVector (our question as numbers)
      //   2. Compares it to EVERY embedding in the chunks collection
      //   3. Uses COSINE SIMILARITY to measure how close they are
      //      (cosine similarity 1.0 = identical, 0.0 = unrelated)
      //   4. Returns the top `limit` closest chunks
      {
        $vectorSearch: {
          index: "autoembed_index", // Name of the Vector Search index in Atlas
                                    // Must match EXACTLY what you named it in Atlas UI
          path: "embedding",        // The field in MongoDB that stores our vectors
                                    // Matches the "embedding" field in Document.js
          queryVector,              // Our question converted to 384 numbers
                                    // MongoDB compares this to every stored embedding
          // NO filter here — we search ALL students' chunks
          // MongoDB returns whichever chunks are most similar
          // regardless of which student they belong to
          numCandidates: 100,       // Check 100 candidates before picking the best ones
                                    // Higher = more accurate but slightly slower
                                    // Rule: numCandidates should be at least 10x limit
          limit,                    // Return only the top N results (default: 4)
                                    // These are the most semantically similar chunks
        },
      },

      // ── STAGE 2: $project ─────────────────────────────────────
      // $project selects which fields to include in the results
      // 1 = include this field
      // 0 = exclude this field
      {
        $project: {
          content:     1, // The actual Q&A text — what Claude will read
          studentName: 1, // Who wrote this — so we see names not IDs
          score: { $meta: "vectorSearchScore" }, // Similarity score (0–1)
                                                  // Higher = more similar to query
          _id: 0, // Exclude MongoDB's internal document ID (we don't need it)
        },
      },
    ]);

    // Log how many results came back — useful for debugging
    console.log(`   Found ${results.length} results`);

    // ── STEP 4: Handle no results ──────────────────────────────
    // If the database returned nothing, tell Claude
    // Claude will then tell the user the topic wasn't found
    if (results.length === 0) {
      return "No matching content found.";
    }

    // ── STEP 5: Format results for Claude ──────────────────────
    // Transform the array of result objects into a readable string
    // Claude reads this string and uses it to compose its answer
    //
    // Each result looks like:
    //   [Desmend Jetton] [score: 0.921]
    //   Q: What job would you like after CodeSquad? A: CTO of a dope startup
    //   ---
    //   [Desmend Jetton] [score: 0.874]
    //   Q: Where do you see yourself in 3-5 years? A: Happy and blessed...
    //
    // The studentName in brackets tells Claude WHOSE answer this is
    // The score tells Claude HOW CONFIDENT the match is
    // The content is what Claude uses to formulate the answer
    return results
      .map(
        (r) =>
          `[${r.studentName}] [score: ${r.score.toFixed(3)}]\n${r.content}`
      )
      .join("\n---\n"); // Separate each result with --- for readability
  },

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // TOOL METADATA
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {
    // The name Claude uses internally to call this tool
    name: "search_student_sheet",

    // Claude reads this to decide WHEN to use this tool
    // "ALWAYS call this FIRST" is critical — without it Claude might
    // try to answer from its training data instead of our database
    description:
      "Searches ALL student Q&A responses in the database by meaning. " +
      "Call this FIRST for every single question. Always. No exceptions. " +
      "Returns the most relevant chunks from whichever student matches best.",

    // What Claude must pass when calling this tool
    schema: z.object({
      question: z.string().describe(
        "The full question to search for. " +
        "Example: 'What is Desmend's dream job?' or 'What is Lewis' salary goal?'"
      ),
      limit: z.number().optional().describe(
        "Max number of chunks to return. Default is 4. " +
        "Use 6-8 for broad questions about multiple students."
      ),
    }),
  }
);