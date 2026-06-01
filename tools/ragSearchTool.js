// tools/ragSearchTool.js
// The RAG Search Tool — finds relevant student Q&As in MongoDB
// This is the RETRIEVAL part of RAG (Retrieval Augmented Generation)

import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { Chunk, connectDB } from "../models/Document.js";
import { pipeline } from "@xenova/transformers";

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// STEP 1: Load the embedding model
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Xenova is a free, local embedding model (runs on your machine)
// It converts text → 384 numbers that represent meaning
// We use it BOTH to:
//   1. Create embeddings when saving chunks (ingestTool)
//   2. Create embeddings when searching (this tool)
// Same model = same vector space = search works
const embed = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2"
);

// Helper function: convert text to a vector (384 numbers)
async function getEmbedding(text) {
  const output = await embed(text, { pooling: "mean", normalize: true });
  return Array.from(output.data); // Convert to plain array of numbers
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// STEP 2: Define the search tool
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// This is what Claude can call when it wants to search
export const ragSearchTool = tool(
  // The actual function that runs when Claude calls this tool
  async ({ question, limit = 4 }) => {
    // Connect to MongoDB (if not already connected)
    await connectDB();
    console.log(`\n🔍 Searching: "${question}"`);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // CONVERT QUESTION TO VECTOR
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // "What is Lewis' ideal salary?" becomes [0.23, -0.81, 0.14, ...]
    // This vector goes into the same 384-dimensional space as the chunks
    const queryVector = await getEmbedding(question);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // SEARCH MONGODB WITH VECTOR SEARCH
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // MongoDB's $vectorSearch aggregation stage:
    //   1. Takes your queryVector
    //   2. Finds all stored chunks with similar embeddings
    //   3. Returns them sorted by similarity (highest first)
    // This is the core of RAG — it finds RELEVANT chunks, not all chunks
    const results = await Chunk.aggregate([
      {
        $vectorSearch: {
          index: "autoembed_index", // Name of the Vector Search index in Atlas
          path: "embedding", // Field in MongoDB that stores the vectors
          queryVector, // Your question converted to numbers
          // NO filter — search all students' chunks
          numCandidates: 100, // Check 100 chunks before returning best ones
          limit, // Return top 4 chunks (or whatever limit is set)
        },
      },
      {
        // $project: which fields to include in the results
        $project: {
          content: 1, // Include the actual text chunk
          studentName: 1, // Include the student's name so we know whose answer it is
          score: { $meta: "vectorSearchScore" }, // Include similarity score (0-1)
          _id: 0, // Exclude MongoDB's internal ID field
        },
      },
    ]);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // HANDLE NO RESULTS
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // If no chunks match the question, tell Claude that
    if (results.length === 0) {
      return `No matching content found.`;
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // FORMAT RESULTS FOR CLAUDE
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Take each result and format it nicely so Claude can read it
    // Include:
    //   - Student name: so Claude knows whose answer this is
    //   - Score: so Claude sees how confident the match is
    //   - Content: the actual Q&A
    const chunks = results
      .map(
        (r) =>
          `[${r.studentName}]: [score: ${r.score.toFixed(3)}]\n${r.content}`
      )
      .join("\n---\n"); // Separate chunks with --- for readability

    return chunks;
  },

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // TOOL DEFINITION (metadata)
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // This tells Claude:
  //   - What this tool is called
  //   - When to use it
  //   - What parameters it accepts
  {
    name: "search_student_sheet",
    description:
      "Searches ALL students' Q&A sheets by meaning using vector similarity. " +
      "Returns results from whichever student's sheet matches best. " +
      "ALWAYS call this FIRST before answering any question.",
    schema: z.object({
      question: z.string().describe(
        "The question or topic to search for. Example: 'What is Lewis ideal salary?' or 'Where does Desmend see himself in 5 years?'"
      ),
      limit: z.number().optional().describe(
        "Max number of chunks to return. Default 4. Use more for broad topics."
      ),
    }),
  }
);