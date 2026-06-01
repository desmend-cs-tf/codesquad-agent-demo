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

    // ── STEP 2: Extract student name from question (if present) ─
    // Look for patterns like "Desmend's", "Desmend", etc.
    const knownFirstNames = [
      "Desmend",
      "Lewis", 
      "Miguel",
      "Jaiden",
      "Andrew",
    ];
    
    let extractedStudentName = null;
    
    // Try to find a student name mentioned in the question
    for (const firstName of knownFirstNames) {
      const regex = new RegExp(`\\b${firstName}\\b`, "i");
      if (regex.test(question)) {
        extractedStudentName = firstName;
        console.log(`   Filtering results to: "${firstName}"`);
        break;
      }
    }

    // ── STEP 3: Convert question to vector ─────────────────────
    const queryVector = await getEmbedding(question);

    // ── STEP 4: Run MongoDB $vectorSearch (search ALL students) ─
    // Get the top results from all students, then filter after
    const results = await Chunk.aggregate([
      {
        $vectorSearch: {
          index: "autoembed_index",
          path: "embedding",
          queryVector,
          numCandidates: 100,
          limit: limit * 3, // Get extra results so we have enough after filtering
        },
      },

      {
        $project: {
          content:     1,
          studentName: 1,
          score: { $meta: "vectorSearchScore" },
          _id: 0,
        },
      },
    ]);

    // Log how many results came back from MongoDB
    console.log(`   Found ${results.length} total results from database`);
    
    // ── STEP 5: Filter results by student if one was identified ─
    let filteredResults = results;
    if (extractedStudentName) {
      // Use case-insensitive matching for student name prefix
      // E.g., "Desmend" matches "Desmend Jetton"
      const regex = new RegExp(`^${extractedStudentName}`, "i");
      filteredResults = results.filter(r => regex.test(r.studentName));
      
      console.log(`   Filtered to ${filteredResults.length} results for "${extractedStudentName}"`);
      
      // If no results for this student, fall back to showing top results from anyone
      if (filteredResults.length === 0) {
        console.log(`   (No results for "${extractedStudentName}", showing best matches)`);
        filteredResults = results.slice(0, limit);
      } else if (filteredResults.length > limit) {
        filteredResults = filteredResults.slice(0, limit);
      }
    } else {
      // No student specified, just take the top N results
      filteredResults = results.slice(0, limit);
    }

    // ── STEP 6: Handle no results ──────────────────────────────
    // If the database returned nothing, tell Claude
    // Claude will then tell the user the topic wasn't found
    if (filteredResults.length === 0) {
      return "No matching content found.";
    }

    // ── STEP 7: Format results for Claude ──────────────────────
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
    return filteredResults
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