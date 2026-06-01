// tools/ingestTool.js
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// THE INGEST TOOL
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Purpose: Load student Q&A responses into MongoDB
// When:    Run ONCE via ingest-all.js (before asking questions)
// What it does:
//   1. Receives a student's raw Q&A text
//   2. Splits it into individual Q&A pairs (chunks)
//   3. Converts each chunk to a vector using Xenova (free, local AI)
//   4. Saves chunk + vector to MongoDB
//
// After this runs, the student's answers are in MongoDB
// and ready to be searched by ragSearchTool
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// dotenv MUST be first — loads .env so process.env.MONGODB_URI works
import "dotenv/config";

// tool() and z are from LangChain/Zod
// tool() wraps our function so Claude can call it
// z defines the schema (shape) of the inputs Claude must pass
import { tool } from "@langchain/core/tools";
import { z }    from "zod";

// Import the Chunk model and connectDB from our schema file
// Chunk = the Mongoose model we use to save/find documents
// connectDB = the function that connects to MongoDB Atlas
import { Chunk, connectDB } from "../models/Document.js";

// pipeline() is from @xenova/transformers
// It loads a free, local AI model that converts text → numbers
// No API key needed — runs entirely on your machine
import { pipeline } from "@xenova/transformers";

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// LOAD THE EMBEDDING MODEL
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// "feature-extraction" = the task (convert text to numbers)
// "Xenova/all-MiniLM-L6-v2" = the model name
//   - "all-MiniLM" = works for all types of text
//   - "L6" = 6 transformer layers (small, fast)
//   - "v2" = version 2
//   - Outputs: 384 numbers per chunk
//   - Cost: FREE (runs locally, no API calls)
//   - First run: downloads ~90MB and caches it (30 sec wait)
//   - Future runs: uses cached copy (instant)
//
// We load it here at the module level (outside any function)
// so it only loads ONCE when the file is imported
// not every time ingestTool is called
const embed = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2"
);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HELPER: getEmbedding(text)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Converts any text string into a vector (array of 384 numbers)
//
// Input:  "Q: What job would you like? A: CTO of a dope startup"
// Output: [-0.096, -0.044, -0.043, 0.020, 0.092, ..., 384 total]
//
// pooling: "mean"  → averages all token vectors into one
// normalize: true  → scales to unit length (required for cosine similarity)
//
// We call this for EVERY chunk we save (ingest)
// AND for EVERY question we search (ragSearchTool)
// SAME model = SAME vector space = searching works correctly
async function getEmbedding(text) {
  const output = await embed(text, { pooling: "mean", normalize: true });
  return Array.from(output.data); // Convert to plain JS array of 384 numbers
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// THE INGEST TOOL DEFINITION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// tool(function, metadata) creates a tool Claude can call
//   - First argument:  the actual function that runs
//   - Second argument: metadata (name, description, schema)
//     Claude reads the description to know WHEN to use this tool
//     Claude reads the schema to know WHAT to pass as inputs
export const ingestTool = tool(

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // THE FUNCTION THAT RUNS
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // Parameters come from Claude (validated by the schema below)
  // studentId:   "student_099"
  // studentName: "Desmend Jetton"
  // fileName:    "student_099_answers.txt"
  // rawText:     "Q: What is your name? A: Desmend Jetton\n\n
  //               Q: What job would you like? A: CTO of a dope startup\n\n..."
  async ({ studentId, studentName, fileName, rawText }) => {

    // ── STEP 1: Connect to MongoDB ──────────────────────────────
    // connectDB() checks if already connected — skips if yes
    // Uses process.env.MONGODB_URI from .env
    await connectDB();

    // ── STEP 2: Split rawText into chunks ──────────────────────
    // rawText comes in with Q&A pairs separated by \n\n (double newline)
    // We split on \n\n to get individual Q&A pairs
    //
    // EXAMPLE:
    // rawText = "Q: What is your name? A: Desmend\n\nQ: Dream job? A: CTO"
    // After split("\n\n"):
    //   chunks = [
    //     "Q: What is your name? A: Desmend",   ← chunk 0
    //     "Q: Dream job? A: CTO"                ← chunk 1
    //   ]
    //
    // ⚠️  CRITICAL: Must be "\n\n" (DOUBLE newline) NOT "\n" (single)
    //     Single newline splits Q and A apart (breaks search)
    //     Double newline keeps them together (search works correctly)
    const chunks = rawText
      .split("\n\n")        // Split on DOUBLE newline to keep Q&A paired
      .map((ln) => ln.trim()) // Remove leading/trailing whitespace
      .filter((ln) => ln.length > 8); // Skip blank or too-short lines

    // Safety check: if no chunks found, something is wrong with the format
    if (chunks.length === 0) {
      return "No valid chunks found — check your text file formatting.";
    }

    // ── STEP 3: Save each chunk to MongoDB ─────────────────────
    let saved = 0;   // Track how many chunks we successfully saved
    let skipped = 0; // Track how many we skipped (duplicates)

    // Loop through each chunk with its index
    // chunks.entries() gives us [index, value] pairs:
    //   [0, "Q: What is your name? A: Desmend"]
    //   [1, "Q: Dream job? A: CTO"]
    //   etc.
    for (const [i, chunk] of chunks.entries()) {

      // Check if this exact chunk already exists for this student
      // Prevents duplicates if you run ingest twice on the same file
      const exists = await Chunk.findOne({ studentId, content: chunk });
      if (exists) {
        // Already in database — skip and move to next chunk
        skipped++;
        continue;
      }

      // Convert the chunk text to a vector (384 numbers)
      // This is the core of RAG:
      //   We embed at SAVE TIME so we can search at QUERY TIME
      //   "Q: Dream job? A: CTO of a dope startup"
      //   → [-0.096, -0.044, -0.043, ..., 384 numbers]
      const embedding = await getEmbedding(chunk);

      // Save the chunk to MongoDB with ALL its fields
      // Chunk.create() inserts a new document into the "chunks" collection
      await Chunk.create({
        studentId,   // "student_099" — who does this belong to?
        studentName, // "Desmend Jetton" — human-readable name for display
        fileName,    // "student_099_answers.txt" — where did it come from?
        content: chunk,  // The actual Q&A text Claude will read
        chunkIndex: i,   // 0, 1, 2... — position in the original file
        embedding,   // The 384 numbers — powers the vector search
        // createdAt is auto-set by the schema (Date.now)
      });

      saved++; // Increment saved counter
    }

    // ── STEP 4: Return summary to Claude ───────────────────────
    // This string is what Claude sees after the tool runs
    // Claude uses it to confirm the ingestion was successful
    return `Done. Saved ${saved} chunks for ${studentName}. Skipped ${skipped} duplicates.`;
  },

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // TOOL METADATA
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // Claude reads this metadata to decide when and how to call this tool
  {
    // The name Claude uses to call this tool internally
    name: "ingest_document",

    // Claude reads this description to decide WHEN to use this tool
    // Good descriptions are specific about WHEN to call and WHAT to pass
    description:
      "Loads a student Q&A sheet into MongoDB with embeddings. " +
      "Call this when the user wants to ingest or load a document. " +
      "Each Q&A pair becomes one searchable chunk. " +
      "Skips duplicates automatically.",

    // z.object() defines exactly what Claude must pass when calling this tool
    // Each field has a type and a .describe() that helps Claude know what to put
    schema: z.object({
      // Claude must pass all four of these
      studentId: z.string().describe(
        "Unique student ID from the CSV. Example: 'student_099'"
      ),
      studentName: z.string().describe(
        "Student's real name. Example: 'Desmend Jetton'"
      ),
      fileName: z.string().describe(
        "Name of the source file. Example: 'student_099_answers.txt'"
      ),
      rawText: z.string().describe(
        "Full Q&A text. Each Q&A pair must be separated by a double newline (\\n\\n)"
      ),
    }),
  }
);