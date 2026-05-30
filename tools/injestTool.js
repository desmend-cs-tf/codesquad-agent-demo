// tools/ingestTool.js
import { tool }              from "@langchain/core/tools";
import { z }                 from "zod";
import { Chunk, connectDB }  from "../models/Document.js";
import { pipeline }          from "@xenova/transformers";

const embed = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

async function getEmbedding(text) {
  const output = await embed(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

export const ingestTool = tool(
  async ({ studentId, studentName, fileName, rawText }) => {
    await connectDB();

    const chunks = rawText
      .split("\n")
      .map((ln) => ln.trim())
      .filter((ln) => ln.length > 8);

    if (chunks.length === 0) {
      return "No valid chunks found — check your text file formatting.";
    }

    let saved = 0;
    let skipped = 0;

    for (const [i, chunk] of chunks.entries()) {
      const exists = await Chunk.findOne({ studentId, content: chunk });
      if (exists) { skipped++; continue; }

      const embedding = await getEmbedding(chunk);

      await Chunk.create({
        studentId,
        studentName, // ← saved on every chunk
        fileName,
        content:    chunk,
        chunkIndex: i,
        embedding,
      });

      saved++;
    }

    return `Done. Saved ${saved} chunks for ${studentName} (${studentId}). Skipped ${skipped} duplicates.`;
  },
  {
    name: "ingest_document",
    description:
      "Loads a student Q&A sheet into MongoDB with embeddings. " +
      "Call when the user wants to ingest a document. " +
      "Each line becomes one searchable chunk.",
    schema: z.object({
      studentId:   z.string().describe("Unique ID, e.g. 'student_001'"),
      studentName: z.string().describe("Student's real name, e.g. 'Maria Chen'"), // ← add
      fileName:    z.string().describe("Original filename"),
      rawText:     z.string().describe("Full text content — each line becomes one chunk"),
    }),
  }
);