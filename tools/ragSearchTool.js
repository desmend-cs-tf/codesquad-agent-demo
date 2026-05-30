// tools/ragSearchTool.js
import { tool }              from "@langchain/core/tools";
import { z }                 from "zod";
import { Chunk, connectDB }  from "../models/Document.js";
import { pipeline }          from "@xenova/transformers";

const embed = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

async function getEmbedding(text) {
  const output = await embed(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

export const ragSearchTool = tool(
  async ({ studentId, question, limit = 4 }) => {
    await connectDB();

    const queryVector = await getEmbedding(question);

    const results = await Chunk.aggregate([
      {
        $vectorSearch: {
          index:         "chunk_vector_index",
          path:          "embedding",
          queryVector,
          filter:        { studentId },
          numCandidates: 50,
          limit,
        },
      },
      {
        $project: {
          content:     1,
          studentName: 1, // ← include in results
          score:       { $meta: "vectorSearchScore" },
          _id:         0,
        },
      },
    ]);

    if (results.length === 0) {
      return "No matching content found for this student.";
    }

    // Show the name in the header so it's clear whose data this is
    const name = results[0].studentName;
    const chunks = results
      .map((r) => `[score: ${r.score.toFixed(3)}]\n${r.content}`)
      .join("\n---\n");

    return `Results for ${name} (${studentId}):\n\n${chunks}`;
  },
  {
    name: "search_student_sheet",
    description:
      "Searches a student's Q&A sheet by meaning. " +
      "ALWAYS call this FIRST before answering any question. " +
      "Do NOT answer from your own knowledge — only from what this tool returns.",
    schema: z.object({
      studentId: z.string().describe("The student to search, e.g. 'student_001'"),
      question:  z.string().describe("The question or topic to search for"),
      limit:     z.number().optional().describe("Max chunks to return. Default 4."),
    }),
  }
);