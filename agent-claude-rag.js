// agent.js  —  run: node agent.js "your question here"
import "dotenv/config";
import { ChatAnthropic }       from "@langchain/anthropic";
// import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { createReactAgent }    from "@langchain/langgraph/prebuilt";
import { ingestTool }          from "./tools/ingestTool.js";
import { ragSearchTool }       from "./tools/ragSearchTool.js";

// ── Model — swap these two lines to use Gemini instead ────────
const model = new ChatAnthropic({
  model:       "claude-haiku-4-5",
  maxTokens:   1024,
  temperature: 0,
});

// const model = new ChatGoogleGenerativeAI({
//   model:           "gemini-2.5-flash",
//   maxOutputTokens: 1024,
//   temperature:     0,
// });

// ── Agent ─────────────────────────────────────────────────────
export const ragAgent = createReactAgent({
  llm:   model,
  tools: [ingestTool, ragSearchTool],
  messageModifier: `You are a study assistant for a CodeSquad class.
    Rules:
    1. ALWAYS call search_student_sheet before answering any question.
    2. Answer ONLY from the retrieved content. Never guess or use your training data.
    3. If the answer is not in the retrieved chunks, say:
       "I could not find that in this student's sheet."
    4. Always refer to the student by their name, not their ID.`,
});

// ── Runner — only executes when called directly ───────────────
// This lets agent.js be both imported (by ingest-all.js) and run directly
if (process.argv[1].endsWith("agent.js")) {

  const studentId = process.argv[2] || "student_001";
  const question  = process.argv[3] || "What is this student's name?";

  console.log(`\nStudent:  ${studentId}`);
  console.log(`Question: ${question}`);
  console.log("=".repeat(60));

  const result = await ragAgent.invoke({
    messages: [{
      role:    "user",
      content: `Student: ${studentId}. Question: ${question}`,
    }],
  });

  console.log("\nAnswer:", result.messages.at(-1).content);
  console.log("=".repeat(60));
}