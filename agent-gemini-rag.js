// agent-claude-rag.js
// RAG Agent — Retrieval Augmented Generation
// This agent searches a MongoDB database of student Q&As and answers questions
// based ONLY on what's in the database, not from its training data.

import "dotenv/config";
console.log("Starting...");

// Import the LLM (Large Language Model) — we're using Claude Haiku
import { ChatAnthropic } from "@langchain/anthropic";

// Import LangGraph's agent creator — handles the reasoning loop
import { createReactAgent } from "@langchain/langgraph/prebuilt";

// Import our custom tools — these are the functions the agent can call
import { ingestTool } from "./tools/ingestTool.js";
import { ragSearchTool } from "./tools/ragSearchTool.js";

console.log("Imports loaded. Initializing model...");

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// STEP 1: Create the Claude model
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// We use Claude Haiku because:
//   - It's fast (good for demos)
//   - It's cheap (low token cost)
//   - It's accurate enough for this task
// maxTokens: limit the response length to 1024 tokens (~750 words)
// temperature: 0 means "be deterministic" (same input = same output)
const model = new ChatAnthropic({
  model: "claude-haiku-4-5",
  maxTokens: 1024,
  temperature: 0,
});

console.log("Model initialized. Creating agent...");

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// STEP 2: Create the RAG agent
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// createReactAgent does the ReAct loop automatically:
//   1. Reason: Claude thinks about what to do
//   2. Act: Claude calls a tool (search_student_sheet)
//   3. Observe: Claude reads the tool result
//   4. Loop: Repeat until it has an answer
//
// The tools array tells Claude what functions it can call.
// The messageModifier is a system prompt that tells Claude HOW to use the tools.
export const ragAgent = createReactAgent({
  llm: model,
  tools: [ragSearchTool, ingestTool],

  // System prompt — this is what makes it "RAG" instead of a regular chatbot
  messageModifier: `You are a study assistant for a CodeSquad class.
    
    CRITICAL RULES:
    1. For ANY question, you MUST call search_student_sheet FIRST — no exceptions.
       Never answer from your training data. Always search first.
    2. Answer ONLY from the retrieved chunks. Do NOT add your own knowledge.
    3. If search returns no results, tell the user the topic wasn't found.
    4. Always mention the student's name when answering about them.
    
    Remember: Always search first. Answer from results only.`,
});

console.log("Agent created. Ready to answer questions.");

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// STEP 3: Run the agent
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// This block only runs when you call: node agent-claude-rag.js "question"
// It does NOT run when other files import this agent (like ingest-all.js)
if (process.argv[1].endsWith("agent-claude-rag.js")) {
  // Get the question from the command line
  // process.argv[2] is the first argument after "node agent-claude-rag.js"
  // Default question is "What is Desmend's dream job?" if none provided
  const question = process.argv[2] || "What is Desmend's dream job?";

  console.log("\n" + "=".repeat(60));
  console.log("Model:    Claude Haiku");
  console.log(`Question: ${question}`);
  console.log("=".repeat(60));
  console.log("Invoking agent...\n");

  try {
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // INVOKE: Call the agent with the question
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // This runs the ReAct loop:
    //   - Claude reads your question
    //   - Claude decides to call search_student_sheet
    //   - search_student_sheet finds similar chunks in MongoDB
    //   - Claude reads the results
    //   - Claude writes an answer based ONLY on those results
    const result = await ragAgent.invoke({
      messages: [
        {
          role: "user",
          content: `Answer this question by searching our student database: ${question}`,
        },
      ],
    });

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // GET THE ANSWER
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // result.messages is an array of all messages in the conversation
    // The LAST message is always Claude's final answer
    // .at(-1) gets the last item in the array
    // .content is the text of that message
    const finalAnswer = result.messages.at(-1).content;

    console.log("Answer:", finalAnswer);
    console.log("=".repeat(60));
  } catch (err) {
    // If something goes wrong, print the error
    console.error("Error:", err.message);
    process.exit(1); // Exit with error code
  }
}