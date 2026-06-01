// agent-claude-rag.js
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// THE RAG AGENT — Entry point for asking questions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// This file does two things:
//   1. Creates and EXPORTS the ragAgent (used by ingest-all.js)
//   2. Runs the agent when called directly from the terminal
//
// Run it: node agent-claude-rag.js "What is Desmend's dream job?"
//
// What happens when you run it:
//   1. Claude reads your question
//   2. Claude calls search_student_sheet (from ragSearchTool.js)
//   3. ragSearchTool converts question to vector, searches MongoDB
//   4. MongoDB returns the closest matching Q&A chunks
//   5. Claude reads those chunks and writes an answer
//   6. You see the answer in the terminal
//
// This is RAG in action:
//   R (Retrieval)  = steps 2-4 (search MongoDB)
//   A (Augmented)  = step 5 (chunks added to Claude's context)
//   G (Generation) = step 5 (Claude generates answer from chunks)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// dotenv MUST be the VERY FIRST import
// It loads .env before any other code runs
// Without this, process.env.MONGODB_URI is undefined everywhere
import "dotenv/config";
console.log("Starting...");

// ChatAnthropic = the Claude model class from LangChain
// This is what actually sends requests to Anthropic's API
import { ChatAnthropic } from "@langchain/anthropic";

// createReactAgent = LangGraph's agent creator
// It implements the ReAct loop automatically:
//   Reason → Act → Observe → Reason → Act → Observe → (until done)
import { createReactAgent } from "@langchain/langgraph/prebuilt";

// Our two custom tools:
//   ingestTool    = loads student data into MongoDB (used by ingest-all.js)
//   ragSearchTool = searches MongoDB by meaning (used every time a question is asked)
import { ingestTool }    from "./tools/ingestTool.js";
import { ragSearchTool } from "./tools/ragSearchTool.js";

console.log("Imports loaded. Initializing model...");

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// STEP 1: Create the Claude model
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// We configure the LLM (Large Language Model) here
// This object is passed to createReactAgent below
//
// model: "claude-haiku-4-5"
//   We use Haiku (not Sonnet or Opus) because:
//   - It's the fastest Claude model
//   - It's the cheapest (important for a classroom with many queries)
//   - It's accurate enough for this RAG task
//   - The heavy work is done by MongoDB, not the LLM
//
// maxTokens: 1024
//   Limits Claude's response to ~750 words
//   Prevents runaway responses that waste API credits
//
// temperature: 0
//   0 = fully deterministic (same question → same answer every time)
//   1 = creative/random
//   We use 0 because we want consistent, factual answers from the DB
const model = new ChatAnthropic({
  model:       "claude-haiku-4-5",
  maxTokens:   1024,
  temperature: 0,
});

console.log("Model initialized. Creating agent...");

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// STEP 2: Create the RAG agent
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// createReactAgent() sets up the full ReAct reasoning loop
//
// THE REACT LOOP (happens automatically inside the agent):
//   1. REASON: Claude reads the question and decides what to do
//   2. ACT:    Claude calls a tool (search_student_sheet)
//   3. OBSERVE: Claude reads the tool's results
//   4. REASON: Claude decides if it has enough info to answer
//   5. Answer: Claude writes the final response
//   (loops back to step 2 if it needs more info)
//
// llm:   the model to use for reasoning (Claude Haiku above)
// tools: the functions Claude is allowed to call
//        - ragSearchTool: search MongoDB (used for questions)
//        - ingestTool:    load documents (used by ingest-all.js)
//        Tools listed FIRST are slightly preferred by the model
//
// messageModifier: a SYSTEM PROMPT that tells Claude how to behave
//   This is what makes it RAG instead of a regular chatbot
//   Without "ALWAYS call search_student_sheet", Claude might
//   just answer from its training data and hallucinate
//
// export: we export ragAgent so ingest-all.js can import it
//   import { ragAgent } from "./agent-claude-rag.js"
export const ragAgent = createReactAgent({
  llm:   model,
  tools: [ragSearchTool, ingestTool],

  // System prompt — Claude reads this before doing anything
  // Think of it as giving Claude its "job description"
  messageModifier: `You are a student career assistant for CodeSquad.
    You have access to a database of student Q&A responses.

    RULES — follow these exactly:
    1. ALWAYS call search_student_sheet first. No exceptions.
       Never answer from your training data. Always search first.
    2. Pass the full question as-is to the search tool.
    3. Answer ONLY from what the tool returns.
    4. Never say the database is down. Never apologize. Just search.
    5. If results come back empty, say "I couldn't find that in the database."`,
});

console.log("Agent created. Ready to answer questions.");

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// STEP 3: Run the agent (only when called directly)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// process.argv[1] = the file that was run directly by Node
//
// WHY THIS CHECK?
//   When ingest-all.js does: import { ragAgent } from "./agent-claude-rag.js"
//   Node runs this ENTIRE file to get the export
//   Without this check, the runner block below would ALSO execute
//   which would try to answer a question in the middle of ingesting
//
//   This check says:
//   "Only run the question-answering block if THIS file was called directly"
//   node agent-claude-rag.js → runs the block  ✓
//   import from agent-claude-rag.js → skips the block ✓
if (process.argv[1].endsWith("agent-claude-rag.js")) {

  // ── GET THE QUESTION ────────────────────────────────────────
  // process.argv is an array of command line arguments:
  //   process.argv[0] = "node"
  //   process.argv[1] = "agent-claude-rag.js"
  //   process.argv[2] = your question (if you passed one)
  //
  // Examples:
  //   node agent-claude-rag.js "What is Desmend's dream job?"
  //     → question = "What is Desmend's dream job?"
  //
  //   node agent-claude-rag.js
  //     → question = "What is Desmend's dream job?" (default)
  const question = process.argv[2] || "What is Desmend's dream job?";

  console.log("=".repeat(60));
  console.log("Model:    Claude Haiku");
  console.log(`Question: ${question}`);
  console.log("=".repeat(60));
  console.log("Invoking agent...");

  // ── INVOKE THE AGENT ────────────────────────────────────────
  // agent.invoke() starts the ReAct loop
  //
  // messages: array of message objects (same format as OpenAI/Anthropic)
  //   role: "user" = the human's message
  //   content: the actual question text
  //
  // We wrap the question with context:
  //   "Search the student database and answer:"
  // This nudges Claude to search before answering
  // (the messageModifier above also enforces this)
  const result = await ragAgent.invoke({
    messages: [{
      role:    "user",
      content: `Search the student database and answer: ${question}`,
    }],
  });

  // ── GET THE FINAL ANSWER ────────────────────────────────────
  // result.messages = array of ALL messages in the conversation:
  //   [0] user question
  //   [1] Claude's decision to call search_student_sheet
  //   [2] search_student_sheet's result (the chunks from MongoDB)
  //   [3] Claude's final answer ← this is what we want
  //
  // .at(-1) = get the LAST item in the array (the final answer)
  // .content = the text of that message
  const finalAnswer = result.messages.at(-1).content;

  console.log("\nAnswer:", finalAnswer);
  console.log("=".repeat(60));
}