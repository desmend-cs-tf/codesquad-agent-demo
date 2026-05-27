import "dotenv/config";
import { ChatAnthropic } from "@langchain/anthropic";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { getLessonTopicsTool, postToSlackTool } from "./tools/slackTool.js";
// import { sendEmailTool } from "./tools/emailTool.js";

// ── Model ──────────────────────────────────────────────────────
const model = new ChatAnthropic({
  model: "claude-haiku-4-5",
  maxTokens: 512,
  temperature: 0,
});

// ── Tools ──────────────────────────────────────────────────────
// const tools = [getLessonTopicsTool, postToSlackTool, sendEmailTool];
const tools = [getLessonTopicsTool, postToSlackTool];


// ── Agent ──────────────────────────────────────────────────────
// No more pull() or AgentExecutor needed — langgraph handles it
const agent = createReactAgent({
  llm: model,
  tools,
});

// ── Goal ───────────────────────────────────────────────────────
const input =
  process.argv[2] ||
  "Get today's lesson topics, write a short friendly recap, " +
  "and post it to the class Slack channel.";

console.log("Running agent with Claude Haiku...\n");
console.log("Goal:", input);
console.log("=".repeat(60));

const result = await agent.invoke({
  messages: [{ role: "user", content: input }],
});

// langgraph returns a messages array — last message is the final answer
const finalMessage = result.messages[result.messages.length - 1];

console.log("=".repeat(60));
console.log("\nFinal Answer:", finalMessage.content);