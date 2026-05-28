import "dotenv/config";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { getLessonTopicsTool, postToSlackTool } from "./tools/slackTool.js";
// import { sendEmailTool } from "./tools/emailTool.js";

// ── ONLY THIS BLOCK CHANGES vs Claude ─────────────────────────
const model = new ChatGoogleGenerativeAI({
  model: "gemini-1.5-pro", 
  maxOutputTokens: 512,
  temperature: 0,
});

// ── Everything below is identical ─────────────────────────────
const tools = [getLessonTopicsTool, postToSlackTool];

const agent = createReactAgent({
  llm: model,
  tools,
});

const input =
  process.argv[2] ||
  "Get today's lesson topics, write a short friendly recap, " +
  "and post it to the class Slack channel.";

console.log("Running agent with Gemini Flash...\n");
console.log("Goal:", input);
console.log("=".repeat(60));

const result = await agent.invoke({
  messages: [{ role: "user", content: input }],
});

const finalMessage = result.messages[result.messages.length - 1];

console.log("=".repeat(60));
console.log("\nFinal Answer:", finalMessage.content);