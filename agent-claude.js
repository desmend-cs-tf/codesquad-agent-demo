import "dotenv/config";
import { ChatAnthropic }         from "@langchain/anthropic";
import { createReactAgent }      from "@langchain/langgraph/prebuilt";
import { fetchTechNewsTool }     from "./tools/guardianTool.js";
import { postToDiscordTool }     from "./tools/discordTool.js";
// import { sendEmailDigestTool }   from "./tools/emailTool.js";

// ── Model ──────────────────────────────────────────────────────
const model = new ChatAnthropic({
  model:       "claude-haiku-4-5",
  maxTokens:   1024,
  temperature: 0,
});

// ── Tools ──────────────────────────────────────────────────────
// Three tools now. Agent decides which ones to call based on the goal.
// Give it a goal that only mentions Discord — it skips email.
// Give it a goal that mentions both — it calls both.
const tools = [fetchTechNewsTool, postToDiscordTool];
// const tools = [fetchTechNewsTool, postToDiscordTool, sendEmailDigestTool];


// ── Agent ──────────────────────────────────────────────────────
const agent = createReactAgent({
  llm:   model,
  tools: tools,
});

// ── Goal ───────────────────────────────────────────────────────
const goal = process.argv[2] ||
  "Fetch the top 10 technology articles from The Guardian, " +
  "pick the 5 most relevant for junior software developers, " +
  "post a short friendly digest with titles and URLs to the class Discord channel, " +
  "then send a more detailed version with full summaries to the instructor by email.";

console.log("\nRunning agent with Claude Haiku...");
console.log("Goal:", goal);
console.log("=".repeat(60));

const result = await agent.invoke({
  messages: [{ role: "user", content: goal }],
});

console.log("=".repeat(60));
console.log("\nFinal Answer:", result.messages.at(-1).content);