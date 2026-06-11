import "dotenv/config";
import { ChatAnthropic }         from "@langchain/anthropic";
import { createReactAgent }      from "@langchain/langgraph/prebuilt";
import { fetchTechNewsTool }     from "./tools/guardianTool.js";
import { postToDiscordTool }     from "./tools/discordTool.js";
import { deployToHereNowTool }   from "./tools/hereNowTool.js";
import { sendEmailDigestTool }   from "./tools/emailTool.js";

const model = new ChatAnthropic({
  model:       "claude-haiku-4-5",
  maxTokens:   4096,
  temperature: 0,
});

const tools = [fetchTechNewsTool, postToDiscordTool, sendEmailDigestTool, deployToHereNowTool];

const agent = createReactAgent({
  llm:   model,
  tools: tools,
});

const goal = process.argv[2] ||
  "Fetch the top 10 technology articles from The Guardian, " +
  "pick the 5 most relevant for junior software developers, " +
  "post a short friendly digest with titles and URLs to the class Discord channel, " +
  "then send a more detailed version with full summaries to the instructor by email. " +
  "Finally, build a complete HTML page yourself from the articles and deploy it using the deployToHereNow tool, then return the live URL."

console.log("\nRunning agent with Claude Haiku...");
console.log("Goal:", goal);
console.log("=".repeat(60));

const result = await agent.invoke(
  { messages: [{ role: "user", content: goal }] },
  { recursionLimit: 25 }
);

console.log("=".repeat(60));
console.log("\nFinal Answer:", result.messages.at(-1).content);