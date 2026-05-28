import { tool } from "@langchain/core/tools";
import { z }    from "zod";

export const postToDiscordTool = tool(
  async ({ message }) => {
    try {
      const res = await fetch(process.env.DISCORD_WEBHOOK_URL, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ content: message }), // Discord uses "content" not "text"
      });

      if (res.ok) {
        return "Posted to Discord successfully.";
      } else {
        return `Discord post failed. Status: ${res.status} ${res.statusText}`;
      }
    } catch (err) {
      return `Discord error: ${err.message}`;
    }
  },
  {
    name: "post_to_discord",
    description:
      "Posts a message to the CodeSquad class Discord channel. " +
      "Call this AFTER you have fetched the news and written the digest. " +
      "Format the message with bullet points and emoji so it is easy to read. " +
      "Keep the message under 300 words. " +
      "Input: the fully formatted digest message as a plain string.",
    schema: z.object({
      message: z.string().describe(
        "The formatted tech digest to post. Should include article titles, " +
        "URLs, and a short summary of each story."
      ),
    }),
  }
);