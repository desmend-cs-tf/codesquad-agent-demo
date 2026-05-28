import { tool } from "@langchain/core/tools";
import { z }    from "zod";

export const fetchTechNewsTool = tool(
  async ({ query, limit }) => {
    try {
      const params = new URLSearchParams({
        q:              query || "",
        section:        "technology",
        "page-size":    String(limit),
        "show-fields":  "trailText",
        "order-by":     "newest",
        "api-key":      process.env.GUARDIAN_API_KEY,
      });

      const res  = await fetch(`https://content.guardianapis.com/search?${params}`);
      const data = await res.json();
      console.table(`THIS IS THE DATA WE GOT BACK FIRST!!!${data}`);

      if (!data.response?.results?.length) {
        return `No articles found for query: "${query}"`;
      }

      return data.response.results
        .map(a =>
          `TITLE: ${a.webTitle}\n` +
          `URL: ${a.webUrl}\n` +
          `DATE: ${a.webPublicationDate.split("T")[0]}\n` +
          `SUMMARY: ${a.fields?.trailText || "No summary available."}`
        )
        .join("\n\n---\n\n");

    } catch (err) {
      return `Guardian API error: ${err.message}`;
    }
  },
  {
    name: "fetch_tech_news",
    description:
      "Fetches recent technology articles from The Guardian newspaper. " +
      "Call this FIRST before writing any digest or posting to Discord. " +
      "Returns article titles, URLs, dates, and summaries. " +
      "Input: a search query (e.g. 'artificial intelligence', 'javascript', 'startups') " +
      "and how many articles to return.",
    schema: z.object({
      query: z.string().describe(
        "Search term to filter articles. Examples: 'AI', 'javascript', 'startups', 'cybersecurity'. " +
        "Leave empty string for general tech news."
      ),
      limit: z.number().describe(
        "Number of articles to return. Must be between 5 and 20."
      ),
    }),
  }
);