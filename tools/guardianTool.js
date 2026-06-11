import { tool } from "@langchain/core/tools";
import { z } from "zod";

export const fetchTechNewsTool = tool(
  async ({ pageSize = 10 }) => {
    try {
      const apiKey = process.env.GUARDIAN_API_KEY;
      const url = `https://content.guardianapis.com/search?section=technology&page-size=${pageSize}&show-fields=trailText&api-key=${apiKey}`;

      const response = await fetch(url);
      const data = await response.json();

      const articles = data.response.results.map(article => ({
        title:   article.webTitle,
        url:     article.webUrl,
        section: article.sectionName,
        date:    article.webPublicationDate,
        summary: article.fields?.trailText || "No summary available.",
      }));

      return JSON.stringify(articles, null, 2);
    } catch (error) {
      return `Failed to fetch articles: ${error.message}`;
    }
  },
  {
    name: "fetchTechNews",
    description: "Fetches the latest technology articles from The Guardian API. Returns an array of articles with title, url, section, date, and summary.",
    schema: z.object({
      pageSize: z.number().optional().describe("Number of articles to fetch. Defaults to 10."),
    }),
  }
);