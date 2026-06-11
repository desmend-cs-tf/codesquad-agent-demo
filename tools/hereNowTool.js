import { tool } from "@langchain/core/tools";
import { z } from "zod";

export const deployToHereNowTool = tool(
  async ({ htmlContent, slug }) => {
    try {
      if (!htmlContent || htmlContent.trim().length < 50) {
        return "Error: htmlContent is empty or too short.";
      }

      const encoder = new TextEncoder();
      const contentSize = encoder.encode(htmlContent).length;

      // ── Step 1: Create site ──────────────────────────────────
      console.log(`\n🚀 Step 1: Creating site on here.now...`);
      const publishRes = await fetch("https://here.now/api/v1/publish", {
        method: "POST",
        headers: {
          "Content-Type":    "application/json",
          "X-HereNow-Client": "codesquad-agent/1.0",
        },
        body: JSON.stringify({
          ...(slug ? { slug } : {}),
          files: [{
            path:        "index.html",
            size:        contentSize,
            contentType: "text/html; charset=utf-8",
          }],
        }),
      });

      if (!publishRes.ok) {
        const err = await publishRes.text();
        return `Step 1 failed: ${publishRes.status} — ${err}`;
      }

      const publishData = await publishRes.json();
      console.log(`📨 Step 1 response:`, JSON.stringify(publishData, null, 2));

      const uploadUrl  = publishData.upload?.uploads?.[0]?.url;
      const versionId  = publishData.upload?.versionId;
      const finalizeUrl = publishData.upload?.finalizeUrl;
      const siteUrl    = publishData.siteUrl;
      const claimUrl   = publishData.claimUrl;

      if (!uploadUrl)   return `Step 1 error: no upload URL. Got: ${JSON.stringify(publishData)}`;
      if (!finalizeUrl) return `Step 1 error: no finalizeUrl. Got: ${JSON.stringify(publishData)}`;
      if (!versionId)   return `Step 1 error: no versionId. Got: ${JSON.stringify(publishData)}`;

      // ── Step 2: Upload HTML ──────────────────────────────────
      console.log(`\n📤 Step 2: Uploading HTML (${contentSize} bytes)...`);
      const uploadRes = await fetch(uploadUrl, {
        method:  "PUT",
        headers: { "Content-Type": "text/html; charset=utf-8" },
        body:    htmlContent,
      });

      if (!uploadRes.ok) {
        const err = await uploadRes.text();
        return `Step 2 failed: ${uploadRes.status} — ${err}`;
      }
      console.log(`✅ Step 2 status: ${uploadRes.status}`);

      // ── Step 3: Finalize ─────────────────────────────────────
      console.log(`\n🏁 Step 3: Finalizing...`);
      const finalizeRes = await fetch(finalizeUrl, {
        method:  "POST",
        headers: {
          "Content-Type":    "application/json",
          "X-HereNow-Client": "codesquad-agent/1.0",
        },
        body: JSON.stringify({ versionId }),  // ← required per docs
      });

      if (!finalizeRes.ok) {
        const err = await finalizeRes.text();
        return `Step 3 failed: ${finalizeRes.status} — ${err}`;
      }

      const finalizeData = await finalizeRes.json();
      console.log(`📨 Step 3 response:`, JSON.stringify(finalizeData, null, 2));

      console.log(`\n🎉 Live at: ${siteUrl}`);

      // Surface claimUrl so the site can be kept permanently
      const claimNote = claimUrl
        ? `\n🔗 Claim URL (save this to keep it permanently): ${claimUrl}`
        : "";

      return `✅ Deployed! Live URL: ${siteUrl}${claimNote}`;
    } catch (error) {
      console.error(`❌ Full error:`, error);
      return `Deployment failed: ${error.name}: ${error.message}`;
    }
  },
  {
    name: "deployToHereNow",
    description:
      "Deploys an article digest as a live public webpage on here.now. " +
      "YOU must construct a complete HTML document from the articles and pass it as htmlContent. " +
      "Do not call this tool without first building the full HTML string yourself. " +
      "Returns a live public URL at <slug>.here.now.",
    schema: z.object({
      htmlContent: z.string().describe(
        "Complete HTML document you build from the fetched articles. Must include <!DOCTYPE html> and all article titles, summaries, and links."
      ),
      slug: z.string().optional().describe(
        "Optional URL slug e.g. 'cs-digest-june-10'. Site will live at slug.here.now"
      ),
    }),
  }
);