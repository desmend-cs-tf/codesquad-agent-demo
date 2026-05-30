import "dotenv/config";
import fs from "fs";
import path from "path";
import * as pdfjsLib from "pdfjs-dist";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone } from "@pinecone-database/pinecone";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatAnthropic } from "@langchain/anthropic";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

// Set PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = 
  `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;

// ── TODO 1a: Extract text from PDF ─────────────────────────────────────
async function extractTextFromPDF(filePath) {
  try {
    const fileBuffer = fs.readFileSync(filePath);
    const pdf = await pdfjsLib.getDocument({ data: fileBuffer }).promise;
    
    let fullText = "";
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      const pageText = textContent.items.map(item => item.str).join(" ");
      fullText += `[Page ${i}]\n${pageText}\n\n`;
    }
    return fullText;
  } catch (err) {
    console.error(`Error reading PDF ${filePath}:`, err);
    return "";
  }
}

// ── TODO 1b: Ingest all PDFs from docs/ folder ──────────────────────────
async function ingestPDFs() {
  try {
    const docsDir = "./docs";
    
    if (!fs.existsSync(docsDir)) {
      console.log("docs/ folder not found. Creating it...");
      fs.mkdirSync(docsDir);
      console.log("Please add PDF files to docs/ folder and run again.");
      return;
    }

    const pdfFiles = fs.readdirSync(docsDir).filter(f => f.endsWith(".pdf"));
    
    if (pdfFiles.length === 0) {
      console.log("No PDF files found in docs/ folder.");
      console.log("Please download PDFs and save them to docs/ folder");
      return;
    }

    console.log(`Found ${pdfFiles.length} PDFs. Processing...`);

    // Extract text from all PDFs
    const documents = [];
    for (const file of pdfFiles) {
      console.log(`  Extracting: ${file}`);
      const filePath = path.join(docsDir, file);
      const text = await extractTextFromPDF(filePath);
      if (text) {
        documents.push({
          pageContent: text,
          metadata: { source: file, type: "pdf" }
        });
      }
    }

    if (documents.length === 0) {
      console.log("No text extracted from PDFs.");
      return;
    }

    // Split into chunks
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    });

    const chunks = [];
    for (let i = 0; i < documents.length; i++) {
      const doc = documents[i];
      const docChunks = await splitter.splitText(doc.pageContent);
      chunks.push(...docChunks.map((chunk, idx) => ({
        pageContent: chunk,
        metadata: { source: doc.metadata.source, chunkIdx: idx }
      })));
    }

    console.log(`Created ${chunks.length} chunks`);

    // Generate embeddings
    const embeddings = new GoogleGenerativeAIEmbeddings({
      model: "embedding-001",
      apiKey: process.env.GOOGLE_API_KEY,
    });

    // Store in Pinecone
    const pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });
    const index = pinecone.Index(process.env.PINECONE_INDEX);

    console.log("Uploading to Pinecone...");
    await PineconeStore.fromDocuments(chunks, embeddings, {
      pineconeIndex: index,
    });

    console.log(`Success! Ingested ${chunks.length} chunks into Pinecone\n`);

  } catch (err) {
    console.error("Ingestion error:", err);
  }
}

// ── TODO 2: Implement retrievalTool ────────────────────────────────────
export const retrievalTool = tool(
  async ({ query }) => {
    try {
      const embeddings = new GoogleGenerativeAIEmbeddings({
        model: "embedding-001",
        apiKey: process.env.GOOGLE_API_KEY,
      });

      const pinecone = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
      });
      const index = pinecone.Index(process.env.PINECONE_INDEX);

      const vectorStore = new PineconeStore(embeddings, { pineconeIndex: index });
      const retriever = vectorStore.asRetriever({ k: 3 });

      const docs = await retriever.getRelevantDocuments(query);

      if (docs.length === 0) {
        return "No relevant sections found in documents.";
      }

      return docs
        .map((doc, idx) => 
          `Result ${idx + 1}:\n` +
          `Source: ${doc.metadata.source}\n` +
          `Content: ${doc.pageContent.substring(0, 300)}...\n`
        )
        .join("\n---\n");

    } catch (err) {
      return `Retrieval error: ${err.message}`;
    }
  },
  {
    name: "search_documents",
    description:
      "Search ingested PDF documents for relevant information. " +
      "Returns the top 3 most relevant sections with source filenames. " +
      "Use this to find supporting context before answering user questions.",
    schema: z.object({
      query: z.string().describe("The search query or question to find in the documents")
    })
  }
);

// ── TODO 3: Create agent with retrieval ────────────────────────────────
async function main() {
  console.log("=".repeat(60));
  console.log("RAG Classwork with PDFs");
  console.log("=".repeat(60));

  // Step 1: Ingest PDFs
  console.log("\nStep 1: Ingesting PDFs from docs/ folder...");
  await ingestPDFs();

  // Step 2: Test retrieval
  console.log("Step 2: Testing retrieval tool...");
  try {
    const testResult = await retrievalTool.invoke({
      query: "What is the main topic?"
    });
    console.log("Retrieval Test:");
    console.log(testResult);
  } catch (err) {
    console.log("Retrieval test (may fail if no docs ingested yet)");
  }

  // Step 3: Create and test agent
  console.log("\nStep 3: Creating agent with retrieval...");
  
  const model = new ChatAnthropic({
    model: "claude-haiku-4-5",
    maxTokens: 1024,
  });

  const agent = createReactAgent({
    llm: model,
    tools: [retrievalTool],
  });

  // Test agent
  const goal = "Based on the documents I uploaded, summarize the main points and key concepts.";
  
  console.log("\nAgent Goal:", goal);
  console.log("\n" + "=".repeat(60));
  
  try {
    const result = await agent.invoke({
      messages: [{ role: "user", content: goal }]
    });

    console.log("\nAgent Response:");
    console.log(result.messages.at(-1).content);
  } catch (err) {
    console.log("Agent invocation failed:", err.message);
  }

  console.log("\n" + "=".repeat(60));
}

main().catch(console.error);
