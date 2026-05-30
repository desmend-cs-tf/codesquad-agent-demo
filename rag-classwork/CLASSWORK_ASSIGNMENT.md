# Classwork: RAG with PDFs (50 Minutes)

## Objective

Build a Retrieval-Augmented Generation chain that:
1. Downloads or ingests PDF files from the internet or local disk
2. Extracts and chunks PDF content
3. Embeds chunks and stores in vector database
4. Retrieves relevant sections based on user queries
5. Integrates retrieval into an agent that answers questions with citations

**Outcome:** An agent that answers questions about CodeSquad articles, news PDFs, or any documents you upload with accurate citations.

---

## Step 0: Download PDFs to Use

### Option A: Use Existing PDFs (Easiest)

Create a `docs/` folder and add PDFs:

```bash
mkdir -p docs
# Then add your PDF files to this folder
```

### Option B: Download PDFs from the Internet

Download a few articles or papers. Here are some easy sources:

**Tech News PDFs:**
- Medium articles (print to PDF via browser)
- Dev.to posts (many available as PDFs)
- GitHub READMEs (convert to PDF)

**Academic Papers (Direct Links):**
- Attention is All You Need: https://arxiv.org/pdf/1706.03762.pdf
- RAG Paper: https://arxiv.org/pdf/2005.11401.pdf

**How to Download:**
```bash
# Using curl to download from URL
curl -o docs/article.pdf https://arxiv.org/pdf/1706.03762.pdf

# Or use browser: Right-click → Save as PDF
```

### Option C: Create Test PDFs

Save any text as PDF:
1. Write article/notes in Google Docs or Word
2. File → Download → PDF
3. Save to `docs/` folder

---

## Prerequisites (Setup: 5 minutes)

1. Clone or checkout the `rag-lecture` branch
2. Create `docs/` folder with PDFs
3. Install dependencies:
   ```bash
   npm install pdfjs-dist
   ```
4. Set up .env with:
   ```
   GOOGLE_API_KEY=your-google-key
   PINECONE_API_KEY=your-key
   PINECONE_INDEX=codesquad-pdfs
   PINECONE_ENVIRONMENT=us-east-1
   ```

Get free API keys:
- Google: https://ai.google.dev/tutorials/setup
- Pinecone: https://www.pinecone.io (free tier)

---

## Timeline

| Phase | Duration | Task |
|---|---|---|
| Setup | 5 min | Env vars, dependencies, download PDFs |
| Part 1 | 15 min | Implement document ingestion |
| Part 2 | 15 min | Build retrieval tool |
| Part 3 | 10 min | Integrate into agent + test |
| Buffer | 5 min | Debugging |

---

## Part 1: Document Ingestion (15 minutes)

### Goal
Load PDFs from `docs/` folder, extract text, chunk, embed, and store in Pinecone.

### How PDFs Are Processed

```
PDF File
  ↓
PDF Parser (pdfjs-dist)
  ↓
Extract Raw Text
  ↓
Text Splitter (500 char chunks, 50 char overlap)
  ↓
Generate Embeddings (Gemini)
  ↓
Store in Pinecone with metadata (filename, page number)
```

### Starter Code

Open `starter-rag-agent.js` and complete the `ingestPDFs()` function:

```javascript
import fs from "fs";
import path from "path";
import * as pdfjsLib from "pdfjs-dist";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone } from "@pinecone-database/pinecone";

// Set PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = 
  `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;

// TODO 1a: Extract text from a single PDF file
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

// TODO 1b: Ingest all PDFs from docs/ folder
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
      return;
    }

    console.log(`Found ${pdfFiles.length} PDFs. Processing...`);

    // 1. Extract text from all PDFs
    const documents = [];
    for (const file of pdfFiles) {
      console.log(`  Extracting: ${file}`);
      const filePath = path.join(docsDir, file);
      const text = await extractTextFromPDF(filePath);
      documents.push({
        pageContent: text,
        metadata: { source: file, type: "pdf" }
      });
    }

    // 2. Split documents into chunks
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

    // 3. Generate embeddings
    const embeddings = new GoogleGenerativeAIEmbeddings({
      model: "embedding-001",
      apiKey: process.env.GOOGLE_API_KEY,
    });

    // 4. Store in Pinecone
    const pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });
    const index = pinecone.Index(process.env.PINECONE_INDEX);

    console.log("Uploading to Pinecone...");
    await PineconeStore.fromDocuments(chunks, embeddings, {
      pineconeIndex: index,
    });

    console.log(`Success! Ingested ${chunks.length} chunks into Pinecone`);

  } catch (err) {
    console.error("Ingestion error:", err);
  }
}

export { ingestPDFs, extractTextFromPDF };
```

### What to Do

1. Create `docs/` folder
2. Download 2-3 PDF articles (tech articles, CodeSquad materials, news)
3. Place PDFs in `docs/` folder
4. Run `await ingestPDFs()` in main()
5. Check console: should show "Ingested X chunks into Pinecone"

### Verification Checklist
- [ ] `docs/` folder exists with PDF files
- [ ] No errors when running ingestPDFs()
- [ ] Console shows "Ingested X chunks" message
- [ ] Pinecone dashboard shows vectors in index

---

## Part 2: Build Retrieval Tool (15 minutes)

### Goal
Create a tool that searches PDFs and returns relevant sections with source.

### Starter Code

```javascript
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// TODO 2: Implement retrievalTool
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

      // Initialize retriever
      const vectorStore = new PineconeStore(embeddings, { pineconeIndex: index });
      const retriever = vectorStore.asRetriever({ k: 3 });

      // Search PDFs
      const docs = await retriever.getRelevantDocuments(query);

      if (docs.length === 0) {
        return "No relevant documents found for your query.";
      }

      // Format results with source
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
      "Returns the top 3 most relevant sections with their source files. " +
      "Use this to find supporting context before answering user questions.",
    schema: z.object({
      query: z.string().describe("The search query or question about the documents")
    })
  }
);
```

### Test Retrieval

```javascript
// Test with sample query
const result = await retrievalTool.invoke({
  query: "What is machine learning?"
});
console.log("Retrieval Result:");
console.log(result);
```

**Expected Output:**
```
Result 1:
Source: article.pdf
Content: Machine learning is a subset of artificial intelligence...

Result 2:
Source: paper.pdf
Content: ML algorithms learn patterns from data...
```

### Verification Checklist
- [ ] Tool returns relevant document sections
- [ ] Source filename is shown
- [ ] Content snippets are readable
- [ ] Multiple sources appear (if relevant)

---

## Part 3: Integrate into Agent (10 minutes)

### Goal
Add retrieval tool to agent so it uses PDFs to answer questions.

### Starter Code

```javascript
import { ChatAnthropic } from "@langchain/anthropic";
// OR for Gemini:
// import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

async function main() {
  // Step 1: Ingest PDFs
  console.log("Step 1: Ingesting PDFs...");
  await ingestPDFs();

  // Step 2: Create agent with retrieval
  console.log("\nStep 2: Creating agent...");
  
  const model = new ChatAnthropic({
    model: "claude-haiku-4-5",
    maxTokens: 1024,
  });

  const agent = createReactAgent({
    llm: model,
    tools: [retrievalTool],
  });

  // Step 3: Test with questions
  console.log("\nStep 3: Testing agent with PDFs...");

  const goal = "Based on the documents, what is the main topic discussed?";

  const result = await agent.invoke({
    messages: [{ role: "user", content: goal }]
  });

  console.log("\nAgent Response:");
  console.log(result.messages.at(-1).content);
}

main().catch(console.error);
```

### Test Queries

Try these with your PDFs:

1. **Basic retrieval:** "What are the key points in the documents?"
2. **Specific question:** "Summarize the main argument"
3. **Comparison:** "What are the differences between concepts A and B?"

### What to Verify
- [ ] Agent calls `search_documents` tool
- [ ] Retrieved sections appear in agent reasoning
- [ ] Agent cites source filenames
- [ ] Answer is accurate and grounded in your PDFs

---

## Submission

### What to Submit
- Completed `starter-rag-agent.js`
- Console output showing successful ingestion, retrieval, and agent response
- List of PDF filenames you used

### Success Criteria
- Part 1: PDFs ingested, chunks created, embeddings stored (30 pts)
- Part 2: Retrieval tool returns relevant sections with sources (30 pts)
- Part 3: Agent uses retrieval to answer questions with citations (40 pts)

### Bonus Challenges
1. **Add filtering:** Only retrieve from specific PDF files
2. **Add ranking:** Re-rank results by relevance before showing agent
3. **Add summarization:** Have agent summarize all retrieved sections
4. **Add multi-agent:** One agent finds documents, another synthesizes answer

---

## Troubleshooting

| Issue | Solution |
|---|---|
| "docs/ folder not found" | Create folder: `mkdir -p docs` |
| "No PDF files found" | Add PDFs to docs/ folder, run again |
| "PDF parsing errors" | Ensure PDFs are valid (not corrupted). Try different PDFs. |
| "Pinecone connection failed" | Check API key in .env file |
| "Embeddings timeout" | Google API might be rate limited. Wait 30s and retry. |
| "Retrieved sections are irrelevant" | PDFs might not match query. Try different queries or PDFs. |

---

## Key Takeaways

- PDFs can be parsed into text using `pdfjs-dist`
- Chunking large documents keeps ingestion costs low
- Metadata (source filename) enables citation tracking
- RAG with real PDFs is more practical than demo data
- Students can use their own articles for personalized learning
