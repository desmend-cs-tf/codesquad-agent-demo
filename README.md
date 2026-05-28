# CodeSquad Agent Demo

A multi-agent system for fetching technology news and distributing curated digests across multiple communication platforms (Discord, Email, Slack). Built with LangChain and LangGraph, supporting both Claude and Gemini LLMs.

## Project Overview

This project demonstrates the ReAct (Reason, Act, Observe) pattern to build AI agents that autonomously orchestrate tools to accomplish complex tasks. Agents are given a goal (e.g., "fetch tech news and post to Discord"), reason about which tools to use, execute them, and adapt based on results — without explicit step-by-step instructions.

The system showcases how different LLM providers (Anthropic Claude, Google Gemini) can be swapped interchangeably while maintaining identical tool integrations, proving model-agnostic agent architecture.

---

## Dependencies & Why We Need Them

| Dependency | Version | Purpose |
|---|---|---|
| `@langchain/anthropic` | ^1.4.0 | Connects Claude LLM (ChatAnthropic) for intelligent reasoning and tool selection. Powers the Claude agent. |
| `@langchain/google-genai` | ^2.1.31 | Connects Google Gemini LLM for alternative agent inference. Allows swapping Claude ↔ Gemini without changing tool code. |
| `@langchain/core` | ^1.1.48 | Core LangChain abstractions: tool definitions (zod schemas), message structures, and LLM interfaces. Foundation for all agents. |
| `@langchain/langgraph` | ^1.3.2 | Graph-based agent orchestration. Implements ReAct loops and enables createReactAgent prebuilt workflow with automatic tool scheduling. |
| `langchain` | ^1.4.2 | Higher-level LangChain utilities and chains. Provides structured output parsing and agent utilities. |
| `dotenv` | ^17.4.2 | Loads environment variables from .env file (API keys for Guardian, Discord, Email, Slack). Keeps secrets out of source code. |
| `resend` | ^6.12.4 | Email delivery service. Allows agents to send formatted digests to instructors with reliable SMTP infrastructure. |

---

## Agents

### Claude Agent (`agent-claude.js`)

Uses Anthropic's Claude Haiku 4.5 as the reasoning engine. The agent:

1. Receives a high-level goal (fetch news, curate, post to Discord and email)
2. Independently decides which tools to invoke and in what order
3. Reasons about tool outputs and refines behavior based on results

**Available Tools:**
- `fetch_tech_news` — Queries The Guardian API for top tech articles
- `post_to_discord` — Posts curated digest to Discord webhook
- `send_email_digest` — Emails detailed digest to instructor (commented out by default)

**Key Feature:** Tool descriptions guide Claude's decision-making. For example, the Discord tool says "call AFTER posting to Discord if the goal mentions email," which teaches the agent the implicit ordering constraint.

### Gemini Agent (`agent-gemini.js`)

Uses Google's Gemini 2.5 Flash Lite as the reasoning engine. Configuration is nearly identical to Claude, proving interchangeability:

- Same tool set (Gemini ignores email tool in current setup)
- Same goal structure
- Different model parameters (maxOutputTokens instead of maxTokens — vendor-specific naming)
- Demonstrates that agent logic remains unchanged when swapping LLMs

### Tool Descriptions as Prompting

Both agents rely on natural language tool descriptions to understand when and how to use tools. This is why descriptions include phrases like:

- "Call this AFTER posting to Discord if the goal mentions email"
- "This high-priority tool MUST be executed first"

These are teaching the agent implicit workflow constraints without hard-coding orchestration logic.

---

## Tools Breakdown

### fetchTechNewsTool
Queries The Guardian API with search parameters (query, limit) and returns formatted articles with title, URL, publication date, and summary. Returns up to 20 articles.

### postToDiscordTool
Sends a message to a Discord webhook. Expects a pre-formatted string (the agent formats the digest, this tool just sends it). Used for broadcasting to the class channel.

### sendEmailDigestTool
Sends formatted email via Resend API. Input: subject line and full email body. Email can be longer and more detailed than Discord version since email has more space.

### postToSlackTool & getLessonTopicsTool (Slack Integration)
Alternative communication channels. Get lesson topics returns today's curriculum for context; post to Slack sends messages to a Slack webhook. Included for extending the agent to multiple platforms.

---

## Supervisor Agent Concept

A **Supervisor Agent** is a higher-level coordinator that:

1. Receives a complex goal (e.g., "daily tech digest workflow")
2. Decomposes it into sub-goals for specialized agents
3. Routes each sub-goal to the appropriate agent (e.g., "fetch news" → fetch agent, "post to platforms" → broadcast agent)
4. Aggregates results and handles failures

In this project, a supervisor could:

- Route "fetch" requests to Guardian tool
- Route "Discord" requests to Discord agent
- Route "email" requests to Email agent
- Handle prioritization (if Discord fails, still send email)
- Maintain context across multiple agents (e.g., "use the same 5 articles for both Discord and email")

**Why Supervisors Matter:** As your agent system grows (more tools, more APIs, more platforms), a single agent becomes bottlenecked. Supervisors enable parallel execution and graceful degradation — if the Discord agent fails, email can still succeed.

---

## Suggested Multi-Agent Projects for CodeSquad

### 1. Customer Support Routing System

**Concept:** A supervisor agent that triages customer tickets to specialized agents:

- **Classification Agent:** Reads incoming support ticket, categorizes (billing, technical, account)
- **Technical Agent:** Has tools for checking system logs, running diagnostics
- **Billing Agent:** Has tools for querying database, processing refunds
- **Response Agent:** Drafts reply based on diagnosis

**Skills Taught:** Multi-agent routing, specialized tool domains, error handling across agents

### 2. Research & Analysis Pipeline

**Concept:** Supervisor coordinates a research workflow:

- **Search Agent:** Finds academic papers, news, blog posts on a topic
- **Summarize Agent:** Condenses long documents into key points
- **Synthesis Agent:** Combines multiple summaries into cohesive report
- **Presentation Agent:** Formats final report for different audiences (PDF, HTML, email)

**Skills Taught:** Chain-of-thought reasoning, context passing between agents, structured output

### 3. Autonomous Deployment Pipeline

**Concept:** Multi-agent CI/CD that handles deployment safely:

- **Test Agent:** Runs unit and integration tests, decides if code is deployable
- **Security Agent:** Scans dependencies for vulnerabilities
- **Review Agent:** Checks code quality and coverage metrics
- **Deployment Agent:** Only runs if all prior agents approve
- **Notification Agent:** Alerts team on Slack/Discord

**Skills Taught:** Conditional tool execution, sequential gating, DevOps automation

---

## Classwork 1: Build the Resend Email Tool

**Objective:** Implement `sendEmailDigestTool` step-by-step and integrate it into an agent.

### Prerequisites
- Resend account (free tier: 100 emails/day)
- Resend API key in `.env` file: `RESEND_API_KEY=your_key`
- Target email in `.env`: `DIGEST_EMAIL=instructor@example.com`

### Steps

#### Step 1: Create Tool Schema
Define the tool input parameters using Zod:

```javascript
const schema = z.object({
  subject: z.string().describe("Email subject line"),
  body: z.string().describe("Email body content")
});
```

**Why:** Zod enforces type safety and teaches the LLM what inputs the tool expects.

#### Step 2: Implement the Tool Function
Wrap Resend API call in error handling:

```javascript
const emailFunction = async ({ subject, body }) => {
  try {
    const { data, error } = await resend.emails.send({
      from: "CodeSquad <onboarding@resend.dev>",
      to: [process.env.DIGEST_EMAIL],
      subject: subject,
      text: body
    });
    
    if (error) return `Email failed: ${error.message}`;
    return `Email sent. Message ID: ${data.id}`;
  } catch (err) {
    return `Email error: ${err.message}`;
  }
};
```

**Key Point:** Always return descriptive success/error messages so the agent knows what happened.

#### Step 3: Export the Tool
Use LangChain's `tool()` helper:

```javascript
export const sendEmailDigestTool = tool(emailFunction, {
  name: "send_email_digest",
  description: "Sends digest email to instructor...",
  schema: schema
});
```

#### Step 4: Add to Agent
Uncomment the import and add to tools array in `agent-claude.js`:

```javascript
import { sendEmailDigestTool } from "./tools/emailTool.js";

const tools = [fetchTechNewsTool, postToDiscordTool, sendEmailDigestTool];
```

#### Step 5: Test the Agent
Run with a goal that mentions email:

```bash
node agent-claude.js "Fetch tech articles, post to Discord, and send me a detailed email."
```

#### Step 6: Verify
- Check that agent called the email tool
- Confirm email arrived in your inbox
- Verify subject and body match expected format

#### Step 7: Extend (Bonus)
- Add HTML email support (use `html` field instead of `text`)
- Add attachments (CSV of article links)
- Format email with Markdown → HTML conversion

---

## Classwork 2: Build a RAG Agent with PDFs (50 Minutes)

**Branch:** `rag-lecture`

Retrieve-Augmented Generation (RAG) is a technique that allows AI agents to answer questions based on documents you provide. Instead of relying on training data, the agent retrieves relevant sections from your PDFs and uses them to generate accurate answers.

### Quick Start

1. Checkout the `rag-lecture` branch:
   ```bash
   git checkout rag-lecture
   ```

2. Create `docs/` folder and add PDFs:
   ```bash
   mkdir docs
   # Add your PDF files to this folder
   ```

3. Install dependencies:
   ```bash
   npm install pdfjs-dist
   ```

4. Set up .env:
   ```
   GOOGLE_API_KEY=your-key
   PINECONE_API_KEY=your-key
   PINECONE_INDEX=codesquad-pdfs
   PINECONE_ENVIRONMENT=us-east-1
   ```

5. Run classwork:
   ```bash
   node rag-classwork/starter-rag-agent.js
   ```

### What You'll Learn

- How to parse PDFs and extract text
- Chunking large documents for efficient processing
- Generating embeddings (semantic representations of text)
- Storing embeddings in a vector database
- Building a retrieval tool that finds relevant documents
- Integrating retrieval into an agent that cites sources

### Detailed Materials

See the `rag-classwork/` folder for:
- **RAG_CONCEPTS.md** — Complete theory and architecture
- **CLASSWORK_ASSIGNMENT.md** — Step-by-step 50-minute lab
- **starter-rag-agent.js** — Template code with TODO markers
- **solution/solution-rag-agent.js** — Reference implementation

---

## Running the Project

### Setup
1. Clone repo
2. `npm install`
3. Create `.env` with API keys:
   ```
   GUARDIAN_API_KEY=your_key
   DISCORD_WEBHOOK_URL=your_webhook
   RESEND_API_KEY=your_key
   DIGEST_EMAIL=email@example.com
   GOOGLE_API_KEY=your_key
   PINECONE_API_KEY=your_key
   PINECONE_INDEX=codesquad-articles
   PINECONE_ENVIRONMENT=us-east-1
   ```

### Run Claude Agent
```bash
node agent-claude.js
```

### Run Gemini Agent
```bash
node agent-gemini.js "Your custom goal here"
```

### Run with Custom Goal
```bash
node agent-claude.js "Fetch top 10 AI articles and post only to Discord"
```

### Run RAG Classwork
```bash
git checkout rag-lecture
node rag-classwork/starter-rag-agent.js
```

---

## Key Concepts Reinforced

- **ReAct Loop:** Agent reasons (should I call fetch?), acts (calls fetch), observes (reads results), repeats
- **Tool-Guided Reasoning:** LLM's tool descriptions teach it workflow constraints
- **Model Interchangeability:** Same tools work with Claude, Gemini, or any LLM
- **Graceful Degradation:** If one tool fails, agent adapts (e.g., no email → still post Discord)
- **Composability:** New tools = new agent capabilities without changing core logic
- **Retrieval-Augmented Generation:** Grounding LLM responses in actual documents reduces hallucination

---

## Additional Resources

- [LangChain Documentation](https://js.langchain.com)
- [LangGraph ReAct Agent](https://langchain-ai.github.io/langgraphjs)
- [Anthropic API Reference](https://docs.anthropic.com)
- [Google Gemini API](https://ai.google.dev)
- [The Guardian API](https://open-platform.theguardian.com)
- [Pinecone Vector Database](https://www.pinecone.io)

---
