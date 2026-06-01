// models/Document.js
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// This file does two things:
//   1. Defines the SCHEMA — the blueprint for what a "chunk" looks like in MongoDB
//   2. Exports connectDB() — the function that connects to MongoDB
//
// A "chunk" = one Q&A pair from a student's intake form
// Example chunk:
//   "Q: What job would you like after CodeSquad? A: CTO of a dope startup"
//
// Every chunk gets saved to MongoDB with these fields:
//   studentId, studentName, fileName, content, chunkIndex, embedding, createdAt
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// dotenv MUST be first — loads .env before anything tries to use process.env
import "dotenv/config";

// Mongoose is the library that lets us talk to MongoDB with JavaScript
// It gives us schemas, models, and helper methods like .create() and .findOne()
import mongoose from "mongoose";

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SCHEMA
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// A schema is a blueprint — it tells MongoDB:
//   "Every document in this collection must have these fields"
//   "These fields must be these types"
//   "Some fields are required, some are optional"
//
// Think of it like defining a table structure in SQL
// except MongoDB is NoSQL, so we define it in JavaScript
const ChunkSchema = new mongoose.Schema({

  // ── studentId ────────────────────────────────────────────────
  // What:    Unique identifier for the student
  // Type:    String (text)
  // Required: yes — every chunk MUST know who it belongs to
  // Example: "student_099"
  // Used by: ingestTool (saves it), ragSearchTool (filters by it)
  studentId: { type: String, required: true },

  // ── studentName ──────────────────────────────────────────────
  // What:    The student's real name
  // Type:    String
  // Required: yes
  // Example: "Desmend Jetton"
  // Used by: ragSearchTool returns this so we see names not IDs
  studentName: { type: String, required: true },

  // ── fileName ─────────────────────────────────────────────────
  // What:    The source file this chunk came from
  // Type:    String
  // Required: yes
  // Example: "student_099_answers.txt"
  // Used by: audit trail — helps debug where data came from
  fileName: { type: String, required: true },

  // ── content ──────────────────────────────────────────────────
  // What:    The actual Q&A text — this is what we search and display
  // Type:    String
  // Required: yes
  // Example: "Q: What job would you like? A: CTO of a dope startup"
  // Used by: ragSearchTool returns this to Claude as the answer
  content: { type: String, required: true },

  // ── chunkIndex ───────────────────────────────────────────────
  // What:    Position of this chunk in the original file
  // Type:    Number
  // Default: 0 (auto-set if not provided)
  // Example: 0 = first Q&A, 1 = second Q&A, 2 = third Q&A...
  // Used by: nice to have — keeps track of original order
  chunkIndex: { type: Number, default: 0 },

  // ── embedding ← THE MAGIC FIELD ──────────────────────────────
  // What:    An array of 384 numbers that represent the MEANING of content
  // Type:    [Number] (array of numbers)
  // Default: undefined (we set it in ingestTool after running Xenova)
  //
  // How it works:
  //   Xenova reads "CTO of a dope startup" and converts it to
  //   [0.23, -0.81, 0.14, 0.05, ..., 384 numbers total]
  //   These numbers live in a 384-dimensional "vector space"
  //   Similar meanings = similar numbers = close together in that space
  //
  // Why we need it:
  //   MongoDB $vectorSearch uses this field to find chunks
  //   that are SEMANTICALLY similar to the search question
  //   "dream job" finds "CTO of a dope startup" even though
  //   those exact words don't appear in the question
  //
  // Used by: MongoDB $vectorSearch in ragSearchTool
  embedding: { type: [Number], default: undefined },

  // ── createdAt ────────────────────────────────────────────────
  // What:    Timestamp of when this chunk was saved
  // Type:    Date
  // Default: Date.now (automatically set to right now)
  // Example: "2026-05-30T14:23:45.123Z"
  // Used by: audit trail — helps debug timing issues
  createdAt: { type: Date, default: Date.now },
});

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CREATE THE MODEL FROM THE SCHEMA
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// mongoose.model(name, schema) does two things:
//   1. Creates a JavaScript class called "Chunk" we can use in code
//   2. Links it to a MongoDB collection called "chunks" (auto-pluralized)
//
// After this line:
//   await Chunk.create({...})   → inserts a document into MongoDB
//   await Chunk.findOne({...})  → finds one document in MongoDB
//   await Chunk.aggregate([...])→ runs a search pipeline in MongoDB
//
// NOTE: "Chunk" (capital C) is the Mongoose MODEL (JavaScript class)
//       "chunks" (lowercase) is the COLLECTION in MongoDB (the actual data)
export const Chunk = mongoose.model("Chunk", ChunkSchema);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// connectDB() — connect to MongoDB Atlas
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// We call this at the start of every tool before using the database
//
// Why check readyState?
//   readyState === 1 means "already connected"
//   If ingestTool and ragSearchTool both call connectDB(),
//   we don't want to connect twice — that causes errors
//   So we check first, and skip if already connected
//
// process.env.MONGODB_URI comes from .env file:
//   MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/rag_demo
export async function connectDB() {
  // If already connected (readyState 1 = connected), return early
  if (mongoose.connection.readyState === 1) return;

  // Otherwise connect using the URI from .env
  await mongoose.connect(process.env.MONGO_URI);

  // Confirm connection in the terminal
  console.log("WE ARE CONNECTED TO MONGO DB!!!");
}