// diagnostic.js — run: node diagnostic.js
import "dotenv/config";
import { Chunk, connectDB } from "./models/Document.js";

await connectDB();

const chunks = await Chunk.find({ studentName: "Desmend Jetton" }).limit(10);
console.log(`Found ${chunks.length} chunks for Desmend:\n`);

chunks.forEach((chunk, i) => {
  console.log(`Chunk ${i}:`);
  console.log(`  Content: "${chunk.content.substring(0, 100)}..."`);
  console.log(`  Embedding: [${chunk.embedding.slice(0, 3).map(n => n.toFixed(3)).join(", ")}...]`);
  console.log();
});

process.exit();