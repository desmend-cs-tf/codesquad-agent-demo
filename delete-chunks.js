// delete-chunks.js
import "dotenv/config";
import { Chunk, connectDB } from "./models/Document.js";

await connectDB();

const result = await Chunk.deleteMany({});
console.log(`Deleted ${result.deletedCount} chunks`);

process.exit();