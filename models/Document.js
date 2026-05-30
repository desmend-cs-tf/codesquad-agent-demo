import mongoose from "mongoose"

const ChunkSchema = new mongoose.Schema({
    studentId: { type: String, required: true},
    studentName: { type: String, required: true},
    fileName: {type: String, required: true },
    content: { type: String, required: true },
    chunkIndex: {type: Number, default: 0 },

    embedding: {type: [Number], default: undefined },

    createdAt: {type: Date, default: Date.now }

})

export const Chunk = mongoose.model("Chunk", ChunkSchema);

export async function connectDB() {
    if (mongoose.connection.readyState === 1) return;
    await mongoose.connect(process.env.MONGO_URI);
    console.log("WE ARE CONNECTED TO MONGO DB!!!");
}