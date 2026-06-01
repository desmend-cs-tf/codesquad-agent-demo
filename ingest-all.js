// ingest-all.js  —  run: node ingest-all.js
import "dotenv/config";
import { createReadStream }  from "fs";
import { parse }             from "csv-parse";
import { ragAgent }          from "./agent-claude-rag.js";
// import { ragAgent } from "./agent-gemini.js";

const HEADERS = {
  studentId:   "What is your Student ID?",
  studentName: "Your Full Name",
  job:         "What job would you like after CodeSquad?",
  salary:      "What is your goal salary range?",
  industry:    "What industry would you like to go into and why?",
  future:      "Where do you see yourself in the next 3-5 years?",
};

const records = await new Promise((resolve, reject) => {
  const rows = [];
  createReadStream("./responses.csv")
    .pipe(parse({ columns: true, trim: true, skip_empty_lines: true }))
    .on("data", (row) => rows.push(row))
    .on("end",  () => resolve(rows))
    .on("error", reject);
});

console.log(`Found ${records.length} student(s) to ingest.\n`);
console.log("=".repeat(60));

for (const row of records) {
  const studentId   = row[HEADERS.studentId]?.trim();
  const studentName = row[HEADERS.studentName]?.trim();

  if (!studentId || !studentName) {
    console.log("⚠  Skipping row — missing studentId or name:", row);
    continue;
  }

  const rawText = [
    `Q: What is your name?\nA: ${studentName}`,
    `Q: What job would you like after CodeSquad?\nA: ${row[HEADERS.job] || "No answer"}`,
    `Q: What is your goal salary range?\nA: ${row[HEADERS.salary] || "No answer"}`,
    `Q: What industry would you like to go into and why?\nA: ${row[HEADERS.industry] || "No answer"}`,
    `Q: Where do you see yourself in the next 3-5 years?\nA: ${row[HEADERS.future] || "No answer"}`,
  ].join("\n\n");

  const fileName = `${studentId}_answers.txt`;

  console.log(`\nIngesting: ${studentName} (${studentId})...`);

  const result = await ragAgent.invoke({
    messages: [{
      role:    "user",
      content:
        `Please ingest this document. ` +
        `studentId: ${studentId} ` +
        `studentName: ${studentName} ` +
        `fileName: ${fileName} ` +
        `rawText: ${rawText}`,
    }],
  });

  console.log("✓", result.messages.at(-1).content);
}

console.log("\n" + "=".repeat(60));
console.log("All students ingested. Run node agent-claude.js to start asking questions.");