require("dotenv").config();
const express = require("express");
const multer = require("multer");
const fs = require("fs");
const crypto = require("crypto");
const { InferenceClient } = require("@huggingface/inference");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const { ChromaClient } = require("chromadb");

const app = express();
app.use(express.json());

// --- Load environment variables ---
const HF_API_KEY = process.env.HF_API_KEY;
const EMBED_MODEL_NAME =
  process.env.EMBED_MODEL_NAME || "sentence-transformers/all-MiniLM-L6-v2";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const LLM_MODEL_NAME = process.env.LLM_MODEL_NAME || "gemini-2.5-flash";
const RAG_DATA_DIR = process.env.RAG_DATA_DIR || "./data";
const CHUNK_THRESHOLD = parseFloat(process.env.CHUNK_LENGTH) || 0.75;
const PORT = parseInt(process.env.PORT) || 8000;

// --- Setup clients ---
const hf = new InferenceClient(HF_API_KEY);
const genAi = new GoogleGenerativeAI(GEMINI_API_KEY);
const llmModel = genAi.getGenerativeModel({ model: LLM_MODEL_NAME });

const client = new ChromaClient({
  host: process.env.CHROMA_HOST,
  port: 443,
  ssl: true,
  tenant: process.env.CHROMA_TENANT,
  database: process.env.CHROMA_DATABASE,
  headers: {
    "x-chroma-token": process.env.CHROMA_API_KEY,
  },
});

// --- Multer setup ---
const upload = multer({ dest: RAG_DATA_DIR });
let collection;

// --- Cosine similarity ---
function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

// --- Semantic chunking ---
async function semanticChunking(text, threshold = CHUNK_THRESHOLD) {
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
  const cleanSentences = sentences.map((s) => s.trim()).filter((s) => s.length);
  if (!cleanSentences.length) return [];

  console.log(`> Embedding ${cleanSentences.length} sentences...`);
  const embeddings = await getEmbeddings(cleanSentences);
  const vecs = Array.isArray(embeddings[0]) ? embeddings : [embeddings];

  const chunks = [];
  let currentChunk = cleanSentences[0];
  let currentVec = vecs[0];

  for (let i = 1; i < cleanSentences.length; i++) {
    const sim = cosineSimilarity(currentVec, vecs[i]);
    if (sim >= threshold) {
      currentChunk += " " + cleanSentences[i];
    } else {
      chunks.push(currentChunk);
      currentChunk = cleanSentences[i];
      currentVec = vecs[i];
    }
  }
  if (currentChunk) chunks.push(currentChunk);
  return chunks;
}

// HuggingFace embeddings
async function getEmbeddings(texts) {
  const inputs = Array.isArray(texts) ? texts : [texts];
  const originalLog = console.log;
  console.log = () => {};

  const resp = await hf.featureExtraction({ model: EMBED_MODEL_NAME, inputs });

  console.log = originalLog;
  return resp;
}

// --- Ingestion endpoint ---
app.post("/index", upload.array("files"), async (req, res) => {
  try {
    if (!req.files || !req.files.length)
      return res.status(400).send({ error: "No files uploaded" });

    const allChunks = [];
    const context =
      req.get("context") || `ctx-${crypto.randomUUID().slice(0, 8)}`;

    collection = await client.getOrCreateCollection({
      name: "Damian_Ingested",
      embeddingFunction: null, // manual embeddings
    });

    for (const file of req.files) {
      const text = fs.readFileSync(file.path, "utf-8");
      const chunks = await semanticChunking(text);

      chunks.forEach((chunk, i) => {
        allChunks.push({
          text: chunk,
          metadata: { source: file.originalname, part: i, context },
        });
      });

      fs.unlinkSync(file.path);
    }

    const embeddings = await getEmbeddings(allChunks.map((c) => c.text));
    allChunks.forEach((c, i) => (c.embedding = embeddings[i]));

    await collection.upsert({
      ids: allChunks.map((_, i) => `${context}-${i}`),
      documents: allChunks.map((c) => c.text),
      embeddings: allChunks.map((c) => c.embedding),
      metadatas: allChunks.map((c) => c.metadata),
    });

    console.log(
      `✅ Indexed ${allChunks.length} chunks for context "${context}"`
    );
    res.send({ ok: true, msg: `Indexed ${allChunks.length} chunks.` });
  } catch (err) {
    console.error("❌ Error in /index:", err);
    res.status(500).send({ error: err.message });
  }
});

// --- Query endpoint ---
app.post("/query", async (req, res) => {
  try {
    const { q, k = 5, c = null } = req.body;
    if (!q) return res.status(400).send({ error: "Missing query" });

    const [queryEmbedding] = await getEmbeddings([q]);

    collection = await client.getOrCreateCollection({
      name: "Damian_Ingested",
      embeddingFunction: null,
    });

    const result = await collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: k,
      where: c ? { context: c } : undefined,
    });

    const retrieved = result.documents[0].map((doc, i) => ({
      text: doc,
      metadata: result.metadatas[0][i],
    }));

    const contextText = retrieved
      .map((r) => `Source: ${r.metadata.source}#${r.metadata.part}\n${r.text}`)
      .join("\n\n---\n\n");

    const prompt = `Use the following context to answer the question.\n\n${contextText}\n\nQuestion: ${q}\nAnswer:`;

    const answerObj = await llmModel.generateContent(prompt);
    const answer = answerObj.response.text();

    console.log(`✅ Query answered: "${q}"`);
    res.send({ answer, retrieved });
  } catch (err) {
    console.error("❌ Error in /query:", err);
    res.status(500).send({ error: err.message });
  }
});

app.listen(PORT, () =>
  console.log(`RAG server running on http://localhost:${PORT}`)
);
