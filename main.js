require("dotenv").config();
const express = require("express");
const multer = require("multer");
const fs = require("fs");
const { ChromaClient } = require("chromadb");
const crypto = require("crypto");
const { InferenceClient } = require("@huggingface/inference");
const { GoogleGenerativeAI } = require("@google/generative-ai");

const app = express();
app.use(express.json());

// --- Configuration ---
const HF_API_KEY = process.env.HF_API_KEY;
const EMBED_MODEL_NAME =
  process.env.EMBED_MODEL_NAME || "sentence-transformers/all-MiniLM-L6-v2";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const CHROMA_DB_HOST = process.env.CHROMA_DB_HOST || "localhost";
const LLM_MODEL_NAME = process.env.LLM_MODEL_NAME || "gemini-2.5-flash";
const RAG_DATA_DIR = process.env.RAG_DATA_DIR || "./data";
const CHUNK_THRESHOLD = parseFloat(process.env.CHUNK_LENGTH) || 0.75;
const PORT = 5000;

// Ensure data directory exists
if (!fs.existsSync(RAG_DATA_DIR)) fs.mkdirSync(RAG_DATA_DIR);

// --- Setup clients ---
const hf = new InferenceClient(HF_API_KEY);
const genAi = new GoogleGenerativeAI(GEMINI_API_KEY);
const llmModel = genAi.getGenerativeModel({ model: LLM_MODEL_NAME });

// ChromaClient defaults to http://localhost:8000
const client = new ChromaClient({
  host: CHROMA_DB_HOST,
  port: 8000,
  ssl: false,
});

const upload = multer({ dest: RAG_DATA_DIR });

// --- Cosine similarity ---
function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

// --- HuggingFace embeddings ---
async function getEmbeddings(texts) {
  const inputs = Array.isArray(texts) ? texts : [texts];
  try {
    const resp = await hf.featureExtraction({
      model: EMBED_MODEL_NAME,
      inputs,
    });
    return resp;
  } catch (err) {
    console.error("❌ Embedding Error:", err.message);
    throw err;
  }
}

// --- Semantic chunking ---
async function semanticChunking(text, threshold = CHUNK_THRESHOLD) {
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
  const cleanSentences = sentences.map((s) => s.trim()).filter((s) => s.length);
  if (!cleanSentences.length) return [];

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

// Health endpoint
app.get("/health", (req, res) => {
  res.status(200).send("OK");
});

// --- Ingestion endpoint ---
app.post("/upload", upload.array("files"), async (req, res) => {
  try {
    if (!req.files || !req.files.length)
      return res.status(400).send({ error: "No files" });

    const context =
      req.get("context") || `ctx-${crypto.randomUUID().slice(0, 8)}`;
    const collection = await client.getOrCreateCollection({
      name: "Damian_Ingested",
      embeddingFunction: null,
    });

    const allChunks = [];
    for (const file of req.files) {
      const text = fs.readFileSync(file.path, "utf-8");
      const chunks = await semanticChunking(text);

      for (let i = 0; i < chunks.length; i++) {
        allChunks.push({
          text: chunks[i],
          metadata: { source: file.originalname, part: i, context },
        });
      }
      fs.unlinkSync(file.path);
    }

    const embeddings = await getEmbeddings(allChunks.map((c) => c.text));

    await collection.upsert({
      ids: allChunks.map((_, i) => `${context}-${Date.now()}-${i}`),
      documents: allChunks.map((c) => c.text),
      embeddings: embeddings,
      metadatas: allChunks.map((c) => c.metadata),
    });

    res.send({
      ok: true,
      msg: `Indexed ${allChunks.length} chunks for context: ${context}`,
    });
  } catch (err) {
    console.error("❌ Error in /index:", err);
    res.status(500).send({ error: err.message });
  }
});

// --- Query endpoint ---
app.post("/chat", async (req, res) => {
  try {
    const { q, k = 5, c = null } = req.body;
    if (!q) return res.status(400).send({ error: "Missing query" });

    const [queryEmbedding] = await getEmbeddings([q]);
    const collection = await client.getOrCreateCollection({
      name: "Damian_Ingested",
      embeddingFunction: null,
    });

    const result = await collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: k,
      where: c ? { context: c } : undefined,
    });

    if (!result.documents[0].length)
      return res.send({
        answer: "No matching documents found.",
        retrieved: [],
      });

    const retrieved = result.documents[0].map((doc, i) => ({
      text: doc,
      metadata: result.metadatas[0][i],
    }));

    const contextText = retrieved
      .map((r) => `Source: ${r.metadata.source}\n${r.text}`)
      .join("\n\n---\n\n");

    const prompt = `Use the following context to answer the question.\n\n${contextText}\n\nQuestion: ${q}\nAnswer:`;
    const answerObj = await llmModel.generateContent(prompt);

    res.send({ answer: answerObj.response.text(), retrieved });
  } catch (err) {
    console.error("❌ Error in /query:", err);
    res.status(500).send({ error: err.message });
  }
});

app.listen(PORT, () =>
  console.log(`RAG server running on http://localhost:${PORT}`)
);
