require("dotenv").config();
const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { InferenceClient } = require("@huggingface/inference");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const { ChromaClient } = require("chromadb");

const app = express();
app.use(express.json());

const HF_API_KEY = process.env.HF_API_KEY;
const EMBED_MODEL_NAME =
  process.env.EMBED_MODEL_NAME || "sentence-transformers/all-MiniLM-L6-v2";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const LLM_MODEL_NAME = process.env.LLM_MODEL_NAME || "gemini-2.5-flash";
// const CHROMA_DB_HOST = process.env.CHROMA_DB_HOST || "chroma_db";
const RAG_DATA_DIR = process.env.RAG_DATA_DIR || "uploads";
const CHUNK_LENGTH = parseInt(process.env.CHUNK_LENGTH) || 150;
const PORT = parseInt(process.env.PORT) || 3000;

const upload = multer({ dest: RAG_DATA_DIR });

//  Setup HuggingFace & Gemini clients
const hf = new InferenceClient(HF_API_KEY);
const genAi = new GoogleGenerativeAI(GEMINI_API_KEY);

// ChromaDB client
const client = new ChromaClient({
  host: "http://localhost:8000",
  port: 8000,
  ssl: false,
  embeddingFunction: null,
});

let collection;

function semanticChunk(text, targetWords = CHUNK_LENGTH) {
  const sentences = text.split(/(?<=[.!?])\s+/);
  const chunks = [];
  let cur = [];
  let curWords = 0;

  for (const s of sentences) {
    const w = s.split(/\s+/).length;
    if (curWords + w > targetWords && cur.length) {
      chunks.push(cur.join(" "));
      cur = [s];
      curWords = w;
    } else {
      cur.push(s);
      curWords += w;
    }
  }
  if (cur.length) chunks.push(cur.join(" "));
  return chunks;
}

// Generate embeddings
async function embed(texts) {
  const embedding = await hf.featureExtraction({
    model: EMBED_MODEL_NAME,
    inputs: texts,
  });
  return embedding;
}

app.post("/index", upload.array("files"), async (req, res) => {
  try {
    const allChunks = [];
    const context =
      req.get("context") ||
      `ctx-${crypto.randomUUID().replace(/-/g, "").slice(0, 8)}`;

    for (const file of req.files) {
      const txt = fs.readFileSync(file.path, "utf-8");
      const chunks = semanticChunk(txt, CHUNK_LENGTH);

      chunks.forEach((c, i) =>
        allChunks.push({
          text: c,
          metadata: { src: file.originalname, part: i, context },
        })
      );

      fs.unlinkSync(file.path);
    }

    // Embed all chunks
    const embeddings = await embed(allChunks.map((c) => c.text));
    allChunks.forEach((c, i) => (c.embedding = embeddings[i]));

    // Chroma collection
    collection = await client.getOrCreateCollection({ name: "my_docs" });

    await collection.add({
      ids: allChunks.map((c, i) => `${context}-${i}`),
      metadatas: allChunks.map((c) => c.metadata),
      embeddings: allChunks.map((c) => c.embedding),
      documents: allChunks.map((c) => c.text),
    });

    res.send({
      ok: true,
      msg: `Indexed ${allChunks.length} chunks under context ${context}`,
    });
  } catch (err) {
    res.status(500).send({ error: err.message });
  }
});

app.post("/query", async (req, res) => {
  try {
    const { q, k = 5, c = null } = req.body;
    if (!q) return res.status(400).send({ error: "missing q" });

    // Embed the query
    const [queryEmbedding] = await embed([q]);

    // Query ChromaDB
    collection = await client.getOrCreateCollection({ name: "my_docs" });
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
      .map((r) => `Source: ${r.metadata.src}#${r.metadata.part}\n${r.text}`)
      .join("\n\n---\n\n");

    const prompt = `Use the following context to answer the question.\n\n${contextText}\n\nQuestion: ${q}\nAnswer:`;

    const answer = await callGemini(prompt);
    res.send({ answer, retrieved });
  } catch (err) {
    res.status(500).send({ error: err.message });
  }
});

async function callGemini(prompt) {
  if (!GEMINI_API_KEY) throw new Error("GEMINI_API_KEY not set in .env");

  const model = genAi.getGenerativeModel({
    model: LLM_MODEL_NAME,
  });

  const result = await model.generateContent(prompt);
  return result.response.text();
}

app.listen(PORT, () =>
  console.log(`Javascript RAG server running on http://localhost:${PORT}`)
);
