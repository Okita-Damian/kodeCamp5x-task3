# RAG Backend with ChromaDB (Docker)

This project is a **Retrieval-Augmented Generation (RAG) backend** built with **Node.js + Express**, **ChromaDB (Docker)**, **Hugging Face embeddings**, and **Google Gemini** for answers.

You can:

- Upload text files
- Chunk + embed them
- Store embeddings in ChromaDB
- Ask questions and get answers grounded in your documents

---

## Project Structure

```
kodeCamp5x-task3/
│── main.js
│── example.txt
├── Dockerfile
│── .env
│── package.json
│── node_modules/
│── data/            # temporary uploaded files
```

---

## Requirements

- Node.js (v18+ recommended)
- Docker Desktop
- Hugging Face account
- Google Gemini API key
- Windows CMD or PowerShell

---

## Environment Variables (.env)

Create a `.env` file **inside `kodeCamp5x-task3`**:

```
HF_API_KEY=your_huggingface_token
GEMINI_API_KEY=your_gemini_api_key
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL_NAME=gemini-2.5-flash
CHROMA_DB_HOST=localhost
```

> Hugging Face token permission: **READ** only

---

## Step 1: Start ChromaDB with Docker

Open **Terminal 1 (CMD or PowerShell)**:

````cmd
docker run -d --name chromadb -p 8000:8000 chromadb/chroma```

Verify ChromaDB:

```cmd
curl http://localhost:8000/api/v2/heartbeat
````

Expected:

```json
{ "nanosecond_heartbeat": 123456789 }
```

---

## Step 2: Install Node Dependencies

Open **Terminal 2**:

```cmd
npm install
```

---

## Step 3: Start the RAG Server

Still in **Terminal 2**:

```cmd
node main.js
```

Expected:

```
RAG server running on http://localhost:5000
```

---

## Step 4: Index a Document (CMD)

Ensure `example.txt` exists in the project folder.

Open **Terminal 3**:

```cmd
curl -X POST http://localhost:5000/upload -F "files=@example.txt"
```

Expected:

```json
{
  "ok": true,
  "msg": "Indexed 2 chunks for context: ctx-xxxx"
}
```

---

## Step 5: Query from Terminal (CMD)

```cmd
curl -X POST http://localhost:5000/chat -H "Content-Type: application/json" -d "{\"q\":\"What is this document about?\"}"
```

Expected response:

```json
{
  "answer": "The document talks about love and money",
  "retrieved": [...]
}
```

---

## Step 6: Test with Postman

### Upload File

- Method: `POST`
- URL: `http://localhost:5000/upload`
- Body → `form-data`

  - Key: `files`
  - Type: File
  - Value: `example.txt`

---

### Query Document

- Method: `POST`
- URL: `http://localhost:5000/chat`
- Headers:

  - `Content-Type: application/json`

- Body (raw → JSON):

```json
{
  "q": "What is this document about?"
}
```

---

## Health Check

```cmd
curl http://localhost:5000/health
```

Expected:

```
OK
```

---

## Common Errors

### 401 from Hugging Face

- Token missing or invalid
- Token must have **READ** permission

### curl -F not working

- Ensure you are in the folder containing the file
- Use `curl.exe` explicitly in PowerShell

---

## What i Have Built

- Dockerized vector database
- Semantic chunking
- Embeddings + retrieval
- LLM grounded answers
- Fully local RAG backend

---
