# Medical Chatbot (RAG with LangGraph + Pinecone)

A conversational AI chatbot that answers medical questions using **Retrieval-Augmented Generation (RAG)**. Instead of relying only on an LLM's built-in knowledge, it retrieves relevant passages from your own PDF documents and uses them as context to generate answers. It also remembers earlier messages in the conversation.

## ⚠️ AI Note

This project uses AI language models to generate responses. Please keep the following in mind:

- **Not medical advice.** Responses are for educational and informational purposes only. They are not a substitute for diagnosis, treatment, or advice from a qualified healthcare professional.
- **AI can be wrong.** Language models may produce incomplete, outdated, or incorrect information, even when grounded in source documents. Always verify important information with a trusted medical source.
- **Answers depend on your data.** Response quality is limited by the PDFs in the `data/` folder and by how well the retrieval step finds relevant passages.
- **Emergencies.** Do not use this chatbot in a medical emergency. Contact your local emergency services or a healthcare provider immediately.
- **Privacy.** Do not enter personal health information or sensitive data. Queries are sent to third-party services (the LLM provider and Pinecone) and are subject to their policies.

## How It Works

1. **Index (one-time):** `vectorstore.py` loads PDFs from `data/`, cleans and splits them into chunks, converts them to embeddings (Hugging Face), and stores them in a Pinecone index named `medical-chatbot`.
2. **Chat:** `main.py` starts a command-line loop. Each question is passed to a LangGraph workflow that retrieves relevant chunks and generates an answer with an LLM.
3. **Memory:** Conversation state is saved per `thread_id` using LangGraph's `MemorySaver`, so follow-up questions work naturally.

## Tech Stack

- **Orchestration:** LangChain, LangGraph
- **LLM providers:** Groq, OpenAI (via `langchain-groq`, `langchain-openai`)
- **Embeddings:** Hugging Face (`langchain-huggingface`)
- **Vector database:** Pinecone (serverless, cosine similarity, 384 dimensions)
- **Document parsing:** pypdf

## Project Structure

```
chatbot/
├── data/              # Source PDF documents
├── notebook/          # Experiments and prototyping
├── src/               # Core logic (graph, helpers)
├── main.py            # CLI chat application
├── vectorstore.py     # One-time script to build the Pinecone index
├── requirements.txt
├── setup.py
└── LICENSE
```

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key        # or OPENAI_API_KEY, depending on your setup
```

### 3. Add your documents

Place your PDF files in the `data/` folder.

### 4. Build the vector index (run once)

```bash
python vectorstore.py
```

### 5. Start the chatbot

```bash
python main.py
```

Type your question at the `You:` prompt. Enter `exit`, `quit`, or `end` to stop.

