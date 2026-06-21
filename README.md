# AI-Internship-Assignment
# 🤖 RAG Chatbot with PDF Knowledge Base

A Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **Pinecone**, **LangChain**, and **Llama models**. Upload PDF documents, generate embeddings, store them in a vector database, and interact with your documents through a conversational AI interface.

---

## 🚀 Features

* 📄 Upload and process multiple PDF documents
* 🔍 Extract and chunk PDF text intelligently
* 🧠 Generate embeddings using Hugging Face Sentence Transformers
* 📦 Store vectors in Pinecone for fast semantic retrieval
* 🤖 Query documents using open-source Llama models
* 📚 Display source documents used for responses
* 💬 Interactive Streamlit-based chat interface
* ⚡ Optional GPU acceleration for faster inference
* 📝 Session-based chat history

---

## 🏗️ Architecture

```text
PDF Documents
      │
      ▼
Text Extraction (PyPDF2)
      │
      ▼
Text Chunking
      │
      ▼
Embedding Generation
(HuggingFace MiniLM)
      │
      ▼
Pinecone Vector Database
      │
      ▼
Retriever (LangChain)
      │
      ▼
Llama Model (LlamaCpp)
      │
      ▼
Answer + Source Documents
```

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Vector Database

* Pinecone

### LLM Framework

* LangChain

### Embedding Model

* `sentence-transformers/all-MiniLM-L6-v2`

### LLMs Supported

* Llama-2-7B-Chat
* CodeLlama-7B-Instruct
* Vicuna-7B

### PDF Processing

* PyPDF2

### Model Hosting

* Hugging Face Hub

---

## 📋 Prerequisites

Before running the application, ensure you have:

* Python 3.9+
* Pinecone Account & API Key
* Hugging Face Account (optional for gated models)
* 8GB+ RAM recommended
* CUDA-compatible GPU (optional)

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/macOS

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📄 Example requirements.txt

```txt
streamlit
PyPDF2
pinecone-client
langchain
langchain-pinecone
sentence-transformers
torch
llama-cpp-python
huggingface-hub
```

---

## 🔑 Configuration

Launch the application and configure the following in the sidebar:

| Parameter            | Description                        |
| -------------------- | ---------------------------------- |
| Pinecone API Key     | Your Pinecone API Key              |
| Pinecone Environment | Pinecone deployment region         |
| Index Name           | Pinecone index name                |
| HuggingFace Token    | Optional token for model downloads |
| Llama Model          | Select preferred model             |

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 📚 Usage

### Step 1: Initialize System

1. Enter your Pinecone credentials.
2. Select a Llama model.
3. Click **Initialize System**.

### Step 2: Upload Documents

1. Upload one or more PDF files.
2. Click **Process PDFs**.
3. Documents are:

   * Extracted
   * Chunked
   * Embedded
   * Stored in Pinecone

### Step 3: Ask Questions

Enter a question such as:

```text
What are the key findings in the uploaded research paper?
```

The chatbot will:

* Retrieve relevant document chunks
* Generate a contextual response
* Display source references

---

## 🧠 Supported Models

| Model           | Purpose                             |
| --------------- | ----------------------------------- |
| Llama-2-7B-Chat | General-purpose conversations       |
| CodeLlama-7B    | Programming and technical documents |
| Vicuna-7B       | Instruction-following and Q&A       |

---

## 📂 Project Structure

```text
rag-chatbot/
│
├── app.py
├── models/
├── requirements.txt
├── README.md
│
└── data/
    └── uploaded_pdfs/
```

---

## ⚙️ How It Works

### PDF Processing

* Extracts text using PyPDF2
* Splits text into manageable chunks
* Preserves context using overlap

### Embedding Generation

Uses:

```python
sentence-transformers/all-MiniLM-L6-v2
```

Vector dimension:

```text
384
```

### Retrieval

* Semantic similarity search
* Top-k relevant chunks returned
* Source metadata preserved

### Response Generation

LangChain RetrievalQA pipeline:

```python
RetrievalQA.from_chain_type(...)
```

Combines retrieved context with Llama-generated answers.

---

## 🔒 Security Notes

* Never commit API keys.
* Use environment variables for production deployments.
* Restrict Pinecone index access appropriately.
* Validate uploaded files before processing.

---

## 🚀 Future Improvements

* Chat memory persistence
* Multi-user authentication
* Hybrid search (BM25 + Vector Search)
* OCR support for scanned PDFs
* Document management dashboard
* Streaming responses
* OpenAI / Mistral support
* Citation highlighting

---

## 🐛 Troubleshooting

### Pinecone Connection Issues

Verify:

```bash
PINECONE_API_KEY
```

and index configuration.

### Model Download Failures

Ensure:

```bash
HUGGINGFACE_HUB_TOKEN
```

is configured correctly.

### CUDA Not Detected

Check:

```python
torch.cuda.is_available()
```

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

* LangChain
* Pinecone
* Hugging Face
* Streamlit
* Llama.cpp
* Sentence Transformers

Built with ❤️ for document-based conversational AI.
