import os
import streamlit as st
import PyPDF2
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import torch
from huggingface_hub import hf_hub_download

class RAGChatbot:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.pinecone_client = None
        self.index = None
        
    def setup_environment(self):
        """Setup API keys and environment variables"""
        st.sidebar.title("Configuration")
        
        # Pinecone API Key
        pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password")
        if pinecone_api_key:
            os.environ["PINECONE_API_KEY"] = pinecone_api_key
        
        # Pinecone Environment
        pinecone_env = st.sidebar.text_input("Pinecone Environment", value="us-west1-gcp-free")
        
        # Index name
        index_name = st.sidebar.text_input("Pinecone Index Name", value="rag-chatbot")
        
        # HuggingFace Token (optional for some models)
        hf_token = st.sidebar.text_input("HuggingFace Token (optional)", type="password")
        if hf_token:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
            
        return pinecone_api_key, pinecone_env, index_name

    def initialize_pinecone(self, api_key: str, environment: str, index_name: str):
        """Initialize Pinecone connection and create index if needed"""
        try:
            # Initialize Pinecone
            self.pinecone_client = Pinecone(api_key=api_key)
            
            # Check if index exists, create if it doesn't
            existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
            
            if index_name not in existing_indexes:
                st.info(f"Creating new index: {index_name}")
                self.pinecone_client.create_index(
                    name=index_name,
                    dimension=384,  # For sentence-transformers/all-MiniLM-L6-v2
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
            
            self.index = self.pinecone_client.Index(index_name)
            return True
        except Exception as e:
            st.error(f"Error initializing Pinecone: {str(e)}")
            return False

    def setup_embeddings(self):
        """Initialize embedding model"""
        try:
            # Using a lightweight, efficient embedding model
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            return True
        except Exception as e:
            st.error(f"Error setting up embeddings: {str(e)}")
            return False

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for embedding"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def store_embeddings_in_pinecone(self, chunks: List[str], filename: str, index_name: str):
        """Store text chunks and their embeddings in Pinecone"""
        try:
            self.vectorstore = PineconeVectorStore.from_texts(
                texts=chunks,
                embedding=self.embeddings,
                index_name=index_name,
                pinecone_api_key=os.environ["PINECONE_API_KEY"],
                namespace="default",
                metadatas=[{"source": filename, "chunk_id": i} for i in range(len(chunks))]
            )
            return True
        except Exception as e:
            st.error(f"Error storing embeddings: {str(e)}")
            return False

    def setup_llama_model(self, model_choice: str):
        """Setup Llama model for generation"""
        try:
            # Model configurations
            model_configs = {
                "Llama-2-7B-Chat": {
                    "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
                    "filename": "llama-2-7b-chat.Q4_K_M.gguf"
                },
                "Code-Llama-7B": {
                    "repo_id": "TheBloke/CodeLlama-7B-Instruct-GGUF",
                    "filename": "codellama-7b-instruct.Q4_K_M.gguf"
                },
                "Vicuna-7B": {
                    "repo_id": "TheBloke/vicuna-7B-v1.5-GGUF",
                    "filename": "vicuna-7b-v1.5.Q4_K_M.gguf"
                }
            }
            
            if model_choice not in model_configs:
                st.error(f"Model {model_choice} not supported")
                return False
            
            config = model_configs[model_choice]
            
            # Download model if not exists
            model_path = hf_hub_download(
                repo_id=config["repo_id"],
                filename=config["filename"],
                cache_dir="./models"
            )
            
            # Callback manager for streaming
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            # Initialize LlamaCpp
            llm = LlamaCpp(
                model_path=model_path,
                temperature=0.1,
                max_tokens=2000,
                top_p=1,
                callback_manager=callback_manager,
                verbose=True,
                n_ctx=4096,  # Context window
                n_gpu_layers=35 if torch.cuda.is_available() else 0  # GPU acceleration
            )
            
            # Create QA chain
            if self.vectorstore:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=self.vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}
                    ),
                    return_source_documents=True
                )
                
            return True
            
        except Exception as e:
            st.error(f"Error setting up Llama model: {str(e)}")
            return False

    def query_chatbot(self, query: str) -> Dict[str, Any]:
        """Query the RAG chatbot"""
        if not self.qa_chain:
            return {"error": "Chatbot not properly initialized"}
        
        try:
            response = self.qa_chain({"query": query})
            return {
                "answer": response["result"],
                "source_documents": response["source_documents"]
            }
        except Exception as e:
            return {"error": f"Error processing query: {str(e)}"}

def main():
    st.set_page_config(
        page_title="RAG Chatbot with PDF Knowledge Base",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG Chatbot with PDF Knowledge Base")
    st.markdown("Upload PDF documents and query them using advanced RAG with Llama models!")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    
    chatbot = st.session_state.chatbot
    
    # Setup environment
    pinecone_api_key, pinecone_env, index_name = chatbot.setup_environment()
    
    # Model selection
    st.sidebar.subheader("Model Configuration")
    model_choice = st.sidebar.selectbox(
        "Select Llama Model",
        ["Llama-2-7B-Chat", "Code-Llama-7B", "Vicuna-7B"]
    )
    
    # Initialize button
    if st.sidebar.button("Initialize System"):
        if not pinecone_api_key:
            st.error("Please provide Pinecone API key")
        else:
            with st.spinner("Initializing system..."):
                # Initialize Pinecone
                if chatbot.initialize_pinecone(pinecone_api_key, pinecone_env, index_name):
                    st.success("✅ Pinecone initialized")
                    
                    # Setup embeddings
                    if chatbot.setup_embeddings():
                        st.success("✅ Embeddings model loaded")
                        
                        # Setup Llama model
                        if chatbot.setup_llama_model(model_choice):
                            st.success("✅ Llama model loaded")
                            st.session_state.system_initialized = True
                        else:
                            st.error("❌ Failed to load Llama model")
                    else:
                        st.error("❌ Failed to load embeddings model")
                else:
                    st.error("❌ Failed to initialize Pinecone")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Upload PDF Knowledge Base")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process PDFs"):
            if not hasattr(st.session_state, 'system_initialized'):
                st.error("Please initialize the system first")
            else:
                with st.spinner("Processing PDFs..."):
                    all_chunks = []
                    filenames = []
                    
                    for uploaded_file in uploaded_files:
                        # Extract text
                        text = chatbot.extract_text_from_pdf(uploaded_file)
                        if text:
                            # Chunk text
                            chunks = chatbot.chunk_text(text)
                            all_chunks.extend(chunks)
                            filenames.extend([uploaded_file.name] * len(chunks))
                            
                            st.success(f"✅ Processed {uploaded_file.name}: {len(chunks)} chunks")
                    
                    if all_chunks:
                        # Store in Pinecone
                        if chatbot.store_embeddings_in_pinecone(all_chunks, "combined_docs", index_name):
                            st.success(f"✅ Stored {len(all_chunks)} chunks in Pinecone")
                            st.session_state.docs_processed = True
                        else:
                            st.error("❌ Failed to store embeddings")
    
    with col2:
        st.subheader("💬 Chat with Your Documents")
        
        # Query input
        query = st.text_area("Ask a question about your documents:", height=100)
        
        if st.button("Get Answer") and query:
            if not hasattr(st.session_state, 'docs_processed'):
                st.error("Please process some PDFs first")
            else:
                with st.spinner("Generating answer..."):
                    response = chatbot.query_chatbot(query)
                    
                    if "error" in response:
                        st.error(response["error"])
                    else:
                        st.subheader("🤖 Answer:")
                        st.write(response["answer"])
                        
                        # Show sources
                        if response.get("source_documents"):
                            st.subheader("📚 Sources:")
                            for i, doc in enumerate(response["source_documents"]):
                                with st.expander(f"Source {i+1}"):
                                    st.write(doc.page_content)
                                    st.write(f"**Metadata:** {doc.metadata}")
        
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if st.session_state.chat_history:
            st.subheader("💬 Chat History")
            for i, (q, a) in enumerate(st.session_state.chat_history[-5:]):  # Show last 5
                with st.expander(f"Q{i+1}: {q[:50]}..."):
                    st.write(f"**Question:** {q}")
                    st.write(f"**Answer:** {a}")
    
    # System status
    st.sidebar.subheader("System Status")
    status_items = [
        ("Pinecone", hasattr(st.session_state, 'system_initialized')),
        ("Embeddings", hasattr(st.session_state, 'system_initialized')),
        ("Llama Model", hasattr(st.session_state, 'system_initialized')),
        ("Documents", hasattr(st.session_state, 'docs_processed'))
    ]
    
    for item, status in status_items:
        if status:
            st.sidebar.success(f"✅ {item}")
        else:
            st.sidebar.error(f"❌ {item}")
    
    # Instructions
    with st.expander("📋 Setup Instructions"):
        st.markdown("""
        ### Prerequisites:
        1. **Pinecone Account**: Sign up at [pinecone.io](https://pinecone.io) and get your API key
        2. **GPU (Optional)**: For faster inference, use a system with CUDA-compatible GPU
        
        ### Setup Steps:
        1. Enter your Pinecone API key and environment in the sidebar
        2. Click "Initialize System" to load models and connect to Pinecone
        3. Upload PDF files and click "Process PDFs"
        4. Start asking questions about your documents!
        
        ### Features:
        - **Multiple PDF Support**: Upload and process multiple PDFs
        - **Advanced Chunking**: Intelligent text splitting for better retrieval
        - **Vector Storage**: Embeddings stored in Pinecone for fast similarity search
        - **Open Source Models**: Uses Llama family models for generation
        - **Source Attribution**: Shows which document chunks were used for answers
        - **Chat History**: Keeps track of your conversations
        """)

if __name__ == "__main__":
    main()