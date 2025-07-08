# 💡 Intelligent Complaint Analysis for Financial Services

A modular Retrieval-Augmented Generation (RAG) system for analyzing consumer complaints from the Consumer Financial Protection Bureau (CFPB). This project transforms textual complaint data into a vector-searchable format, enabling semantic exploration, evaluation, and structured generation. Built with low-resource environments in mind, it supports both notebook-based workflows and a Streamlit UI powered by a quantized Mistral model via `llama_cpp`.

---

## 📁 Project Structure
📁 project_root/
├── 📁 notebooks/                            # Jupyter notebooks for development & evaluation
│   ├── EDA_and_Data_Preprocessing.ipynb          # EDA, normalization, word filtering
│   ├── text-chunking_embedding_vector-indexing.ipynb  # Chunking, embeddings, FAISS indexing
│   └── rag-core-logic-and-evaluation.ipynb       # Retrieval, prompting, generation & analysis
│
├── 📁 src/                                  # Source code for RAG pipeline and app
│   ├── 📁 core/                                   # Core logic: preprocessing, embedding, retrieval
│   │   ├── ComplaintVectorPipeline.py                # Chunker, embedder, retriever, index builder
│   │   └── utils.py                                  # Text normalization, loaders, plots
│   │
│   ├── 📁 models/                                 # Model orchestration & RAG abstraction
│   │   └── RAG_Pipeline.py                           # PromptBuilder, LLMClient, RAGAgent classes
│   │
│   └── 📁 services/
│       └── 📁 streamlit/                            # Streamlit app frontend
│           └── rag_interface.py                        # Interface using quantized Mistral model
│
├── 📁 data/                                 # Data storage
│   ├── 📁 raw/                                    # Original CFPB complaint CSVs
│   └── 📁 processed/                              # Filtered & chunked data for retrieval
│
├── 📁 vector_store/                        # FAISS vector index
│   └── complaints_index.faiss                  # Prebuilt FAISS index
│
├── 📁 models/                             # LLMs and embeddings
│   └── 📁 mistral_condensed/                    # Quantized Mistral model (.gguf format)
│
├── 📁 tests/                              # Unit and integration tests
│   ├── test_ComplaintVectorPipeline.py         # Chunking, embeddings, retrieval tests
│   ├── test_RAGAgent.py                        # Prompting and generation logic
│   ├── test_utils.py                           # Normalizer and utility function tests
│   └── test_rag_interface_minimal.py           # Minimal app test (without full model load)
│
└── requirements.txt                      # Python dependencies for the project


---

## 🚀 Core Components

### ✅ Data Processing (`ComplaintVectorPipeline.py`)
- Recursive chunking (`langchain.text_splitter`)
- Embedding with `sentence-transformers`
- FAISS inner-product index for similarity search
- Metadata mapping for traceability

### 🎨 Prompting & Generation (`RAG_Pipeline.py`)
- `PromptBuilder`: Inserts context and user query into structured template
- `LLMClient`: Interfaces with Hugging Face pipelines
- `LLMClient_For_Llama`: Custom wrapper for quantized Mistral (`llama_cpp`)
- `RAGAgent`: Glues together search → prompt → generate

### 📊 Evaluation (`rag-core-logic-and-evaluation.ipynb`)
- Simulated RAG pipeline run over 9 curated financial questions
- Answers analyzed via tabular summaries
- Tests Mistral's contextual accuracy and prompt shaping

---

## 🌐 Streamlit Interface (`rag_interface.py`)

A streamlined interactive app using:

-  Quantized Mistral (7B) via `llama_cpp` for low-resource CPU inference
-  Cached embeddings and FAISS index
-  Real-time question input and answer rendering
-  Stateless prompt wrapping with `LLMClient_For_Llama`

### Setup

```bash
# for notebook implementaion create a .venv file with python 3.13.
python -m venv .venv
source .venv/Scripts/activate   # On bash: .\llama_venv\Scripts\Activate
# Create virtual environment with Python 3.10(for streamlit app)
python3.10 -m venv llama_venv
source llama_venv/Scripts/activate  # On bash: .\llama_venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run src/services/streamlit/rag_interface.py
