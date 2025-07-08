# 🧠 Notebooks: Intelligent Complaint Analysis for Financial Services

This directory documents the notebook-based workflow behind the Intelligent Complaint Analysis project. These notebooks detail the full pipeline: from cleaning raw complaints data to building a retrieval-augmented generation (RAG) system powered by chunking, embedding, indexing, prompting, and response evaluation.

---

## 📓 Available Notebooks

### 1. `EDA_and_Data_Preprocessing.ipynb`

🔍 **Purpose:**  
Explore and clean raw complaint data from the CFPB to prepare it for semantic indexing.

🧩 **Key Steps:**
- Load original data and inspect basic structure
- Analyze product-level complaint distributions
- Visualize word count distributions
- Filter narratives based on word count and key product proxies
- Normalize complaint text (e.g., punctuation, dates, unicode)
- Export cleaned dataset in `.csv` and `.parquet` formats

📊 **Key Insight:**  
Filtered dataset contains substantive complaints across five major financial products, cleaned and normalized for downstream language modeling.

---

### 2. `text-chunking_embedding_vector-store_indexing.ipynb`

🧠 **Purpose:**  
Transform cleaned complaints into a vector-searchable format using chunking, embeddings, and FAISS indexing.

⚙️ **Key Steps:**
- Initialize `ComplaintVectorPipeline`
- Perform recursive chunking using LangChain's `RecursiveCharacterTextSplitter`
- Embed chunks via `sentence-transformers` using the `all-MiniLM-L6-v2` model
- Normalize vectors and build a FAISS index (`IndexFlatIP`)
- Attach metadata (product, issue, ID) to each chunk
- Save artifacts: chunked DataFrame and FAISS index

📦 **Output:**  
Vector database ready for semantic querying of financial complaint excerpts.

---

### 3. `rag-core-logic-and-evaluation.ipynb`

🧪 **Purpose:**  
Execute and validate the full RAG loop — retrieval, prompt assembly, model response, and answer analysis.

🔁 **Key Steps:**
- Load cleaned data and vector index
- Configure the `RAGAgent`, composed of:
  - `ComplaintVectorPipeline.search()` for semantic chunk retrieval
  - `PromptBuilder` to inject context into structured instruction prompt
  - `LLMClient` or `LLMClient_For_Llama` to generate responses using Mistral
- Evaluate system on 9 representative user queries
- Record results in structured summary table for analysis

📈 **Key Outcome:**  
System generates coherent, bullet-pointed answers to financial queries based on complaint corpus context — validated through comparison and qualitative scoring.

---

## 🧑‍💻 Usage Notes

- Launch notebooks using Jupyter Lab or VS Code
- Ensure all required modules (in `requirements.txt`) are installed
- Generated visualizations and text outputs are inline and reproducible
- Notebook outputs feed directly into the `src/` modules and the Streamlit app

---

## 📌 Notes

- Quantized Mistral model used via `llama_cpp` for offline generation
- Evaluation demonstrates prompt engineering and model alignment with task goals
- Output summaries follow structured financial QA guidance

---

Feel free to extend the pipeline with new product categories, evaluation prompts, or model variants. These notebooks form the analytical backbone of the Intelligent Complaint Analysis system.