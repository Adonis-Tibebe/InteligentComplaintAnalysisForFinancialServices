# Intelligent Complaint Analysis for Financial Services

A comprehensive NLP project for analyzing consumer complaint data from the Consumer Financial Protection Bureau (CFPB). This project provides a full framework for cleaning, analyzing, and transforming complaint narratives into a vector-searchable format for semantic exploration and retrieval.

---

## 📁 Project Structure

```
├── notebooks/                          # Analysis workflows
│   ├── EDA_and_Data_Preprocessing.ipynb       # Data cleaning & EDA
│   └── text-chunking_embedding_vector-indexing.ipynb  # NLP pipeline
├── src/                                # Core code
│   ├── core/ComplaintVectorPipeline.py # Main processing logic
│   └── utils/utils.py                  # Helper functions
├── data/                               # Dataset storage
│   ├── raw/                            # Original CFPB data
│   └── processed/                      # Cleaned datasets
└── requirements.txt                    # Python dependencies
```

---

## 🚀 Key Features

- Efficient data processing using **pandas**, **NumPy**, and **Matplotlib**
- Smart chunking and embedding of complaint narratives via **LangChain** and **sentence-transformers**
- Vector indexing and semantic search using **FAISS**
- Modular and reproducible workflows with Jupyter notebooks and Python scripts
- Scalable setup for Retrieval-Augmented Generation (RAG) applications

---

## 🔧 Getting Started

### Step 1: Clone the Repository

```bash
git clone https:https://github.com/Adonis-Tibebe/InteligentComplaintAnalysisForFinancialServices
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Launch the Notebooks

Navigate to the `notebooks/` directory and launch with:

```bash
jupyter notebook
# or
code notebooks/
```

---

## 📓 Notebooks Overview

### `EDA_and_Data_Preprocessing.ipynb`

- Perform exploratory data analysis (EDA)
- Clean and normalize the complaints
- Generate word count distributions and filter short entries

### `text-chunking_embeding_and_vector-store_indexing.ipynb`

- Chunk long complaints using a recursive splitting strategy
- Embed each chunk using `all-MiniLM-L6-v2`
- Normalize vectors and store them in a FAISS index
- Enable semantic search and contextual retrieval

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

Have questions or ideas? Open an issue or start a discussion — we’d love to collaborate!
