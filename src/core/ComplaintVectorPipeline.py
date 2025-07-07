import pandas as pd
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss

class ComplaintVectorPipeline:
    def __init__(self, chunk_size=800, chunk_overlap=100, embedding_model="all-MiniLM-L6-v2"):
        # Initialize chunking strategy and embedding model
        """
        Initialize a ComplaintVectorPipeline object.

        :param chunk_size: The maximum size of a chunk in characters (default: 800)
        :param chunk_overlap: The minimum size of chunk overlap in characters (default: 100)
        :param embedding_model: The Sentence-BERT model to use for embedding (default: 'all-MiniLM-L6-v2')
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]  # Prioritized fallback when chunking
        )
        self.model = SentenceTransformer(embedding_model)  # Sentence-BERT embedding model
        self.index = None  # FAISS index placeholder
        self.chunk_df = None  # Will hold all chunk records + metadata
        self.filtered_df = None  # Original complaints that pass filtering

    def preprocess_and_chunk(self, df: pd.DataFrame, text_column="Normalized Consumer complaint narrative", min_words=20, metadata_fields: List[str] = []):
        # Filter out very short complaints (optional, prevents empty chunks)
        df = df[df["word_count"] >= min_words].copy()
        df = df.reset_index(drop=True)
        self.filtered_df = df  # Save for later retrieval of full complaints

        def chunk_row(text):
            return self.splitter.split_text(text) if isinstance(text, str) else []

        df["chunks"] = df[text_column].apply(chunk_row)  # Split each complaint into chunks

        records = []
        for idx, row in df.iterrows():
            for i, chunk in enumerate(row["chunks"]):  # Generate one row per chunk
                record = {
                    "source_row": idx,  # Reference to original complaint in filtered_df
                    "chunk_id": i,  # Order of chunk within the original complaint
                    "chunk_text": chunk,
                    "word_count": len(chunk.split()),
                    "Complaint ID": row["Complaint ID"],
                    "Product": row["Target_Product"],
                    "Issue": row["Issue"],
                    "Sub-issue": row["Sub-issue"]
                }
                # Dynamically attach any additional metadata specified by the user(generic pipeline for other data formats)
                for field in metadata_fields:
                    record[field] = row.get(field, None)
                records.append(record)

        self.chunk_df = pd.DataFrame(records)
        return self.chunk_df

    def embed_chunks(self, convert_to_numpy=True, normalize=True):
        # Embed each chunk into a 384-dimension vector
        texts = self.chunk_df["chunk_text"].tolist()
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=True
        )
        if normalize:
            # Normalize each vector to unit length — required(more optimized) for cosine-style similarity via dot product
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        self.chunk_df["embedding"] = embeddings.tolist()
        return embeddings

    def build_faiss_index(self):
        # Convert embeddings to float32 format expected by FAISS!!!
        embeddings = np.array(self.chunk_df["embedding"].tolist()).astype("float32")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Use inner product for cosine-style similarity
        index.add(embeddings)
        self.index = index
        return index

    def search(self, query: Union[str, List[str]], top_k=5, return_full_text=False, text_column="Normalized Consumer complaint narrative"):
        # Encode the user query and normalize for inner product search
        if isinstance(query, str):
            query = [query]
        query_vecs = self.model.encode(query, convert_to_numpy=True)
        query_vecs = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)

        # Perform FAISS search over top_k most similar chunks
        distances, indices = self.index.search(query_vecs, top_k)

        results = []
        for i, idx_list in enumerate(indices):
            result = self.chunk_df.iloc[idx_list].copy()
            result["similarity"] = distances[i]

            # Optionally add full complaint text from filtered_df
            if return_full_text and self.filtered_df is not None:
                source_rows = result["source_row"].values
                full_texts = self.filtered_df.iloc[source_rows]
                result = result.join(full_texts[text_column].reset_index(drop=True), rsuffix="_full")

            results.append(result)

        return results[0] if len(results) == 1 else results

    def save_index(self, path: str):
        if self.index:
            faiss.write_index(self.index, path)  # Serialize FAISS index to disk

    def load_index(self, path: str):
        self.index = faiss.read_index(path)  # Load FAISS index from disk

    def save_chunk_df(self, path: str):
        self.chunk_df.to_csv(path, index=False)  # Save chunked metadata as CSV

    def load_chunk_df(self, path: str):
        self.chunk_df = pd.read_csv(path)  # Load previously saved chunk_df