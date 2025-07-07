import pandas as pd
import matplotlib.pyplot as plt
import re 
import unicodedata

def load_data(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format.")
    

def plot_complaint_distribution(complaint_distribution_data):
    # plot the complaint count as a horizontal bar chart
    plt.figure(figsize=(10, 6), dpi=200)
    complaint_distribution_data.plot(kind='barh', color='skyblue')
    plt.xlabel("number of complaints")
    plt.ylabel("Product")
    plt.title("Complaint Distribution per Product")
    plt.gca().invert_yaxis() # so that product with the highest complaint appear at the top

    plt.xscale("log") # so that products with low complaints could be better visalized
    plt.tight_layout()
    plt.show()


def plot_word_count_distribution(word_count_data):
    plt.figure(figsize=(10, 6), dpi=200)
    word_count_data.hist(bins=50, color="teal", edgecolor="black")
    plt.xlabel("word count")
    plt.ylabel("number of complaints")
    plt.title("Word Count Distribution")

    plt.yscale("log") # for better visualizing minority entries
    plt.tight_layout()
    plt.show()
def normalize_for_rag(text: str) -> str:
    """
    Optimized Cleaning for RAG embeddings & retrieval:
      1. Drop placeholders & dates
      2. Unicode-normalize + lowercase
      3. Remove stray non-printable/special chars
      4. Collapse whitespace
    """
    if pd.isna(text) or not text.strip():
        return "" # handle empty strings
    
    # 1) Remove common placeholders, dollar-amount tokens, and dates
    text = re.sub(r"[xX]{2,}", " ", text)
    
    text = re.sub(r"\{\$(?:\d+(?:\.\d{2})?)\}", " $ ", text)

    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", " ", text)
    
    # 2) Unicode normalize & lowercase
    text = unicodedata.normalize("NFKC", text).lower()
    
    # 3) Strip control chars & uncommon punctuation (keep . , ; : ! ? ' -)
    text = re.sub(r"[^a-z0-9\s\.\,\;\:\!\?\'\-\$]", " ", text)
    
    # 4) Collapse any sequence of whitespace to a single space
    text = re.sub(r"\s+", " ", text).strip()
    
    return text