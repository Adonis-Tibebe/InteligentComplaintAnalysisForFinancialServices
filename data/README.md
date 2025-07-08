# Data Directory
This directory contains the raw and processed data used in the Intelligent Complaint Analysis for Financial Services project.

## Raw Data
The raw data is stored in the raw subdirectory and consists of:

complaints.csv: A CSV file containing the raw complaint data from the Consumer Financial Protection Bureau (CFPB).
## Processed Data
The processed data is stored in the processed subdirectory and consists of:

filtered_complaints.csv: A CSV file containing the cleaned and preprocessed complaint data.
final_filtered_complaints.csv: A final version of the filtered_complaints.csv with the narative with less than 20 words were removed to accomodate for noise in the rag pipeline
Chunked_complaints.csv: the end result of the cunking and embedding which is directly mapped to each vector in our faiss vector index database and contains metadata and source row column which mapps back to the final_filtered_complaints.csv to trace original text narative

The data used in this project is sourced from the Consumer Financial Protection Bureau (CFPB) and is publicly available.

### Data Description
The complaint data includes information such as:

Complaint ID
Product
Issue
Sub-issue
State
ZIP code
Date received
Date sent to company
Company response
Consumer disputed?
### Data Preprocessing
The raw data is preprocessed using the following steps:

- Data cleaning: Removing missing and duplicate values.
- Data transformation: Converting data types and formatting.
- Data normalization: Normalizing the data to a appropriate narative text which could be fed into a RAG pipeline
### Data Vectorization
The preprocessed data is vectorized using the following techniques:

- chunking: was done using langchains RecursiveCharacterTextSplitter
- Vectorization: vectorized each chunk using the all-MiniLM-L6-v2 vectorizor model and converted each chunk into a 384 dimension vector