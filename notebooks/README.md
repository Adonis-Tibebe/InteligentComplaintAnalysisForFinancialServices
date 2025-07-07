# Notebooks

This directory contains Jupyter notebooks related to the **Intelligent Complaint Analysis for Financial Services** project. The notebooks document the step-by-step process of data exploration, cleaning, preprocessing, and analysis of consumer complaint data from the CFPB.

## Current Notebooks

### EDA_and_Data_Preprocessing.ipynb

- **Purpose:**  
  Performs exploratory data analysis (EDA) and preprocessing on the raw complaints dataset.
- **Key Steps:**
  - Loads and inspects the raw data.
  - Analyzes the distribution of complaints across product categories.
  - Calculates and visualizes word counts for complaint narratives.
  - Filters the dataset to focus on five key product categories using defined proxies.
  - Removes complaints without narrative text.
  - Normalizes complaint narratives to improve text quality for downstream tasks.
  - Saves the cleaned and processed data in both CSV and Parquet formats.

- **Key Findings:**  
  The majority of complaints are concentrated in a few product categories. After filtering and cleaning, the dataset is focused on five relevant categories with only substantive complaint narratives retained. Text normalization was successfully applied, preparing the data for further analysis or modeling.

---

## How to Use

- Open any notebook in this directory with Jupyter or VS Code.
- Follow the markdown and code cells for a step-by-step walkthrough of the analysis.
- Outputs and visualizations are generated inline for transparency and reproducibility.

## Template for Documenting Future Changes

When adding or updating notebooks, please use the following template:

### [Notebook Name].ipynb

- **Purpose:**  
  [Briefly describe the notebook's objective.]
- **Key Steps:**
  - [List the main steps or sections in the notebook.]
- **Key Findings / Results:**  
  [Summarize the main findings, results, or outputs.]

---

## Changelog

- *[YYYY-MM-DD]*: [Short description of the change, e.g., "Added EDA_and_Data_Preprocessing notebook."]
- *[YYYY-MM-DD]*: [Describe further updates or new notebooks.]

---

*For questions or suggestions, please contact the project