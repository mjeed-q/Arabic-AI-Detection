# Scalable Real-time Detection of AI-Generated Arabic Text 🚀
### *A Distributed Data Pipeline Approach (MSBDA 882 Final Project)*

---

## 📌 Project Overview
The rapid proliferation of Large Language Models (LLMs) poses a significant challenge to academic integrity. This project delivers an end-to-end, high-performance Big Data pipeline tailored specifically for detecting AI-generated Arabic academic abstracts. Leveraging **Apache Spark's** distributed computing framework, the pipeline efficiently ingests, cleans, and classifies massive textual data in real-time.

---

## 🏗️ System Architecture & Pipeline
The framework features a modular and distributed architecture built entirely on **Spark ML Pipeline**:
1. **Data Ingestion & Persistance:** Processing **16,895 balanced records** optimized via distributed **Parquet storage** for maximum memory efficiency.
2. **Preprocessing:** Text cleaning and morphological normalization integrated with the `Camel Tools` library.
3. **Hybrid Feature Engineering:**
   - **Lexical Baselines:** Extracted via `HashingTF` and `IDF` (tuned to 2,000 features).
   - **Stylometric Signatures:** 5 custom linguistic features tracking word variance, characters per paragraph, command verbs, average sentence length, and formality scores.
4. **Real-time Auditing:** Operationalized using **Spark Structured Streaming** to detect AI text on live streaming inputs.

---

## 📂 Repository Structure
```text
├── data/
│   ├── raw/          # Complete original dataset (Full_Raw_Dataset.xlsx)
│   └── processed/    # Model predictions & engineered features (Full_Dataset_Predictions.csv)
├── models/
│   ├── rf_model/     # Trained Spark MLlib Random Forest Model (Champion)
│   ├── gbt_model/    # Trained Gradient Boosted Trees Model
│   ├── nb_model/     # Trained Naive Bayes Model
│   └── scaler/       # Saved StandardScaler configurations
├── scripts/
│   └── MSBDA882_Project_Final.ipynb   # Core PySpark execution Notebook
└── docs/
    ├── Research_Paper.pdf
    └── Presentation.html