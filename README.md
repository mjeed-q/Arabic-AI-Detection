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
# Detection of AI-Generated Arabic Text:A Data Mining Approach

**Course:** MSIS-822 Advanced Data Analytic Techniques  
**University:** Taibah University  
**Student Name:** Abdulmajeed Alqahtani  
**Student ID:** 4714235  
**Date:** December 14, 2025

---

## 📌 Project Overview
Large Language Models (LLMs) like GPT-4 and Jais have revolutionized content creation but also introduced risks regarding academic dishonesty and misinformation. Detecting machine-generated text in **Arabic** is particularly challenging due to the language's rich morphology and complex syntax.

This project presents a **Data Mining Pipeline** to distinguish between human-written and AI-generated Arabic abstracts. Unlike black-box deep learning models, we utilize an **Explainable Stylometric Approach**, engineering interpretable features that capture the statistical "fingerprints" of AI generation.

## 📂 Dataset
* **Source:** `KFUPM-JRCAI/arabic-generated-abstracts` (Hugging Face).
* **Content:** Original human abstracts paired with AI-generated versions from models like OpenAI GPT, Llama-2, Jais, and Allam.
* **Preprocessing:** The dataset was balanced using random undersampling to ensure a fair evaluation.
* **Final Size:** 16,776 samples (50% Human, 50% AI).

## ⚙️ Methodology & Feature Engineering
We engineered **five specific stylometric features** based on Arabic linguistics to detect the "structural rigidity" of AI text:

| Code | Feature Name | Description & Hypothesis |
| :--- | :--- | :--- |
| **F15** | **Word Length Distribution** | **Logic:** Mean & Std Dev of character counts.<br>**Hypothesis:** AI prefers high-probability words (uniform length), while humans have wider vocabulary variance. |
| **F38** | **Chars per Paragraph** | **Logic:** Avg characters per newline block.<br>**Hypothesis:** AI texts often appear as dense, perfectly balanced blocks, unlike irregular human formatting. |
| **F61** | **Instructional Tone** | **Logic:** Count of imperative verbs (e.g., "notice", "compare", "write").<br>**Hypothesis:** AI models often accidentally adopt a "tutorial" style due to instruction-tuning. |
| **F84** | **Avg Sentence Length** | **Logic:** Mean number of words per sentence.<br>**Hypothesis:** AI maintains a "safe" rhythmic uniformity for readability. Humans exhibit "burstiness" (mix of short/long sentences). |
| **F107** | **Formality Score** | **Logic:** Ratio of formal markers (e.g., "Al-", "iyya") vs. informal pronouns.<br>**Hypothesis:** AI text is often hyper-correct and formal, lacking personal nuances. |
