# Detection of AI-Generated Arabic Text: A Stylometric Analysis Approach

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

## 🚀 Installation & Usage

### 1. Prerequisites
Install the required packages using `pip`:
```bash
pip install -r requirements.txt
