# Data Dictionary & Methodology

## 1. Feature Description
The following stylometric features were engineered to detect AI-generated Arabic text:

| Feature Code | Feature Name | Description |
| :--- | :--- | :--- |
| **F15** | Word Length Distribution | Measures vocabulary complexity (Mean & Std Dev of character counts). |
| **F38** | Characters per Paragraph | Measures information density and layout structure. |
| **F61** | Number of Commands | Counts imperative verbs (e.g., "انظر", "قارن") indicating instructional tone. |
| **F84** | Average Sentence Length | Measures rhythmic structure; found to be the strongest predictor. |
| **F107** | Formality Score | Ratio of MSA strictness (formal markers) vs. informal indicators. |

## 2. Dataset Overview
* **Source:** KFUPM-JRCAI/arabic-generated-abstracts
* **Total Samples:** 16,776 (Balanced 50/50 split).
* **Classes:**
    * `0`: Human-written text.
    * `1`: AI-generated text.

## 3. Models Saved
The trained models are stored in the `models/` directory:
1.  **random_forest.pkl**: (Best Performer - 99.08% Accuracy).
2.  **svm.pkl**: Support Vector Machine.
3.  **logistic_regression.pkl**: Baseline Model.