# 🔎 Quora Duplicate Question Detection

A production-grade NLP project that detects whether two questions are semantically similar (duplicates) using **Sentence Transformers, feature engineering, and machine learning models**.

---

## 🚀 Project Overview

Duplicate questions are a major challenge in platforms like Quora, StackOverflow, and customer support systems. This project builds a **full end-to-end machine learning pipeline** to identify duplicate questions by combining:

- Semantic understanding (transformer embeddings)
- Lexical similarity features
- Supervised ML models
- Threshold tuning for optimal classification

---

## 🎯 Business Problem

Duplicate questions lead to:
- Poor user experience
- Redundant content
- Inefficient search systems

### 💡 Solution Impact

- Improves **search relevance**
- Reduces **duplicate content**
- Enhances **knowledge retrieval**
- Optimizes **user engagement**

### 🏢 Real-world Applications

- Quora duplicate detection
- FAQ deduplication
- Customer support ticket clustering
- Search engine optimization

---

## 🧠 Solution Approach

### 1. Data Understanding

Input dataset contains:

- `question1`
- `question2`
- `is_duplicate` (target)

---

### 2. Text Representation

We use transformer-based embeddings:

SentenceTransformer("all-MiniLM-L6-v2")

- 384-dimensional vectors
- Captures **semantic meaning**
- Context-aware (better than TF-IDF)

---

### 3. Feature Engineering

#### 🔹 Semantic Features
- Cosine similarity between embeddings

#### 🔹 Lexical Features
- Word count (q1, q2)
- Character count (q1, q2)
- Word count difference
- Character count difference
- Jaccard similarity
- Token overlap ratio

#### 🔹 Embedding-Based Features
- Absolute difference: |e1 - e2|
- Element-wise product: e1 * e2

---

### 4. Model Training

Models evaluated:

- Logistic Regression
- Random Forest
- XGBoost (best performing model)

---

### 5. Threshold Tuning

Instead of default 0.5:

- Threshold optimized on validation set
- Improves F1-score
- Balances Precision and Recall

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## 🏗️ Project Pipeline

Raw Text  
↓  
Text Cleaning  
↓  
Sentence Embeddings  
↓  
Feature Engineering  
↓  
Model Training  
↓  
Threshold Tuning  
↓  
Final Model  
↓  
Deployment (Streamlit)

---

## 💻 Streamlit Application

An interactive web application for real-time predictions.

### Features

- 🔍 Single prediction (manual input)
- 📂 Batch prediction (CSV upload)
- 📊 Probability score visualization
- 📈 Feature-level insights

---

## 📁 Project Structure

Project--Quora-Duplicate-Question-Detection-with-Streamlit/  
│  
├── app.py                      # Streamlit application  
├── requirements.txt  
├── README.md  
│  
├── artifacts/  
│   ├── quora_duplicate_classifier.joblib  
│   └── metadata.json  
│  
├── Training_phase_with_GPU/  
│   └── Project_Quora_Duplicate_Question_Detection.ipynb  

---

## ⚙️ Installation

### 1. Clone Repository

git clone https://github.com/BIRJUNG/-Project--Quora-Duplicate-Question-Detection-with-streamlit.git
cd -Project--Quora-Duplicate-Question-Detection-with-streamlit

---

### 2. Create Virtual Environment

python3 -m venv venv  
source venv/bin/activate  

---

### 3. Install Dependencies

pip install -r requirements.txt  

---

### 4. Run Application

streamlit run app.py  

---

## 🧪 Example

Input:

Q1: How can I learn Python quickly?  
Q2: What is the fastest way to learn Python?  

Output:

Prediction: Duplicate  
Probability: 0.91  

---

## 🧠 Key Learnings

- Difference between **vectorization vs embeddings**
- Importance of **semantic similarity in NLP**
- Feature engineering for text data
- Threshold tuning vs default classification
- Model comparison and selection
- Deployment challenges (XGBoost, environment issues)

---

## ⚠️ Deployment Note (Important)

XGBoost models saved with joblib may cause compatibility issues.

Recommended approach:

model.save_model("model.json")  

Then load using:

model.load_model("model.json")  

---

## 🔥 Future Improvements

- Use **cross-encoder models** for higher accuracy
- Implement **Siamese Neural Networks**
- Deploy with **FastAPI + Docker**
- Add **real-time API endpoints**
- Integrate **vector databases (FAISS)**

---

## 📌 Tech Stack

- Python  
- Scikit-learn  
- XGBoost  
- Sentence Transformers  
- Pandas / NumPy  
- Streamlit  

---

## 👨‍💻 Author

**Birjung Thapa**  
Master’s in Data Science  
University of Colorado Boulder  

---

## ⭐ Support

If you found this project useful:

⭐ Star the repository  
📢 Share it with others  

---
