# 🚨 Fraud Deception in Network

This project is a dual-module Machine Learning system designed to detect:
- 🕵️‍♂️ **Phishing URLs**
- 💳 **Credit Card Fraudulent Transactions**

The goal is to simulate a multi-layered fraud detection system within a network security context.

---

## 🔍 Modules Overview

### 1. Phishing URL Detection
- Dataset: [UCI ML - Phishing Websites Dataset](https://archive.ics.uci.edu/ml/datasets/phishing+websites)
- Output: `Legit` or `Phishing`
- Features: Based on structural properties of the URL (length, presence of '@', redirection, etc.)
- Model: `Random Forest Classifier`

### 2. Credit Card Fraud Detection
- Dataset: [Kaggle - Credit Card Fraud Detection (European Bank)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Output: `Genuine` or `Fraudulent`
- Features: PCA-reduced numerical transaction features
- Class imbalance handled using `SMOTE`
- Model: `Random Forest Classifier`

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- Required libraries:
```bash
pip install pandas scikit-learn imbalanced-learn seaborn matplotlib
````

---

## 🚀 How to Run

### Phishing Detection

```python
python phishing_detection.py
```

### Credit Card Fraud Detection

```python
python creditcard_fraud_detection.py
```

---

## 📊 Results

### ✅ Phishing URL Detection

* **Accuracy**: \~95%+
* Includes confusion matrix and classification report

### ✅ Credit Card Fraud Detection

* **Accuracy**: \~99% (after SMOTE balancing)
* Also includes confusion matrix and classification report

---

## 📁 Project Structure

```
fraud-deception-network/
│
├── phishing.csv                         # UCI phishing dataset
├── creditcard.csv                       # Kaggle credit card fraud dataset
│
├── phishing_detection.py                # Phishing ML model
├── creditcard_fraud_detection.py        # Credit card ML model
│
├── phishing_model.pkl                   # Saved phishing model
├── creditcard_model.pkl                 # Saved credit card model
│
├── README.md                            # This file
```

---

## 🎯 Future Enhancements

* 🔄 Integrate both models into a unified Flask web app
* 📡 Add real-time detection and logging interface
* 📊 Deploy dashboard with charts and alerts

---

## 📚 References

* [UCI ML Repository – Phishing Websites Dataset](https://archive.ics.uci.edu/ml/datasets/phishing+websites)
* [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 🧠 Author

D. Ravi Kiran
*Machine Learning | AI | Cybersecurity Enthusiast*

---
