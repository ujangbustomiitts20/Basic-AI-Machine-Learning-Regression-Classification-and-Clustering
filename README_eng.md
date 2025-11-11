### 🎯 Project Description

This repository contains a **demonstration of foundational AI & Machine Learning concepts** using Python and Scikit-Learn.
It covers three essential areas:

1. **Regression (Supervised Learning)** — Predicting synthetic house prices based on area, number of rooms, and building age.
2. **Classification (Supervised Learning)** — Identifying *Iris* flower species using *Logistic Regression*.
3. **Clustering (Unsupervised Learning)** — Grouping 300 individuals by height and weight using the *KMeans* algorithm.

This project serves as a beginner-friendly introduction to understanding how AI models are built, trained, evaluated, and applied to new data.

---

### 🧩 Project Structure

```
ai_basics_demo/
├── tahap1_ai_ml_demo.py
├── model_regresi.pkl
├── model_iris.pkl
├── requirements.txt
└── README.md
```

---

### 🚀 How to Run

#### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

#### 2️⃣ Run the main script

```bash
python tahap1_ai_ml_demo.py
```

#### 3️⃣ Expected Outputs:

* **Regression:** Displays MAE and R² scores, plus a plot comparing actual vs predicted prices.
* **Classification:** Shows accuracy and a confusion matrix for the Iris dataset.
* **Clustering:** Visualizes how 300 people are grouped by height and weight.

---

### 📦 Saved Models

* `model_regresi.pkl` → trained Linear Regression model.
* `model_iris.pkl` → trained Logistic Regression model for Iris classification.

You can reuse these models using `joblib.load()` to make predictions without retraining.

---

### 📊 Technologies & Libraries

* **Python 3.10+**
* **scikit-learn**
* **pandas**
* **numpy**
* **matplotlib**
* **joblib**
* **Pillow**

---

### 👨‍💻 Author

Developed by [Aulia Ikhwanudin](https://github.com/ujangbustomiitts20)
As part of the *AI & Machine Learning from Fundamentals to Implementation* learning series.

---
