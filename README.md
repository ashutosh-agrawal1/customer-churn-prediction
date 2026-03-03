# 📊 Customer Churn Prediction  
### Logistic Regression (From Scratch + Scikit-learn Comparison)

---

## ⭐ Project Highlights

- End-to-End Machine Learning Pipeline  
- Logistic Regression implemented from scratch using NumPy  
- Comparison with scikit-learn implementation  
- Stratified train-test split for fair evaluation  
- ROC-AUC, Precision, Recall, F1-score, and Confusion Matrix evaluation  
- Business-oriented churn analysis  

---

## 📌 Project Overview

This project builds an end-to-end customer churn prediction model using the Telco Customer Churn dataset.

Logistic Regression was implemented from scratch using NumPy to demonstrate strong understanding of the mathematical foundations behind binary classification models, including:

- Sigmoid activation  
- Binary cross-entropy loss  
- Gradient descent optimization  

The implementation is benchmarked against scikit-learn’s `LogisticRegression` model using the same train-test split and preprocessing pipeline to ensure a fair and consistent comparison.

The objective extends beyond prediction accuracy to identifying key churn drivers and generating actionable business insights.

---

## 📂 Dataset

**Telco Customer Churn Dataset**  
Source: IBM Sample Data (Kaggle)

The dataset contains customer demographic, contract, and billing information along with churn labels.

- Total samples: ~7,000  
- Churn rate: ~26–27%  

Due to class imbalance, evaluation emphasizes Recall and ROC-AUC in addition to Accuracy.

---

## 📊 Exploratory Data Analysis (EDA)

Key insights from analysis:

- Customers with **low tenure (< 12 months)** have significantly higher churn probability.
- **Month-to-month contracts** show nearly 2× higher churn rate compared to two-year contracts.
- Higher **monthly charges** increase churn likelihood.
- Long-term contracts represent the most stable customer segment.

These findings guided feature selection and modeling decisions.

---

## 🧠 Model Implementation

### 1️⃣ NumPy Implementation (From Scratch)

The custom Logistic Regression model includes:

- Sigmoid activation function  
- Binary cross-entropy loss  
- Gradient descent optimization  
- Probability prediction (`predict_proba`)  
- Adjustable classification threshold  

This implementation demonstrates understanding of the mathematical derivation of logistic regression.

---

### 2️⃣ Scikit-learn Implementation

The scikit-learn model is trained using:

- `LogisticRegression`
- `class_weight='balanced'` (to handle class imbalance)
- Identical normalized features
- Stratified train-test split

This ensures a fair performance comparison.

---

## ⚙️ Data Preprocessing

- Stratified 80-20 train-test split  
- Feature normalization using training data statistics only  
- Division-by-zero protection during normalization  

---

## 📈 Model Evaluation

Both models are evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  
- Confusion Matrix  

### Why Accuracy Alone Is Not Enough

Since churn rate is ~26%, predicting all customers as non-churn would already yield ~73% accuracy.

Therefore, Recall and ROC-AUC are prioritized to better detect at-risk customers.

Typical performance:
- Accuracy: ~77–80%
- Recall: ~65–70%
- ROC-AUC: ~0.80+

---

## 🔬 Model Comparison

Both models are trained on identical splits and preprocessing steps to ensure fairness.

The comparison validates that the from-scratch NumPy implementation produces results comparable to scikit-learn’s optimized solver.

This validates the correctness of the manual gradient descent implementation and demonstrates alignment with industry-standard tools.

---

## 💼 Business Interpretation

From model coefficients and evaluation:

1. Low-tenure customers are highest risk.
2. Month-to-month contracts significantly increase churn probability.
3. Higher monthly charges correlate with increased churn.
4. Longer contract durations improve retention stability.

### Suggested Business Actions

- Prioritize retention campaigns for early-stage customers.
- Encourage migration to long-term contracts.
- Provide targeted offers to high-cost, short-tenure customers.
- Use churn probability scores for proactive outreach.

---

## ▶️ How to Run

Clone the repository and run:
```bash
python train.py


This will:

- Load the dataset  
- Perform stratified train-test split  
- Normalize features  
- Train both NumPy and scikit-learn models  
- Print evaluation metrics and confusion matrices  
```
---


## 📂 Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── data_set.csv
│
├── churn_analysis.ipynb          # Exploratory data analysis
├── logistic_regression_numpy.py  # Logistic Regression from scratch
├── utils.py                      # Feature loading & preprocessing
├── train.py                      # Training + comparison pipeline
│
└── README.md                   
```



---

## 🚀 Key Learning Outcomes

- Mathematical understanding of Logistic Regression  
- Manual gradient descent implementation  
- Handling class imbalance  
- Threshold tuning for business optimization  
- Fair benchmarking against industry tools  
- Translating model results into business strategy  

---

## 📌 Limitations

- Linear decision boundary (may not capture nonlinear patterns)  
- Limited feature set  
- Single train-test split (no cross-validation)  
- No regularization implemented  

---

## 🔮 Future Improvements

- Add L2 Regularization  
- Implement k-fold cross-validation  
- Add ROC curve visualization  
- Explore tree-based models (Random Forest, XGBoost)  
- Hyperparameter tuning  

---

## 👨‍💻 Author

**Ashutosh Agrawal**  
Electronics & Communication Engineer  
Aspiring Data Analyst / Machine Learning Engineer
