# 📊 Customer Churn Prediction
### Logistic Regression — From Scratch (NumPy) vs Scikit-learn

---

## ⭐ Project Highlights

- End-to-end ML pipeline on a real business dataset
- Logistic Regression implemented from scratch using NumPy
- Side-by-side comparison with scikit-learn's implementation
- Stratified train-test split for fair, reproducible evaluation
- Business-oriented churn analysis with actionable recommendations

---

## 📌 Project Overview

This project builds a customer churn prediction model on the Telco Customer Churn dataset.

Logistic Regression was implemented from scratch using NumPy to demonstrate a strong understanding of the mathematical foundations behind binary classification, including:

- Sigmoid activation
- Binary cross-entropy loss
- Gradient descent optimization with numerical stability

The scratch implementation is benchmarked against scikit-learn's `LogisticRegression` on identical train-test splits and preprocessing pipelines to ensure a fair comparison.

The objective goes beyond prediction accuracy — the project identifies key churn drivers and translates model outputs into actionable business strategy.

---

## 📂 Dataset

**Telco Customer Churn Dataset**
Source: [IBM Sample Data via Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

> ⚠️ Dataset not included in this repo. Download from Kaggle and place the CSV file in the `/data/` folder before running.

- Total samples: ~7,000
- Features: Customer demographics, contract type, billing info, service usage
- Churn rate: ~26–27% (imbalanced)

---

## 📊 Exploratory Data Analysis

Key insights from analysis:

- Customers with **tenure < 12 months** show significantly higher churn probability
- **Month-to-month contracts** have nearly 2× the churn rate of two-year contracts
- Higher **monthly charges** correlate with increased churn likelihood
- Long-term contracts represent the most stable customer segment

These findings directly guided feature selection and modeling decisions.

---

## 🧠 Model Implementation

### 1️⃣ NumPy — From Scratch

```python
class LogisticRegression:
    def fit(self, X, y):      # gradient descent training loop
    def predict_proba(self, X): # sigmoid output
    def predict(self, X, threshold=0.5): # binary classification
```

Key implementation details:
- `np.clip` for numerical stability in sigmoid
- Epsilon smoothing in binary cross-entropy loss
- Vectorized gradient computation (no loops)
- Adjustable classification threshold for business tuning

### 2️⃣ Scikit-learn Baseline

- `LogisticRegression` with `class_weight='balanced'`
- Identical normalized features and stratified split
- Used as ground truth to validate the scratch implementation

---

## ⚙️ Data Preprocessing

- Stratified 80-20 train-test split
- Feature normalization using training statistics only (no data leakage)
- Division-by-zero protection in normalization

---

## 📈 Results

> Evaluated on identical stratified test splits for fair comparison.

| Metric | NumPy (Scratch) | Scikit-learn |
|---|---|---|
| Accuracy | 79.2% | 80.1% |
| Precision | 63.4% | 65.2% |
| Recall | 67.3% | 71.0% |
| F1 Score | 65.3% | 68.0% |
| ROC-AUC | 0.821 | 0.838 |

### Why not just use Accuracy?

With a ~26% churn rate, predicting every customer as "not churned" already gives ~73% accuracy. **Recall and ROC-AUC** are the metrics that matter — missing a churner is more costly than a false alarm.

---

## 🔬 Key Finding: Scratch vs Scikit-learn

The NumPy implementation achieves within **~2% ROC-AUC** of scikit-learn's optimized solver — validating the correctness of the manual gradient descent implementation.

The small gap is expected: scikit-learn uses L2 regularization and a LBFGS solver by default, while the scratch model uses vanilla gradient descent.

---

## 💼 Business Interpretation

From model coefficients and EDA:

1. **Low-tenure customers** are the highest churn risk — target them early
2. **Month-to-month contracts** are the strongest churn predictor
3. **High monthly charges + short tenure** = highest risk segment
4. Long-term contracts significantly improve retention stability

### Recommended Actions

| Segment | Action |
|---|---|
| Tenure < 3 months | Onboarding loyalty offer |
| Month-to-month, high charges | Targeted discount to switch contract |
| High churn probability score | Proactive outreach before next billing cycle |
| Long-term contracts | Reward and upsell — lowest churn risk |

---

## ▶️ How to Run
# Clone the repo
```bash
git clone https://github.com/ashutosh-agrawal1/customer-churn-prediction
```
```
cd customer-churn-prediction
```
# Install dependencies
```
pip install -r requirements.txt
```
# Download dataset from Kaggle and place in /data/
```
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
```
# Run training and comparison
```
python train.py
```

---

## 📂 Project Structure

```
customer-churn-prediction/
│
├── data/                          # Add dataset here (not tracked in git)
│   └── .gitkeep
│
├── churn_analysis.ipynb           # EDA and feature insights
├── logistic_regression_numpy.py   # Logistic Regression from scratch
├── utils.py                       # Preprocessing and feature loading
├── train.py                       # Training + evaluation pipeline
├── requirements.txt               # Pinned dependencies
└── README.md
```

---

## 📦 Requirements

```
numpy==1.24.0
scikit-learn==1.3.0
pandas==2.0.3
matplotlib==3.7.2
```

---

## 📌 Limitations

- Linear decision boundary — may miss nonlinear churn patterns
- No regularization in scratch implementation (L2 planned)
- Single train-test split — no cross-validation yet
- Limited to features available in this dataset

---

## 🔮 Future Improvements

- [ ] Add L2 regularization to scratch implementation
- [ ] K-fold cross-validation
- [ ] ROC curve and precision-recall curve visualizations
- [ ] Tree-based models — Random Forest, XGBoost comparison
- [ ] Streamlit deployment for live churn probability scoring

---

## 👨‍💻 Author

**Ashutosh Agrawal**
ECE Undergraduate

[GitHub](https://github.com/ashutosh-agrawal1) · [LinkedIn](https://www.linkedin.com/in/ashutosh-agrawal-823753238)

---
