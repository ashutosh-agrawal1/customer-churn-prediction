# 📊 Customer Churn Prediction

### Logistic Regression --- From Scratch (NumPy) vs Production Pipeline



## 🚀 Live Demo

👉 https://your-app-link.streamlit.app



## 💡 Why this project?

Built to demonstrate logistic regression from first principles on a
real-world churn problem, and translate predictions into actionable
business decisions through a deployable, user-facing application.



## ⭐ Project Highlights

-   End-to-end ML pipeline on a real business dataset\
-   Logistic Regression implemented from scratch using NumPy\
-   Production-ready implementation using scikit-learn Pipeline\
-   Interactive Streamlit app for real-time churn prediction\
-   Business-oriented insights with actionable recommendations\
-   Clear comparison between theoretical and production ML approaches



## 🖥️ App Preview

### 🔍 Input Panel

Users can input: - Tenure (months)\
- Monthly Charges\
- Contract Type

### 📊 Output

-   Churn probability\
-   Risk classification (Low / Medium / High)\
-   Key churn drivers\
-   Recommended business actions



## 📌 Project Overview

Customer churn is a critical business problem --- retaining existing
customers is significantly cheaper than acquiring new ones.

This project builds a churn prediction system using the Telco dataset
and focuses on:

-   Understanding the mathematics behind Logistic Regression\
-   Building a production-ready ML pipeline\
-   Translating predictions into business decisions



## 📊 Exploratory Data Analysis

Key insights:

-   Customers with tenure \< 12 months show significantly higher churn\
-   Month-to-month contracts have the highest churn rate\
-   Higher monthly charges increase churn probability\
-   Long-term contracts correlate with strong retention



## 🧠 Model Implementation

### NumPy --- From Scratch

-   Sigmoid activation\
-   Binary cross-entropy loss\
-   Gradient descent

### Production Model --- Pipeline

-   StandardScaler + LogisticRegression\
-   Ensures consistent preprocessing\
-   Prevents train--inference mismatch



## 📈 Results

| Metric | NumPy | Pipeline |
|--------|------|----------|
| Accuracy | 77.36% | 71.89% |
| Precision | 58.5% | 48.2% |
| Recall | 50.5% | 78.6% |
| F1 Score | 54.2% | 59.7% |
| ROC-AUC | 0.821 | 0.823 |



## 💼 Business Interpretation

-   Low tenure customers are high risk\
-   Month-to-month contracts increase churn\
-   High charges increase churn probability



## 🧠 Business Story

Instead of just predicting churn, this system explains:

-   Why the customer may churn\
-   What actions should be taken

This turns the model into a decision-making tool.



## 🛠️ Tech Stack

-   Python (NumPy, Pandas, Scikit-learn)\
-   Streamlit\
-   Matplotlib



## ▶️ How to Run

``` bash
git clone https://github.com/ashutosh-agrawal1/customer-churn-prediction
cd customer-churn-prediction
pip install -r requirements.txt
```

Run training:

``` bash
python train.py
```

Run app:

``` bash
streamlit run app.py
```


## 📂 Project Structure

```
customer-churn-prediction/
│
├── Data/                          # Add dataset here (not tracked in git)
│   └── .gitkeep
│
├── churn_analysis.ipynb           # EDA and feature insights
├── logistic_regression_numpy.py   # Logistic Regression from scratch
├── utils.py                       # Preprocessing and feature loading
├── train.py                       # Training + evaluation pipeline
├── requirements.txt               # Pinned dependencies
├── app.py
├── churn_model.pkl             
└── README.md
```

---

## 📦 Requirements

```
python 3.9+ (tested on Python 3.14.2)
numpy==1.24.0
scikit-learn==1.3.0
pandas==2.0.3
matplotlib==3.7.2
streamlit
```



## 📌 Limitations

- Linear decision boundary — may miss nonlinear churn patterns
- Single train-test split — no cross-validation yet
- Limited to features available in this dataset



## 🔮 Future Improvements

- [ ] Add L2 regularization to scratch implementation
- [ ] K-fold cross-validation for more robust evaluation
- [ ] ROC curve and precision-recall curve visualizations
- [ ] Tree-based models — Random Forest, XGBoost comparison
- [ ] Streamlit improvement (UI/UX + advanced explanations)
- [ ] Threshold tuning for business-specific optimization
- [ ] Enhanced feature set (TotalCharges, InternetService, etc.)



## 👨‍💻 Author

**Ashutosh Agrawal**
ECE Undergraduate
## 👨‍💻 Author

**Ashutosh Agrawal**  
ECE Undergraduate  

- GitHub: https://github.com/ashutosh-agrawal1  
- LinkedIn: https://www.linkedin.com/in/ashutosh-agrawal-823753238  
- X: https://x.com/hey_its_ashh  
- Email: ashutosh69003@gmail.com  

