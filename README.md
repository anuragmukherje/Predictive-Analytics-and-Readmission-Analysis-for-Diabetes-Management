# Predictive Analytics and Readmission Analysis for Diabetes Management

## 📌 Project Overview
This project focuses on **predictive analytics and readmission analysis** for diabetes management using machine learning techniques. The primary objectives include:

- Developing a **comprehensive machine learning pipeline** for diabetes prediction and readmission analysis.
- Implementing **K-Nearest Neighbors (KNN) imputation** to handle missing values in clinical features.
- Training a **Logistic Regression model** for diabetes prediction with performance evaluation.
- Conducting an **in-depth readmission analysis** for diabetic patients.
- Utilizing **bivariate and trivariate analyses** to extract meaningful insights.
- Employing **data visualization techniques** for clear communication of findings.

---

## 🚀 Features & Capabilities
- **Data Imputation using KNN**
- **Predictive Modeling using Logistic Regression**
- **Comprehensive Model Evaluation (Accuracy, Precision, Recall, ROC-AUC)**
- **Advanced Data Analysis (Bivariate and Trivariate Analysis)**
- **Data Visualization (Heatmaps, Bar Plots, Clustered Plots)**
- **Insights into Readmission Trends in Diabetic Patients**

---

## 📊 Dataset
This project uses two primary datasets:

1. **Diabetes Prediction Dataset**
   - Contains patient medical history and lab test results.
   - Features include blood glucose levels, BMI, HbA1c, and medication details.

2. **Readmission Analysis Dataset**
   - Focuses on diabetic patient hospital readmissions.
   - Includes key indicators such as **HbA1c levels**, **medication changes**, and **primary diagnoses**.

📌 **Source:** [UCI Machine Learning Repository - Diabetes Readmission Dataset](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)

---

## ⚙️ Installation & Setup
To run this project, follow these steps:

### 🔹 Step 1: Clone the Repository
```bash
git clone https://github.com/your-repo/diabetes-predictive-analysis.git
cd diabetes-predictive-analysis
```

### 🔹 Step 2: Create a Virtual Environment (Optional)
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

### 🔹 Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
---

## 🔬 Methodology

### 📌 Data Preprocessing
- **Handling Missing Data**: Used **KNN Imputation** to replace missing values with the most probable values based on feature similarity.
- **Feature Engineering**: Selected the most relevant features affecting diabetes prediction and readmission rates.
- **Data Normalization**: Applied MinMax scaling for improved model performance.

### 📌 Machine Learning Model - Logistic Regression
- **Trained a Logistic Regression model** to classify diabetic vs. non-diabetic patients.
- **Evaluated performance using**:
  - **Accuracy**
  - **Precision & Recall**
  - **ROC-AUC Score**
  - **Confusion Matrix**

### 📌 Readmission Analysis
- **Bivariate & Trivariate Analysis**:
  - Examined relationships between **HbA1c levels, medication changes, and readmission rates**.
  - Identified trends affecting patient readmissions.
- **Data Visualization**:
  - Used **heatmaps** to highlight key correlations.
  - Applied **clustered bar plots** for categorical insights.

---

## 📈 Results & Findings

| Model | Accuracy | Precision | Recall | ROC-AUC |
|-------|----------|------------|--------|---------|
| Logistic Regression | 76.19% | 67.92% | 0.48 | 83.62% |

### 🔹 Key Insights
- **Early HbA1c Monitoring** significantly reduces readmission rates.
- **Medication Adjustments** play a crucial role in readmission probability.
- **Primary Diagnoses** have varying impacts on readmission likelihood.

---

## 📌 Usage Instructions
### 🔹 Running the Model
```python
from src.train_model import train_logistic_regression
train_logistic_regression()
```

### 🔹 Performing Readmission Analysis
```python
from src.readmission_analysis import analyze_readmission_trends
analyze_readmission_trends()
```

---

## 📌 Technologies Used
- **Python** (Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn)
- **Machine Learning** (Logistic Regression, KNN Imputation)
- **Data Visualization** (Heatmaps, Clustered Bar Plots)

---

## 💡 Future Enhancements
- Implement **Random Forest** or **XGBoost** for improved predictive accuracy.
- Introduce **Deep Learning Models (e.g., ANN, LSTM)** for time-series analysis.
- Develop a **real-time predictive dashboard** for hospitals.

---

## 📜 License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 🤝 Contribution Guidelines
We welcome contributions! Feel free to:
- Fork the repository
- Raise issues
- Submit pull requests

For major changes, please open an issue first to discuss what you'd like to change.

---

## 👨‍💻 Author
Anurag Mukherjee - Data Science & Machine Learning Enthusiast 🚀

📌 **GitHub**: Anurag Mukherjee(https://github.com/anuragmukherje)
📌 **LinkedIn**: Anurag Mukherjee(https://www.linkedin.com/in/anurag-mukherjee21/)

---

🚀 **Empowering Healthcare with Data Science!** 📊💡
