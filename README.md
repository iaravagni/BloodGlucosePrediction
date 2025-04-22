# 🩸 Blood Glucose Level Prediction

This project explores multiple approaches for predicting blood glucose levels using multimodal data from the Big Ideas dataset. It includes data from wearables (accelerometer), nutritional inputs, and demographic/medical history features. Our goal is to evaluate and compare naive, machine learning, and deep learning methods to provide personalized and accurate glucose level forecasts.

---

## 🚀 Problem Statement

Blood glucose levels are influenced by numerous variables including physical activity, food intake, and individual physiology. Early prediction can empower individuals to manage and prevent health complications, especially for those with diabetes. This project seeks to build models that can forecast glucose levels based on a combination of behavioral and clinical data.

---

## 📊 Dataset Preparation

Data is structured from the **Big Ideas dataset** to include the following features:

- Glucose Level (target)
- Accelerometer
- Calories
- Carbs
- Sugar
- Gender
- HbA1c
- Age

The dataset is split as follows:
- **Train:** 13 patients (~80%)
- **Validation:** 2 patients from the training set (~15%)
- **Test:** 3 patients (~20%)

### Generated Files:
- Individual CSV for each patient
- Combined dataset CSV
- Train/Validation/Test CSVs

---

## 🧠 Modeling Approaches

We evaluated 3 approaches to predict glucose levels:

### 1. 🧮 Naive Approach
- Model: `ibm-granite/granite-timeseries-ttm-r2` (zero-shot)
- No fine-tuning

### 2. 🌲 Machine Learning
- **Model:** XGBoost Regressor
```python
xgb_model = xgb.XGBRegressor(
    n_estimators=50, 
    learning_rate=0.2, 
    max_depth=5, 
    objective='reg:squarederror', 
    random_state=42
)
```

### 3. 🤖 Deep Learning
- **Model:** Fine-tuned `granite-timeseries-ttm-r2` on the structured dataset

---

## 📈 Results

| Approach | RMSE |
|----------|------|
| Naive    | 3.7812 |
| ML       | 3.9681 |
| DL       | 3.8762 |

---

## 🌐 Streamlit Web App

The interactive app allows users to:
- Upload the 3 required CSVs
- Select preloaded patient samples
- View and compare predictions from:
  - Naive model
  - Machine Learning model
  - Deep Learning model

> 📍 A working version is deployed and publicly accessible [**here**](#) *(add your link)*

---

## 📽️ Final Deliverables

- 🔗 [10-minute video presentation](#) 
- 🔗 [Live Streamlit app](https://huggingface.co/spaces/iaravagni/BloodGlucosePrediction)

---

## 🔍 Ethics Statement

While this project focuses on non-invasive glucose prediction, it is not intended for real-time clinical use. Any deployment of such tools must undergo rigorous validation and ethical review. We aim to respect user privacy, and datasets used in this project are anonymized and publicly available.

---

## ▶️ How to Run

Follow these steps to set up the environment and launch the Streamlit app:

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/glucose-prediction.git
cd glucose-prediction
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app:**
```bash
streamlit run glucose_app.py --server.maxUploadSize=1000
```

This will launch the interactive app in your browser, allowing you to upload data and view predictions from the three modeling approaches.
