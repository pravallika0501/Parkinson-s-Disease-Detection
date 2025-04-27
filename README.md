# Parkinson's Disease Prediction Web Application

## 🧠 Project Overview

This project is a machine learning-powered web application that predicts the likelihood of Parkinson's disease based on various vocal and biomedical measurements. Users can input 8 specific health parameters, and the system will predict if the person is likely to have Parkinson’s disease.

---

## 🎯 Aim

- Create a user-friendly UI where users can input relevant parameters.
- Predict the likelihood of Parkinson’s disease using a trained machine learning model.

---

## 📚 Technologies Used

- **Python**
- **Flask** (Web Framework)
- **HTML/CSS** (Frontend)
- **Pandas** (Data Processing)
- **Scikit-learn** (Model Building)
- **Joblib** (Model Serialization)

---

## 📊 Dataset

- The dataset used is the **Parkinson's Disease Dataset** from [DPhi](https://github.com/dphi-official/Datasets/blob/master/parkinsons_data.csv).
- Target Variable: `status` (0: healthy, 1: Parkinson's patient)

---

## 🛠 Features Used

The model is trained on the following 8 features:
- MDVP:Fo(Hz)
- MDVP:Jitter(%)
- MDVP:RAP
- MDVP:Shimmer
- NHR
- HNR
- RPDE
- DFA


## ✨ How It Works
User fills in the 8 parameters in the web form.
The values are scaled using the pre-fitted StandardScaler.
The scaled features are passed to the Random Forest Classifier model.
The model returns whether the person is likely or unlikely to have Parkinson’s disease.

