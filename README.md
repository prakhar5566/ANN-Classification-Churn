# ANN-Classification-Churn

# 📊 Customer Churn Prediction using ANN

![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)

## 🔗 Live Demo
[![Open in Streamlit](https://ann-classification-churn-mghwjyw8t2jd3maxdhxdrr.streamlit.app/)

---

## 📌 Project Overview
This is an **Artificial Neural Network (ANN)** based application that predicts whether a bank customer is likely to churn (leave) or stay, based on their account and demographic details.

The app is built with **Streamlit** and **TensorFlow** and deployed on **Streamlit Cloud**.

---

## 🛠 Features
- 📂 View and download the dataset
- 📈 Model accuracy and confusion matrix
- 🔍 Make predictions for individual customers
- ⚡ Real-time prediction using a trained ANN
- 🌐 Hosted online — no installation required

---

## 📊 Dataset
We use the **Churn_Modelling.csv** dataset, containing features such as:
- Credit Score  
- Age  
- Tenure  
- Balance  
- Number of Products  
- Has Credit Card  
- Is Active Member  
- Estimated Salary  

---

## 🧠 Model Details
- **Architecture**: 3 hidden layers with ReLU activation, 1 output layer with Sigmoid activation  
- **Loss function**: Binary Crossentropy  
- **Optimizer**: Adam  
- **Early Stopping**: Enabled to prevent overfitting  
- **Scaling**: StandardScaler  

---

## 📦 Requirements
- Python 3.10  
- TensorFlow 2.16.1  
- NumPy 1.26.4  
- pandas, scikit-learn, matplotlib, streamlit  

---

## 🙌 Acknowledgements
- [Streamlit](https://streamlit.io)
- [TensorFlow](https://www.tensorflow.org)
- Dataset source: Provided in repository
