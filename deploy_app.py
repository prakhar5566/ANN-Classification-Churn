import streamlit as st 
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf

# Page Config
st.set_page_config(page_title="Customer Churn Prediction ANN", layout="centered")

st.title("📊 Customer Churn Prediction using ANN")
st.write("Upload customer data and predict churn probability using a trained Artificial Neural Network.")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Churn_Modelling.csv")
    return df
df = load_data()
st.subheader("Sample of Dataset")
st.write(df.head())

# Download full dataset
csv_data = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Full Dataset",
    data=csv_data,
    file_name='Churn_Modelling.csv',
    mime='text/csv'
)

# Load Model, Scaler & History
@st.cache_resource
def load_model_scaler_history():
    model = tf.keras.models.load_model("model.h5")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    history = pickle.load(open("model_history.pkl", "rb"))  # Ensure you saved this after training
    return model, scaler, history

model, sc, model_history = load_model_scaler_history()

# Evaluate Model
X = df.iloc[:, 3:13].drop(df.columns[4:7], axis=1)
y = df.iloc[:, 13]
X_scaled = sc.transform(X)

y_pred = (model.predict(X_scaled) > 0.5)
cm = confusion_matrix(y, y_pred)
acc = accuracy_score(y, y_pred)

# st.subheader("Confusion Matrix")
# st.write(pd.DataFrame(cm, columns=['Predicted No', 'Predicted Yes'], index=['Actual No', 'Actual Yes']))

# Accuracy Plot
st.subheader("📈 Model Accuracy Over Epochs")
fig_acc, ax_acc = plt.subplots()
ax_acc.plot(model_history['accuracy'])
ax_acc.plot(model_history['val_accuracy'])
ax_acc.set_title('Model Accuracy')
ax_acc.set_ylabel('Accuracy')
ax_acc.set_xlabel('Epoch')
ax_acc.legend(['Train', 'Validation'], loc='upper left')
st.pyplot(fig_acc)

# # Loss Plot
# st.subheader("📉 Model Loss Over Epochs")
# fig_loss, ax_loss = plt.subplots()
# ax_loss.plot(model_history['loss'])
# ax_loss.plot(model_history['val_loss'])
# ax_loss.set_title('Model Loss')
# ax_loss.set_ylabel('Loss')
# ax_loss.set_xlabel('Epoch')
# ax_loss.legend(['Train', 'Validation'], loc='upper left')
# st.pyplot(fig_loss)

st.subheader("Accuracy Score")
st.write(f"{acc * 100:.2f}%")

# Single Prediction
st.subheader("🔍 Predict for a Single Customer")

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", value=60000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card? (1 = Yes, 0 = No)", [1, 0])
is_active_member = st.selectbox("Is Active Member? (1 = Yes, 0 = No)", [1, 0])
estimated_salary = st.number_input("Estimated Salary", value=50000.0)

if st.button("Predict Churn"):
    new_data = np.array([[credit_score, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary]])
    new_data_scaled = sc.transform(new_data)
    prediction = model.predict(new_data_scaled)
    if prediction > 0.5:
        st.error("❌ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

st.caption("Model trained on Churn_Modelling.csv dataset.")

