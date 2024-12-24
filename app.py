import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Title and description
st.title("Advanced Email Spam Classifier")
st.markdown("**Classify emails as Spam or Ham and explore advanced functionalities like batch classification and insights.**")

# Sidebar options
st.sidebar.title("Options")
option = st.sidebar.radio("Choose functionality:", ["Single Email Prediction", "Batch Prediction", "Insights"])

# Single email prediction
if option == "Single Email Prediction":
    st.subheader("Single Email Prediction")
    user_input = st.text_area("Enter the email content here:")

    if st.button("Classify"):
        if user_input.strip():
            input_data_features = vectorizer.transform([user_input])
            prediction = model.predict(input_data_features)
            confidence = model.predict_proba(input_data_features).max() * 100
            result = "Ham mail" if prediction[0] == 1 else "Spam mail"
            
            st.success(f"Prediction: {result}")
            st.info(f"Confidence: {confidence:.2f}%")
        else:
            st.error("Please enter valid email content to classify.")

# Batch prediction
elif option == "Batch Prediction":
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file with a 'Message' column", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        if 'Message' in data.columns:
            data['Prediction'] = model.predict(vectorizer.transform(data['Message']))
            data['Confidence'] = model.predict_proba(vectorizer.transform(data['Message'])).max(axis=1) * 100
            data['Prediction'] = data['Prediction'].map({0: 'Spam', 1: 'Ham'})
            st.write("Predictions:")
            st.dataframe(data[['Message', 'Prediction', 'Confidence']])
        else:
            st.error("CSV file must contain a 'Message' column.")

# Insights
elif option == "Insights":
    st.subheader("Model Insights")
    st.markdown("### Spam vs. Ham Distribution in Training Data")

    raw_mail_data = pd.read_csv("mail_data.csv")
    mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
    mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

    fig, ax = plt.subplots()
    mail_data['Category'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Category Distribution")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Spam', 'Ham'], rotation=0)
    st.pyplot(fig)