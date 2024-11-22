import streamlit as st
import torch
import pickle
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from docx import Document
from PyPDF2 import PdfReader

# Load pre-trained model, tokenizer, and label encoder
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=70  # Update this based on your number of labels
    )
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

@st.cache_resource
def load_label_encoder():
    # Make sure to replace the file path with the actual one for the label encoder
    with open('label_encoder.pkl', 'rb') as file:
        return pickle.load(file)

# Load model, tokenizer, and label encoder
model = load_model()
tokenizer = load_tokenizer()
label_encoder = load_label_encoder()

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    return '\n'.join([p.text for p in doc.paragraphs])

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return '\n'.join([page.extract_text() for page in reader.pages])

# App Header
st.title("Job Classification App")
st.write("Predict job categories based on input text using a fine-tuned BERT model.")

# File uploader
uploaded_file = st.file_uploader("Upload a Word or PDF file", type=["docx", "pdf"])

if uploaded_file is not None:
    # Extract text from the uploaded file
    if uploaded_file.name.endswith(".docx"):
        text = extract_text_from_docx(uploaded_file)
    elif uploaded_file.name.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file)

    st.text_area("Extracted Text", text, height=300)  # Show the extracted text

    # Tokenize and Predict
    if st.button("Predict"):
        if text.strip():
            # Tokenizing the extracted text
            inputs = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()

            # Display Prediction
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
            st.success(f"Predicted Job Category: **{predicted_label}**")

            # Display Probabilities
            st.write("Prediction Probabilities:")
            prob_df = pd.DataFrame({
                'Category': label_encoder.classes_,
                'Probability': probs.squeeze().tolist()
            })

            # Sorting the probabilities in descending order
            prob_df = prob_df.sort_values(by='Probability', ascending=False)

            # Display as a dataframe
            st.dataframe(prob_df)

            # Plotting the probabilities as a bar chart
            fig, ax = plt.subplots()
            sns.barplot(x='Category', y='Probability', data=prob_df.head(10), ax=ax)
            ax.set_title("Top 10 Prediction classes")
            ax.set_xlabel("Category")
            ax.set_ylabel("Probability")
            st.pyplot(fig)
        else:
            st.warning("No text to classify.")
