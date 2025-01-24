import streamlit as st
from google.cloud import language_v1
import pdfplumber

# Function to analyze sentiment using Google Cloud API
def analyze_sentiment(text_content):
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=text_content, type_=language_v1.Document.Type.PLAIN_TEXT)
    response = client.analyze_sentiment(document=document)
    sentiment = response.document_sentiment
    return sentiment.score, sentiment.magnitude

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
    return " ".join(pages) if pages else ""

# Function to interpret the sentiment score into labels
def interpret_sentiment(score):
    if score > 0.25:
        return "Positive"
    elif score < -0.25:
        return "Negative"
    else:
        return "Neutral"

# Streamlit interface
st.title('PDF Sentiment Analysis using Google Cloud Natural Language API')

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    if text:
        st.write("Extracted Text:")
        st.write(text)
        if st.button('Analyze Sentiment'):
            score, magnitude = analyze_sentiment(text)
            sentiment = interpret_sentiment(score)
            st.success(f"Sentiment score: {score}, Magnitude: {magnitude}, Interpreted Sentiment: {sentiment}")
    else:
        st.error("No text could be extracted from the PDF.")
