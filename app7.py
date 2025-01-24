import streamlit as st
import pdfplumber
from textblob import TextBlob

# Function to extract text from a PDF file
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""  # Append text or an empty string if None is returned
    return text

# Streamlit app
def main():
    st.title("PDF Sentiment Analysis Using TextBlob")
    st.write("Upload a PDF file to analyze its sentiment.")

    # File uploader allows user to add their own PDF
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner('Extracting text from PDF...'):
            text = extract_text_from_pdf(uploaded_file)

        if text:
            # Using TextBlob to analyze sentiment
            blob = TextBlob(text)
            sentiment_polarity = blob.sentiment.polarity
            sentiment_subjectivity = blob.sentiment.subjectivity

            # Display results
            st.write("Sentiment Analysis Completed!")
            st.write(f"**Polarity**: {sentiment_polarity:.2f} (Range: -1 to 1, where -1 is negative, 0 is neutral, and 1 is positive)")
            st.write(f"**Subjectivity**: {sentiment_subjectivity:.2f} (Range: 0 to 1, where 0 is objective and 1 is subjective)")

            # Interpreting the polarity score
            if sentiment_polarity > 0:
                st.success("The overall sentiment is Positive.")
            elif sentiment_polarity < 0:
                st.error("The overall sentiment is Negative.")
            else:
                st.info("The overall sentiment is Neutral.")
        else:
            st.error("No text could be extracted from the file.")

if __name__ == "__main__":
    main()
