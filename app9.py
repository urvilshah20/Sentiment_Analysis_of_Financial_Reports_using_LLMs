from transformers import RobertaTokenizer, pipeline
import streamlit as st
import pdfplumber

# Load the sentiment analysis model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model_name)

def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)

def analyze_sentiment_in_chunks(text):
    label_map = {'LABEL_0': 'Negative', 'LABEL_1': 'Positive', 'LABEL_2': 'Neutral'}
    chunk_size = 500
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Initialize counters
    sentiment_scores = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    total_confidence = 0

    # Analyze each chunk
    for chunk in text_chunks:
        if chunk.strip() == "":
            continue
        results = classifier(chunk)
        for result in results:
            sentiment = label_map.get(result['label'], 'Unknown')
            confidence = result['score']
            sentiment_scores[sentiment] += confidence
            total_confidence += confidence

    # Calculate weighted average sentiment
    overall_sentiment = max(sentiment_scores, key=sentiment_scores.get)
    return overall_sentiment, sentiment_scores, total_confidence

st.title('Document Sentiment Analysis with RoBERTa')
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    extracted_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Text", extracted_text, height=250)
    if st.button('Analyze Sentiment'):
        overall_sentiment, scores, total_conf = analyze_sentiment_in_chunks(extracted_text)
        st.write(f"Overall Sentiment: {overall_sentiment}")
        st.write(f"Details: {scores}")
else:
    st.write("Upload a PDF to extract and analyze text.")
