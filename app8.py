from transformers import BertTokenizer, BertForSequenceClassification, pipeline, set_seed
import streamlit as st
import pdfplumber

# Set seed for reproducibility
set_seed(42)

# Initialize tokenizer and model explicitly and only once
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Define a mapping from labels to sentiments
sentiment_map = {
    'LABEL_0': 'Neutral',
    'LABEL_1': 'Positive',
    'LABEL_2': 'Negative'
}

def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
    return "\n".join(pages)

st.title('PDF Sentiment Analysis')

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    extracted_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Text", extracted_text, height=250)

    if st.button('Analyze Sentiment'):
        results = classifier(extracted_text[:512])  # Model limit consideration
        for result in results:
            sentiment_desc = sentiment_map.get(result['label'], 'Unknown')
            st.write(f"Label: {result['label']}, Score: {result['score']:.4f}, Sentiment: {sentiment_desc}")
else:
    st.write("Upload a PDF to extract and analyze text.")
