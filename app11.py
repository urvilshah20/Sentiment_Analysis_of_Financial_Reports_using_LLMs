import streamlit as st
import boto3
import PyPDF2
from io import BytesIO

def extract_text_from_pdf(file_data):
    """ Extract text from a PDF file. """
    with BytesIO(file_data.getvalue()) as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Check if text extraction was successful
                text += page_text + ' '
    return text

def analyze_sentiment(text):
    """ Analyze sentiment of the provided text using AWS Comprehend. """
    comprehend = boto3.client(
        service_name='comprehend',
        region_name='us-east-1',
        aws_access_key_id='********************',  #add your own aws access key id
        aws_secret_access_key='****************************************'  #add your own secret access key
    )

    # Function to send a chunk to Comprehend
    def get_sentiment(chunk):
        return comprehend.detect_sentiment(Text=chunk, LanguageCode='en')

    # Split the text into chunks that do not exceed the byte limit
    chunks = []
    current_chunk = ''
    words = text.split()
    for word in words:
        if len((current_chunk + word).encode('utf-8')) + 1 <= 5000:
            current_chunk += word + ' '
        else:
            chunks.append(current_chunk)
            current_chunk = word + ' '
    if current_chunk:
        chunks.append(current_chunk)  # Add the last chunk

    # Analyze each chunk and collect results
    sentiments = [get_sentiment(chunk) for chunk in chunks]
    
    # Aggregate results: example approach is averaging sentiment scores
    avg_sentiment = {'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Mixed': 0}
    for sentiment in sentiments:
        for key in avg_sentiment:
            avg_sentiment[key] += sentiment['SentimentScore'][key]
    for key in avg_sentiment:
        avg_sentiment[key] /= len(sentiments)  # Normalize by number of chunks

    # Determine the dominant sentiment
    dominant_sentiment = max(avg_sentiment, key=avg_sentiment.get)
    return dominant_sentiment, avg_sentiment[dominant_sentiment]

st.title('PDF Sentiment Analysis using Amazon API')
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")
if uploaded_file is not None:
    text_content = extract_text_from_pdf(uploaded_file)
    if st.button('Analyze Sentiment'):
        dominant_sentiment, score = analyze_sentiment(text_content)
        st.write(f"The dominant sentiment is {dominant_sentiment} with a score of {score:.2%}")
