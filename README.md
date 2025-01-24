# Sentiment_Analysis_of_Financial_Reports_using_LLMs

**Financial Report Sentiment Analysis using LLMs:**

This project is developing a sentiment analysis tool that leverages GPT-3.5 architecture, and advanced NLP techniques to analyze corporate earnings reports and financial documents. The tool provides instant sentiment insights to assist financial professionals in making informed decisions during after-market trading.

**Key Features:**

- Context-aware sentiment analysis using n-gram processing and custom phrase weighting
- Integration with GPT-3.5 for nuanced understanding of complex financial language
- Batch processing capability for multiple documents
- User-friendly interface with adjustable parameters and downloadable results
- Specialized handling of financial terminology and industry-specific jargon
  
**Technical Implementation:**

- **NLP Pipeline:-**
  - Custom text preprocessing using NLTK
  - N-gram analysis for context-aware interpretation
  - Optional phrase-weighting system for financial terms
  - PDF text extraction using PyPDF2 and pdfplumber

- **Model Architecture:-**
  - Integration with GPT-3.5 API
  - Streamlit-based interface
  - Sentiment scoring algorithm based on positive/negative term frequency
  - Support for batch processing multiple documents

**Performance Analysis:**

- Tested against actual market outcomes
- Successfully predicted sentiment alignment with stock price movements
- Outperformed traditional sentiment analysis tools in financial contexts
- Demonstrated effectiveness in handling complex financial language

**Results:**

- Outperformed traditional sentiment analysis tools (TextBlob, BERT, RoBERTa) in financial context
- Successfully predicted market sentiment aligned with stock price movements
- Demonstrated robust performance in processing diverse financial documents
- Provided actionable insights for after-market trading decisions

**Collaboration:**

Developed in partnership with Citibank Innovation Lab as part of MSc Business Analytics dissertation project at Bayes Business School, London (formerly Cass).
