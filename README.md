# Sentiment_Analysis_of_Financial_Reports_using_LLMs

### **Financial Report Sentiment Analysis using LLMs**

This project is developing a sentiment analysis tool that leverages GPT-3.5 architecture, and advanced NLP techniques to analyze corporate earnings reports and financial documents. The tool provides instant sentiment insights to assist financial professionals in making informed decisions during after-market trading.

### **Key Features**

- Context-aware sentiment analysis using n-gram processing and custom phrase weighting
- Integration with GPT-3.5 for nuanced understanding of complex financial language
- Batch processing capability for multiple documents
- User-friendly interface with adjustable parameters and downloadable results
- Specialized handling of financial terminology and industry-specific jargon
  
### **Technical Implementation**

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

### **Performance Analysis**

- Tested against actual market outcomes
- Successfully predicted sentiment alignment with stock price movements
- Outperformed traditional sentiment analysis tools in financial contexts
- Demonstrated effectiveness in handling complex financial language

### **Results**

- Outperformed traditional sentiment analysis tools (TextBlob, BERT, RoBERTa) in financial context
- Successfully predicted market sentiment aligned with stock price movements
- Demonstrated robust performance in processing diverse financial documents
- Provided actionable insights for after-market trading decisions

### **Getting started**

**Project Structure:**

The repository contains the following key files:

- `Final_code.py`: Main sentiment analysis implementation using GPT-3.5
- Comparison Model Implementations:
  
  - `app7.py`: TextBlob implementation
  - `app8.py`: BERT implementation
  - `app9.py`: RoBERTa implementation
  - `app10.py`: Google Cloud API implementation
  - `app11.py`: AWS Comprehend implementation
- Supporting Files:

  - `config.toml`: Configuration settings
  - `.env`: Environment variables (API keys)
  - `Loughran-McDonald_MasterDictionary_1993-2021.csv`: Financial terms dictionary

**Prerequisites:**

Required Python libraries:

```
streamlit
openai
pandas
nltk
textblob
transformers
torch
google-cloud-language
boto3
pdfplumber
PyPDF2
```

**Installation & Setup:**

1) Clone the repository or download all files
2) Install required dependencies:
```
pip install streamlit openai pandas nltk textblob transformers torch google-cloud-language boto3 pdfplumber PyPDF2
```
3) Set up your OPENAI API key in the `.env` file


### **Usage**

**Main Implementation:**

Place all files in the same directory
Run the main sentiment analysis code:
```
python Final_code.py
```

**Comparison Models:**
Each comparison model can be run independently:

- TextBlob Analysis:
```
python app7.py
```
- BERT Analysis:
```
python app8.py
```
- RoBERTa Analysis:
```
python app9.py
```
- Google Cloud Analysis:
```
python app10.py
```
- AWS Comprehend Analysis:
```
python app11.py
```

**Important Notes**

- Ensure all files remain in the same directory when running any of the codes
- Configure API keys properly in the `.env` file before running cloud-based implementations
- Make sure you have sufficient API credits/permissions for cloud services
