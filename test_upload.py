import streamlit as st

def main():
    st.title("File Upload Test")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        st.write(f"File name: {uploaded_file.name}")
        st.write(f"File size: {uploaded_file.size} bytes")
        st.write("File uploaded successfully!")

if _name_ == "_main_":
    main()