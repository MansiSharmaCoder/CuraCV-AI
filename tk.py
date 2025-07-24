# save this as app.py and run: streamlit run app.py

import streamlit as st
import fitz  # PyMuPDF
import nltk

# Download NLTK punkt tokenizer (first time)
nltk.download('punkt')

from nltk.tokenize import word_tokenize

st.title("📄 PDF Token Counter using NLP")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    # Read PDF
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""

    for page in pdf:
        text += page.get_text()

    # Display extracted text
    st.subheader("Extracted Text:")
    st.write(text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Show total number of tokens
    st.subheader("Token Count:")
    st.write(f"Total tokens found: **{len(tokens)}**")

    # Optionally, display the tokens
    if st.checkbox("Show tokens"):
        st.write(tokens)
else:
    st.info("Please upload a PDF file to proceed.")
