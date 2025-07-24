import streamlit as st
import fitz  # PyMuPDF
import os

st.title("PDF Text Extractor using PyMuPDF")

# --- Function to extract text ---
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        all_text += f"--- Page {page_num + 1} ---\n{text}\n\n"
    doc.close()
    return all_text

# --- Option 1: User uploads PDF file ---
st.header("Upload a PDF File")
uploaded_file = st.file_uploader("Choose a PDF file to upload", type="pdf")

if uploaded_file is not None:
    with open("uploaded_temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("File uploaded successfully!")

    # Extract text from uploaded PDF
    extracted_text = extract_pdf_text("uploaded_temp.pdf")

    st.subheader("Extracted Text from Uploaded PDF")
    st.text_area("PDF Content", extracted_text, height=400)

    # Option to download extracted text
    st.download_button("Download Extracted Text", extracted_text, file_name="extracted_text.txt")


# --- Option 2: User inputs PDF path ---
st.header("Extract Text by Giving PDF Path")
pdf_path_input = st.text_input("Enter PDF file path (e.g. /home/user/sample.pdf)")

if pdf_path_input:
    if os.path.exists(pdf_path_input):
        extracted_text = extract_pdf_text(pdf_path_input)
        st.subheader("Extracted Text from Input Path PDF")
        st.text_area("PDF Content", extracted_text, height=400)

        # Option to download extracted text
        st.download_button("Download Extracted Text", extracted_text, file_name="extracted_text.txt")
    else:
        st.error("File not found. Please enter a valid path.")
