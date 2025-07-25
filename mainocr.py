import streamlit as st
import pathlib
import os
import fitz  # PyMuPDF for PDF to image conversion
import easyocr
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Securely load API key from .env
API_KEY = os.getenv("API_KEY")

# Configure Google Gemini client
client = genai.Client(api_key=API_KEY)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if CUDA is installed

# Streamlit app layout
st.title("🩺 Medical PDF Summariser with EasyOCR")

uploaded_file = st.file_uploader("Upload a medical PDF file", type=["pdf"])

if uploaded_file:
    pdf_path = f"temp_{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded {uploaded_file.name}")

    if st.button("Extract Text with EasyOCR and Generate Summary"):
        with st.spinner("Extracting text using EasyOCR..."):

            # Extract images from PDF pages
            pdf_doc = fitz.open(pdf_path)
            full_text = ""

            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)
                pix = page.get_pixmap()

                image_path = f"page_{page_num}.png"
                pix.save(image_path)

                # OCR on each page image
                results = reader.readtext(image_path, detail=0)
                page_text = "\n".join(results)
                full_text += f"Page {page_num + 1}:\n{page_text}\n\n"

                os.remove(image_path)  # clean up temporary image

            pdf_doc.close()

            st.subheader("🔍 Extracted Text (OCR)")
            st.text_area("OCR Output", value=full_text, height=300)

        with st.spinner("Generating medical summary using Gemini..."):
            prompt = (
                "Generate a concise and clear medical summary with recommended "
                "medical tests, diet, basic exercises or physical activities for this text:\n"
                f"{full_text}"
            )

            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[
                        types.Part.from_text(prompt)
                    ]
                )

                summary = response.text
            except Exception as e:
                summary = f"Error: {e}"

            st.subheader("📝 Medical Summary")
            st.write(summary)

    # Optionally delete temporary PDF after use
    os.remove(pdf_path)
