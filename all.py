import streamlit as st
import fitz  # PyMuPDF
import easyocr
import tempfile
import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

# Gemini 2.0 Flash API setup
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={API_KEY}"

def extract_text_pymupdf(pdf_file):
    try:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text.strip()
    except Exception as e:
        return ""

def extract_text_easyocr(pdf_file):
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name

        text = ""
        doc = fitz.open(tmp_path)
        for page_num in range(len(doc)):
            pix = doc[page_num].get_pixmap()
            image_path = f"{tmp_path}_page_{page_num}.png"
            pix.save(image_path)

            result = reader.readtext(image_path, detail=0, paragraph=True)
            text += "\n".join(result) + "\n"

            os.remove(image_path)

        os.remove(tmp_path)
        return text.strip()
    except Exception as e:
        return ""

def analyze_with_gemini(text_or_base64):
    headers = {"Content-Type": "application/json"}
    
    if isinstance(text_or_base64, str) and text_or_base64.strip().startswith("data:"):
        # Assume it's base64 PDF input (future-proofing)
        prompt_text = "Analyze this medical report from PDF and generate a summary with recommendations."
        parts = [{"inline_data": {"mime_type": "application/pdf", "data": text_or_base64}}]
    else:
        # Regular text input
        prompt_text = f"Generate a clear medical summary and health recommendations (tests, diet, physical activities) from the following report:\n\n{text_or_base64}"
        parts = [{"text": prompt_text}]

    data = {
        "contents": [{
            "role": "user",
            "parts": parts
        }]
    }

    response = requests.post(GEMINI_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        res = response.json()
        return res['candidates'][0]['content']['parts'][0]['text']
    else:
        return "❌ Gemini API error: " + response.text

# Streamlit App
st.set_page_config(page_title="Medical PDF Analyzer", layout="centered")
st.title("🩺 Medical Report Analyzer")
st.markdown("Upload your **medical PDF** and get a summary with smart health recommendations.")

uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("🔍 Extracting text using PyMuPDF..."):
        pymupdf_text = extract_text_pymupdf(uploaded_file)

    if len(pymupdf_text.strip()) > 100:
        st.success("✅ Text extracted using PyMuPDF.")
        final_text = pymupdf_text
    else:
        st.warning("⚠️ PyMuPDF failed or found little text. Trying EasyOCR...")
        uploaded_file.seek(0)
        with st.spinner("🧠 Extracting text using OCR..."):
            easyocr_text = extract_text_easyocr(uploaded_file)

        if len(easyocr_text.strip()) > 100:
            st.success("✅ Text extracted using EasyOCR.")
            final_text = easyocr_text
        else:
            st.error("❌ OCR also failed. Sending PDF to Gemini directly.")
            uploaded_file.seek(0)
            b64_pdf = uploaded_file.read().encode("base64")
            final_text = "data:application/pdf;base64," + b64_pdf.decode()

    if st.button("📊 Generate AI Summary & Recommendations"):
        with st.spinner("🧠 Gemini is analyzing..."):
            result = analyze_with_gemini(final_text)
        st.subheader("📄 Medical Summary & Recommendations")
        st.markdown(result)
