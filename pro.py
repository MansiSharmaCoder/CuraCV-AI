import streamlit as st
import fitz  # PyMuPDF
import base64
import os
import requests
from dotenv import load_dotenv

# Load Gemini API key
load_dotenv()
API_KEY = os.getenv("API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# ------------------ PDF Processing Functions ------------------

def extract_text_and_images(file):
    text = ""
    images = []
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_img = doc.extract_image(xref)
                    image_bytes = base_img["image"]
                    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                    mime_type = base_img["ext"]
                    images.append({
                        "mime_type": f"image/{mime_type}",
                        "data": image_base64
                    })
    except Exception as e:
        st.error(f"❌ Error extracting: {e}")
    return text.strip(), images

def call_gemini_with_text_and_images(text, images):
    parts = []
    if text:
        parts.append({"text": f"""You are a medical expert AI assistant. Analyze the following medical report carefully.

Your task is to:

1. Provide a clear and accurate summary of the key medical findings.
2. Highlight any diagnosed conditions, abnormal results, or observations.
3. Recommend appropriate follow-up medical tests, if needed.
4. Suggest a suitable diet plan based on the report findings.
5. Recommend general physical activities or exercises suitable for the patient.

Use simple and understandable language. Avoid medical jargon unless necessary. Keep the summary concise, informative, and actionable.

Medical Report:\n{text}"""})
    for img in images:
        parts.append({
            "inline_data": {
                "mime_type": img["mime_type"],
                "data": img["data"]
            }
        })

    body = {
        "contents": [{"role": "user", "parts": parts}]
    }

    response = requests.post(GEMINI_URL, headers={"Content-Type": "application/json"}, json=body)
    if response.ok:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f"❌ Gemini Error: {response.text}"

def call_gemini_with_pdf(pdf_file):
    b64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")
    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "application/pdf",
                        "data": b64_pdf
                    }
                },
                {
                    "text": "This is a medical report in PDF. Please extract relevant information and give a short and accurate summary with useful recommendations for health tests, diet, and exercise."
                }
            ]
        }]
    }

    response = requests.post(GEMINI_URL, headers={"Content-Type": "application/json"}, json=body)
    if response.ok:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f"❌ Gemini Error: {response.text}"

# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="🧠 Medical PDF Analyzer", layout="centered")
st.title("🩺 Medical Report Analyzer")
st.markdown("Upload your **medical PDF report** and get an AI-generated summary and recommendations.")

# File Upload UI
uploaded_file = st.file_uploader("📄 Upload PDF Report", type=["pdf"])

# Show instructions if no file uploaded yet
if not uploaded_file:
    st.info("📥 Please upload a medical PDF file to begin analysis.")
    st.stop()

# Process uploaded file
uploaded_file.seek(0)
with st.spinner("🔍 Extracting text and images..."):
    text, images = extract_text_and_images(uploaded_file)

uploaded_file.seek(0)

# AI Summary if text/images found
if len(text) > 100 or len(images) > 0:
    st.success("✅ Text and/or images successfully extracted from PDF.")
    if st.button("⚡ Get AI Summary"):
        with st.spinner("🧠 Gemini is analyzing the extracted content..."):
            summary = call_gemini_with_text_and_images(text, images)
        st.subheader("📋 AI Summary & Recommendations")
        st.markdown(summary)
else:
    st.warning("⚠️ Not enough content extracted. Sending full PDF to Gemini for direct analysis.")
    if st.button("📨 Send Full PDF to Gemini"):
        with st.spinner("🔎 Gemini is analyzing the full PDF..."):
            summary = call_gemini_with_pdf(uploaded_file)
        st.subheader("📋 AI Summary & Recommendations")
        st.markdown(summary)
