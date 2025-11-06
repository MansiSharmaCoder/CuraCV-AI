import streamlit as st
import fitz  # PyMuPDF
import base64
import os
import requests
import numpy as np
import faiss
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ------------------ Setup & Config ------------------

load_dotenv()
API_KEY = os.getenv("API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={API_KEY}"

model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index_path = "medical_index.faiss"
metadata_path = "vector_data.json"

# ------------------ Load or Init Vector DB ------------------

if os.path.exists(index_path):
    index = faiss.read_index(index_path)
else:
    index = faiss.IndexFlatIP(dimension)

vector_data = []
if os.path.exists(metadata_path):
    with open(metadata_path, "r") as f:
        vector_data = json.load(f)

# ------------------ Utility Functions ------------------

def save_vector_db():
    faiss.write_index(index, index_path)
    with open(metadata_path, "w") as f:
        json.dump(vector_data, f)

# ------------------ PDF Extraction ------------------

def extract_text_and_images(file):
    text, images = "", []
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
        st.error(f"❌ Error extracting PDF: {e}")
    return text.strip(), images

# ------------------ Gemini API ------------------

def call_gemini(prompt):
    body = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    }
    response = requests.post(GEMINI_URL, headers={"Content-Type": "application/json"}, json=body)
    try:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"❌ Gemini parsing error: {str(e)}\nRaw response: {response.text}"

def call_gemini_with_text_and_images(text, images):
    parts = []
    if text:
        parts.append({"text": f"""
You are a medical expert AI assistant. Analyze the following medical report carefully.

1. Summary of key findings  
2. Diagnosis / Abnormalities  
3. Recommended tests  
4. Diet suggestions  
5. Physical activities

Medical Report:
{text}
"""})
    for img in images:
        parts.append({
            "inline_data": {
                "mime_type": img["mime_type"],
                "data": img["data"]
            }
        })

    body = {"contents": [{"role": "user", "parts": parts}]}
    response = requests.post(GEMINI_URL, headers={"Content-Type": "application/json"}, json=body)
    try:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"❌ Gemini parsing error: {str(e)}\nRaw response: {response.text}"

# ------------------ FAISS Functions ------------------

def store_to_vector_db(text, summary, source="report"):
    embedding = model.encode([summary], normalize_embeddings=True)[0].astype('float32')
    index.add(np.array([embedding]))
    vector_data.append({
        "type": source,
        "text": text,
        "summary": summary
    })
    save_vector_db()

def search_vector_db(query, k=2):
    q_vector = model.encode([query], normalize_embeddings=True)[0].astype('float32')
    D, I = index.search(np.array([q_vector]), k)
    results = []
    if I is not None and len(I) > 0:
        for i in I[0]:
            if 0 <= i < len(vector_data):
                results.append(vector_data[i])
    return results

# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="🧠 Medical Report Analyzer", layout="centered")
st.title("🩺 Medical PDF Analyzer + Chatbot")

st.markdown("Upload your **medical PDF report**, get AI summary, and ask health-related questions.")

# ----------- Step 1: Upload PDF -----------

uploaded_file = st.file_uploader("📄 Upload medical report (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("🔍 Extracting PDF..."):
        text, images = extract_text_and_images(uploaded_file)

    if len(text.strip().split()) > 50 or images:
        st.success("✅ Extraction complete.")
        if st.button("⚡ Analyze Report"):
            with st.spinner("🤖 Gemini analyzing..."):
                summary = call_gemini_with_text_and_images(text, images)
            st.subheader("📋 Gemini Summary & Recommendations")
            st.markdown(summary)
            store_to_vector_db(text, summary, source="report")
    else:
        st.warning("⚠️ Not enough content found in the report.")
else:
    st.info("📥 Upload a medical PDF to begin.")

# ----------- Step 2: Chatbot -----------

st.divider()
st.subheader("💬 Ask Questions About Reports")

with st.form("chat_form"):
    user_query = st.text_input("Ask a health-related question (e.g., What tests were recommended?)")
    submit_btn = st.form_submit_button("Submit")

if submit_btn and user_query:
    with st.spinner("🔍 Searching relevant report..."):
        matched_reports = search_vector_db(user_query)

    if matched_reports:
        context = "\n\n---\n\n".join([r["summary"] for r in matched_reports])
        prompt = f"""
You are a helpful medical AI.

Here is a patient's summarized report:

\"\"\"{context}\"\"\"

Now answer this question clearly and accurately:

\"\"\"{user_query}\"\"\"

If the answer isn't found in the report, say "Not found in the report."
"""
        with st.spinner("🤖 Generating answer..."):
            response = call_gemini(prompt)

        st.markdown("### 🧠 Gemini's Answer")
        st.markdown(response)

        # Store Q&A in memory
        store_to_vector_db(user_query, response, source="chat")
    else:
        st.warning("No relevant report found to answer this question.")
