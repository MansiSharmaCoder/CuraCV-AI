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

# ------------------ Setup ------------------

load_dotenv()
API_KEY = os.getenv("API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={API_KEY}"

model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index_path = "medical_index.faiss"
metadata_path = "vector_data.json"

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

def call_gemini(prompt):
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
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

Your task is to:

1. Provide a clear and accurate summary of the key medical findings.
2. Highlight any diagnosed conditions, abnormal results, or observations.
3. Recommend appropriate follow-up medical tests, if needed.
4. Suggest a suitable diet plan based on the report findings.
5. Recommend general physical activities or exercises suitable for the patient.

Use simple and understandable language. Avoid medical jargon unless necessary. Keep the summary concise, informative, and actionable.

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

def store_to_vector_db(text, summary, source="report"):
    embedding = model.encode([summary], normalize_embeddings=True)[0].astype('float32')
    index.add(np.array([embedding]))
    vector_data.append({
        "type": source,
        "text": text,
        "summary": summary
    })
    save_vector_db()

def search_vector_db(query, k=5):
    q_vector = model.encode([query], normalize_embeddings=True)[0].astype('float32')
    D, I = index.search(np.array([q_vector]), k)
    results = []
    if I is not None and len(I) > 0:
        for i in I[0]:
            if 0 <= i < len(vector_data):
                results.append(vector_data[i])
    return results

# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="🧠 Medical Report Analyzer", layout="wide")

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
.upload-card {
    background-color: #f9fafa;
    padding: 2rem;
    border: 2px solid #e1e5eb;
    border-radius: 16px;
    box-shadow: 4px 4px 14px rgba(0,0,0,0.05);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #ffffff;
    border: 2px dashed #0d6efd;
    border-radius: 12px;
    padding: 1.2rem;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    background-color: #eaf4ff;
    border-color: #0b5ed7;
}

/* Button hover */
button[kind="secondary"] {
    background-color: #0d6efd !important;
    color: white !important;
    border-radius: 10px;
    font-weight: 500;
    box-shadow: 0 4px 10px rgba(13, 110, 253, 0.2);
}
button[kind="secondary"]:hover {
    background-color: #0b5ed7 !important;
    box-shadow: 0 6px 14px rgba(13, 110, 253, 0.3);
}
</style>
""", unsafe_allow_html=True)

# ------------------ Layout ------------------

col1, col2 = st.columns([2, 2])  # 50%-50% layout

with col1:
    st.markdown("### 📂 Upload & Analyze")
    with st.container():
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload medical report (PDF)", type=["pdf"])
        analyze_btn = False

        if uploaded_file:
            with st.spinner("🔍 Extracting text and images..."):
                text, images = extract_text_and_images(uploaded_file)

            if len(text.strip().split()) > 50 or images:
                st.success("✅ PDF processed.")
                analyze_btn = st.button("⚡ Your Health in a Glance")
            else:
                st.warning("⚠️ Not enough content found in the report.")
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.title("🩺 Medical PDF Analyzer + Chatbot")
    st.markdown("Upload a **medical PDF report**, get a smart **AI-generated summary**, and ask **follow-up questions**.")

    if analyze_btn:
        with st.spinner("🤖 Analyzing report with Gemini..."):
            summary = call_gemini_with_text_and_images(text, images)
            st.session_state.summary_output = summary
            store_to_vector_db(text, summary, source="report")

    if st.session_state.summary_output:
        st.subheader("📋 AI Summary & Recommendations")
        st.markdown(st.session_state.summary_output)

# ------------------ Chatbot Section ------------------

st.divider()
st.subheader("💬 Ask Questions About the Report")

with st.form("chat_form"):
    user_query = st.text_input("Ask a medical question related to the report:")
    submit_btn = st.form_submit_button("Submit")

if submit_btn and user_query:
    with st.spinner("🔍 Searching reports..."):
        matched_reports = search_vector_db(user_query)

    if matched_reports:
        context = "\n\n---\n\n".join([r["summary"] for r in matched_reports])
        prompt = f"""
You are a helpful medical AI.

Here is a patient's summarized report:

\"\"\"{context}\"\"\"

Now answer this question clearly and accurately:

\"\"\"{user_query}\"\"\"

Instructions:
- If the answer is directly found in the report, respond accordingly.
- If it relates to common medical conditions or lab values in the report (e.g., cholesterol, anemia), provide a general medical explanation.
- If the question is unrelated to the report or not medically relevant, respond with: "Not found in the report."
"""
        with st.spinner("🤖 Generating answer..."):
            response = call_gemini(prompt)

        st.markdown("### 🧠 Gemini's Answer")
        st.markdown(response)

        st.session_state.chat_history.append((user_query, response))
        store_to_vector_db(user_query, response, source="chat")

    else:
        st.warning("No relevant report found to answer this question.")

# ------------------ Chat History Viewer ------------------

if st.session_state.chat_history:
    with st.expander("🕘 Show Previous Chat History"):
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")
            st.markdown("---")