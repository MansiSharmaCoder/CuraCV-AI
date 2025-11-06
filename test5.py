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
    if index and vector_data:
        faiss.write_index(index, index_path)
        with open(metadata_path, "w") as f:
            json.dump(vector_data, f)
    else:
        st.warning("⚠️ Nothing to save: FAISS index or vector data is empty.")

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
    try:
        response = requests.post(GEMINI_URL, headers={"Content-Type": "application/json"}, json=body)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"❌ Gemini error: {str(e)}\nResponse: {response.text if 'response' in locals() else 'No response'}"

def call_gemini_with_text_and_images(text, images):
    parts = []

    if not text and not images:
        return "⚠️ No text or images provided. Please upload a valid medical report."

    # Construct dynamic prompt
    if not text and images:
        prompt = """
🔍 You are a compassionate AI Medical Assistant with expertise in both diagnostic imaging and internal medicine.

Your role is to analyze the attached medical report and/or diagnostic images (e.g., blood reports, X-rays, MRIs, CT scans) and provide an easy-to-understand, supportive interpretation of the findings.
🔧 Your Tasks:
1. Summarize Key Medical Findings

    Clearly explain the most important observations from the report and/or image (e.g., abnormal blood values, imaging results, or vitals).

2. Identify Abnormalities or Diagnosed Conditions

    Point out any critical findings or abnormalities (e.g., Vitamin D deficiency, anemia, fractures, lesions, etc.).

    Briefly explain what these conditions may imply in simple terms.

3. Interpret Diagnostic Images (if provided)

    Describe visible abnormalities or clinical signs from any attached imaging (e.g., X-ray, MRI, CT).

    Suggest possible medical interpretations or related conditions.

4. Recommend Follow-Up Actions

    Propose further medical tests, specialist consultations, or monitoring steps that may be appropriate based on the findings.

5. Suggest a Personalized Diet Plan

    Recommend a basic diet suited to the findings (e.g., iron-rich for anemia, low-sodium for hypertension, etc.).

6. Recommend Suitable Physical Activities

    Provide safe and appropriate exercise suggestions aligned with the patient’s current health condition.

🧭 Guidelines:

    Use clear and simple language that a non-medical person can easily understand.

    Avoid complex medical jargon. If necessary, explain terms briefly.

    Present your analysis using headings or bullet points for better readability.

    Maintain a tone that is informative, professional, and empathetic.

    If any information cannot be determined from the report or image, simply state: "Not found in the report/image."

Medical Input:
"""
    elif text and not images:
        prompt = f"""
You are a highly knowledgeable and empathetic AI medical assistant. Your role is to analyze the provided medical report thoroughly and generate an insightful, easy-to-understand health summary for the patient.

Carefully review the report and perform the following tasks:

    Summarize Key Findings: Provide a concise and clear summary of the main medical findings. Focus on results related to major organs, blood tests, imaging, or other diagnostic observations.

    Highlight Abnormalities: List any abnormal results, diagnosed conditions, or unusual patterns observed in the report.

    Recommend Follow-up Actions: Suggest appropriate follow-up medical tests, checkups, or referrals (e.g., cardiologist, endocrinologist) if necessary.

    Suggest a Diet Plan: Recommend a personalized diet plan based on the report findings. Mention specific food groups to include or avoid. Keep in mind any chronic conditions if mentioned.

    Propose Physical Activities: Provide suitable exercise or physical activity suggestions tailored to the patient’s condition, age (if known), and general health.

Guidelines:

    Use simple, non-technical language understandable to a general audience.

    Present the output in clearly labeled sections or bullet points.

    If any important information is missing from the report, clearly note it and explain what else would be needed for better recommendations.

    Be professional, supportive, and informative. Avoid creating anxiety—frame suggestions as supportive advice.

Medical Report to Analyze:
{text}
"""
    else:  # text and images
        prompt = f"""
You are a highly skilled AI medical assistant trained in analyzing diagnostic reports and medical images. Carefully analyze the text-based medical report and any attached diagnostic image(s) (e.g., scans, lab charts, x-rays).

Your task is to:

    Summarize Key Medical Findings

        Extract and clearly summarize the main results from both the report and the images.

        Highlight both normal and abnormal parameters, using reference ranges if available.

    Identify Medical Concerns

        Point out any abnormal findings, potential diagnoses, and health risks.

        Use simple, patient-friendly terms unless medical terms are absolutely necessary (with brief explanations).

    Recommend Follow-Up Actions

        Suggest further medical tests, screenings, or specialist consultations, if applicable.

        Mention why the follow-up is recommended.

    Provide Personalized Lifestyle Suggestions

        Recommend a general diet plan aligned with the findings (e.g., low sugar, heart-healthy, liver support).

        Suggest physical activities or exercises suitable for the patient’s condition.

Important Guidelines:

    Use clear, concise, and understandable language.

    Avoid excessive medical jargon.

    Format the response using bullet points or sections for better readability.

    Focus on being actionable and patient-friendly.

Input Medical Report and Images:
{text}
"""

    parts.append({"text": prompt})

    for img in images:
        parts.append({
            "inline_data": {
                "mime_type": img["mime_type"],
                "data": img["data"]
            }
        })

    try:
        response = requests.post(GEMINI_URL, headers={"Content-Type": "application/json"}, json={"contents": [{"role": "user", "parts": parts}]})
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"❌ Gemini parsing error: {str(e)}\nRaw response: {response.text if 'response' in locals() else 'No response'}"

def translate_text(text, target_language_name):
    if target_language_name.lower() == "english":
        return text
    prompt = f"Translate the following medical content into {target_language_name}:\n\n{text}"
    return call_gemini(prompt)

def store_to_vector_db(text, summary, source="report"):
    try:
        embedding = model.encode([summary], normalize_embeddings=True)[0].astype('float32')
        index.add(np.array([embedding]))
        vector_data.append({
            "type": source,
            "text": text,
            "summary": summary
        })
        save_vector_db()
    except Exception as e:
        st.error(f"❌ Error storing to vector DB: {e}")

def search_vector_db(query, k=5):
    try:
        q_vector = model.encode([query], normalize_embeddings=True)[0].astype('float32')
        D, I = index.search(np.array([q_vector]), k)
        results = []
        if I is not None and len(I) > 0:
            for i in I[0]:
                if 0 <= i < len(vector_data):
                    results.append(vector_data[i])
        return results
    except Exception as e:
        st.error(f"❌ Search error: {e}")
        return []


# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="🧠 Medical Report Analyzer", layout="wide")
st.title("🩺 Medical PDF Analyzer + Chatbot")
st.markdown("Upload a **medical PDF report**, get a smart **AI-generated summary**, and ask **follow-up questions**.")

# ----------- Session State Setup -----------
if "summary_output" not in st.session_state:
    st.session_state.summary_output = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ SIDEBAR ------------------

st.sidebar.header("📂 Upload & Analyze")
uploaded_file = st.sidebar.file_uploader("Upload medical report (PDF)", type=["pdf"])

# ✅ Language selector always visible
language_options = ["English", "Hindi", "Marathi", "Gujarati", "Tamil", "Telugu", "Bengali", "Punjabi", "Kannada", "Urdu"]
selected_language = st.sidebar.selectbox("🌐 Choose language for output:", language_options)

analyze_btn = False
if uploaded_file:
    with st.spinner("🔍 Extracting text and images..."):
        text, images = extract_text_and_images(uploaded_file)
        print(len(text.strip()))
        print("----")
        print(len(images))

    if len(text.strip().split()) >= 0 or images:
        st.sidebar.success("✅ PDF processed.")
        analyze_btn = st.sidebar.button("⚡Your Health in a Glance")
    else:
        st.sidebar.warning("⚠️ Not enough content found in the report.")

# ------------------ Main Area ------------------

if analyze_btn:
    with st.spinner("🤖 Analyzing report with Gemini..."):
        summary = call_gemini_with_text_and_images(text, images)
        st.session_state.summary_output = summary
        store_to_vector_db(text, summary, source="report")

if st.session_state.summary_output:
    st.subheader("📋 AI Summary & Recommendations")
    translated_summary = translate_text(st.session_state.summary_output, selected_language)
    st.markdown(translated_summary)

# ----------- Chatbot Section -----------

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

You are a highly knowledgeable and responsible medical AI assistant. Use the information from the provided medical report to answer user queries accurately.
✅ General Guidelines:

    If the answer is clearly stated in the report, provide a direct, clear response based on the report content.

    If the answer is not found in the report, respond with:
    "Not found in the report."

    If the question relates generally to a condition mentioned in the report (e.g., Vitamin D deficiency, leukopenia), and the report does not give a detailed explanation:

        Use your general medical knowledge to explain the condition briefly in simple, layperson-friendly language.

        Clearly mention that this is a general explanation, not report-specific.

    If the question is unrelated to the report or unrelated to medical topics, respond only with:
    "Not found in the report."

⚠️ For Medicine or Supplement Recommendations:

If the user asks for medicine or supplement suggestions (e.g., for treating a condition or deficiency):

    First, always display this warning message in bold:

    ⚠️ Please consult your doctor before taking any medicine or supplement. This is just a general suggestion and not a substitute for professional medical advice.

    Then, only if it's medically appropriate, provide a commonly suggested over-the-counter medicine or supplement.

        Keep it general and safe.

        Mention it only if it is directly related to the user’s medical report or condition.

🧠 Language and Style:

    Use clear, simple, and understandable language.

    Avoid complex medical jargon unless necessary, and explain any technical term used.
"""
        with st.spinner("🤖 Generating answer..."):
            response = call_gemini(prompt)

        translated_response = translate_text(response, selected_language)
        st.markdown("### 🧠 Gemini's Answer")
        st.markdown(translated_response)

        st.session_state.chat_history.append((user_query, response))
        store_to_vector_db(user_query, response, source="chat")

    else:
        st.warning("No relevant report found to answer this question.")

# ----------- Chat History Viewer -----------

if st.session_state.chat_history:
    with st.expander("🕘 Show Previous Chat History"):
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
            st.markdown(f"**Q{i}:** {q}")
            translated_a = translate_text(a, selected_language)
            st.markdown(f"**A{i}:** {translated_a}")
            st.markdown("---")
