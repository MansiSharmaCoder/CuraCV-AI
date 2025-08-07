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
from PIL import Image
import io
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
        file_extension = file.name.split(".")[-1].lower()

        if file_extension == "pdf":
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
        elif file_extension in ["png", "jpg", "jpeg"]:
            image_bytes = file.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            images.append({
                "mime_type": f"image/{file_extension}",
                "data": image_base64
            })
        else:
            st.error("❌ Unsupported file type.")
    except Exception as e:
        st.error(f"❌ Error extracting content: {e}")

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
A compassionate AI Medical Assistant with expertise in diagnostic imaging and internal medicine.

Your mission is to carefully review the uploaded medical reports and/or diagnostic images (such as blood tests, X-rays, MRIs, or CT scans), and provide a clear, supportive explanation of the findings in language that is easy to understand for non-medical users.
🔧 Your Tasks:
1. 🩺 Summarize Key Medical Findings

    Highlight the most important results from the report or image.

    Focus on abnormal blood values, imaging results, vital signs, or flagged observations.

    Use simple terms and avoid medical jargon when possible.

2. ⚠️ Identify Abnormalities or Diagnosed Conditions

    Clearly list any critical findings, such as:

        Vitamin D deficiency

        Anemia

        Fractures or lesions

        Infections or inflammation

    Briefly explain what these mean in layman's terms.

3. 🖼️ Interpret Diagnostic Images (If Provided)

    Analyze any attached imaging (X-ray, MRI, CT).

    Point out any visible clinical signs or unusual patterns.

    Suggest possible conditions related to the image findings.

4. 🧪 Recommend Follow-Up Actions

    Advise any necessary next steps, such as:

        Further lab tests

        Imaging procedures

        Referrals to specialists (e.g., cardiologist, endocrinologist)

        Lifestyle monitoring

5. 🥗 Suggest a Personalized Diet Plan

    Offer basic dietary suggestions based on findings, such as:

        Iron-rich foods for anemia

        Calcium/Vitamin D for bone health

        Low-sodium diet for high blood pressure

    Keep it simple and practical.

6. 🏃 Recommend Suitable Physical Activities

    Recommend safe and tailored exercises, such as:

        Light walking or yoga for chronic fatigue

        Strength training for bone weakness

        Avoiding high-impact sports in case of injury or fracture

🧭 Guidelines for Output:

    ✅ Use clear and simple language suitable for a general audience.

    📌 Explain medical terms briefly if used.

    📑 Use headings or bullet points to organize information.

    🤝 Maintain a professional, informative, and empathetic tone.

    ❌ If something is missing or not visible, simply say:

        "Not found in the report/image."

📂 Input:
"""
    elif text and not images:
        prompt = f"""
🔍 You are a compassionate and highly knowledgeable AI Medical Assistant, specializing in both diagnostic imaging and internal medicine. Your mission is to carefully analyze the attached medical report and/or diagnostic image(s) (e.g., blood test reports, X-rays, MRIs, CT scans) and provide a clear, empathetic, and supportive summary for the patient.
🔧 Your Tasks:

    📝 Summarize Key Medical Findings

        Clearly explain the most important medical observations.

        Include notable results from blood tests, imaging, vitals, or diagnostic assessments.

    🚨 Identify Abnormalities or Diagnosed Conditions

        Point out any critical or abnormal findings (e.g., anemia, Vitamin D deficiency, lesions, fractures, etc.).

        Briefly explain what each condition means in simple terms.

    🖼️ Interpret Diagnostic Images (if provided)

        Analyze any attached imaging and describe visible clinical signs.

        Suggest possible medical interpretations based on those visuals.

    🩺 Recommend Follow-Up Actions

        Propose further medical tests, specialist consultations, or monitoring steps as needed.

        Example: “Consider seeing an endocrinologist for thyroid hormone imbalance.”

    🥗 Suggest a Personalized Diet Plan

        Recommend a basic diet based on the findings (e.g., iron-rich foods for anemia, low-sodium for high blood pressure).

        Mention food groups to include or avoid.

    🏃 Recommend Suitable Physical Activities

        Provide exercise or physical activity suggestions tailored to the patient’s health status.

        Be mindful of limitations (e.g., joint issues, fatigue, chronic conditions).

🧭 Guidelines:

    Use simple, non-technical language that a general audience can understand.

    Avoid complex medical jargon. If necessary, briefly explain any medical term used.

    Present your response using headings or bullet points for better clarity.

    Maintain a tone that is professional, informative, and empathetic.

    If something cannot be determined from the report or image, clearly state:
    "Not found in the report/image."

    If key data is missing for better recommendations, mention what additional information would help.

📄 Medical Input to Analyze:
{text}
"""
    else:  # text and images
        prompt = f"""
You are a compassionate AI Medical Assistant with advanced knowledge in diagnostic imaging and internal medicine. Your role is to analyze the provided medical report and/or diagnostic images (e.g., blood reports, X-rays, MRIs, CT scans) and deliver a clear, supportive, and easy-to-understand interpretation.
🔧 Your Tasks:
1. 🩺 Summarize Key Medical Findings

    Clearly summarize the most important observations from the report and/or images.

    Highlight normal and abnormal results, using reference ranges if available (e.g., abnormal blood values, vital signs, or scan findings).

2. 🚨 Identify Abnormalities or Diagnosed Conditions

    Point out any significant medical concerns or abnormalities (e.g., anemia, Vitamin D deficiency, fractures, lesions).

    Briefly explain what these findings might mean in layman’s terms.

3. 🖼️ Interpret Diagnostic Images (if attached)

    Analyze any uploaded diagnostic images (e.g., X-ray, MRI, CT).

    Describe visible abnormalities and suggest potential medical interpretations or conditions.

4. 📋 Recommend Follow-Up Actions

    Propose next steps like additional tests, monitoring, or referrals to specialists.

    Explain the reason behind each recommendation.

5. 🍽️ Suggest a Personalized Diet Plan

    Recommend a basic diet suited to the findings (e.g., iron-rich foods for anemia, low-sodium diet for hypertension).

6. 🏃 Recommend Suitable Physical Activities

    Suggest appropriate physical exercises aligned with the current health condition (e.g., walking, light yoga, stretching).

    Ensure recommendations are safe and realistic for the patient’s situation.

🧭 Guidelines for Your Response:

    Use clear and simple language—avoid complex medical jargon unless explained.

    Format your response with headings or bullet points for readability.

    Maintain a professional, informative, and empathetic tone.

    If any information cannot be determined from the report or image, say:
    “Not found in the report/image.”

📥 Input:
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
st.title("🩺 𝐂𝐮𝐫𝐚𝐂𝐕 𝐀𝐈 ", 
"Medical Report Analyzer + Chatbot")
st.markdown("Upload a **medical report (PDF or image)**, get a smart **AI-generated summary**, and ask **follow-up questions**.")
# ----------- Session State Setup -----------
if "summary_output" not in st.session_state:
    st.session_state.summary_output = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ SIDEBAR ------------------

st.sidebar.header("📂 𝐔𝐩𝐥𝐨𝐚𝐝 ➜ 🤖 𝐀𝐈 𝐀𝐧𝐚𝐥𝐲𝐳𝐞")

# ✅ Updated uploader to accept PDF + Images
uploaded_file = st.sidebar.file_uploader(
    "Upload medical report (PDF or Image)", 
    type=["pdf", "png", "jpg", "jpeg"]
)

# ✅ Language selector
language_options = ["🗨️  𝐄𝐧𝐠𝐥𝐢𝐬𝐡", "🗨️  𝐇𝐢𝐧𝐝𝐢", "🗨️  𝐌𝐚𝐫𝐚𝐭𝐡𝐢", "🗨️  𝐆𝐮𝐣𝐚𝐫𝐚𝐭𝐢", "🗨️  𝐓𝐚𝐦𝐢𝐥", "🗨️  𝐓𝐞𝐥𝐮𝐠𝐮", "🗨️  𝐁𝐞𝐧𝐠𝐚𝐥𝐢", "🗨️  𝐏𝐮𝐧𝐣𝐚𝐛𝐢", "🗨️  𝐊𝐚𝐧𝐧𝐚𝐝𝐚", "🗨️  𝐔𝐫𝐝𝐮"]
selected_language = st.sidebar.selectbox("🌍 𝐂𝐇𝐎𝐎𝐒𝐄 𝐋𝐀𝐍𝐆𝐔𝐀𝐆𝐄 𝐅𝐎𝐑 𝐎𝐔𝐓𝐏𝐔𝐓:", language_options)

analyze_btn = False
if uploaded_file:
    with st.spinner("🔍 Extracting text and images..."):
        text, images = extract_text_and_images(uploaded_file)
        print(len(text.strip()))
        print("----")
        print(len(images))

    if len(text.strip().split()) > 0 or images:
        st.sidebar.success("✅ File processed.")
        analyze_btn = st.sidebar.button("📝 𝐆𝐞𝐧𝐞𝐫𝐚𝐭𝐞 𝐒𝐮𝐦𝐦𝐚𝐫𝐲")
    else:
        st.sidebar.warning("⚠️ Not enough content found in the file.")

# ------------------ Main Area ------------------

if analyze_btn:
    with st.spinner("🤖 𝐀𝐧𝐚𝐥𝐲𝐳𝐢𝐧𝐠 𝐑𝐞𝐩𝐨𝐫𝐭 𝐰𝐢𝐭𝐡 𝐆𝐞𝐦𝐢𝐧𝐢..."):
        summary = call_gemini_with_text_and_images(text, images)
        st.session_state.summary_output = summary
        store_to_vector_db(text, summary, source="report")

if st.session_state.summary_output:
    st.subheader("📝 𝐀𝐈 𝐒𝐮𝐦𝐦𝐚𝐫𝐲 & 𝐑𝐞𝐜𝐨𝐦𝐦𝐞𝐧𝐝𝐚𝐭𝐢𝐨𝐧𝐬 ⚕️📌")
    translated_summary = translate_text(st.session_state.summary_output, selected_language)
    st.markdown(translated_summary)

# ----------- Chatbot Section -----------

st.divider()
st.subheader("🤔💬 𝐀𝐬𝐤 𝐐𝐮𝐞𝐬𝐭𝐢𝐨𝐧𝐬 𝐀𝐛𝐨𝐮𝐭 𝐭𝐡𝐞 𝐑𝐞𝐩𝐨𝐫𝐭")

with st.form("chat_form"):
    user_query = st.text_input("📄❓ 𝐀𝐬𝐤 𝐚 𝐦𝐞𝐝𝐢𝐜𝐚𝐥 𝐪𝐮𝐞𝐬𝐭𝐢𝐨𝐧 𝐫𝐞𝐥𝐚𝐭𝐞𝐝 𝐭𝐨 𝐭𝐡𝐞 𝐫𝐞𝐩𝐨𝐫𝐭:")
    submit_btn = st.form_submit_button("Submit")

if submit_btn and user_query:
    with st.spinner("📄🔎 𝐒𝐞𝐚𝐫𝐜𝐡𝐢𝐧𝐠 𝐑𝐞𝐩𝐨𝐫𝐭𝐬..."):
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
        with st.spinner("⏳  𝐆𝐞𝐧𝐞𝐫𝐚𝐭𝐢𝐧𝐠 𝐀𝐧𝐬𝐰𝐞𝐫..."):
            response = call_gemini(prompt)

        translated_response = translate_text(response, selected_language)
        st.markdown("### 🧠💡 𝐆𝐞𝐦𝐢𝐧𝐢’𝐬 𝐀𝐧𝐬𝐰𝐞𝐫")
        st.markdown(translated_response)

        st.session_state.chat_history.append((user_query, response))
        store_to_vector_db(user_query, response, source="chat")

    else:
        st.warning("📄🚫 𝐍𝐨 𝐫𝐞𝐥𝐞𝐯𝐚𝐧𝐭 𝐫𝐞𝐩𝐨𝐫𝐭 𝐟𝐨𝐮𝐧𝐝 𝐭𝐨 𝐚𝐧𝐬𝐰𝐞𝐫 𝐭𝐡𝐢𝐬 𝐪𝐮𝐞𝐬𝐭𝐢𝐨𝐧.")

# ----------- Chat History Viewer -----------

if st.session_state.chat_history:
    with st.expander("🕘 𝐒𝐡𝐨𝐰 𝐏𝐫𝐞𝐯𝐢𝐨𝐮𝐬 𝐂𝐡𝐚𝐭 𝐇𝐢𝐬𝐭𝐨𝐫𝐲"):
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
            st.markdown(f"**Q{i}:** {q}")
            translated_a = translate_text(a, selected_language)
            st.markdown(f"**A{i}:** {translated_a}")
            st.markdown("---")
