# # filename: medical_summary_app.py

# import streamlit as st
# import fitz  # PyMuPDF for PDF extraction
# import requests
# import os
# from dotenv import load_dotenv    

# load_dotenv()

# # Securely load API key (set in your environment variables in production)
# # API_KEY = "AIzaSyDu8ssrjVXWmMZuIHQx0FGSqphQfFYV8t4"
# API_KEY = os.getenv("API_KEY")

# # Function to extract text from uploaded PDF using PyMuPDF
# def extract_text_from_pdf(pdf_file):
#     text = ""
#     with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
#         for page in doc:
#             text += page.get_text()
#     return text

# # Function to call Google Gemini 2.0 Flash API for summarisation
# def generate_medical_summary(input_text):
#     url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={API_KEY}"  
#     headers = {
#         "Content-Type": "application/json"
#     }

#     data = {
#         "contents": [
#             {
#                 "parts": [
#                     {
#                         "text": f"Generate a concise and clear medical summary and Recommended medical tests, diet,basic exercises or physical activities for the following text.  :\n{input_text}"
#                     }
#                 ]
#             }
#         ]
#     }

#     response = requests.post(url, headers=headers, json=data)

#     if response.status_code == 200:
#         result = response.json()
#         try:
#             summary = result['candidates'][0]['content']['parts'][0]['text']
#         except Exception as e:
#             summary = "Error parsing summary response: " + str(e)
#         return summary
#     else:
#         return f"API Error {response.status_code}: {response.text}"

# # Streamlit app layout
# st.title("🩺 Medical PDF Summariser ")

# uploaded_file = st.file_uploader("Upload a medical PDF file", type=["pdf"])

# if uploaded_file:
#     with st.spinner("Extracting text from PDF..."):
#         pdf_text = extract_text_from_pdf(uploaded_file)
#     #     st.subheader("📄 Extracted PDF Text (first 100000000 characters):")
#     #     st.write(pdf_text[:100000000] + "..." if len(pdf_text) > 100000000 else pdf_text)

#     if st.button("Generate Medical Summary"):
#         with st.spinner("Generating summary using Gemini 2.0 Flash API..."):
#             summary = generate_medical_summary(pdf_text)
#             st.subheader("📝 Medical Summary")
#             st.write(summary)


# filename: medical_summary_app.py

import streamlit as st
import pathlib
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Securely load API key from .env
API_KEY = os.getenv("API_KEY")

# Configure Google Gemini client
# genai.configure(api_key=API_KEY)
client = genai.Client(api_key=API_KEY)

# Streamlit app layout
st.title("🩺 Medical PDF Summariser")

uploaded_file = st.file_uploader("Upload a medical PDF file", type=["pdf"])

if uploaded_file:
    pdf_path = f"temp_{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded {uploaded_file.name}")

    if st.button("Generate Medical Summary"):
        with st.spinner("Generating summary using Gemini API..."):

            prompt = "Generate a concise and clear medical summary with recommended medical tests, diet, basic exercises or physical activities for this document."

            # Generate content using Gemini Vision model that supports PDFs
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",   # Use appropriate Vision-enabled model here
                    contents=[
                        types.Part.from_bytes(
                            data=pathlib.Path(pdf_path).read_bytes(),
                            mime_type='application/pdf',
                        ),
                        prompt
                    ]
                )

                summary = response.text
            except Exception as e:
                summary = f"Error: {e}"

            st.subheader("📝 Medical Summary")
            st.write(summary)

    # Optionally delete temporary PDF after use
    os.remove(pdf_path)
