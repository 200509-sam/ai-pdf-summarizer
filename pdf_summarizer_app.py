import streamlit as st
import PyPDF2
from transformers import pipeline
from pathlib import Path

# ---------- Model + Helper Functions ----------

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def chunk_text(text, max_chars=2000):
    words = text.split()
    chunks, current, length = [], [], 0
    for w in words:
        length += len(w) + 1
        if length > max_chars:
            chunks.append(" ".join(current))
            current = [w]
            length = len(w)
        else:
            current.append(w)
    if current:
        chunks.append(" ".join(current))
    return chunks

def summarize_text(text):
    summarizer = load_summarizer()
    chunks = chunk_text(text)
    summaries = []
    progress = st.progress(0, text="✨ Summarizing... please wait ✨")
    total = len(chunks)
    for i, chunk in enumerate(chunks, start=1):
        summary = summarizer(chunk, max_length=200, min_length=60, do_sample=False)[0]["summary_text"]
        summaries.append(summary)
        progress.progress(i / total)
    progress.text("✅ Done summarizing!")
    return " ".join(summaries)

# ---------- Page Setup ----------

st.set_page_config(
    page_title="AI PDF Summarizer",
    page_icon="📚",
    layout="wide"
)

# ---------- Colorful Background CSS ----------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #ff7eb3, #ff758c, #42a5f5, #7c4dff);
    background-size: 400% 400%;
    animation: gradientMove 10s ease infinite;
    color: #fff;
}
@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

.title {
    font-size: 2.8rem;
    font-weight: 900;
    text-align: center;
    background: -webkit-linear-gradient(45deg, #fff, #e0f7fa, #ffe082);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #f5f5f5;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

.glass {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 20px 25px;
    border: 1px solid rgba(255,255,255,0.3);
    box-shadow: 0px 8px 40px rgba(0,0,0,0.25);
}

.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 25px;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    transition: all 0.3s ease-in-out;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #f093fb, #f5576c);
    transform: scale(1.05);
}
.stFileUploader div[data-baseweb="file-uploader"] {
    border-radius: 18px;
    background-color: rgba(255,255,255,0.25);
}
.summary-box {
    background: rgba(255,255,255,0.1);
    padding: 16px 20px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.4);
    color: #fff;
    line-height: 1.6;
    font-size: 1rem;
    box-shadow: inset 0 0 20px rgba(255,255,255,0.1);
}
.footer {
    text-align:center;
    font-size: 0.85rem;
    color: #fff;
    margin-top: 2rem;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header Section ----------

st.markdown("<div class='title'>🌈 AI PDF Summarizer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a PDF and get a vibrant, AI-powered summary in seconds!</div>", unsafe_allow_html=True)

# ---------- Layout ----------

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📂 Step 1: Upload PDF")
    pdf = st.file_uploader("Choose your file", type=["pdf"])
    st.write("💡 Tip: Works best with lecture notes, chapters, or articles.")
    summarize = st.button("✨ Generate Summary")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("🧾 Step 2: Your AI Summary")
    summary_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Logic ----------

if summarize:
    if pdf is None:
        st.error("Please upload a PDF first!")
    else:
        with st.spinner("Extracting text..."):
            text = extract_text_from_pdf(pdf)
        st.success(f"Extracted {len(text)} characters.")
        summary = summarize_text(text)
        summary_placeholder.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)

        filename = Path(pdf.name).with_suffix(".summary.txt").name
        st.download_button("⬇️ Download Summary", summary, file_name=filename, mime="text/plain")

# ---------- Footer ----------

st.markdown("<div class='footer'>💖 Made with AI | Designed by SANGEETHA</div>", unsafe_allow_html=True)
