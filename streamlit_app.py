
############################################################
# TN VAZIKATTI AI — FINAL CLEAN STREAMLIT APP (RAG + GROQ)
############################################################

import os
import json
import re
from pathlib import Path
from io import BytesIO

import streamlit as st
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Optional extractors
import pdfplumber
import docx
from bs4 import BeautifulSoup
import requests

############################################################
# FIX NLTK PUNKT (Needed for Streamlit Cloud)
############################################################
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


############################################################
# PATHS / DIRECTORIES
############################################################
DATA_RAW = Path("tn_docs_raw")
TEXT_DIR = Path("tn_texts")
INDEX_DIR = Path("faiss_index")

DATA_RAW.mkdir(exist_ok=True)
TEXT_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

FAISS_INDEX_FILE = INDEX_DIR / "tn_faiss.index"
META_FILE = INDEX_DIR / "tn_meta.json"

############################################################
# MODELS
############################################################
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
LLM_MODEL = "openai/gpt-oss-120b"   # Groq free model


############################################################
# TEXT EXTRACTION HELPERS
############################################################
def extract_text_from_pdf_bytes(b: bytes):
    text_parts = []
    try:
        with pdfplumber.open(BytesIO(b)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    text_parts.append(txt)
    except Exception:
        return ""
    return "\n".join(text_parts).strip()


def extract_text_from_docx_bytes(b: bytes):
    try:
        tmp = BytesIO(b)
        doc = docx.Document(tmp)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception:
        return ""


def extract_text_from_html_string(html):
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.extract()
    parts = soup.find_all(["p", "li", "h1", "h2", "h3"])
    return "\n".join([p.get_text(" ", strip=True) for p in parts])


def clean_text(text):
    text = text.replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page\s*\d+", "", text)
    return text.strip()


############################################################
# SAVE + EXTRACT
############################################################
def save_uploaded_file(uploaded_file):
    path = DATA_RAW / uploaded_file.name
    path.write_bytes(uploaded_file.getvalue())
    return path


def extract_and_save_text_from_file(path: Path):
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            txt = extract_text_from_pdf_bytes(path.read_bytes())
        elif ext == ".docx":
            txt = extract_text_from_docx_bytes(path.read_bytes())
        elif ext in [".html", ".htm"]:
            txt = extract_text_from_html_string(path.read_text("utf-8", errors="ignore"))
        elif ext == ".txt":
            txt = path.read_text("utf-8", errors="ignore")
        else:
            txt = ""

    except Exception:
        txt = ""

    if not txt or len(txt.strip()) < 20:
        return None

    txt = clean_text(txt)
    out_txt = TEXT_DIR / (path.stem + ".txt")
    out_txt.write_text(txt, encoding="utf-8")
    return out_txt


############################################################
# CHUNKING
############################################################
from nltk.tokenize import sent_tokenize

def chunk_text_by_sentences(text, max_tokens=180, overlap_sentences=1):
    sentences = sent_tokenize(text)
    chunks = []
    current = []
    length = 0

    for s in sentences:
        words = s.split()
        if length + len(words) > max_tokens and current:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            length = sum(len(x.split()) for x in current)

        current.append(s)
        length += len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks


############################################################
# EMBEDDINGS (HuggingFace)
############################################################
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_texts(texts):
    model = load_embed_model()
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")


############################################################
# FAISS INDEX
############################################################
def build_faiss_index(docs, metas):
    vecs = embed_texts(docs)
    dim = vecs.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    faiss.write_index(index, str(FAISS_INDEX_FILE))
    META_FILE.write_text(json.dumps({"docs": docs, "meta": metas}, ensure_ascii=False, indent=2))
    return index


def load_faiss_index():
    if FAISS_INDEX_FILE.exists() and META_FILE.exists():
        return faiss.read_index(str(FAISS_INDEX_FILE)), json.load(open(META_FILE, encoding="utf-8"))
    return None, None


############################################################
# RETRIEVAL
############################################################
def retrieve(query, k=4):
    index, meta = load_faiss_index()
    if index is None:
        return []

    q_vec = embed_texts([query])
    D, I = index.search(q_vec, k)

    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(meta["docs"]):
            continue
        results.append({
            "text": meta["docs"][idx],
            "source": meta["meta"][idx]["source_file"]
        })
    return results


############################################################
# GROQ CLIENT
############################################################
@st.cache_resource(show_spinner=False)
def get_groq_client():
    key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if not key:
        st.error("❌ ERROR: GROQ_API_KEY not set.")
        return None

    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")


############################################################
# PROMPT GENERATION
############################################################
def build_prompt(query, ctx):
    context_text = "\n\n".join([c["text"] for c in ctx])

    return f"""
Context:
{context_text}

User Question: {query}

Instructions:
1. First answer in **English**.
2. Then give the **same answer in Tamil**.
3. If answer not found, say:
   English: "Information not available."
   Tamil: "தகவல் கிடைக்கவில்லை."

Answer:
"""


def is_small_talk(q):
    q = q.lower().strip()
    greetings = ["hi", "hello", "hey", "vanakkam"]
    return q in greetings or any(g in q for g in greetings)


def generate_with_groq(prompt):
    client = get_groq_client()
    if client is None:
        return "LLM not ready."

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful Tamil Nadu Government Assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=600
    )
    return resp.choices[0].message.content


def tn_answer(query):
    if is_small_talk(query):
        return generate_with_groq(f"Respond politely in English and Tamil: {query}"), []

    ctx = retrieve(query, k=4)
    prompt = build_prompt(query, ctx)
    answer = generate_with_groq(prompt)
    return answer, ctx


############################################################
# STREAMLIT UI  — BEAUTIFUL MODERN CHAT GPT STYLE
############################################################
st.set_page_config(page_title="TN Vazikatti AI", layout="wide")

# ===== CSS =====
st.markdown("""
<style>
body { background: #eef2ff; }

.chat-box {
  background: white;
  padding: 28px;
  border-radius: 25px;
  width: 100%;
  max-width: 700px;
  margin: auto;
  box-shadow: 0 6px 18px rgba(0,0,0,0.1);
}

.message-user {
  background: #2563eb;
  color: white;
  padding: 12px 18px;
  border-radius: 16px;
  margin: 10px 0;
  text-align:right;
}

.message-bot {
  background: #f3f4f6;
  color: #111827;
  padding: 12px 18px;
  border-radius: 16px;
  margin: 10px 0;
  text-align:left;
}

</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])


############################################################
# SIDEBAR (Upload + Build Index)
############################################################
with col2:
    st.title("TN Vazikatti AI")
    st.write("**Tamil Nadu Government Assistant**")
    st.write("---")

    index, meta = load_faiss_index()
    if index is None:
        st.error("FAISS index not available.")
    else:
        st.success(f"Index Loaded ({len(meta['docs'])} chunks)")

    uploaded = st.file_uploader("Upload docs (.pdf, .txt, .html, .docx)", accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            raw_path = save_uploaded_file(f)
            saved = extract_and_save_text_from_file(raw_path)
            st.write(saved.name if saved else "Extraction failed")

    if st.button("Build / Rebuild Index"):
        st.info("Extracting documents...")
        raw_files = DATA_RAW.glob("*")
        for p in raw_files:
            extract_and_save_text_from_file(p)

        st.info("Chunking...")
        docs = []
        metas = []
        for txt in TEXT_DIR.glob("*.txt"):
            content = txt.read_text("utf-8", errors="ignore")
            chunks = chunk_text_by_sentences(content)
            for i, ch in enumerate(chunks):
                docs.append(ch)
                metas.append({"source_file": txt.name, "chunk_id": f"{txt.stem}_{i}"})

        if docs:
            st.info("Building FAISS...")
            build_faiss_index(docs, metas)
            st.success("Index built!")

    st.write("---")
    st.caption("Add docs → Build index → Chat")


############################################################
# MAIN CHAT UI
############################################################
with col1:
    st.markdown("""
        <div style='text-align:center; padding:20px 0;'>
            <h1>TN Vazikatti AI</h1>
            <p style='font-size:18px; color:#4b5563;'>Ask anything about Tamil Nadu Government Services</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    for user_msg, bot_msg, _ in st.session_state.history:
        st.markdown(f"<div class='message-user'>{user_msg}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='message-bot'>{bot_msg.replace('\\n','<br>')}</div>", unsafe_allow_html=True)

    user_query = st.text_input("Type your message...")
    if st.button("Send"):
        if user_query.strip():
            with st.spinner("Thinking..."):
                ans, ctx = tn_answer(user_query)
            st.session_state.history.append((user_query, ans, ctx))
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
