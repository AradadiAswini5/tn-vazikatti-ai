# streamlit_app.py

import os
import json
import re
from pathlib import Path
from io import BytesIO
import html

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
from tqdm import tqdm

# ---------------------------
# Config / constants
# ---------------------------
DATA_RAW = Path("tn_docs_raw")
TEXT_DIR = Path("tn_texts")
INDEX_DIR = Path("faiss_index")
INDEX_DIR.mkdir(exist_ok=True)
DATA_RAW.mkdir(exist_ok=True)
TEXT_DIR.mkdir(exist_ok=True)

EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"   # good multilingual embedding (Tamil+English)
LLM_MODEL = "openai/gpt-oss-120b"                     # your Groq model name
FAISS_INDEX_FILE = INDEX_DIR / "tn_faiss.index"
META_FILE = INDEX_DIR / "tn_meta.json"

# ---------------------------
# Utilities: text extraction
# ---------------------------
def extract_text_from_pdf_bytes(b: bytes):
    text_parts = []
    try:
        with pdfplumber.open(BytesIO(b)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    text_parts.append(txt)
    except Exception:
        # fallback empty
        return ""
    return "\n".join(text_parts).strip()

def extract_text_from_docx_bytes(b: bytes):
    try:
        tmp = BytesIO(b)
        doc = docx.Document(tmp)
        texts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(texts)
    except Exception:
        return ""

def extract_text_from_html_string(html):
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.extract()
    main = soup.find("main") or soup.find("article") or soup.body
    if main:
        texts = [p.get_text(separator=" ", strip=True) for p in main.find_all(["p","li","h1","h2","h3"])]
    else:
        texts = [p.get_text(separator=" ", strip=True) for p in soup.find_all(["p","li","h1","h2","h3"])]
    return "\n".join([t for t in texts if t])

def clean_text(text):
    text = text.replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"Page\s*\d+", "", text, flags=re.I)
    text = text.replace("₹", "Rs.")
    return text

# ---------------------------
# Ingest uploaded files or local raw files
# ---------------------------
def save_uploaded_file(uploaded_file):
    # save raw copy
    out = DATA_RAW / uploaded_file.name
    out.write_bytes(uploaded_file.getvalue())
    return out

def extract_and_save_text_from_file(path: Path):
    # returns path to saved .txt file or None
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            txt = extract_text_from_pdf_bytes(path.read_bytes())
            if not txt or len(txt) < 30:
                txt = ""  # skip OCR fallback for simplicity
        elif ext in [".docx"]:
            txt = extract_text_from_docx_bytes(path.read_bytes())
        elif ext in [".html", ".htm"]:
            txt = extract_text_from_html_string(path.read_text(encoding="utf-8", errors="ignore"))
        elif ext == ".txt":
            txt = path.read_text(encoding="utf-8", errors="ignore")
        else:
            txt = ""
    except Exception as e:
        txt = ""
    if not txt or len(txt.strip()) < 20:
        return None
    txt = clean_text(txt)
    out_name = TEXT_DIR / (path.stem + ".txt")
    out_name.write_text(txt, encoding="utf-8")
    return out_name

# ---------------------------
# Chunking (sentence-aware)
# ---------------------------
import nltk
from nltk.tokenize import sent_tokenize
# Uncomment once if punkt not present:
# nltk.download('punkt')

def chunk_text_by_sentences(text, max_tokens=180, overlap_sentences=1):
    sents = sent_tokenize(text)
    chunks = []
    cur = []
    cur_len = 0
    for s in sents:
        words = s.split()
        if cur_len + len(words) > max_tokens and cur:
            chunks.append(" ".join(cur).strip())
            # overlap
            cur = cur[-overlap_sentences:] if overlap_sentences>0 else []
            cur_len = sum(len(x.split()) for x in cur)
        cur.append(s)
        cur_len += len(words)
    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks

# ---------------------------
# Embeddings (HF)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

def embed_texts(texts, batch_size=32):
    model = load_embed_model()
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size)
    return vecs.astype("float32")

# ---------------------------
# FAISS index build / load
# ---------------------------
def build_faiss_index(docs, metas):
    vecs = embed_texts(docs)
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine (we normalized)
    index.add(vecs)
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump({"docs": docs, "meta": metas}, f, ensure_ascii=False, indent=2)
    return index

def load_faiss_index():
    if FAISS_INDEX_FILE.exists() and META_FILE.exists():
        index = faiss.read_index(str(FAISS_INDEX_FILE))
        meta = json.load(open(META_FILE, encoding="utf-8"))
        return index, meta
    return None, None

# ---------------------------
# Retrieval
# ---------------------------
def retrieve(query, k=4):
    index, meta = load_faiss_index()
    if index is None:
        return []
    qv = embed_texts([query])
    D, I = index.search(qv, k)
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(meta["docs"]): 
            continue
        results.append({
            "text": meta["docs"][idx],
            "source": meta["meta"][idx].get("source_file", "")
        })
    return results

# ---------------------------
# Groq client init (OpenAI SDK pointed to Groq)
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_groq_client():
    key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if not key:
        st.error("GROQ_API_KEY not set. Set env var before running.")
        return None
    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    return client

# Small talk detector
def is_small_talk(text):
    t = text.lower().strip()
    greetings = ["hi","hello","hey","vanakkam","good morning","good evening","good afternoon","thanks","thank you"]
    if t in greetings:
        return True
    if len(t.split()) <= 2 and any(g in t for g in greetings):
        return True
    return False

# Build prompt with context (EN + TA)
def build_prompt(query, context_chunks):
    context_text = "\n\n".join([c["text"] for c in context_chunks]) if context_chunks else ""
    prompt = f"""
You are a Tamil Nadu Government Assistant.

Context:
{context_text}

User Question:
{query}

Instructions:
1) First answer in clear ENGLISH using ONLY the context above.
2) Then provide the SAME answer in clear TAMIL.
3) If the context does NOT contain the answer, reply:
   English: "Information not available."
   Tamil: "தகவல் கிடைக்கவில்லை."

Answer:
"""
    return prompt

# ---------------------------
# Generate answer (Groq)
# ---------------------------
def generate_with_groq(prompt):
    client = get_groq_client()
    if client is None:
        return "LLM client not configured."
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role":"system", "content":"You are a helpful Tamil Nadu Government Assistant."},
            {"role":"user", "content": prompt}
        ],
        max_tokens=512
    )
    return resp.choices[0].message.content

def tn_answer(query):
    # small-talk path
    if is_small_talk(query):
        prompt = f"Respond in English and Tamil, politely, to: {query}"
        return generate_with_groq(prompt)
    # retrieval path
    ctx = retrieve(query, k=4)
    prompt = build_prompt(query, ctx)
    answer = generate_with_groq(prompt)
    return answer, ctx

# ---------------------------
# Streamlit UI layout
# ---------------------------
st.set_page_config(page_title="TN Vazikatti AI (TN Government Assistant)", layout="wide")



# ---------------- CSS (exact screenshot style) ----------------
st.markdown("""
<style>
:root {
    --bg-start: #dcecff;
    --bg-end: #f4f9ff;
    --card: #ffffff;
    --blue: #3b82f6;
    --bubble: #e8f1ff;
}

body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, var(--bg-start), var(--bg-end));
}

.center {
    display: flex;
    justify-content: center;
    margin-top: 40px;
}

.chat-card {
    background: var(--card);
    width: 460px;
    padding: 30px 34px 40px 34px;
    border-radius: 22px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.08);
    position: relative;
    text-align: center;
}

.bot-icon-wrapper {
    width: 80px;
    height: 80px;
    background: #ffffff;
    border-radius: 50%;
    position: absolute;
    top: -40px; 
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    justify-content: center;
    align-items: center;
    box-shadow: 0px 8px 24px rgba(0,0,0,0.08);
}

.bot-icon svg {
    width: 48px;
    height: 48px;
}

.title-text {
    font-size: 28px;
    font-weight: 700;
    margin-top: 50px;
    margin-bottom: 6px;
    color: #0f172a;
}

.sub-text {
    font-size: 15px;
    color: #475569;
    margin-bottom: 24px;
}

/* Bubbles */
.bubble {
    background: var(--bubble);
    padding: 14px 18px;
    border-radius: 16px;
    text-align: left;
    font-size: 15px;
    margin-bottom: 14px;
}

/* Input row */
.input-row {
    display: flex;
    background: #ffffff;
    border: 1px solid #e7ecf5;
    border-radius: 12px;
    padding: 4px 6px;
    margin-top: 10px;
}

.input-row input {
    flex: 1;
    border: none;
    padding: 12px;
    outline: none;
    font-size: 15px;
    border-radius: 10px;
}

.send-btn {
    background: var(--blue);
    color: white;
    width: 44px;
    height: 40px;
    border-radius: 10px;
    border: none;
    font-size: 22px;
    cursor: pointer;
}

/* Start Chat button */
.start-btn {
    margin-top: 22px;
    background: linear-gradient(90deg, #4c80f4, #3b82f6);
    padding: 14px 22px;
    color: white;
    border: none;
    border-radius: 24px;
    font-weight: 600;
    width: 70%;
    font-size: 16px;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# ------------------- UI Skeleton -------------------
st.markdown('<div class="center"><div class="chat-card">', unsafe_allow_html=True)

# Bot icon
st.markdown("""
<div class="bot-icon-wrapper">
    <div class="bot-icon">
        <svg fill="none" stroke="#3b82f6" stroke-width="2" viewBox="0 0 24 24">
            <rect x="5" y="8" width="14" height="10" rx="3" stroke-linecap="round"/>
            <circle cx="9" cy="13" r="1.5" fill="#3b82f6"/>
            <circle cx="15" cy="13" r="1.5" fill="#3b82f6"/>
            <path d="M12 3v3" stroke-linecap="round"/>
        </svg>
    </div>
</div>
""", unsafe_allow_html=True)

# Title + Subtitle
st.markdown('<div class="title-text">Hi! How can I assist you?</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Ask anything about TN Govt Schemes</div>', unsafe_allow_html=True)

# Example bubbles
st.markdown('<div class="bubble">What are the benefits of the Amma Scooter Scheme?</div>', unsafe_allow_html=True)
st.markdown('<div class="bubble">The Amma Scooter Scheme provides financial assistance for purchasing a scooter.</div>', unsafe_allow_html=True)

# ------------------- Input Box -------------------
user_input = st.text_input("", placeholder="Type your message...")

# ------------------- Send Handler -------------------
if user_input:
    st.session_state["last_query"] = user_input
    st.success(f"You asked: {user_input}")
    st.stop()

# ------------------- Start Chat Button -------------------
st.markdown('<button class="start-btn">Start Chat</button>', unsafe_allow_html=True)

st.markdown('</div></div>', unsafe_allow_html=True)

st.caption("Built with local HuggingFace embeddings + FAISS + Groq LLM. Add docs, build index, then chat.")
