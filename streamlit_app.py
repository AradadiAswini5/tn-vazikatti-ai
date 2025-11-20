# streamlit_app.py
# TN Vazikatti AI — Redesigned modern UI + RAG backend (FAISS + HF embeddings + Groq)
# Paste this entire file and run with: streamlit run streamlit_app.py

import os
import json
import re
import datetime
from pathlib import Path
from io import BytesIO
import textwrap
import base64

import streamlit as st
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Optional extractors
import pdfplumber
import docx
from bs4 import BeautifulSoup

# NLTK punkt fix for cloud
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

from nltk.tokenize import sent_tokenize

# ---------------------------
# Paths & config
# ---------------------------
DATA_RAW = Path("tn_docs_raw")
TEXT_DIR = Path("tn_texts")
INDEX_DIR = Path("faiss_index")

DATA_RAW.mkdir(exist_ok=True)
TEXT_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

FAISS_INDEX_FILE = INDEX_DIR / "tn_faiss.index"
META_FILE = INDEX_DIR / "tn_meta.json"

EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
LLM_MODEL = "openai/gpt-oss-120b"

# ---------------------------
# Text extraction helpers
# ---------------------------
def extract_text_from_pdf_bytes(b: bytes):
    try:
        text_parts = []
        with pdfplumber.open(BytesIO(b)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    text_parts.append(txt)
        return "\n".join(text_parts).strip()
    except Exception:
        return ""

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
    parts = soup.find_all(["p","li","h1","h2","h3"])
    return "\n".join([p.get_text(" ", strip=True) for p in parts])

def clean_text(text):
    text = text.replace("\r"," ").replace("\t"," ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page\s*\d+", "", text, flags=re.I)
    return text.strip()

# ---------------------------
# Save + extract files
# ---------------------------
def save_uploaded_file(uploaded_file):
    out = DATA_RAW / uploaded_file.name
    out.write_bytes(uploaded_file.getvalue())
    return out

def extract_and_save_text_from_file(path: Path):
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            txt = extract_text_from_pdf_bytes(path.read_bytes())
        elif ext == ".docx":
            txt = extract_text_from_docx_bytes(path.read_bytes())
        elif ext in [".html", ".htm"]:
            txt = extract_text_from_html_string(path.read_text(encoding="utf-8", errors="ignore"))
        elif ext == ".txt":
            txt = path.read_text(encoding="utf-8", errors="ignore")
        else:
            txt = ""
    except Exception:
        txt = ""
    if not txt or len(txt.strip()) < 20:
        return None
    txt = clean_text(txt)
    out_name = TEXT_DIR / (path.stem + ".txt")
    out_name.write_text(txt, encoding="utf-8")
    return out_name

# ---------------------------
# Chunking
# ---------------------------
def chunk_text_by_sentences(text, max_tokens=220, overlap_sentences=1):
    sents = sent_tokenize(text)
    chunks = []
    cur = []
    cur_len = 0
    for s in sents:
        words = s.split()
        if cur_len + len(words) > max_tokens and cur:
            chunks.append(" ".join(cur).strip())
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
    index = faiss.IndexFlatIP(d)
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
            "source": meta["meta"][idx].get("source_file","")
        })
    return results

# ---------------------------
# GROQ client init
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_groq_client():
    key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if not key:
        st.error("GROQ_API_KEY not set. Add it to Streamlit Secrets.")
        return None
    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    return client

# ---------------------------
# Prompt & LLM
# ---------------------------
def is_small_talk(text):
    t = text.lower().strip()
    greetings = ["hi","hello","hey","vanakkam","thanks","thank you"]
    return t in greetings or any(g in t for g in greetings)

def build_prompt(query, context_chunks):
    context_text = "\n\n".join([c["text"] for c in context_chunks]) if context_chunks else ""
    prompt = f"""You are a Tamil Nadu Government Assistant.
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

def generate_with_groq(prompt):
    client = get_groq_client()
    if client is None:
        return "LLM client not configured."
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role":"system","content":"You are a helpful Tamil Nadu Government Assistant."},
            {"role":"user","content":prompt}
        ],
        max_tokens=600
    )
    return resp.choices[0].message.content

def tn_answer(query):
    if is_small_talk(query):
        return generate_with_groq(f"Respond in English and Tamil, politely, to: {query}"), []
    ctx = retrieve(query, k=4)
    prompt = build_prompt(query, ctx)
    answer = generate_with_groq(prompt)
    return answer, ctx

# ---------------------------
# UI - final redesigned
# ---------------------------
st.set_page_config(page_title="TN Vazikatti AI", layout="wide")

# CSS + background image + emojis + avatars
st.markdown("""
<style>
:root{
  --accent:#0b67ff;
  --bg1: linear-gradient(180deg, #e6f0ff 0%, #ffffff 60%);
}
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg1);
}
.app-shell {
  padding-top: 18px;
}
.header {
  display: flex;
  gap: 16px;
  align-items:center;
}
.header .logo {
  width:72px;
  height:72px;
  border-radius:18px;
  background: linear-gradient(135deg,#0b67ff,#60a5fa);
  display:flex;
  align-items:center;
  justify-content:center;
  color:white;
  font-weight:700;
  font-size:34px;
  box-shadow: 0 6px 18px rgba(11,103,255,0.18);
}
.header h1 { margin:0; font-size:22px; color:#063069; }
.header p { margin:0; color:#475569; }

/* Chat container */
.container {
  max-width:980px;
  margin: 12px auto 60px;
  display: grid;
  grid-template-columns: 1fr 360px;
  gap: 20px;
  align-items:start;
}

/* left chat card */
.chat-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(250,250,255,0.95));
  border-radius:18px;
  padding:18px;
  box-shadow: 0 8px 30px rgba(9,30,66,0.06);
}

/* make chat area smaller height */
.chat-area {
  max-height: 620px;
  overflow:auto;
  padding: 12px;
  border-radius:12px;
  border: 1px solid rgba(15,23,42,0.04);
  background: linear-gradient(180deg,#ffffff,#fbfdff);
}

/* bubbles */
.bubble-user {
  background: var(--accent);
  color: white;
  padding: 10px 14px;
  border-radius: 18px 18px 6px 18px;
  display: inline-block;
  margin: 8px 2px;
  max-width:78%;
  word-wrap:break-word;
  float:right;
  clear:both;
}
.bubble-bot {
  background: #f6f9ff;
  color: #0b1a33;
  padding: 10px 14px;
  border-radius: 18px 18px 18px 6px;
  display: inline-block;
  margin: 8px 2px;
  max-width:78%;
  word-wrap:break-word;
  float:left;
  clear:both;
  border: 1px solid rgba(11,103,255,0.06);
}

/* little meta */
.meta { font-size:12px; color:#6b7280; margin-top:6px; }

/* input area - floating feel */
.input-area {
  display:flex;
  gap:10px;
  margin-top:12px;
  align-items:center;
}
.input-area textarea {
  flex:1;
  border-radius:12px;
  border:1px solid rgba(15,23,42,0.06);
  padding:10px 12px;
  min-height:44px;
  font-size:14px;
  resize:none;
}
.send-btn {
  background: var(--accent);
  color:white;
  padding:10px 16px;
  border-radius:12px;
  border:none;
  cursor:pointer;
}
.send-btn:active { transform: translateY(1px); }
.small {
  font-size:13px;
  color:#475569;
}

/* sidebar */
.side-card {
  background: white;
  padding:16px;
  border-radius:12px;
  box-shadow: 0 6px 18px rgba(9,30,66,0.04);
}

/* typing indicator */
.typing {
  display:inline-block;
  width: 44px;
  text-align:center;
}
.dot {
  height:8px; width:8px; margin:0 2px; display:inline-block; border-radius:50%; background:#90caf9; animation: blink 1s infinite;
}
.dot:nth-child(2){ animation-delay:0.15s;}
.dot:nth-child(3){ animation-delay:0.3s;}
@keyframes blink {
  0%,80%,100%{opacity:0.2}
  40%{opacity:1}
}

/* badges / emoji */
.badge { font-size:14px; padding:6px 10px; border-radius:999px; background:#eef2ff; color:#0b67ff; border:1px solid rgba(11,103,255,0.06); }

/* clear floats */
.clearfix::after { content:""; clear:both; display:block; }
</style>
""", unsafe_allow_html=True)

# Top header
st.markdown("""
<div class="header">
  <div class="logo">TN</div>
  <div>
    <h1>TN Vazikatti AI <span class="small">— Tamil Nadu Government Services</span></h1>
    <p class="small">Ask about services in English or Tamil • Powered by FAISS + Groq + HF embeddings</p>
  </div>
</div>
""", unsafe_allow_html=True)

# layout container
st.markdown("<div class='container'>", unsafe_allow_html=True)

# ---------------------------
# Left: Chat area
# ---------------------------
st.markdown("<div class='chat-card'>", unsafe_allow_html=True)
st.markdown("<div class='chat-area' id='chat-area'>", unsafe_allow_html=True)

# initialize
if "history" not in st.session_state:
    st.session_state.history = []  # list of (user, bot, sources)

# render messages
for user_text, bot_text, sources in st.session_state.history:
    # user bubble
    st.markdown(f"<div class='bubble-user'>{user_text}</div>", unsafe_allow_html=True)
    # bot bubble (allow markup)
    safe_bot = bot_text.replace("\n","<br>")
    st.markdown(f"<div class='bubble-bot'>{safe_bot}</div>", unsafe_allow_html=True)
    if sources:
        st.markdown(f"<div class='meta'>Sources: {', '.join(sources)}</div>", unsafe_allow_html=True)
    st.markdown("<div class='clearfix'>&nbsp;</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # end chat-area

# input area (floating)
st.markdown("""
<div class="input-area">
  <textarea id="user_input" placeholder="Type your question here (English / Tamil)"></textarea>
  <button class="send-btn" id="send_btn">➤</button>
</div>
<script>
const sendBtn = document.getElementById('send_btn');
const input = document.getElementById('user_input');
sendBtn.onclick = () => {
  const val = input.value;
  if (!val) return;
  // use Streamlit setComponentValue via form submit fallback:
  const el = document.createElement('a');
  el.href='/?_send='+encodeURIComponent(val);
  el.click();
};
</script>
""", unsafe_allow_html=True)

# small helpers: capture GET param _send (when clicking send button)
query_from_front = None
params = st.experimental_get_query_params()
if "_send" in params:
    q = params["_send"][0]
    query_from_front = q

# old fallback text_input for browsers without JS
fallback = st.text_input("or type here and press Send", key="fallback_input")
if fallback and st.button("Send (fallback)"):
    query_from_front = fallback

# process query if provided
if query_from_front:
    q = query_from_front.strip()
    if q:
        st.session_state.history.append((q, "…", []))
        st.rerun()

# If last appended is placeholder '…' we will generate reply
# generate replies for any placeholder entries
updated = False
for i,(u,b,s) in enumerate(st.session_state.history):
    if b == "…":
        # call model
        with st.spinner("Vazikatti is typing..."):
            ans, ctx = tn_answer(u)
        sources = list({c["source"] for c in ctx}) if ctx else []
        st.session_state.history[i] = (u, ans, sources)
        updated = True

if updated:
    st.rerun()

# small area: download chat
def get_chat_text():
    txt = ""
    for u,b,s in st.session_state.history:
        txt += f"USER: {u}\nBOT:\n{b}\n"
        if s:
            txt += f"SOURCES: {', '.join(s)}\n"
        txt += "\n---\n\n"
    return txt

st.markdown("<div style='margin-top:12px; display:flex; gap:8px; align-items:center;'>", unsafe_allow_html=True)
if st.button("💾 Download Chat (TXT)"):
    content = get_chat_text()
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/text;base64,{b64}" download="tn_vazikatti_chat_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt">Click to download</a>'
    st.markdown(href, unsafe_allow_html=True)

# optionally provide JSON download
if st.button("📁 Save as JSON"):
    fname = f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump([{"user":u,"bot":b,"sources":s} for u,b,s in st.session_state.history], f, ensure_ascii=False, indent=2)
    st.success(f"Saved {fname}")

st.markdown("</div>", unsafe_allow_html=True)  # end input area
st.markdown("</div>", unsafe_allow_html=True)  # end chat-card

# ---------------------------
# Right: Sidebar card (upload + index)
# ---------------------------
st.markdown("<div class='side-card'>", unsafe_allow_html=True)

st.markdown("<h3 style='margin-top:0;'>Index & Upload</h3>", unsafe_allow_html=True)
index, meta = load_faiss_index()
if index is None:
    st.error("FAISS index not loaded. Upload docs & click Build.")
else:
    st.success(f"Index loaded — {len(meta['docs'])} chunks")

uploaded = st.file_uploader("Upload documents (.pdf, .txt, .docx, .html)", accept_multiple_files=True)
if uploaded:
    for f in uploaded:
        p = save_uploaded_file(f)
        saved = extract_and_save_text_from_file(p)
        st.write(f"Saved: {saved.name if saved else 'Extraction failed'}")

if st.button("Build / Rebuild Index"):
    st.info("Extracting and chunking documents...")
    raw_files = sorted(DATA_RAW.glob("*"))
    for p in raw_files:
        extract_and_save_text_from_file(p)
    docs = []
    metas = []
    for txtfile in sorted(TEXT_DIR.glob("*.txt")):
        txt = txtfile.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text_by_sentences(txt, max_tokens=220, overlap_sentences=1)
        for i,ch in enumerate(chunks):
            docs.append(ch)
            metas.append({"source_file": txtfile.name, "chunk_id": f"{txtfile.stem}__{i}"})
    if not docs:
        st.warning("No text chunks found.")
    else:
        st.info("Building FAISS (this may take a while)...")
        build_faiss_index(docs, metas)
        st.success(f"Index built with {len(docs)} chunks")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='small'>Model: " + LLM_MODEL + "</div>", unsafe_allow_html=True)
st.markdown("<div style='margin-top:8px;'><span class='badge'>English + Tamil</span>  <span style='margin-left:8px' class='small'>FAQ • Upload docs • Build index</span></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)  # end side-card

st.markdown("</div>", unsafe_allow_html=True)  # end container
