# streamlit_app.py
# Updated UI: colorful hero + centered chat card (Aranai style)
import os
import json
import re
from pathlib import Path
from io import BytesIO
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
import requests
from tqdm import tqdm

import nltk

# Streamlit Cloud FIX — download tokenizers only if missing
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# For updated NLTK versions
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

# ---------------------------
# Config / constants (unchanged)
# ---------------------------
DATA_RAW = Path("tn_docs_raw")
TEXT_DIR = Path("tn_texts")
INDEX_DIR = Path("faiss_index")
INDEX_DIR.mkdir(exist_ok=True)
DATA_RAW.mkdir(exist_ok=True)
TEXT_DIR.mkdir(exist_ok=True)

EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
LLM_MODEL = "openai/gpt-oss-120b"
FAISS_INDEX_FILE = INDEX_DIR / "tn_faiss.index"
META_FILE = INDEX_DIR / "tn_meta.json"

# ---------------------------
# Existing utilities (unchanged) - text extraction, chunking, embeddings...
# (I copied your existing helper functions so functionality stays the same)
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

def save_uploaded_file(uploaded_file):
    out = DATA_RAW / uploaded_file.name
    out.write_bytes(uploaded_file.getvalue())
    return out

def extract_and_save_text_from_file(path: Path):
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            txt = extract_text_from_pdf_bytes(path.read_bytes())
            if not txt or len(txt) < 30:
                txt = ""
        elif ext in [".docx"]:
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

# Chunking
from nltk.tokenize import sent_tokenize

def chunk_text_by_sentences(text, max_tokens=180, overlap_sentences=1):
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

# Embeddings
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

def embed_texts(texts, batch_size=32):
    model = load_embed_model()
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size)
    return vecs.astype("float32")

# FAISS
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

@st.cache_resource(show_spinner=False)
def get_groq_client():
    key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if not key:
        st.error("GROQ_API_KEY not set. Set env var before running.")
        return None
    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    return client

def is_small_talk(text):
    t = text.lower().strip()
    greetings = ["hi","hello","hey","vanakkam","good morning","good evening","good afternoon","thanks","thank you"]
    if t in greetings:
        return True
    if len(t.split()) <= 2 and any(g in t for g in greetings):
        return True
    return False

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
    if is_small_talk(query):
        prompt = f"Respond in English and Tamil, politely, to: {query}"
        return generate_with_groq(prompt)
    ctx = retrieve(query, k=4)
    prompt = build_prompt(query, ctx)
    answer = generate_with_groq(prompt)
    return answer, ctx

# ---------------------------
# New UI: hero + centered card, updated CSS
# ---------------------------
st.set_page_config(page_title="TN Vazikatti AI - Government Services Assistant", layout="wide")

# helper to embed local image (if present) as base64 for CSS background/icon usage
def img_to_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

# Path to the uploaded image you mentioned (adjust if different)
local_bot_img = "/mnt/data/WhatsApp Image 2025-11-20 at 3.26.11 PM.jpeg"
bot_base64 = img_to_base64(local_bot_img)

# CSS to recreate the screenshot style
st.markdown(
    f"""
    <style>
    :root{{
      --bg-start: #e9f1ff;
      --bg-end: #f8fbff;
      --card:#ffffff;
      --muted:#8b96a8;
      --blue:#3b82f6;
      --pill:#e8f1ff;
      --shadow: 0 10px 30px rgba(14,30,70,0.08);
    }}

    html, body, [data-testid="stAppViewContainer"] {{
      height: 100%;
      background: linear-gradient(180deg, var(--bg-start), var(--bg-end));
    }}

    /* Centered hero area */
    .hero {{
      display:flex;
      flex-direction:column;
      align-items:center;
      justify-content:center;
      gap:12px;
      padding:48px 10px 10px 10px;
      text-align:center;
    }}
    .hero h1 {{
      margin:0;
      font-size:44px;
      color:#0f172a;
      font-weight:700;
      letter-spacing: -0.5px;
    }}
    .hero p {{
      margin:0;
      color:var(--muted);
      font-size:18px;
    }}

    /* Chat card */
    .chat-card {{
      width:720px;
      max-width:92%;
      background: var(--card);
      border-radius:20px;
      box-shadow: var(--shadow);
      padding:28px;
      margin: 22px auto 48px;
      position:relative;
      overflow:hidden;
    }}

    /* small robot circle on top */
    .robot-circle {{
      width:72px;
      height:72px;
      background: linear-gradient(180deg,#ffffff,#eef6ff);
      border-radius:50%;
      border:6px solid #fff;
      display:flex;
      align-items:center;
      justify-content:center;
      position:absolute;
      top:-36px;
      left:50%;
      transform:translateX(-50%);
      box-shadow:0 8px 20px rgba(59,130,246,0.08);
    }}
    .robot-circle img{{ width:54px; height:54px; object-fit:cover; border-radius:10px; }}

    .chat-title {{
      text-align:center;
      margin-top:18px;
      font-weight:700;
      font-size:22px;
      color:#0b1220;
    }}
    .chat-sub {{ text-align:center; color:var(--muted); margin-bottom:18px; }}

    /* message bubbles */
    .bubble-user {{
      background: var(--blue);
      color: white;
      padding:12px 16px;
      border-radius:20px 20px 4px 20px;
      display:inline-block;
      font-size:15px;
      margin:10px 0;
      max-width:78%;
      float:right;
      clear:both;
    }}
    .bubble-bot {{
      background: #f1f7ff;
      color:#07203b;
      padding:12px 16px;
      border-radius:20px 20px 20px 4px;
      display:inline-block;
      font-size:15px;
      margin:10px 0;
      max-width:78%;
      float:left;
      clear:both;
    }}
    .chat-area {{ min-height:220px; padding:8px 4px; }}

    /* input row */
    .input-row {{
      display:flex;
      gap:10px;
      margin-top:18px;
      align-items:center;
    }}
    .text-input {{
      flex:1;
      padding:12px 16px;
      border-radius:14px;
      border:1px solid #e6eefc;
      font-size:15px;
      outline:none;
    }}
    .send-btn {{
      background: linear-gradient(90deg,#4f46e5,#3b82f6);
      color:white;
      padding:12px 20px;
      border-radius:14px;
      border:none;
      font-weight:600;
      cursor:pointer;
    }}
    .start-btn {{
      width:240px;
      display:block;
      margin:18px auto 0;
      padding:12px 22px;
      border-radius:30px;
      background: linear-gradient(90deg,#3b82f6,#60a5fa);
      color:white;
      border:none;
      font-size:16px;
      font-weight:700;
      cursor:pointer;
      box-shadow: 0 8px 20px rgba(59,130,246,0.18);
    }}

    /* small meta */
    .meta {{ font-size:12px; color:#94a3b8; margin-top:6px; text-align:center; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero
st.markdown(
    """
    <div class="hero">
      <h1>TN Vazikatti AI - TN services Assistant</h1>
      <p>Ask anything about TN Government Services</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Chat card wrapper (we will render interactive parts underneath using Streamlit; but keep the same visual)
card_html = f"""
<div class="chat-card">
  <div class="robot-circle">
    {"<img src='data:image/jpeg;base64," + bot_base64 + "' alt='bot' />" if bot_base64 else "<div style='width:38px;height:38px;border-radius:8px;background:#e6f0ff'></div>"}
  </div>
  <div style="padding-top:30px;">
    <div class="chat-title">Hi! How can I assist you?</div>
    <div class="chat-sub">Type your message below and press Enter</div>
    <div class="chat-area" id="chat-area">
      <!-- chat bubbles will be rendered by Streamlit below -->
    </div>
  </div>
</div>
"""
st.markdown(card_html, unsafe_allow_html=True)

# Sidebar / right column controls (keeping previous functionality but relocated)
col1, col2 = st.columns([3,1])
with col2:
    st.markdown("### Admin")
    st.markdown("---")
    index, meta = load_faiss_index()
    if index is None:
        st.warning("FAISS index not found. Ingest docs & build index.")
    else:
        st.success(f"Index loaded — {len(meta['docs'])} chunks")
    st.markdown("**Upload documents** (.txt, .pdf, .docx, .html)")
    uploaded = st.file_uploader("", accept_multiple_files=True, key="uploader_right")
    if uploaded:
        for f in uploaded:
            p = save_uploaded_file(f)
            saved = extract_and_save_text_from_file(p)
            st.write(f"Saved text: {saved.name if saved else 'Extraction failed'}")
        st.success("Saved uploaded files to raw folder. Now click Build Index.")
    if st.button("Build / Rebuild Index"):
        st.info("Extracting texts from raw files...")
        raw_files = sorted(DATA_RAW.glob("*"))
        for p in raw_files:
            extract_and_save_text_from_file(p)
        st.info("Creating chunks...")
        docs = []
        metas = []
        for txtfile in sorted(TEXT_DIR.glob("*.txt")):
            txt = txtfile.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_text_by_sentences(txt, max_tokens=180, overlap_sentences=1)
            for i,ch in enumerate(chunks):
                docs.append(ch)
                metas.append({"source_file": txtfile.name, "chunk_id": f"{txtfile.stem}__{i}", "title": txtfile.stem})
        if not docs:
            st.warning("No text chunks found.")
        else:
            st.info("Embedding and building FAISS index (this may take a while)...")
            build_faiss_index(docs, metas)
            st.success(f"Index built with {len(docs)} chunks")
    st.markdown("---")
    st.markdown("**Settings**")
    st.write("Model:", LLM_MODEL)
    st.write("Embedding model:", EMBED_MODEL_NAME)

# Chat column: render messages inside the card area by reusing session_state
with col1:
    # ensure history exists
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (user, bot, sources)
    st.markdown("<div style='max-width:720px;margin:0 auto;'>", unsafe_allow_html=True)

    # Input: use text_input and a send button stylized below
    query = st.text_input("", placeholder="Type your message...", key="query_input_ui", label_visibility="collapsed")
    send_clicked = st.button("Send", key="send_button")

    # handle send / enter
    if (send_clicked or (query and st.session_state.get("last_query") != query)):
        if query:
            st.session_state.last_query = query
            with st.spinner("Thinking..."):
                out = tn_answer(query)
            if isinstance(out, tuple):
                answer, ctx = out
            else:
                answer, ctx = out, []
            sources = list({c["source"] for c in ctx})
            st.session_state.history.append((query, answer, sources))
            # clear the input (Streamlit requires rerun to clear)
            st.experimental_rerun()

    # Render chat history within the styled chat card area (we reproduce bubble HTML)
    # We'll render the card again with bubble content (so it appears inside the visual card)
    def render_bubbles(history):
        bubbles_html = ""
        # show last ~8 messages
        for user_text, bot_text, sources in history[-12:]:
            # bot bubble
            bot_html = f"<div class='bubble-bot'>{bot_text.replace(chr(10), '<br>')}</div>"
            user_html = f"<div class='bubble-user'>{user_text}</div>"
            bubbles_html += f"<div style='clear:both;padding:6px 0'>{bot_html}</div>"
            bubbles_html += f"<div style='clear:both;padding:6px 0'>{user_html}</div>"
            if sources:
                bubbles_html += f"<div style='clear:both;padding:4px 0;text-align:center;font-size:12px;color:#94a3b8'>Sources: {', '.join(sources)}</div>"
        return bubbles_html

    # Inject bubble HTML into the card area
    bubbles = render_bubbles(st.session_state.history)
    card_inner = f"""
      <div class="chat-card" style="margin-top:6px;">
        <div style="padding-top:30px;">
          <div style="padding:12px 6px;">
            <div class="chat-area">{bubbles}</div>
            <div class="input-row">
              <input class="text-input" id="streamlit-text-input" placeholder="Ask about govt schemes, services..." value="" />
              <button class="send-btn" onclick="document.querySelector('button[kind=primary]').click()">Ask</button>
            </div>
            <button class="start-btn" onclick="document.querySelector('button[kind=primary]').click()">Start Chat</button>
          </div>
        </div>
      </div>
    """
    st.markdown(card_inner, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with local embeddings + FAISS + Groq LLM. Add docs, build index, then chat.")
