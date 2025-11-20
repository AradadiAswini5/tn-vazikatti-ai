# streamlit_app.py
# Streamlit UI: Screenshot replica (Option B1) — single working input + "Start Chat" resets history
import streamlit as st
from pathlib import Path
import html
import datetime

st.set_page_config(page_title="TN Vazikatti AI — Chat", layout="wide")

# ---------- Try to reuse user's existing tn_answer function if available ----------
try:
    # if tn_answer is defined elsewhere (e.g. in your original file), import it
    from tn_vazikatti_backend import tn_answer  # try user-provided backend module (optional)
except Exception:
    try:
        # maybe tn_answer exists in globals from other imports
        tn_answer  # reference to check existence
    except Exception:
        # fallback placeholder (returns (answer_text, context_chunks_list))
        def tn_answer(query):
            ans = f"**English:**\nThis is a placeholder answer for: {html.escape(query)}\n\n**Tamil:**\nஇது குறுநூல் பதில்: {html.escape(query)}"
            return ans, []
# ---------- End backend hookup ----------

# Basic CSS to reproduce the screenshot-like style (no images)
st.markdown(
    """
    <style>
    :root{
      --bg-start: #eaf3ff;
      --bg-end: #fbfdff;
      --card: #ffffff;
      --muted:#6b7280;
      --blue:#2563eb;
      --shadow: 0 10px 30px rgba(14,30,70,0.08);
    }
    html, body, [data-testid="stAppViewContainer"] {
      background: linear-gradient(180deg, var(--bg-start), var(--bg-end));
      height:100%;
    }
    .center {
      display:flex;
      justify-content:center;
      margin-top:28px;
    }
    .chat-card {
      width:720px;
      max-width:94%;
      background: var(--card);
      border-radius:18px;
      box-shadow: var(--shadow);
      padding:20px 24px 16px 24px;
    }
    .title { text-align:center; font-weight:700; font-size:22px; color:#0b1220; margin-top:6px;}
    .subtitle { text-align:center; color:var(--muted); margin-bottom:12px; }
    .chat-area { min-height:240px; padding:8px 6px; max-height:520px; overflow:auto; }
    .bubble-user {
      background: var(--blue);
      color:white;
      padding:10px 14px;
      border-radius:18px 18px 4px 18px;
      display:inline-block;
      margin:8px 0;
      font-size:15px;
      float:right;
      clear:both;
      max-width:78%;
    }
    .bubble-bot {
      background:#f4f8ff;
      color:#07203b;
      padding:10px 14px;
      border-radius:18px 18px 18px 4px;
      display:inline-block;
      margin:8px 0;
      font-size:15px;
      float:left;
      clear:both;
      max-width:78%;
    }
    .sources {
      font-size:12px;
      color:#94a3b8;
      text-align:center;
      margin-top:6px;
    }
    .input-row { display:flex; gap:10px; align-items:center; margin-top:16px; }
    .text-input { flex:1; padding:12px 14px; border-radius:14px; border:1px solid #e6eefc; font-size:15px; }
    .send-btn { padding:10px 16px; border-radius:12px; background:linear-gradient(90deg,#3b82f6,#60a5fa); color:white; border:none; font-weight:600; cursor:pointer; }
    .start-btn { margin:12px auto 0; display:block; padding:10px 20px; border-radius:999px; background:linear-gradient(90deg,#3b82f6,#4f46e5); color:white; font-weight:700; border:none; cursor:pointer; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Layout ----------------
with st.sidebar:
    st.header("Admin")
    st.markdown("Upload docs and build index (optional).")
    uploaded = st.file_uploader("Upload documents (.pdf, .txt, .docx, .html)", accept_multiple_files=True)
    if uploaded:
        st.write("Files uploaded (saved to /tmp for now):")
        for f in uploaded:
            tmp = Path("/tmp") / f.name
            tmp.write_bytes(f.getvalue())
            st.write("-", f.name)
        st.success("Files saved to /tmp. Now run your ingestion script / Build Index.")
    if st.button("Build / Rebuild Index"):
        st.info("Trigger your build process (not implemented in this UI file).")

# central chat card
st.markdown('<div class="center"><div class="chat-card">', unsafe_allow_html=True)

st.markdown('<div class="title">Hi! How can I assist you?</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Type your message below and press Enter</div>', unsafe_allow_html=True)

# session state initialization
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"user": str, "bot": str, "sources": [str], "ts": str}

# Start Chat button (clears chat)
if st.button("Start Chat"):
    st.session_state.history = []
    # immediate UI refresh
    st.experimental_set_query_params(_ts=str(datetime.datetime.utcnow().timestamp()))
    st.rerun()

# Chat area (rendered)
chat_area_html = '<div class="chat-area">'
if st.session_state.history:
    for item in st.session_state.history:
        user_html = f'<div class="bubble-user">{html.escape(item["user"])}</div>'
        bot_html = f'<div class="bubble-bot">{item["bot"].replace(chr(10), "<br>")}</div>'
        chat_area_html += f'<div style="clear:both;padding:6px 0">{user_html}</div>'
        chat_area_html += f'<div style="clear:both;padding:6px 0">{bot_html}</div>'
        if item.get("sources"):
            srcs = ", ".join(item["sources"])
            chat_area_html += f'<div class="sources">Sources: {html.escape(srcs)}</div>'
else:
    # placeholder first bot greeting (optional)
    greeting = "Hello! How may I assist you today?\nவணக்கம்! இன்று நான் எப்படி உங்களுக்கு உதவலாம்?"
    chat_area_html += f'<div class="bubble-bot">{greeting.replace(chr(10), "<br>")}</div>'

chat_area_html += "</div>"
st.markdown(chat_area_html, unsafe_allow_html=True)

# Input form (single working input)
with st.form(key="chat_form", clear_on_submit=True):
    query = st.text_input(" ", placeholder="Ask about govt schemes, services...", key="input_box", label_visibility="collapsed")
    submit = st.form_submit_button("Send")
    if submit and query:
        # call backend
        with st.spinner("Thinking..."):
            try:
                resp, ctx = tn_answer(query)
            except Exception as e:
                resp = f"Error generating answer: {e}"
                ctx = []
        # normalize ctx sources to list of strings if provided
        try:
            sources = []
            for c in ctx:
                if isinstance(c, dict) and c.get("source"):
                    sources.append(c.get("source"))
                elif isinstance(c, str):
                    sources.append(c)
        except Exception:
            sources = []
        # append in correct order: user then bot
        st.session_state.history.append({"user": query, "bot": resp, "sources": sources, "ts": datetime.datetime.utcnow().isoformat()})
        # re-run to immediately show new message (form clears because clear_on_submit=True)
        st.rerun()

st.markdown('</div></div>', unsafe_allow_html=True)

# footer / small note
st.markdown("<div style='text-align:center;margin-top:8px;color:#94a3b8;font-size:13px'>Built with local embeddings + FAISS + LLM. Add docs in the sidebar & build index.</div>", unsafe_allow_html=True)
