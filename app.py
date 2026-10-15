import os
import streamlit as st
import chromadb

from dotenv import load_dotenv
from google import genai


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Campus AI | Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------- API SETUP ----------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


# ---------- CUSTOM STYLING ----------
# Palette:
#   Black / charcoal : #0B0F14  #11161D  #17191F
#   Greys            : #1F2933  #4B5563  #6B7280  #9CA3AF  #D1D5DB  #E5E7EB  #F3F4F6
#   Light blue       : #7DD3FC  #BAE6FD  #E0F2FE
#   White            : #FFFFFF
st.markdown("""
<style>

    /* ---------- Global ---------- */
    .stApp {
        background-color: #EEF1F5;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* ---------- Sidebar (black / charcoal) ---------- */
    section[data-testid="stSidebar"] {
        background-color: #0B0F14;
        border-right: 1px solid #1F2933;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] label {
        color: #D1D5DB !important;
    }

    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #9CA3AF !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #1F2933 !important;
    }

    /* Sidebar "New chat" button (light blue) */
    section[data-testid="stSidebar"] div.stButton > button {
        background-color: #7DD3FC !important;
        color: #0B0F14 !important;
        border: 1px solid #7DD3FC !important;
        border-radius: 10px !important;
        font-weight: 600;
        min-height: 44px;
        transition: background-color 0.15s ease;
    }

    section[data-testid="stSidebar"] div.stButton > button p {
        color: #0B0F14 !important;
    }

    section[data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #BAE6FD !important;
        border-color: #BAE6FD !important;
    }

    /* ---------- Hero banner ---------- */
    .hero {
        box-sizing: border-box;
        width: 100%;
        min-height: 155px;
        padding: 28px 32px;
        background: linear-gradient(120deg, #0B0F14 0%, #1F2933 70%, #374151 100%);
        margin: 0 auto 28px auto;
        border-radius: 14px;
        border-left: 6px solid #7DD3FC;
        box-shadow: 0 6px 18px rgba(11, 15, 20, 0.18);
    }

    .stApp div.hero h1 {
        color: #FFFFFF !important;
        font-size: clamp(1.6rem, 2.5vw, 2.1rem);
        line-height: 1.25;
        margin: 0 0 12px 0;
    }

    .stApp div.hero p {
        color: #BAE6FD !important;
        font-size: 1rem;
        line-height: 1.5;
        margin: 0;
    }

    /* ---------- Headings and text ---------- */
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] h3,
    .main [data-testid="stMarkdownContainer"] h3 {
        color: #11161D !important;
    }

    [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
    .main [data-testid="stMarkdownContainer"] p {
        color: #1F2933 !important;
    }

    .section-label {
        color: #4B5563 !important;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin: 14px 0 8px 0;
    }

    /* ---------- Suggested question buttons (unchanged look) ---------- */
    div.stButton > button {
        background-color: #17191F !important;
        color: #FFFFFF !important;
        border: 1px solid #3F424B !important;
        border-radius: 10px !important;
        min-height: 55px;
        transition: background-color 0.15s ease;
    }

    div.stButton > button p {
        color: #FFFFFF !important;
    }

    div.stButton > button:hover {
        background-color: #292C34 !important;
        border-color: #818CF8 !important;
    }

    /* ---------- Chat message cards ---------- */
    div[data-testid="stChatMessage"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D1D5DB;
        border-left: 4px solid #7DD3FC;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(17, 22, 29, 0.06);
    }

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] h1,
    div[data-testid="stChatMessage"] h2,
    div[data-testid="stChatMessage"] h3,
    div[data-testid="stChatMessage"] h4,
    div[data-testid="stChatMessage"] strong,
    div[data-testid="stChatMessage"] em {
        color: #11161D !important;
    }

    /* ---------- Chat input ---------- */
    /* Bottom bar that holds the input: match the page background */
    div[data-testid="stBottom"],
    div[data-testid="stBottom"] > div,
    div[data-testid="stBottomBlockContainer"] {
        background: #EEF1F5 !important;
    }

    /* Outer input box */
    div[data-testid="stChatInput"] {
        background-color: #FFFFFF !important;
        border: 1px solid #9CA3AF !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(17, 22, 29, 0.08);
    }

    div[data-testid="stChatInput"]:focus-within {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 3px #BAE6FD !important;
    }

    /* Inner wrappers: no extra background or border */
    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] div[data-baseweb="textarea"],
    div[data-testid="stChatInput"] div[data-baseweb="base-input"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #11161D !important;
        -webkit-text-fill-color: #11161D !important;
        caret-color: #11161D !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #6B7280 !important;
        -webkit-text-fill-color: #6B7280 !important;
        opacity: 1 !important;
    }

    /* Send button */
    div[data-testid="stChatInput"] button {
        background-color: #1F2933 !important;
        border-radius: 8px !important;
    }

    div[data-testid="stChatInput"] button:hover {
        background-color: #38BDF8 !important;
    }

    div[data-testid="stChatInput"] button svg {
        fill: #FFFFFF !important;
        color: #FFFFFF !important;
    }

    /* ---------- Source document expander ---------- */
    div[data-testid="stExpander"] {
        background-color: #F3F4F6;
        border: 1px solid #D1D5DB;
        border-radius: 8px;
    }

    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] p {
        color: #1F2933 !important;
    }

    /* ---------- Divider and footer ---------- */
    .main hr {
        border-color: #D1D5DB !important;
    }

    .main [data-testid="stCaptionContainer"] p {
        color: #6B7280 !important;
    }

</style>
""", unsafe_allow_html=True)


# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("🎓 Campus AI")
    st.caption("University Knowledge Assistant")
    st.divider()

    st.markdown("### Your workspace")
    st.write("Ask questions about your university documents.")

    if st.button("＋ New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("pending_question", None)
        st.rerun()

    st.divider()

    st.markdown("### About")
    st.caption(
        "Powered by Retrieval-Augmented Generation (RAG). "
        "Answers are generated using relevant document excerpts."
    )
    st.caption("Tip: Ask specific questions for better results.")


# ---------- HEADER ----------
st.markdown("""
<div class="hero">
    <h1>University Knowledge Assistant</h1>
    <p>Your academic documents, made easier to explore.</p>
</div>
""", unsafe_allow_html=True)



# ---------- INITIALIZE CHAT HISTORY ----------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- SUGGESTED QUESTION BUTTONS ----------
if not st.session_state.messages:
    st.markdown(
        '<div class="section-label">Try asking</div>',
        unsafe_allow_html=True
    )

    suggested_questions = [
        ("📚 What are the topics in DSA?", "What are the topics in DSA?"),
        ("🗄️ What are the topics in DBMS?", "What are the topics in DBMS?"),
        ("📅 When are term papers starting?", "When are term examinations starting?"),
        ("❄️ When are winter vacations?", "When are winter vacations?"),
    ]

    for row_start in range(0, len(suggested_questions), 2):
        cols = st.columns(2)
        for col, (label, query) in zip(
            cols, suggested_questions[row_start:row_start + 2]
        ):
            with col:
                if st.button(
                    label,
                    use_container_width=True,
                    key=f"suggest_{query}"
                ):
                    st.session_state.pending_question = query
                    st.rerun()


# ---------- DISPLAY CHAT HISTORY ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📄 View source documents"):
                for source in message["sources"]:
                    st.write(source)


# ---------- CHAT INPUT ----------
typed_question = st.chat_input(
    "Ask a question about your university syllabus..."
)

question = typed_question

if not question and "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")


# ---------- PROCESS QUESTION ----------
if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):

            try:
                # Check API key
                if not api_key:
                    raise ValueError(
                        "API key not found. Check your .env file."
                    )

                # Initialize Gemini
                client = genai.Client(api_key=api_key)

                # Connect to ChromaDB
                db = chromadb.PersistentClient(
                    path="./chroma_db"
                )

                collection = db.get_collection(
                    name="university_documents"
                )

                # Embed the question
                embedding_response = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=question,
                    config={
                        "task_type": "RETRIEVAL_QUERY"
                    }
                )

                query_embedding = (
                    embedding_response.embeddings[0].values
                )

                # Retrieve relevant document chunks
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=4,
                    include=["documents", "metadatas"]
                )

                documents = results["documents"][0] or []
                metadatas = results["metadatas"][0] or []

                # Build context
                context = "\n\n".join(
                    document for document in documents
                    if document
                )

                sources = sorted({
                    metadata.get("source", "Unknown")
                    for metadata in metadatas
                    if metadata
                })

                # Generate grounded answer
                if not context.strip():
                    answer = (
                        "I couldn't find relevant information "
                        "in the provided university documents."
                    )

                else:
                    prompt = f"""
You are a helpful University Knowledge Assistant.

Answer using ONLY the provided document context.
Do not invent facts, dates, or university policies.

If the answer is not in the context, say:
"I couldn't find that information in the provided university documents."

Context:
{context}

Question:
{question}

Give a clear, student-friendly answer.
"""

                    response = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=prompt
                    )

                    answer = response.text or (
                        "I couldn't generate an answer. Please try again."
                    )

                # Display answer
                st.markdown(answer)

                # Display source documents
                if sources:
                    with st.expander("📄 View source documents"):
                        for source in sources:
                            st.write(source)

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(
                    "Something went wrong while generating your answer."
                )
                st.caption(f"Error details: {e}")


# ---------- FOOTER ----------
st.divider()

st.caption(
    "Campus AI • University Knowledge Assistant • "
    "Always verify important academic information against official documents."
)