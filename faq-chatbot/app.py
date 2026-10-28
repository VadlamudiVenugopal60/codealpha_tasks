"""Streamlit chat interface for the FAQ chatbot.

Run from the project root:
    streamlit run app.py
"""

import streamlit as st

from src.chatbot import DEFAULT_THRESHOLD, FAQChatbot, FAQDataError

WELCOME = (
    "Hi! I'm a college FAQ bot. Ask me about admissions, fees, exams, hostels, "
    "the library or placements."
)
EXAMPLES = [
    "How do I apply for admission?",
    "What is the minimum attendance required?",
    "What are the library timings?",
    "How can I get a bonafide certificate?",
]

st.set_page_config(page_title="College FAQ Chatbot", page_icon="💬")


@st.cache_resource
def load_bot():
    """Load the CSV and fit TF-IDF once; Streamlit reuses it across reruns."""
    return FAQChatbot()


def new_chat():
    st.session_state.messages = [{"role": "assistant", "content": WELCOME}]


st.title("College FAQ Chatbot")
st.caption(
    "A retrieval-based chatbot built with NLP: it finds the most similar FAQ "
    "using TF-IDF and cosine similarity. It does not generate new answers."
)

try:
    bot = load_bot()
except FAQDataError as error:
    st.error(f"Could not load the FAQ data: {error}")
    st.stop()

if "messages" not in st.session_state:
    new_chat()

with st.sidebar:
    st.header("Try these")
    for example in EXAMPLES:
        if st.button(example):
            st.session_state.pending_question = example
    st.divider()
    if st.button("Clear chat"):
        new_chat()
        st.rerun()
    st.divider()
    debug = st.checkbox("Debug mode (show match and score)")
    threshold = st.slider("Similarity threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)
    bot.threshold = threshold

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if debug and message.get("debug"):
            st.caption(message["debug"])

typed = st.chat_input("Ask a question...")
question = typed or st.session_state.pop("pending_question", None)

if question is not None:
    try:
        response = bot.get_answer(question)
        reply, debug_text = response.message, (
            f"status={response.status} | score={response.score:.2f} | "
            f"best match: {response.matched_question or '-'}"
        )
    except Exception:  # keep the UI alive whatever happens
        reply, debug_text = "Something went wrong. Please try again.", ""
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "debug": debug_text}
    )
    st.rerun()
