"""
Project 1: Conversational Chatbot with Memory
==============================================
A Streamlit chatbot that uses Google Gemini via LangChain.
It remembers conversation history and can adopt different personas.

Key LangChain concepts:
  - ChatGoogleGenerativeAI  (LLM connection)
  - ChatPromptTemplate      (structuring prompts)
  - ConversationBufferMemory (retaining chat history)
  - LLMChain / LCEL          (chaining components)

Run:  streamlit run chatbot/app.py
"""

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ── Load environment variables (.env should contain GOOGLE_API_KEY) ──────────
load_dotenv()

# ── Persona options ──────────────────────────────────────────────────────────
PERSONAS = {
    "Helpful Assistant": "You are a helpful, friendly assistant. Answer clearly and concisely.",
    "Pirate": (
        "You are a pirate captain. Respond in pirate-speak with 'Arrr', 'matey', "
        "'ye', etc. Be entertaining but still answer the question accurately."
    ),
    "Socratic Teacher": (
        "You are a Socratic teacher. Never give the answer directly. Instead, "
        "guide the student by asking probing questions that lead them to discover "
        "the answer on their own."
    ),
    "Stand-up Comedian": (
        "You are a stand-up comedian. Answer every question with humor, jokes, "
        "and witty observations, but still provide useful information."
    ),
    "Explain Like I'm 5": (
        "You explain everything as if you're talking to a 5-year-old. Use simple "
        "words, fun analogies, and short sentences."
    ),
}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return (or create) the chat history for a given session."""
    if session_id not in st.session_state["histories"]:
        st.session_state["histories"][session_id] = InMemoryChatMessageHistory()
    return st.session_state["histories"][session_id]


def build_chain(persona_prompt: str):
    """
    Build the LangChain LCEL chain:
      prompt  ➜  LLM  ➜  string output parser

    Then wrap it with RunnableWithMessageHistory so that conversation
    memory is automatically injected.
    """
    # 1️⃣  LLM — Google Gemini (free tier)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
    )

    # 2️⃣  Prompt template — system persona + history placeholder + user input
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", persona_prompt),
            MessagesPlaceholder(variable_name="history"),  # auto-filled by memory
            ("human", "{input}"),
        ]
    )

    # 3️⃣  Chain (LCEL pipe syntax)
    chain = prompt | llm | StrOutputParser()

    # 4️⃣  Wrap with message history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    return chain_with_history


# ══════════════════════════════════════════════════════════════════════════════
#  Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="LangChain Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 LangChain Chatbot with Memory")
st.caption("Powered by Google Gemini (free tier) + LangChain")

# ── Sidebar: persona selector ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    selected_persona = st.selectbox("Choose a persona:", list(PERSONAS.keys()))

    if st.button("🗑️ Clear conversation"):
        st.session_state["messages"] = []
        st.session_state["histories"] = {}
        st.rerun()

    st.divider()
    st.markdown(
        "**How it works:**\n"
        "1. Your message goes into a **ChatPromptTemplate** "
        "(system persona + history + your input).\n"
        "2. The template is sent to **Google Gemini** via LangChain.\n"
        "3. **RunnableWithMessageHistory** automatically injects "
        "past messages so the bot *remembers* the conversation."
    )

# ── Initialise session state ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "histories" not in st.session_state:
    st.session_state["histories"] = {}

# ── Display existing messages ────────────────────────────────────────────────
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Handle new user input ────────────────────────────────────────────────────
if user_input := st.chat_input("Type your message…"):
    # Show user message immediately
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build chain with selected persona
    chain = build_chain(PERSONAS[selected_persona])

    # Invoke the chain (session_id ties into our memory store)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "default"}},
            )
        st.markdown(response)

    # Save assistant response to display history
    st.session_state["messages"].append({"role": "assistant", "content": response})
