# Chatbot with Memory

A conversational chatbot built with **LangChain** and **Streamlit** that remembers your conversation and can switch between different personas.

## What It Does

- Chat with an AI that **remembers the full conversation** (not just the last message)
- Switch between 5 personas: Helpful Assistant, Pirate, Socratic Teacher, Stand-up Comedian, and Explain Like I'm 5
- Clear conversation history with one click

## LangChain Concepts Used

| Concept | What It Does |
|---|---|
| `ChatGoogleGenerativeAI` | Connects to Google Gemini as the LLM |
| `ChatPromptTemplate` | Structures the prompt (system persona + history + user input) |
| `MessagesPlaceholder` | Injects conversation history into the prompt dynamically |
| `StrOutputParser` | Parses the LLM response into a plain string |
| LCEL pipe syntax (`\|`) | Chains components together: `prompt \| llm \| parser` |
| `RunnableWithMessageHistory` | Automatically manages conversation memory across turns |

## Setup

1. **Get a free Google Gemini API key** from [aistudio.google.com](https://aistudio.google.com)

2. **Create a `.env` file** in the project root:
   ```
   GOOGLE_API_KEY=your_key_here
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run:**
   ```bash
   streamlit run chatbot/app.py
   ```

The app opens at `http://localhost:8501`. Pick a persona from the sidebar and start chatting.
