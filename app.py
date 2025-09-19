# Paste your full app.py code here

import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
import os

# ==============================
# 🔑 API Keys
# ==============================
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# ==============================
# 📦 Pinecone Setup
# ==============================
PINECONE_INDEX_NAME = "diy-kit-support"
index = pc.Index(PINECONE_INDEX_NAME)

# OpenAI embedding model
OPENAI_EMBED_MODEL = "text-embedding-3-small"

# ==============================
# 🧑‍🏫 Mentor-style instruction
# ==============================
MENTOR_STYLE_INSTRUCTION = (
    "You are a friendly mentor helping kids and parents with Butterfly Fields DIY kits. "
    "Always explain things in a clear, step-by-step way, simple enough for a child but also helpful for parents. "
    "Keep your tone encouraging and supportive, like a teacher guiding a curious student. "
    "Give answers in short, bite-sized pieces (2–4 sentences max). "
    "Whenever possible, use simple bullets (•) or numbered steps (1, 2, 3). "
    "If the instruction manuals provide the answer, use only them to give your answer. "
    "Otherwise, say you don't know. "
)

# Predefined kit list
KNOWN_KITS = [
    "Fun With Magnets",
    "Components of food",
    "Integers Positive and Negative",
    "Separation of substances",
    "Mensuration – Area perimeter",
    "Motion and measurement of distance",
    "India Natural Resources Minerals and Rocks",
    "Journey of a water drop",
    "Practical Geometry Constructions",
    "Numbers Factors Multiples"
]
kit_list = "\n".join([f"- {kit}" for kit in KNOWN_KITS])

# ==============================
# 🔎 Query Function
# ==============================
def answer_query_with_confidence_2(user_query, chat_history, threshold=0.5, max_history_turns=10):

    # --- Step 2: Query Pinecone for relevant context ---
    embedding = client.embeddings.create(
        input = user_query,
        model = "text-embedding-3-small",
    ).data[0].embedding

    results = index.query(
        vector=embedding,
        top_k=5,
        namespace="diy_kit_support_chunks",
        include_metadata=True
    )

    best_score = results.matches[0].score if results.matches else 0
    context_texts = [m.metadata.get("text_content", "") for m in results.matches]

    #print(f"Best score: {best_score}")

    # --- Step 3: Build messages ---
    llm_prompt = []

    # 3.1 : Add system instruction to LLM prompt - ROLE: SYSTEM
    llm_prompt.append({"role": "system", "content": MENTOR_STYLE_INSTRUCTION})

    # 3.2 : Add fallback response
    fallback_response = (
          f"🤔 Sorry, I don’t see that in the Butterfly Fields manuals. 🦋  \n"
          "Could you tell me which kit or activity you mean?  \n"
          "\nHere are some kits I can answer about:\n"
          f"{kit_list}"
      )
    
    # 3.2 : Add chat history to LLM prompt - MEMORY
    for msg in chat_history:
        llm_prompt.append(msg)

    # 3.3: Add retrieved docs to LLM prompt - KNOWLEDGE BASE
    if best_score >= threshold:
      context_block = "\n\n".join(context_texts)
      context = f"Relevant context from instruction manuals:\n{context_block}"
      llm_prompt.append({"role": "user", "content": context})
    else:
      context = fallback_response  # empty -> fallback response
      llm_prompt.append({"role": "assistant", "content": context})

    # 3.4: Add user query to LLM prompt
    llm_prompt.append({"role": "user", "content": user_query})

    # --- Step 4: Generate answer ---
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 💰 cheaper model for main chat
        messages=llm_prompt,
        temperature=0.2
    )

    return response.choices[0].message.content, best_score, context_texts, fallback_response

# ==============================
# 🌐 Streamlit Chat App
# ==============================
st.title("🦋 Butterfly DIY Assistant")
st.write("Chat with me about your Butterfly Fields DIY kits. I’ll guide you in building your kits!")

# Reset button
if st.button("🔄 Reset Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# Initialize history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Initialize debug variables
best_score = 0.0
context_texts = []
fallback_response = ""
user_input = None  # Also initialize this to avoid NameError in debug panel

# Input at bottom
if user_input := st.chat_input("Ask me something about your kits..."):
    # Show user input (on screen, not yet in history)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🧠"):
            answer, best_score, context_texts, fallback_response = answer_query_with_confidence_2(
                user_query=user_input,
                chat_history=st.session_state.messages  # history excludes this new input
            )
            st.markdown(answer)

    # Save both messages to history AFTER response
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Optional: Debug panel
with st.expander("🛠 Debug Info"):
    st.markdown(f"**User Query:** {user_input}")
    st.markdown(f"**Best Match Score:** {best_score:.4f}")

    if best_score >= 0.75:
        st.markdown("**Retrieved Context Chunks:**")
        for i, chunk in enumerate(context_texts, 1):
            st.markdown(f"**Chunk {i}:** {chunk}")
    else:
        st.markdown("**Fallback Triggered:** Showing kit list instead of context.")
        st.markdown(f"**Fallback Response:**\n{fallback_response}")
