# app.py

import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

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
    "If the manuals provide the answer, use them directly. "
    "If the manuals don’t fully cover it, try to give a helpful kit-related explanation "
    "without making up unrelated information."
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
def answer_query_with_confidence(user_query, chat_history, threshold=0.5, max_history_turns=10):
    # --- Step 1: Summarize old history if needed ---
    if len(chat_history) > max_history_turns:
        # Take all but the last 5 turns and compress them
        old_history = chat_history[:-5]
        recent_history = chat_history[-5:]

        # Ask the model to summarize old conversation
        summary_prompt = [
            {"role": "system", "content": "Summarize this conversation briefly so it can be used as context."},
            {"role": "user", "content": str(old_history)}
        ]
        summary_resp = client.chat.completions.create(
            model="gpt-3.5-turbo",   # cheaper summarizer
            messages=summary_prompt,
            temperature=0
        )
        summary_text = summary_resp.choices[0].message.content

        # Replace old history with one summary message
        chat_history = [{"role": "system", "content": f"Conversation so far (summary): {summary_text}"}] + recent_history

    # --- Step 2: Query Pinecone for relevant context ---
    results = index.query(
        vector=client.embeddings.create(
            input=user_query,
            model="text-embedding-3-small"
        ).data[0].embedding,
        top_k=3,
        include_metadata=True
    )

    best_score = results.matches[0].score if results.matches else 0
    context_texts = [m.metadata.get("text", "") for m in results.matches]

    # --- Step 3: Build messages ---
    messages = [
        {"role": "system", "content": MENTOR_STYLE_INSTRUCTION}
    ]

    # Add history (already summarized if too long)
    for m in chat_history:
        messages.append(m)

    # Add query with or without retrieved docs
    if best_score >= threshold:
        context_block = "\n\n".join(context_texts)
        user_input = f"User query: {user_query}\n\nRelevant manual sections:\n{context_block}"
    else:
        user_input = f"User query: {user_query}\n\n🤔 Sorry, I don’t see that in the Butterfly Fields manuals. 🦋  \n"
            "I can only answer questions about Butterfly Fields DIY kits.  \n"
            "Could you tell me which kit or activity you mean?  \n"
            "\nHere are some kits I can answer about:\n"
            f"{kit_list}"

    messages.append({"role": "user", "content": user_input})

    # --- Step 4: Generate answer ---
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 💰 cheaper model for main chat
        messages=messages,
        temperature=0.3
    )

    return response.choices[0].message.content

# ==============================
# 🌐 Streamlit Chat App
# ==============================
st.title("🦋 Butterfly DIY Assistant")
st.write("Chat with me about your Butterfly Fields DIY kits. I’ll guide you in building your kits!")

# Initialize history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input at bottom
if user_input := st.chat_input("Ask me something about your kits..."):
    # Show user input (on screen, not yet in history)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🧠"):
            answer = answer_query_with_confidence(
                user_query=user_input,
                chat_history=st.session_state.messages  # history excludes this new input
            )
            st.markdown(answer)

    # Save both messages to history AFTER response
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": answer})
