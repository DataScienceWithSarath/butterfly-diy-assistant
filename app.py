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
def answer_query_with_confidence(
    user_query: str,
    chat_history: list = None,
    namespace: str = "diy_kit_support_chunks",
    top_k: int = 5,
    confidence_threshold: float = 0.5
):
    """
    Answers a query using Pinecone + OpenAI with tiered mentor-style fallbacks.
    Includes chat history for memory.
    """

    # 1️⃣ Get query embedding
    query_emb = client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=user_query
    ).data[0].embedding

    # 2️⃣ Pinecone search
    results = index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )

    if not results.get("matches"):
        return (
            "🤔 I couldn’t find anything in the Butterfly Fields DIY kit manuals.  \n"
            "🦋 I can only help with Butterfly Fields kits.  \n"
            "👉 Try asking again, and please tell me the kit name or the activity name. "
            "\nHere are some kits I can answer about:\n"
            f"{kit_list}"
        )

    # 3️⃣ Confidence check
    best_score = results["matches"][0]["score"]

    if best_score >= confidence_threshold:
        case_instruction = "Use the context below to answer directly from the manual."
    else:
        return (
            "🤔 Sorry, I don’t see that in the Butterfly Fields manuals. 🦋  \n"
            "I can only answer questions about Butterfly Fields DIY kits.  \n"
            "Could you tell me which kit or activity you mean?  \n"
            "\nHere are some kits I can answer about:\n"
            f"{kit_list}"
        )

    # 4️⃣ Build context
    context_parts = []
    for match in results["matches"]:
        meta = match.get("metadata", {})
        title = meta.get("kit_title", "")
        activity = meta.get("activity_title", "")
        section = meta.get("chunk_type", "")
        text = meta.get("text_content", "")
        context_parts.append(f"[{title}] {activity} ({section}):\n{text}")

    context = "\n\n".join(context_parts)

    # 5️⃣ Final prompt with history
    final_instruction = f"{MENTOR_STYLE_INSTRUCTION}\n\n{case_instruction}"
    context_block = f"Context from manuals:\n{context}" if context else "No relevant context found."

    messages = [{"role": "system", "content": final_instruction}]

    if chat_history:
        for m in chat_history:
            messages.append(m)

    messages.append({
        "role": "user",
        "content": f"{context_block}\n\nUser Query: {user_query}"
    })

    # 6️⃣ OpenAI chat
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

# ==============================
# 🌐 Streamlit Chat App
# ==============================
st.title("🦋 Butterfly DIY Assistant")
st.write("Chat with me about your Butterfly Fields DIY kits. I’ll guide you like a friendly mentor!")

# Initialize history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input at bottom
if user_input := st.chat_input("Ask me something about your kits..."):
    # Show user input
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🧠"):
            answer = answer_query_with_confidence(
                user_input,
                chat_history=st.session_state.messages
            )
            st.markdown(answer)

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": answer})
