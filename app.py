# app.py

import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Pinecone index details
PINECONE_INDEX_NAME = "diy-kit-support"
index = pc.Index(PINECONE_INDEX_NAME)

# OpenAI embedding model
OPENAI_EMBED_MODEL = "text-embedding-3-small"

# ==============================
# 🧑‍🏫 Mentor-style instruction (shared across cases)
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
# 🔎 Answer function
# ==============================
def answer_query_with_confidence(
    user_query: str,
    namespace: str = "diy_kit_support_chunks",
    top_k: int = 5,
    confidence_threshold: float = 0.6
):
    """
    Answers a query using Pinecone + OpenAI with tiered mentor-style fallbacks.
    Designed for kids and parents using Butterfly Fields DIY kits.
    """

    # 1️⃣ Get query embedding from OpenAI
    query_emb = client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=user_query
    ).data[0].embedding

    # 2️⃣ Search Pinecone for top matches
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

    # Case-specific instruction
    if best_score >= confidence_threshold:
        case_instruction = "Use the context below to answer directly from the manual."
    elif "kit" in user_query.lower() or "activity" in user_query.lower():
        case_instruction = (
            "The context is not an exact match, but since the question is about a kit or activity, "
            "give general guidance that could still help."
        )
    else:
        return (
            "🤔 Sorry, I don’t see that in the Butterfly Fields manuals. 🦋  \n"
            "👉 Please ask again with the kit name or the activity name so I can help better! \n"
	    "\nHere are some kits I can answer about: \n"
            f"{kit_list}"
        )

    # 4️⃣ Build context string from matches
    context_parts = []
    for match in results["matches"]:
        meta = match.get("metadata", {})
        title = meta.get("kit_title", "")
        activity = meta.get("activity_title", "")
        section = meta.get("chunk_type", "")
        text = meta.get("text_content", "")
        context_parts.append(f"[{title}] {activity} ({section}):\n{text}")

    context = "\n\n".join(context_parts)

    # 5️⃣ Build final prompt
    final_instruction = f"{MENTOR_STYLE_INSTRUCTION}\n\n{case_instruction}"
    prompt = f"""{final_instruction}

Context:
{context}

User Query: {user_query}
"""

    # 6️⃣ Call OpenAI Chat
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

# ==============================
# 🌐 Streamlit Web App
# ==============================
st.title("🦋 Butterfly DIY Assistant")

st.write("Ask me questions about your Butterfly Fields DIY kits. I’ll guide you like a friendly mentor!")

question = st.text_input("🔍 Type your question here:")

if st.button("Get Answer"):
    if question.strip():
        with st.spinner("Thinking... 🧠"):
            answer = answer_query_with_confidence(question)
        st.write("### ✨ Answer")
        st.write(answer)
    else:
        st.warning("Please enter a question first.")
