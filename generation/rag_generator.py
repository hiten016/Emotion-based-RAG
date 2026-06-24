from generation.model import llm


def generate_rag(query, contexts):

    context = "\n".join(contexts)

    prompt = f"""
Use the provided context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

    output = llm(
        prompt,
        max_new_tokens=300
    )

    return output[0]["generated_text"]