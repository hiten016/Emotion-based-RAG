def choose_path(
    intent,
    emotion,
    safety
):

    if safety["score"] > 0.9:

        return "SAFE_RESPONSE"

    if intent == "Knowledge":

        return "RAG"

    if emotion in [
        "sadness",
        "fear",
        "grief"
    ]:

        return "EMPATHETIC_RAG"

    return "CHAT"