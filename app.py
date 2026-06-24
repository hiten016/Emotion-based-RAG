from classifiers.intent_classifier import classify_intent
from classifiers.emotion_classifier import detect_emotion
from classifiers.safety_classifier import check_safety

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import ReRanker

from decision.decision_engine import choose_path

from generation.rag_generator import generate_rag
from generation.direct_generator import chat

from vectorstore.load_documents import load_documents

documents = load_documents()

hybrid = HybridRetriever(documents)

reranker = ReRanker()

while True:

    query = input("\nUser: ")

    if query.lower() == "exit":
        break

    intent = classify_intent(query)

    emotion = detect_emotion(query)

    safety = check_safety(query)

    route = choose_path(
        intent,
        emotion,
        safety
    )

    print("Intent :", intent)

    print("Emotion:", emotion)

    print("Safety :", safety)

    print("Route  :", route)


    if route == "RAG":

        docs = hybrid.search(
            query,
            top_k=10
        )

        print("\nRetrieved Documents:\n")

        for i, doc in enumerate(docs[:3]):

            print(f"\n----- DOC {i+1} -----")

            print(doc[:300])

        docs = reranker.rerank(
            query,
            docs,
            top_k=3
        )

        answer = generate_rag(
            query,
            docs
        )

    else:

        answer = chat(query)

    print("\nAssistant:\n")

    print(answer)

    print("\n")