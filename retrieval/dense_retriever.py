from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient

embedder = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

client = QdrantClient(
    url="http://localhost:6333"
)

COLLECTION = "emotion_rag"


class DenseRetriever:

    def search(self, query, top_k=5):

        query_embedding = embedder.encode(
            query,
            normalize_embeddings=True
        )

        results = client.search(
            collection_name=COLLECTION,
            query_vector=query_embedding,
            limit=top_k
        )

        return [
            r.payload["text"]
            for r in results
        ]