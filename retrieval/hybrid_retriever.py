from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever


class HybridRetriever:

    def __init__(self, documents):

        self.bm25 = BM25Retriever(documents)
        self.dense = DenseRetriever()

    def reciprocal_rank_fusion(
        self,
        bm25_docs,
        dense_docs,
        k=60
    ):

        scores = {}

        for rank, doc in enumerate(bm25_docs):

            scores[doc] = scores.get(doc, 0) + (
                1 / (k + rank + 1)
            )

        for rank, doc in enumerate(dense_docs):

            scores[doc] = scores.get(doc, 0) + (
                1 / (k + rank + 1)
            )

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            doc
            for doc, score in ranked
        ]

    def search(
        self,
        query,
        top_k=10
    ):

        bm25_results = self.bm25.search(
            query,
            top_k=top_k
        )

        dense_results = self.dense.search(
            query,
            top_k=top_k
        )

        fused_results = self.reciprocal_rank_fusion(
            bm25_results,
            dense_results
        )

        return fused_results[:top_k]