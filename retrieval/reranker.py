from sentence_transformers import CrossEncoder

class ReRanker:

    def __init__(self):

        self.model = CrossEncoder(
            "BAAI/bge-reranker-base"
        )

    def rerank(
        self,
        query,
        docs,
        top_k=3
    ):

        pairs = [
            (query, d)
            for d in docs
        ]

        scores = self.model.predict(
            pairs
        )

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            doc
            for doc, score in ranked[:top_k]
        ]