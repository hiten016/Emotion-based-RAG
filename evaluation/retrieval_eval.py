import json
import numpy as np


class RetrievalEvaluator:

    def __init__(self, retriever):
        self.retriever = retriever

    def recall_at_k(
        self,
        retrieved_ids,
        relevant_ids,
        k
    ):

        retrieved_k = retrieved_ids[:k]

        hits = len(
            set(retrieved_k).intersection(
                set(relevant_ids)
            )
        )

        return hits / len(relevant_ids)

    def mrr(
        self,
        retrieved_ids,
        relevant_ids
    ):

        for rank, doc_id in enumerate(
            retrieved_ids,
            start=1
        ):

            if doc_id in relevant_ids:
                return 1 / rank

        return 0

    def hit_rate(
        self,
        retrieved_ids,
        relevant_ids,
        k
    ):

        retrieved_k = retrieved_ids[:k]

        return int(
            len(
                set(retrieved_k).intersection(
                    set(relevant_ids)
                )
            ) > 0
        )

    def evaluate(
        self,
        dataset_path
    ):

        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        recall_scores = []
        mrr_scores = []
        hit_scores = []

        for sample in dataset:

            query = sample["question"]

            relevant_ids = sample[
                "relevant_doc_ids"
            ]

            results = self.retriever.search(
                query,
                top_k=10
            )

            retrieved_ids = [
                r["id"]
                for r in results
            ]

            recall_scores.append(
                self.recall_at_k(
                    retrieved_ids,
                    relevant_ids,
                    k=5
                )
            )

            mrr_scores.append(
                self.mrr(
                    retrieved_ids,
                    relevant_ids
                )
            )

            hit_scores.append(
                self.hit_rate(
                    retrieved_ids,
                    relevant_ids,
                    k=5
                )
            )

        metrics = {
            "Recall@5":
                np.mean(recall_scores),

            "MRR":
                np.mean(mrr_scores),

            "HitRate@5":
                np.mean(hit_scores)
        }

        return metrics