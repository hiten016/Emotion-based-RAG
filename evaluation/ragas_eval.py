

from datasets import Dataset
from ragas import evaluate

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
)


class RagasEvaluator:

    def __init__(self):
        pass

    def run(
        self,
        questions,
        answers,
        contexts,
        ground_truths
    ):

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths
            }
        )

        result = evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(),
                AnswerRelevancy(),
                ContextPrecision(),
                ContextRecall()
            ]
        )

        return result


if __name__ == "__main__":

    questions = [
        "What is RAG?"
    ]

    answers = [
        "Retrieval-Augmented Generation combines retrieval and generation."
    ]

    contexts = [
        [
            "RAG combines retrieval with language generation."
        ]
    ]

    ground_truths = [
        "RAG combines retrieval and generation."
    ]

    evaluator = RagasEvaluator()

    result = evaluator.run(
        questions,
        answers,
        contexts,
        ground_truths
    )

    print(result)