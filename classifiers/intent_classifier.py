import torch
from transformers import pipeline
device = 0 if torch.cuda.is_available() else -1

intent_model = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

def classify_intent(query):

    candidate_labels = ["Knowledge", "Emotional Support", "Casual Chat"]
    
    result = intent_model(query, candidate_labels=candidate_labels)

    return result['labels'][0]

if __name__ == "__main__":
    test_query = "Can you explain how a retrieval augmented generation system works?"
    print("Intent Output:", classify_intent(test_query))