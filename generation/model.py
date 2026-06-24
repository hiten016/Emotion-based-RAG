from transformers import pipeline

llm = pipeline(
    "text-generation",
    model="Qwen/Qwen3-8B"
)