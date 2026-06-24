import os

from sentence_transformers import SentenceTransformer

from langchain.document_loaders import PyPDFLoader

from qdrant_client.models import PointStruct

from qdrant_client import QdrantClient

embedder = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

client = QdrantClient(
    url="http://localhost:6333"
)

COLLECTION_NAME = "emotion_rag"


def ingest_pdfs():

    point_id = 0

    pdf_folder = "data/pdfs"

    for file in os.listdir(pdf_folder):

        if file.endswith(".pdf"):

            loader = PyPDFLoader(
                os.path.join(pdf_folder, file)
            )

            docs = loader.load()

            for doc in docs:

                text = doc.page_content

                embedding = embedder.encode(
                    text,
                    normalize_embeddings=True
                )

                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=embedding.tolist(),
                            payload={
                                "text": text
                            }
                        )
                    ]
                )

                point_id += 1


if __name__ == "__main__":
    ingest_pdfs()