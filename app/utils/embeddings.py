from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import json
from dotenv import load_dotenv
import os

load_dotenv()


class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dir: str = "../data/faiss_index",
    ):
        self.model = SentenceTransformer(model_name)
        self.output_dir = Path(
            output_dir
        ).resolve()  # resolve() is used to get the absolute path of the directory.
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        L2 normalize vectors for cosine similarity.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

    def generate_embeddings(
        self, chunks_file: str = "chunks.json", output_file: str = "embeddings.json"
    ):
        """
        Load chunks from JSON, generate embeddings and save to JSON again.
        """
        chunks_path = self.output_dir / chunks_file
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunk file {chunks_path} does not exist.")

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        texts = [c["text"] for c in chunks]

        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=True
        )

        embeddings = self._normalize(embeddings)

        output_data = []
        for chunk, vector in zip(chunks, embeddings):
            output_data.append(
                {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "embedding": vector.tolist(),  # Convert numpy array to list for JSON serialization
                }
            )

        output_path = self.output_dir / output_file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        return output_data
