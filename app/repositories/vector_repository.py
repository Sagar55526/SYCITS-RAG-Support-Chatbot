from pathlib import Path
import numpy as np
import faiss
import json
import os


class VectorRepository:
    def __init__(self, index_dir: str = None):
        # Go up from app/repositories → app → project root
        project_root = Path(__file__).resolve().parents[1].parent
        if index_dir:
            self.index_dir = project_root / index_dir
        else:
            self.index_dir = project_root / "data/faiss_index"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.index_dir / "index.faiss"
        self.meta_file = self.index_dir / "embeddings.json"

    def build_index(self, embeddings_file: str = "embeddings.json"):
        """
        Build a FAISS index from embeddings stored in a JSON file.
        """
        embeddings_path = self.index_dir / embeddings_file
        if not embeddings_path.exists():
            raise FileNotFoundError(f"file for embeddings {embeddings_path} not found")

        with open(embeddings_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vectors = np.array([d["embedding"] for d in data], dtype="float32")
        dim = vectors.shape[1]

        index = faiss.IndexFlatIP(
            dim
        )  # Using Inner Product (dot product) for cosine similarity
        index.add(vectors)

        faiss.write_index(index, str(self.index_file))

        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"FAISS index built with {index.ntotal} vectors")
        print("Index dimension: ", index.d)
        print("Total vectors:", index.ntotal)
        return index

    def Load_index(self):
        """
        Load a persisted FAISS index from disk.
        """
        if not self.index_file.exists():
            raise FileNotFoundError(f"FAISS index file {self.index_file} not found.")
        return faiss.read_index(str(self.index_file))

    def load_metadata(self):
        """
        Load metadata associated with the FAISS index.
        """
        if not self.meta_file.exists():
            raise FileNotFoundError(f"Metadata file {self.meta_file} not found.")
        with open(self.meta_file, "r", encoding="utf-8") as f:
            return json.load(f)
