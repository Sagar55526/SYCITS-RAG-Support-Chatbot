import google.generativeai as genai
from typing import List, Dict
import faiss
import numpy as np
import os


class ChatbotService:
    def __init__(self, app, top_k=2):
        self.app = app
        self.top_k = top_k

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.llm = genai.GenerativeModel("gemini-1.5-flash")

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalizing query vector for cosine similarity search.
        """
        return vector / np.linalg.norm(vector)

    def retrieve_context(self, question: str) -> List[Dict]:
        """
        Embed user question, search in FAISS index, and retrieve relevant contexts.
        """
        model = self.app.state.embedding_model
        index = self.app.state.faiss_index
        metadata = self.app.state.metadata

        query_vec = model.encode([question], convert_to_numpy=True)
        query_vec = self._normalize(query_vec)
        D, I = index.search(query_vec.astype("float32"), self.top_k)

        retrieved = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            chunk = metadata[idx]
            retrieved.append(
                {
                    "score": float(score),
                    "metadata": chunk["metadata"],
                    "text": chunk["text"],
                }
            )
        return retrieved

    def build_prompt(self, question: str, retrieved: List[Dict]) -> str:
        """
        Construct a RAG prompt with Marathi instructions.
        """
        context_texts = "\n\n".join(
            [f"(पृष्ठ {c['metadata']['page']}) {c['text']}" for c in retrieved]
        )
        prompt = f"""तुम्ही एक बुद्धिमान, सहाय्यक चॅटबॉट आहात जो खालील संदर्भांचा वापर करून मराठीतून प्रश्नांची उत्तरे देतो.
                    संदर्भ: {context_texts}
                    प्रश्न: {question}
                    वरील संदर्भ वापरून अचूक व समजण्यास सोपा उत्तर मराठीत लिहा.
                    जर माहिती उपलब्ध नसेल तर "माहिती उपलब्ध नाही" असे सांग.
                    """
        return prompt.strip()

    def answer(self, question: str) -> Dict:
        """
        End-To-End RAG flow
        """
        retrieved = self.retrieve_context(question)

        prompt = self.build_prompt(question, retrieved)
        response = self.llm.generate_content(prompt)
        answer_text = response.text.strip()

        sources = [
            {"source": c["metadata"]["source"], "page": c["metadata"]["page"]}
            for c in retrieved
        ]
        return {
            "answer": answer_text,
            "sources": sources,
            "retrieved_chunks": retrieved,
        }
