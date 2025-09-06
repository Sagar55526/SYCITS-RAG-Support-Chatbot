from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from app.repositories.vector_repository import VectorRepository
from contextlib import asynccontextmanager
from app.api.routes import router

app = FastAPI(title="Marathi Chatbot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load embeddings model, FAISS index, and metadata on server startup,
    and cleanup if needed on shutdown.
    """
    app.state.embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    repo = VectorRepository()
    app.state.faiss_index = repo.Load_index()
    app.state.metadata = repo.load_metadata()
    print("Embedding model, FAISS index, and metadata loaded successfully.")

    yield

    print("Shutting down... resources released.")


app = FastAPI(title="Marathi Chatbot", lifespan=lifespan)

app.include_router(router, prefix="/api")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
