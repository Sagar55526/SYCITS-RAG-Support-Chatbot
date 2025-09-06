from repositories.pdf_repository import PDFRepository
from utils.embeddings import EmbeddingGenerator
from repositories.vector_repository import VectorRepository

if __name__ == "__main__":
    # Step 1: Load and chunk PDF
    pdf_repo = PDFRepository()
    chunks = pdf_repo.load_and_chunk("1. स्कूल मास्टर सॉफ्टवेअर सुरु करणे.pdf")

    # Step 2: Generate embeddings
    embed_gen = EmbeddingGenerator()
    embeddings = embed_gen.generate_embeddings()

    # Step 3: Create FAISS index
    repo = VectorRepository()
    repo.build_index()

    print("✅ chunks.json, embeddings.json, and index.faiss created successfully!")
