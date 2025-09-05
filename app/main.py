from repositories.vector_repository import VectorRepository

if __name__ == "__main__":
    repo = VectorRepository()
    index = repo.build_index()
    print("Index dimension: ", index.d)
    print("Total vectors:", index.ntotal)

    query = index.reconstruct(0)
    D, I = index.search(query.reshape(1, -1), k=3)
    print("Nerest Neighbors (IDs):", I, "with scores", D)
