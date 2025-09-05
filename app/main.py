from repositories.pdf_repository import PDFRepository

if __name__ == "__main__":
    repo = PDFRepository()
    chunks = repo.load_and_chunk("1. स्कूल मास्टर सॉफ्टवेअर सुरु करणे.pdf")
    print(f"extracted {len(chunks)} chunks from the pdf file")
