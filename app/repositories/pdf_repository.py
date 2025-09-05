from langchain_community.document_loaders import PyPDFLoader
from utils.text_splitter import TextSplitter
from pathlib import Path
import json
import os


class PDFRepository:
    def __init__(
        self, data_dir: str = "../data/pdfs", output_dir: str = "../data/faiss_index"
    ):
        self.data_dir = Path(data_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=  True, exist_ok=True)

    def load_and_chunk(self, filename: str):
        """
        Load a pdf file, split into chunkns, and save to JSON,
        Returns list of chunks with metadata.
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not found.")

        pdf_loader = PyPDFLoader(str(file_path))
        pdf_content = pdf_loader.load()

        splitter = TextSplitter()
        chunks = []
        idx = 0

        for page_num, page in enumerate(pdf_content, start=1):
            page_text = page.page_content.strip()
            page_chunks = splitter.split_text(page_text)

            for i, chunk in enumerate(page_chunks):
                chunks.append(
                    {
                        "id": idx,
                        "text": chunk,
                        "metadata": {
                            "source": filename,
                            "page": page_num,
                            "chunk_id": f"{filename}_{page_num}_{i}",
                        },
                    }
                )
                idx += 1
                print(chunk)

        output_file = self.output_dir / "chunks.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
            print("chunk dumped successfully in chunks.json file")
        return chunks
