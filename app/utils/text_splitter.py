from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextSplitter:
    def __init__(self, chunk_size=1500, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(  # This is a instance variable to bound each object of TextSplitter.
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                "|",
                ".",
                "?",
                "!",
                ",",
                " ",
            ],  # \n\n is for paragraph and so on..
        )

    def split_text(self, text: str):
        return self.splitter.split_text(text)


# splitter = TextSplitter(chunk_size=5, chunk_overlap=1)
# text = "This is the sample text"
# chunks = splitter.split_text(text)
# print(chunks)
