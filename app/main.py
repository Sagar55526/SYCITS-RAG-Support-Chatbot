# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # pdfloader = PyPDFLoader("data/1. स्कूल मास्टर सॉफ्टवेअर सुरु करणे.pdf")
# pdfloader = PyPDFLoader(
#     r"C:/Users/ahire/Desktop/GenAI Projects/SYCITS RAG Chatbot/data\1. स्कूल मास्टर सॉफ्टवेअर सुरु करणे.pdf"
# )
# content = pdfloader.load()
# print(content)
# print(len(content))

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
