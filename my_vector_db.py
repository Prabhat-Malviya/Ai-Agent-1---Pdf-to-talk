from langchain_community.document_loaders import PDFPlumberLoader

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()


    #create chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    return text_splitter.split_documents(document)

#embedding
from langchain_ollama import OllamaEmbeddings

def get_embedded_model(model_name):
    return OllamaEmbeddings(model = model_name)


documents = load_pdf("Unit -1 notes CS601 (ML) (1).pdf")
print(len(documents))


chunks = create_chunks(documents)
print(len(chunks))