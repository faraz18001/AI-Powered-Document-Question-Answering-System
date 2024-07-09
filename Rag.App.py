import os
import tempfile
from typing import Any, List
import logging
import pathlib
from langchain.document_loaders import (
    PyPDFLoader, TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEPubLoader
)
from langchain.schema import Document, BaseRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains.base import Chain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, PromptTemplate
from langchain.callbacks import get_openai_callback

# set the open ai api key
os.environ['OPENAI_API_KEY'] = ''


class EpubReader(UnstructuredEPubLoader):
    def __init__(self, file_path: str | List[str], **kwargs: Any):
        super().__init__(file_path, **kwargs, mode="elements", strategy="fast")


class DocumentLoaderException(Exception):
    pass


class DocumentLoader:
    supported_extensions = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".epub": EpubReader,
        ".docx": UnstructuredWordDocumentLoader,
    }

    @staticmethod
    def load_document(file_path: str) -> List[Document]:
        ext = pathlib.Path(file_path).suffix
        loader = DocumentLoader.supported_extensions.get(ext)
        if not loader:
            raise DocumentLoaderException(
                f'Invalid Extension Type {ext}, cannot load this type of file'
            )
        loader = loader(file_path)
        docs = loader.load()
        logging.info(f"Loaded {len(docs)} documents")
        return docs


def configure_retriever(docs: List[Document], use_compression: bool = False) -> BaseRetriever:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    if not use_compression:
        return retriever

    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings, similarity_threshold=0.76
    )
    return ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=retriever
    )


def configure_chain(retriever: BaseRetriever) -> Chain:
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=False)  # Disable streaming for terminal
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        verbose=True
    )


def main():
    # Variable for the document file path
    document_path = "text.pdf"  # Replace with your document path

    docs = DocumentLoader.load_document(document_path)
    retriever = configure_retriever(docs=docs)
    qa_chain = configure_chain(retriever=retriever)

    while True:
        user_query = input("Ask a question about the document: ")
        if user_query.lower() == "quit":
            break
        with get_openai_callback() as cb:
            response = qa_chain.run(user_query, callbacks=[cb])
            print(f'Total Input Tokens: {cb.total_tokens}')
            print(f'Total Output Tokens: {cb.total_tokens}')
            print(response)


if __name__ == "__main__":
    main()
