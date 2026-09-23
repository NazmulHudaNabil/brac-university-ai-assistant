from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def get_chunker(chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Returns a text splitter configured for generic text chunking.
    The RecursiveCharacterTextSplitter is structure-aware by splitting on 
    paragraphs, then sentences, then words.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

def chunk_documents(documents: list[Document], chunker=None) -> list[Document]:
    """
    Splits a list of documents into chunks using the provided chunker.
    """
    if chunker is None:
        chunker = get_chunker()
        
    return chunker.split_documents(documents)
