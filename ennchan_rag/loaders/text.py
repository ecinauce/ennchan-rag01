from langchain_community.document_loaders import TextLoader
from typing import Optional

from ennchan_rag.core.interfaces import DocLoader


class TextLoaderAdapter(DocLoader):
    """
    Adapter for loading documents from text files.
    
    This class wraps the LangChain TextLoader to provide a consistent
    interface for loading text content within the RAG system.
    """
    
    def __init__(self, 
        file_path: str,
        encoding: Optional[str] = None, 
        autodetect_encoding: bool = True):
        """
        Initialize the text loader adapter.
        
        Args:
            file_path: Path to the text file
            encoding: Specific encoding to use, or None to use default
            autodetect_encoding: Whether to attempt to autodetect the file encoding
        """
        self.path = file_path
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding


    def load(self):
        """
        Load documents from a text file.
        
        Returns:
            List of Document objects containing the text file content
        """
        loader = TextLoader(self.path, 
                          encoding=self.encoding, 
                          autodetect_encoding=self.autodetect_encoding)

        return loader.load()