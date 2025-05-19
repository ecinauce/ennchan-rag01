from langchain_community.document_loaders import TextLoader
from typing import Optional

from ennchan_rag.core.interfaces import DocLoader


class TextLoaderAdapter(DocLoader):
    def __init__(self, 
        file_path: str,
        encoding: Optional[str] = None, 
        autodetect_encoding: bool = True):
        self.path = file_path
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding


    def load(self):
        """Load documents from a text file."""
        loader = TextLoader(self.path, 
                          encoding=self.encoding, 
                          autodetect_encoding=self.autodetect_encoding)

        return loader.load()