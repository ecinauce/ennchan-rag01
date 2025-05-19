from abc import ABC, abstractmethod
from typing import Dict, List
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.document_loaders.base import BaseLoader
from langchain_huggingface import HuggingFacePipeline


class RetrievalStrategy(ABC):
    """Abstract base class for document retrieval strategies."""
    
    @abstractmethod
    def retrieve(self, query: str, vector_store: VectorStore) -> list[Document]:
        """Retrieve relevant documents based on the query."""


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""
    
    @abstractmethod
    def invoke(self, messages: Dict[str, str]) -> HuggingFacePipeline:
        """Invoke the LLM with the given messages."""


class VectorStoreInterface(ABC):
    @abstractmethod
    def similarity_search(self, query: str) -> List[Document]:
        """Perform a similarity search in the vector store."""


class DocLoader(BaseLoader):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self) -> list[Document]:
        """Load documents from a web source."""