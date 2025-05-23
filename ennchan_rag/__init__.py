"""RAG implementation using LangChain."""

__version__ = "0.1.0"

# Convenience imports
from langchain_core.documents import Document
from ennchan_rag.ask import ask
from ennchan_rag.core.model import QAModel, SearchAugmentedQAModel
from ennchan_rag.core.interfaces import LLMInterface, \
    VectorStoreInterface, RetrievalStrategy, DocLoader