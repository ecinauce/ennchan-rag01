from langchain_core.documents import Document
from typing import List
from ennchan_rag.core.interfaces import RetrievalStrategy

class SimilaritySearchRetrieval(RetrievalStrategy):
    """Retrieval strategy using vector similarity search."""
    
    def retrieve(self, query: str, vector_store) -> List[Document]:
        """Retrieve documents using similarity search."""
        return vector_store.similarity_search(query)
