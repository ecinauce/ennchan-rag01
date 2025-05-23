from langchain_core.documents import Document
from typing import List
from ennchan_rag.core.interfaces import RetrievalStrategy

class MMRRetrieval(RetrievalStrategy):
    """
    Retrieval strategy using Maximum Marginal Relevance.
    
    This strategy balances relevance with diversity in the retrieved documents.
    Higher diversity values (closer to 1.0) will prioritize diversity,
    while lower values (closer to 0.0) will prioritize relevance.
    """
    
    def __init__(self, diversity: float = 0.3, k: int = 4, fetch_k: int = 20):
        """
        Initialize the MMR retrieval strategy.
        
        Args:
            diversity: Float between 0 and 1 that determines the balance between
                relevance and diversity. 0 is all relevance, 1 is all diversity.
            k: Number of documents to return
            fetch_k: Number of documents to consider before reranking
        """
        self.diversity = diversity
        self.k = k
        self.fetch_k = fetch_k
    
    def retrieve(self, query: str, vector_store) -> List[Document]:
        """
        Retrieve documents using Maximum Marginal Relevance search.
        
        This method finds semantically relevant documents and then
        reranks them to ensure diversity in the results.
        """
        try:
            return vector_store.max_marginal_relevance_search(
                query, 
                k=self.k,
                fetch_k=self.fetch_k,
                lambda_mult=self.diversity
            )
        except Exception as e:
            print(f"MMR retrieval failed: {e}. Falling back to similarity search.")
            # Fallback to regular similarity search
            return vector_store.similarity_search(query, k=self.k)