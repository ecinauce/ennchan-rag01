from langchain_core.documents import Document
from typing import List, Optional, Dict, Any
from ennchan_rag.core.interfaces import RetrievalStrategy

class SimilaritySearchRetrieval(RetrievalStrategy):
    """
    Enhanced retrieval strategy using vector similarity search.
    
    This strategy provides configurable parameters and additional features
    like score thresholding and metadata filtering.
    """
    
    def __init__(self, 
                 k: int = 4, 
                 score_threshold: Optional[float] = None,
                 filter: Optional[Dict[str, Any]] = None):
        """
        Initialize the similarity search retrieval strategy.
        
        Args:
            k: Number of documents to return
            score_threshold: Optional minimum similarity score (0-1) for documents
            filter: Optional metadata filter to apply to the search
        """
        self.k = k
        self.score_threshold = score_threshold
        self.filter = filter
    
    def retrieve(self, query: str, vector_store) -> List[Document]:
        """
        Retrieve documents using vector similarity search.
        
        This method finds the most semantically similar documents to the query.
        If score_threshold is set, it will only return documents with similarity
        scores above that threshold.
        """
        try:
            # Check if the vector store supports search with scores
            if hasattr(vector_store, 'similarity_search_with_score'):
                docs_and_scores = vector_store.similarity_search_with_score(
                    query, 
                    k=self.k * 2,  # Get more for filtering
                    filter=self.filter
                )
                
                # Apply score threshold if specified
                if self.score_threshold is not None:
                    filtered_results = [
                        (doc, score) for doc, score in docs_and_scores 
                        if score >= self.score_threshold
                    ]
                    # Return at most k documents
                    return [doc for doc, _ in filtered_results[:self.k]]
                else:
                    # Return at most k documents
                    return [doc for doc, _ in docs_and_scores[:self.k]]
            else:
                # Fallback to regular similarity search
                return vector_store.similarity_search(
                    query, 
                    k=self.k,
                    filter=self.filter
                )
                
        except Exception as e:
            print(f"Similarity search failed: {e}")
            # Try a more basic approach as fallback
            try:
                return vector_store.similarity_search(query, k=self.k)
            except Exception as e2:
                print(f"Fallback similarity search also failed: {e2}")
                return []
    
    def set_filter(self, filter: Dict[str, Any]) -> None:
        """Update the metadata filter."""
        self.filter = filter
    
    def set_k(self, k: int) -> None:
        """Update the number of documents to retrieve."""
        self.k = k
    
    def set_score_threshold(self, threshold: float) -> None:
        """Update the similarity score threshold."""
        self.score_threshold = threshold
