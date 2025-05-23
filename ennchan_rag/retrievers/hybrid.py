from langchain_core.documents import Document
from typing import List, Dict, Any
import re
from ennchan_rag.core.interfaces import RetrievalStrategy
from ennchan_rag.retrievers.keyword import KeywordRetrieval
from ennchan_rag.retrievers.similarity import SimilaritySearchRetrieval

class HybridRetrieval(RetrievalStrategy):
    """
    Hybrid retrieval strategy combining semantic search with keyword matching.
    
    This strategy is effective for technical questions where both semantic
    understanding and specific terminology are important.
    """
    
    def __init__(self, alpha: float = 0.5, k: int = 4):
        """
        Initialize the hybrid retrieval strategy.
        
        Args:
            alpha: Weight between 0 and 1 for blending results.
                  0 is all keyword, 1 is all semantic.
            k: Number of documents to return
        """
        self.alpha = alpha
        self.k = k
        self.keyword_retriever = KeywordRetrieval(k=k*2)  # Get more for reranking
        self.semantic_retriever = SimilaritySearchRetrieval()
    
    def retrieve(self, query: str, vector_store) -> List[Document]:
        """
        Retrieve documents using a hybrid of semantic search and keyword matching.
        
        This method combines results from both approaches and reranks them.
        """
        try:
            # Get results from both retrievers
            keyword_docs = self.keyword_retriever.retrieve(query, vector_store)
            semantic_docs = vector_store.similarity_search(query, k=self.k*2)
            
            # Create a scoring system that combines both approaches
            doc_scores: Dict[str, Dict[str, Any]] = {}
            
            # Score semantic results (normalize by position)
            for i, doc in enumerate(semantic_docs):
                doc_id = self._get_doc_id(doc)
                score = 1.0 - (i / len(semantic_docs)) if semantic_docs else 0
                doc_scores[doc_id] = {
                    "doc": doc,
                    "semantic_score": score,
                    "keyword_score": 0.0
                }
            
            # Score keyword results (normalize by position)
            for i, doc in enumerate(keyword_docs):
                doc_id = self._get_doc_id(doc)
                score = 1.0 - (i / len(keyword_docs)) if keyword_docs else 0
                
                if doc_id in doc_scores:
                    doc_scores[doc_id]["keyword_score"] = score
                else:
                    doc_scores[doc_id] = {
                        "doc": doc,
                        "semantic_score": 0.0,
                        "keyword_score": score
                    }
            
            # Calculate combined scores
            for doc_id, scores in doc_scores.items():
                scores["combined_score"] = (
                    self.alpha * scores["semantic_score"] + 
                    (1 - self.alpha) * scores["keyword_score"]
                )
            
            # Sort by combined score and return top k
            ranked_docs = sorted(
                doc_scores.values(), 
                key=lambda x: x["combined_score"], 
                reverse=True
            )
            
            return [item["doc"] for item in ranked_docs[:self.k]]
            
        except Exception as e:
            print(f"Hybrid retrieval failed: {e}. Falling back to similarity search.")
            # Fallback to regular similarity search
            return vector_store.similarity_search(query, k=self.k)
    
    def _get_doc_id(self, doc: Document) -> str:
        """Generate a unique identifier for a document."""
        # Use metadata if available
        if hasattr(doc, 'metadata') and doc.metadata:
            if 'url' in doc.metadata:
                return doc.metadata['url']
            elif 'source' in doc.metadata:
                return doc.metadata['source']
        
        # Fallback to first 100 chars of content
        return doc.page_content[:100]