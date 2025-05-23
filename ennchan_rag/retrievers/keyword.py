from langchain_core.documents import Document
from typing import List
import re
from ennchan_rag.core.interfaces import RetrievalStrategy

class KeywordRetrieval(RetrievalStrategy):
    """
    Retrieval strategy using traditional keyword matching.
    
    This strategy is useful for queries that require exact term matching
    rather than semantic similarity.
    """
    
    def __init__(self, k: int = 4):
        """
        Initialize the keyword retrieval strategy.
        
        Args:
            k: Maximum number of documents to return
        """
        self.k = k
    
    def retrieve(self, query: str, vector_store) -> List[Document]:
        """
        Retrieve documents using keyword matching.
        
        This method extracts keywords from the query and finds documents
        that contain those keywords.
        """
        try:
            # Get all documents from the vector store
            # Note: This is inefficient for large collections but works for demo purposes
            all_docs = vector_store.get_all_documents()
            
            # Extract keywords (simple implementation - remove stop words and punctuation)
            stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 
                         'were', 'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for',
                         'with', 'by', 'about', 'like', 'through', 'over', 'before',
                         'after', 'between', 'under', 'above', 'of', 'from'}
            
            # Clean query and extract keywords
            clean_query = re.sub(r'[^\w\s]', ' ', query.lower())
            keywords = [word for word in clean_query.split() if word not in stop_words]
            
            # Score documents based on keyword matches
            scored_docs = []
            for doc in all_docs:
                score = 0
                content = doc.page_content.lower()
                
                # Count keyword occurrences
                for keyword in keywords:
                    score += content.count(keyword)
                
                # Add document if it contains any keywords
                if score > 0:
                    scored_docs.append((doc, score))
            
            # Sort by score and take top k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs[:self.k]]
            
        except Exception as e:
            print(f"Keyword retrieval failed: {e}. Falling back to similarity search.")
            # Fallback to regular similarity search
            return vector_store.similarity_search(query, k=self.k)