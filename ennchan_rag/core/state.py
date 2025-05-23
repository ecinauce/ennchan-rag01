from typing_extensions import List, TypedDict, Optional, Dict
from langchain_core.documents import Document


# Define state for application
class State(TypedDict):
    """
    State definition for the RAG application.
    
    This TypedDict defines the structure of the state object used throughout
    the RAG pipeline, containing all necessary information for processing
    a query and generating a response.
    """
    question: str  # The user's original question
    context: List[Document]  # Retrieved documents for context
    answer: str  # The generated answer
    search_queries: Optional[List[str]]  # Added for query tracking
    search_results: Optional[List[Dict]]  # Added for raw search results
    processed_results: Optional[List[Dict]]  # Added for individual summaries
    reference_document: Optional[str]  # Added for compiled document