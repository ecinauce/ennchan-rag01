from ennchan_rag.core.state import State

class ContextProcessor:
    """
    Processes document context for RAG operations.
    
    This class handles the extraction and formatting of document content
    to be used as context for the language model.
    """
    
    def process(self, state: State, max_chars: int):
        """
        Process documents from state into a context string.
        
        This method concatenates document content up to a maximum character limit,
        ensuring the context doesn't exceed the model's input constraints.
        
        Args:
            state: The current state containing documents in the "context" field
            max_chars: Maximum number of characters to include in the context
            
        Returns:
            A string containing the concatenated document content
        """
        docs = state["context"]
        docs_content = ""

        for doc in docs:
            if len(doc.page_content) + len(docs_content) < max_chars:
                docs_content += doc.page_content + "\n\n"
            else:
                break

        if not docs_content and docs:
            docs_content = docs[0].page_content[:1000]

        return docs_content