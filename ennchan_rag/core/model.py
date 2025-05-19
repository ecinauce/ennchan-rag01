from typing import Dict, Optional
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph

from ennchan_rag.core.context import ContextProcessor
from ennchan_rag.core.interfaces import LLMInterface, VectorStoreInterface, RetrievalStrategy
from ennchan_rag.core.state import State
from ennchan_rag.retrievers.similarity import SimilaritySearchRetrieval
from ennchan_search import search as web_search

print("Invoking model...")

class QAModel:
    def __init__(self, 
                 llm: LLMInterface, 
                 vector_store: VectorStoreInterface, 
                 prompt_source: str,
                 context_scope: int,
                 retrieval_strategy: RetrievalStrategy = SimilaritySearchRetrieval()):
        self.prompt = hub.pull(prompt_source)
        self.context_scope = context_scope
        self.llm = llm
        self.vector_store = vector_store
        self.retrieval_strategy = retrieval_strategy or SimilaritySearchRetrieval()

        # Compile application and test
        self.graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        self.graph_builder.add_edge(START, "retrieve")
        self.graph = self.graph_builder.compile()


    # Define application steps
    def retrieve(self, state: State) -> Dict[str, list[Document]]:
        query = state["question"]
        retrieved_docs = self.retrieval_strategy.retrieve(query, self.vector_store) 
        return {"context": retrieved_docs}


    def generate(self, state: State) -> Dict[str, str]:
        context = ContextProcessor()
        messages = self.prompt.invoke({"question": state["question"], "context": context.process(state, self.context_scope)})
        response = self.llm.invoke(messages)

        return {"answer": response}


class SearchAugmentedQAModel(QAModel):
    def __init__(self, 
                 llm: LLMInterface, 
                 vector_store: VectorStoreInterface, 
                 prompt_source: str,
                 context_scope: int,
                 search_config: Optional[Dict] = None):
        super().__init__(llm, vector_store, prompt_source, context_scope)
        self.search_config = search_config
        
        # Rebuild the graph with search step
        self.graph_builder = StateGraph(State).add_sequence([
            self.formulate_query, 
            self.search_web, 
            self.retrieve, 
            self.generate
        ])
        self.graph_builder.add_edge(START, "formulate_query")
        self.graph = self.graph_builder.compile()
    
    def formulate_query(self, state: State) -> Dict:
        """Convert user question to search query"""
        user_question = state["question"]
        
        # For simple cases, use the question directly
        # For more complex cases, you could use the LLM to generate a better query
        prompt = f"""
        Your task is to convert a user's question into an effective search engine query.
        Make the query concise and focused on retrieving factual information.
        Remove any personal elements and focus on the core information need.
        
        User question: {user_question}
        
        Search query:
        """
        
        search_query = self.llm.invoke(prompt).strip()
        
        return {"question": user_question, "search_query": search_query}
    
    
    def search_web(self, state: State) -> Dict:
        """Search the web for relevant information"""
        search_query = state.get("search_query", state["question"])
        
        # Use your search module to get results
        search_results = web_search(search_query, self.search_config)
        
        # Convert search results to documents
        search_documents = []
        for result in search_results:
            if "content" in result and result["content"]:
                doc = Document(
                    page_content=result["content"],
                    metadata={
                        "title": result["title"],
                        "url": result["url"],
                        "source": "web_search"
                    }
                )
                search_documents.append(doc)
        
        # Add search documents to vector store for retrieval
        if search_documents:
            self.vector_store.add_documents(search_documents)
        
        return state
