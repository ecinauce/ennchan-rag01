from typing import Dict, Optional
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph

from ennchan_rag.core.context import ContextProcessor
from ennchan_rag.core.interfaces import LLMInterface, VectorStoreInterface, RetrievalStrategy
from ennchan_rag.core.state import State
from ennchan_rag.retrievers.similarity import SimilaritySearchRetrieval
from ennchan_rag.retrievers.mmr import MMRRetrieval
from ennchan_rag.retrievers.hybrid import HybridRetrieval
from ennchan_rag.retrievers.keyword import KeywordRetrieval
from ennchan_search import search as web_search
import concurrent.futures
print("Invoking model...")


class QAModel:
    def __init__(self, 
                 llm: LLMInterface, 
                 vector_store: VectorStoreInterface, 
                 prompt_source: str,
                 context_scope: int,
                 retrieval_strategy: RetrievalStrategy = SimilaritySearchRetrieval()):
        # self.prompt = hub.pull(prompt_source)
        self.prompt_source = prompt_source
        self.prompt = ChatPromptTemplate([("system",
            """
            {prompt_source}
            Question: {question} 
            Context: {context} 
            Answer:
            """,)])
        self.context_scope = context_scope
        self.llm = llm
        self.vector_store = vector_store
        self.retrieval_strategy = retrieval_strategy or SimilaritySearchRetrieval()

        # Compile application and test
        self.graph_builder = StateGraph(State).add_sequence([
            self.retrieve, 
            self.generate
        ])
        self.graph_builder.add_edge(START, "retrieve")
        self.graph = self.graph_builder.compile()

    # Define application steps
    def retrieve(self, state: State) -> Dict[str, list[Document]]:
        query = state["question"]
        retrieved_docs = self.retrieval_strategy.retrieve(query, self.vector_store) 
        return {"context": retrieved_docs}

    def generate(self, state: State) -> Dict[str, str]:
        context = ContextProcessor()
        messages = self.prompt.invoke({
            "prompt_source": self.prompt_source,
            "question": state["question"], 
            "context": context.process(state, self.context_scope)})
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
            self.process_search_results,  # New step
            self.compile_reference_document,  # New step
            self.retrieve,
            self.generate
        ])
        self.graph_builder.add_edge(START, "formulate_query")
        self.graph = self.graph_builder.compile()
        

    def formulate_query(self, state: State) -> Dict:
        """Convert user question to search query with classification and validation"""
        user_question = state["question"]
        
        # Step 1: Classify the question type
        classification_prompt = f"""
        Analyze the following question and classify it into one of these categories:
        - FACTUAL: Seeking objective information or facts
        - HOW_TO: Seeking instructions or procedures
        - OPINION: Seeking subjective views or evaluations
        - COMPARISON: Seeking to compare multiple items
        - EXPLANATION: Seeking to understand concepts or reasons
        
        Question: {user_question}
        
        Classification (return only the category name):
        """
        
        question_type = self.llm.invoke(classification_prompt).strip()
        
        # Step 2: Generate tailored search queries based on question type
        query_prompt = f"""
        Your task is to convert a user's question into 1-3 effective search engine queries.
        
        Question type: {question_type}
        
        Guidelines:
        - For FACTUAL questions: Focus on key entities and relationships, use neutral terms
        - For HOW_TO questions: Include terms like "tutorial", "guide", "steps", "instructions"
        - For OPINION questions: Include terms like "review", "opinion", "analysis", "perspective"
        - For COMPARISON questions: Include terms like "versus", "compared to", "differences"
        - For EXPLANATION questions: Include terms like "explained", "understanding", "concept"
        
        Examples:
        User: "What were the major causes of World War II?" (FACTUAL)
        Queries: ["main causes World War II historical analysis", "economic political factors leading to World War II"]
        
        User: "How do I build a simple website?" (HOW_TO)
        Queries: ["beginner website creation tutorial", "step by step build simple website guide"]
        
        User question: {user_question}
        
        Return a JSON array of 1-3 search queries (more for complex questions):
        """
        
        # Parse the response as a list of queries
        try:
            import json
            search_queries_text = self.llm.invoke(query_prompt).strip()
            # Handle potential formatting issues in LLM response
            if not search_queries_text.startswith("["):
                search_queries_text = "[" + search_queries_text
            if not search_queries_text.endswith("]"):
                search_queries_text = search_queries_text + "]"
                
            search_queries = json.loads(search_queries_text)
            if not isinstance(search_queries, list):
                search_queries = [search_queries]
        except:
            # Fallback if parsing fails
            search_queries = [user_question]
        
        # Step 3: Validate queries
        valid_queries = []
        for query in search_queries:
            # Basic validation - ensure query is not too short or just repeating the question
            if isinstance(query, str) and len(query) > 5 and query != user_question:
                valid_queries.append(query)
        
        # Fallback if all queries were invalid
        if not valid_queries:
            valid_queries = [user_question]
        
        return {
            "question": user_question,
            "question_type": question_type,
            "search_queries": valid_queries
        }

    def process_search_results(self, state: State) -> Dict:
        """Process and summarize individual search results."""
        raw_results = state.get("raw_search_results", [])
        processed_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            result_threads = {
                executor.submit(
                    self._process_single_result, 
                    result, 
                    state["question"]
                ): result for result in raw_results if result.get("content")
            }

            for future in concurrent.futures.as_completed(result_threads):
                try:
                    processed_result = future.result()
                    if processed_result:
                        processed_results.append(processed_result)
                except Exception as e:
                    print(f"Error processing result: {e}")

        return {**state, "processed_results": processed_results}
        
    def _process_single_result(self, result: Dict, question: str) -> Optional[Dict]:
        """Process a single search result into a summarized version."""
        # Skip results without content
        if not result.get("content"):
            return None
            
        # Create a summary prompt for this specific result
        summary_prompt = f"""
        Summarize the following content in relation to this question: "{question}"
        
        Content from {result.get('title', 'Unknown Source')} ({result.get('url', 'No URL')}):
        {result.get('content')[:2000]}...
        
        Provide a concise summary that captures the key information relevant to the question.
        Include specific facts, figures, and quotes if relevant.
        """
        
        try:
            # Generate summary using LLM
            summary = self.llm.invoke(summary_prompt)
            
            # Return processed result
            return {
                "title": result.get("title", "Unknown Source"),
                "url": result.get("url", ""),
                "summary": summary,
                "original_content": result.get("content", "")
            }
        except Exception as e:
            print(f"Error processing result from {result.get('url', 'unknown URL')}: {e}")
            return None

    def search_web(self, state: State) -> Dict:
        """Search the web for relevant information using multiple queries"""
        search_queries = state.get("search_queries", [state["question"]])
        
        # Use all generated queries
        all_search_results = []
        for query in search_queries:
            try:
                # Use your search module to get results
                results = web_search(query, self.search_config)
                all_search_results.extend(results)
            except Exception as e:
                print(f"Search failed for query '{query}': {e}")
        
        # Remove duplicates based on URL
        unique_results = {}
        for result in all_search_results:
            if "url" in result and result["url"] not in unique_results:
                unique_results[result["url"]] = result
        
        # Convert search results to documents
        search_documents = []
        for result in unique_results.values():
            if "content" in result and result["content"]:
                doc = Document(
                    page_content=result["content"],
                    metadata={
                        "title": result.get("title", "Unknown Title"),
                        "url": result.get("url", ""),
                        "source": "web_search",
                        "query": result.get("query", "")
                    }
                )
                search_documents.append(doc)
        
        # Add search documents to vector store for retrieval
        if search_documents:
            self.vector_store.add_documents(search_documents)
        
        # Update state with search results for later steps
        return {
            **state,
            "raw_search_results": list(unique_results.values()),
            "search_document_count": len(search_documents)
        }
    
    def select_retrieval_strategy(self, state: State) -> Dict:
        """Select the most appropriate retrieval strategy based on question type and content"""
        question_type = state.get("question_type", "FACTUAL")
        question = state["question"]
        
        # Define a prompt to help select the best retrieval strategy
        strategy_prompt = f"""
        Based on the question type and content, select the most appropriate retrieval strategy:
        
        Question: {question}
        Question Type: {question_type}
        
        Available strategies:
        1. SIMILARITY: Vector similarity search (best for semantic understanding and conceptual questions)
        2. MMR: Maximum Marginal Relevance (best for diverse information needs)
        3. HYBRID: Combines keyword and semantic search (best for specific technical questions)
        4. KEYWORD: Traditional keyword search (best for exact term matching)
        
        Select the most appropriate strategy number (1-4):
        """
        
        try:
            strategy_selection = self.llm.invoke(strategy_prompt).strip()
            # Extract just the number if there's additional text
            import re
            match = re.search(r'[1-4]', strategy_selection)
            if match:
                strategy_num = int(match.group(0))
            else:
                strategy_num = 1  # Default to similarity search
        except:
            strategy_num = 1  # Default to similarity search if parsing fails
        
        # Map strategy number to actual strategy
        strategies = {
            1: SimilaritySearchRetrieval(),
            2: MMRRetrieval(diversity=0.7),
            3: HybridRetrieval(alpha=0.5),  # 50% keyword, 50% semantic
            4: KeywordRetrieval()
        }
        
        selected_strategy = strategies.get(strategy_num, SimilaritySearchRetrieval())
        
        return {
            **state,
            "selected_retrieval_strategy": type(selected_strategy).__name__
        }, selected_strategy
    
    def compile_reference_document(self, state: State) -> Dict:
        """Compile processed results into a structured reference document."""
        processed_results = state.get("processed_results", [])
        question = state["question"]
        
        if not processed_results:
            return {**state, "reference_document": ""}
        
        # Create a compilation prompt
        compilation_prompt = f"""
        Based on the following summaries, create a comprehensive reference document that answers this question:
        
        Question: {question}
        
        Summaries:
        """
        
        # Add each summary with source information
        for i, result in enumerate(processed_results, 1):
            compilation_prompt += f"""
            Source {i}: {result.get('title')} ({result.get('url')})
            {result.get('summary')}
            """
        
        compilation_prompt += """
        
        Create a well-structured reference document that:
        1. Synthesizes information from all sources
        2. Organizes content by topic or relevance
        3. Includes proper citations [Source X] for each piece of information
        4. Presents a comprehensive answer to the question
        """
        
        try:
            # Generate compiled document
            reference_document = self.llm.invoke(compilation_prompt)
            
            # Add to vector store for retrieval
            if reference_document:
                doc = Document(
                    page_content=reference_document,
                    metadata={
                        "title": f"Reference Document for: {question}",
                        "source": "compiled_reference",
                        "question": question
                    }
                )
                self.vector_store.add_documents([doc])
            
            return {**state, "reference_document": reference_document}
        except Exception as e:
            print(f"Error compiling reference document: {e}")
            return {**state, "reference_document": ""}
        
    def retrieve(self, state: State) -> Dict[str, list[Document]]:
        """Retrieve documents using dynamically selected strategy"""
        # Select the appropriate retrieval strategy
        updated_state, strategy = self.select_retrieval_strategy(state)
        
        # Use the selected strategy to retrieve documents
        query = state["question"]
        retrieved_docs = strategy.retrieve(query, self.vector_store)
        
        return {
            **updated_state,
            "context": retrieved_docs
        }
