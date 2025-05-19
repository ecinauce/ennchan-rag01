from ennchan_rag import QAModel, SearchAugmentedQAModel
from ennchan_rag.config import load_config
from ennchan_rag.loaders import WebLoaderAdapter
from ennchan_rag.utils.quantization import load_quantization
# from ennchan_rag.retrievers import SimilaritySearchRetrieval
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


def ask(question: str, p_config: str = None) -> str:
    """
    Inquire about a question using the QAModel.

    Args:
        question (str): The question to inquire about.

    Returns:
        str: The answer to the question.
    """
    # Initialize components
    config = load_config(p_config)
    embeddings = HuggingFaceEmbeddings(model_name=config.embeddings_model)
    vector_store = InMemoryVectorStore(embeddings)
    llm = HuggingFacePipeline.from_model_id(
        model_id=config.model_name,
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        ),
        model_kwargs=load_quantization(config),
    )

    # Create and use the model
    model = SearchAugmentedQAModel(
        llm=llm,
        vector_store=vector_store,
        prompt_source=config.prompt_source,
        context_scope=config.context_scope,
    )

    # Ask a question
    answer = model.graph.invoke({"question": question})
    return answer["answer"]