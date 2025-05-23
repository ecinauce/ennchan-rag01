# ennchan_rag/utils/model_cache.py
from typing import Dict, Any, Optional
from langchain_huggingface import HuggingFacePipeline
import time

# Global cache for models
_MODEL_CACHE = {}
_LAST_USED = {}
_MAX_CACHE_SIZE = 2  # Maximum number of models to keep in cache

def get_model(
    model_id: str, 
    task: str = "text-generation", 
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None
) -> HuggingFacePipeline:
    """
    Get a model from cache or load it if not cached.
    
    Args:
        model_id: The Hugging Face model ID
        task: The task for the pipeline
        pipeline_kwargs: Keyword arguments for the pipeline
        model_kwargs: Keyword arguments for the model
        
    Returns:
        The HuggingFacePipeline instance
    """
    # Create a cache key from the parameters
    cache_key = f"{model_id}_{task}"
    
    # Return cached model if available
    if cache_key in _MODEL_CACHE:
        print(f"Using cached model: {model_id}")
        _LAST_USED[cache_key] = time.time()
        return _MODEL_CACHE[cache_key]
    
    # If cache is full, remove least recently used model
    if len(_MODEL_CACHE) >= _MAX_CACHE_SIZE:
        lru_key = min(_LAST_USED.items(), key=lambda x: x[1])[0]
        print(f"Cache full, removing model: {lru_key}")
        del _MODEL_CACHE[lru_key]
        del _LAST_USED[lru_key]
    
    # Load the model
    print(f"Loading model: {model_id}")
    pipeline_kwargs = pipeline_kwargs or {}
    model_kwargs = model_kwargs or {}
    
    model = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task=task,
        pipeline_kwargs=pipeline_kwargs,
        model_kwargs=model_kwargs,
    )
    
    # Cache the model
    _MODEL_CACHE[cache_key] = model
    _LAST_USED[cache_key] = time.time()
    
    return model
