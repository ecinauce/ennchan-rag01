# Ennchan's RAG Project 01
LangChain RAG Project with Custom Search engines

## Installation:


pip install ennchan-rag-*.whl

(*Best do it inside a venv*)

**WARNING!**

You need to install [ennchan_search](https://github.com/ecinauce/ennchan-search01/) beforehand. Give it a visit, follow the installation guide, then come back here.

## How to use:
### Config
Filename: config.json
```json
{
    "BRAVE_API_KEY": "please get your own key",
    "USER_AGENT": "EnnchanLangChainRAG/1.0",
    "LANGSMITH_TRACING": "true",
    "LANGSMITH_API_KEY": "please get your own key",
    "HUGGINGFACEHUB_API_TOKEN": "please get your own token",
    "quantization": 1,
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2",
    "prompt_source": "rlm/rag-prompt",
    "context_scope": 3000,
    "quantization_config": {
        "load_in_4bit": 1,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": 1
    }
}
```
### Code
```python
from ennchan_rag.ask import ask

config = "path/to/config.json"
question = "What are the 3 laws of robotics?"
answers = ask(question, config)
```
