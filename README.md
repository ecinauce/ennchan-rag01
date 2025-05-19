# Ennchan's RAG Project 01
LangChain RAG Project with Custom Search engines

## Installation:


pip install ennchan-rag-*.whl

(*Best do it inside a venv*)

**WARNING!**

You need to install [ennchan_search](https://placeholder.it/) beforehand.

## How to use:
```python
from ennchan_rag.ask import ask

config = "path/to/config.json"
question = "What are the 3 laws of robotics?"
answers = ask(question, config)
```

