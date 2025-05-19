from ennchan_rag.core.state import State

class ContextProcessor:
    def process(self, state: State, max_chars: int):
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
