import bs4
from langchain_community.document_loaders import WebBaseLoader

from ennchan_rag.core.interfaces import DocLoader


class WebLoaderAdapter(DocLoader):
    def __init__(self, url):
        self.url = url


    def load(self):
        loader = WebBaseLoader(
            web_paths=(self.url,),
            bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("mw-content-container")
                )
            ),
        )

        return loader.load()