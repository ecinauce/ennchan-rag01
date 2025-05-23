import bs4
from langchain_community.document_loaders import WebBaseLoader

from ennchan_rag.core.interfaces import DocLoader


class WebLoaderAdapter(DocLoader):
    """
    Adapter for loading documents from web sources.
    
    This class wraps the LangChain WebBaseLoader to provide a consistent
    interface for loading web content within the RAG system.
    """
    
    def __init__(self, url):
        """
        Initialize the web loader adapter.
        
        Args:
            url: The URL to load content from
        """
        self.url = url

    def load(self):
        """
        Load documents from the web URL.
        
        This method fetches content from the specified URL and converts it
        into Document objects. It uses BeautifulSoup to filter content
        to only include the main container.
        
        Returns:
            List of Document objects containing the web content
        """
        loader = WebBaseLoader(
            web_paths=(self.url,),
            bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("mw-content-container")
                )
            ),
        )

        return loader.load()