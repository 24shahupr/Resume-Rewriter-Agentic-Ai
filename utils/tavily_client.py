import os
import asyncio
from tavily import TavilyClient as Tavily

class TavilyClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("Tavily API key not found. Please set TAVILY_API_KEY environment variable.")
        self.client = Tavily(api_key=self.api_key)

    async def search(self, query: str, search_depth: str = "advanced") -> str:
        """
        Performs a web search using Tavily.
        """
        try:
            # Tavily's python client is sync, so we run it in a thread.
            response = await asyncio.to_thread(self.client.search, query=query, search_depth=search_depth)
            results = response.get("results", [])
            
            context = ""
            for res in results[:5]:
                context += f"Source: {res['url']}\nContent: {res['content']}\n\n"
            return context
        except Exception as e:
            print(f"Error in Tavily search: {e}")
            return "No search results found due to an error."
