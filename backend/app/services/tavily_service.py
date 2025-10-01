"""
Tavily AI Research Service
Handles travel research searches
"""

from tavily import TavilyClient
from app.config import settings


class TavilyService:
    """Service class for Tavily research operations"""
    
    def __init__(self):
        """Initialize Tavily client"""
        self.client = TavilyClient(api_key=settings.tavily_api_key)
    
    def search_travel_research(self, query: str, max_results: int = 5):
        """
        Search for travel research and information
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        try:
            # Perform search with travel context
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_images=True
            )
            
            return response
            
        except Exception as e:
            raise Exception(f"Research search error: {str(e)}")
    
    def format_results(self, raw_results):
        """
        Format raw Tavily results into clean structure
        
        Args:
            raw_results: Raw results from Tavily
            
        Returns:
            List of formatted results
        """
        formatted = []
        
        for result in raw_results.get("results", []):
            formatted.append({
                "title": result.get("title", "Untitled"),
                "url": result.get("url", ""),
                "content": result.get("content", "")[:500],
                "score": result.get("score", 0.0),
                "image_url": result.get("image", None)
            })
        
        return formatted


# Global service instance
tavily_service = TavilyService()
