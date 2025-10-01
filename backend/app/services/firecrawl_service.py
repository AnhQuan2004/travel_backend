"""
Firecrawl Web Scraping Service
Handles scraping content and extracting image URLs from URLs
"""
import re
import json
from typing import Any, Dict
from firecrawl import FirecrawlApp
from app.config import settings


def _to_dict(obj: Any) -> Dict[str, Any]:
    """Best-effort convert Firecrawl response to a plain dict."""
    if isinstance(obj, dict):
        return obj
    # Pydantic v2
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Pydantic v1
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return obj.dict()
        except Exception:
            pass
    # JSON fallback
    if hasattr(obj, "json") and callable(getattr(obj, "json")):
        try:
            return json.loads(obj.json())
        except Exception:
            pass
    # Last resort
    return {"raw": str(obj)}

class FirecrawlService:
    """Service class for Firecrawl scraping operations"""

    def __init__(self):
        """Initialize Firecrawl client"""
        self.client = FirecrawlApp(api_key=settings.firecrawl_api_key)

    def scrape(self, url: str):
        """
        Scrape a URL and return its content and image URLs
        
        Args:
            url: The URL to scrape
            
        Returns:
            Dictionary with scraped data, including a list of image URLs
        """
        try:
            # Scrape the URL using markdown format and caching
            scraped_obj = self.client.scrape(url, formats=["markdown"], max_age=3600000)

            # Normalize to dict (handles Document / Pydantic models)
            data = _to_dict(scraped_obj)

            # Some SDKs nest under "data" or return top-level "markdown"
            markdown = None
            if "markdown" in data and isinstance(data["markdown"], str):
                markdown = data["markdown"]
            elif "data" in data and isinstance(data["data"], dict) and "markdown" in data["data"]:
                markdown = data["data"]["markdown"]

            image_urls = []
            if isinstance(markdown, str):
                # Find all markdown image URLs: ![alt](url)
                image_urls = re.findall(r'!\[.*?\]\((.*?)\)', markdown)

            # Build a clean return payload without mutating the SDK object
            return {
                "markdown": markdown,
                "image_urls": image_urls,
                # include the whole normalized dict in case you need more fields
                "raw": data,
            }

        except Exception as e:
            # Return a dict with an error key
            return {"error": f"Scraping error: {str(e)}", "image_urls": []}


# Global service instance
firecrawl_service = FirecrawlService()