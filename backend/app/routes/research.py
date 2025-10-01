"""
Travel research endpoints using Tavily + LangChain
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import ResearchRequest, ResearchResponse, ResearchResult
from app.services.tavily_service import tavily_service
from app.services.firecrawl_service import firecrawl_service
from app.chains.chat_chain import get_chat_response
from datetime import datetime

router = APIRouter(prefix="/api", tags=["Research"])


@router.post("/research", response_model=ResearchResponse)
async def search_travel_research(request: ResearchRequest):
    """
    Search for travel research and information
    Uses Tavily for search + LangChain for summary
    
    Args:
        request: Research request with query and parameters
        
    Returns:
        Research results with AI-generated summary
    """
    try:
        # Search using Tavily
        raw_results = tavily_service.search_travel_research(
            query=request.query,
            max_results=request.max_results
        )
        
        # Format results
        formatted_results = tavily_service.format_results(raw_results)
        
        # Scrape content and get image URLs from top 3 results
        scraped_contents = []
        all_image_urls = []
        for r in formatted_results[:3]:
            scraped_data = firecrawl_service.scrape(r["url"])
            print("🔥 Firecrawl scraped data:", scraped_data)  # Added log
            if "content" in scraped_data:
                scraped_contents.append(f"Source: {r['title']}\n{scraped_data['content']}")
            if "image_urls" in scraped_data and len(scraped_data["image_urls"]) > 0:
                all_image_urls.extend(scraped_data["image_urls"])
                r["image_url"] = scraped_data["image_urls"][0] # Keep the first for the source object

        # Generate summary using LangChain chat
        results_text = "\n\n".join(scraped_contents)
        
        if request.language == "fr":
            synthesis_prompt = f"""Basé sur les informations de voyage suivantes, rédigez une réponse complète et engageante pour la requête de l'utilisateur '{request.query}'. Intégrez les détails clés dans un récit cohérent.

{results_text}

Votre réponse doit être bien structurée, informative et facile à lire."""
        elif request.language == "vi":
            synthesis_prompt = f"""Dựa trên thông tin du lịch sau đây, hãy viết một câu trả lời tổng hợp đầy đủ và hấp dẫn cho truy vấn của người dùng '{request.query}'. Tích hợp các chi tiết chính vào một bài tường thuật mạch lạc.

{results_text}

Câu trả lời của bạn phải có cấu trúc tốt, nhiều thông tin và dễ đọc."""
        else:
            synthesis_prompt = f"""Based on the following travel information, write a comprehensive and engaging synthesized response for the user's query '{request.query}'. Integrate the key details into a coherent narrative.

{results_text}

Your response should be well-structured, informative, and easy to read."""
        
        # Use LangChain chat to generate synthesized response
        synthesized_response = get_chat_response(synthesis_prompt, request.language)
        
        # Filter and limit image URLs based on query keywords
        query_keywords = request.query.lower().split()
        filtered_image_urls = [
            url for url in all_image_urls
            if any(keyword in url.lower() for keyword in query_keywords)
        ][:5]
        
        # Convert to ResearchResult models for sources
        sources = [
            ResearchResult(
                title=r["title"],
                url=r["url"],
                content=r["content"],
                score=r["score"],
                image_url=r.get("image_url")
            )
            for r in formatted_results
        ]
        
        return ResearchResponse(
            query=request.query,
            response=synthesized_response,
            image_urls=filtered_image_urls,
            sources=sources,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research error: {str(e)}")
