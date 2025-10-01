"""
LangChain chains for chat functionality
Simple, beginner-friendly implementation
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config import load_google_llm


def create_chat_chain(language: str = "en"):
    """
    Create a simple chat chain for travel Q&A
    
    How it works:
    1. User sends a question
    2. Prompt adds travel context
    3. LLM generates response
    4. Parser converts to string
    
    Args:
        language: Response language (en/fr)
        
    Returns:
        Runnable chain
    """
    # Load the LLM
    llm = load_google_llm()
    
    # Create prompt template based on language
    if language == "fr":
        system_message = """Vous êtes un assistant de voyage IA expert. Votre objectif est d'inspirer et d'aider les utilisateurs à planifier leurs voyages de rêve.

Vos capacités:
- Fournir des informations détaillées et engageantes sur les destinations, la culture, les activités et la logistique.
- Créer des exemples d'itinéraires.
- Offrir des conseils pratiques pour la budgétisation, la sécurité et les voyages responsables.
- Répondre dans un ton conversationnel et inspirant.

IMPORTANT: Vous n'êtes pas un agent de réservation. Toujours conseiller aux utilisateurs de vérifier les détails tels que les prix et la disponibilité sur les sites de réservation officiels."""
    elif language == "vi":
        system_message = """Bạn là một trợ lý du lịch AI chuyên nghiệp. Mục tiêu của bạn là truyền cảm hứng và giúp người dùng lên kế hoạch cho chuyến đi mơ ước của họ.

Khả năng của bạn:
- Cung cấp thông tin chi tiết và hấp dẫn về các điểm đến, văn hóa, hoạt động và hậu cần.
- Tạo các lịch trình mẫu.
- Đưa ra các mẹo thực tế về ngân sách, an toàn và du lịch có trách nhiệm.
- Phản hồi bằng một giọng điệu trò chuyện và đầy cảm hứng.

QUAN TRỌNG: Bạn không phải là đại lý đặt vé. Luôn khuyên người dùng xác minh các chi tiết như giá cả và tình trạng còn trống trên các trang web đặt vé chính thức."""
    else:
        system_message = """You are an expert AI travel assistant. Your goal is to inspire and help users plan their dream trips.

Your capabilities:
- Provide detailed and engaging information about destinations, culture, activities, and logistics.
- Create sample itineraries.
- Offer practical tips for budgeting, safety, and responsible travel.
- Respond in a conversational and inspiring tone.

IMPORTANT: You are not a booking agent. Always advise users to verify details like prices and availability on official booking sites."""
    
    # Create the chat prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "{user_question}")
    ])
    
    # Create output parser
    parser = StrOutputParser()
    
    # Chain them together using LCEL (LangChain Expression Language)
    # This is like: prompt → llm → parser
    chain = prompt | llm | parser
    
    return chain


def get_chat_response(message: str, language: str = "en"):
    """
    Get a chat response from the AI
    
    Args:
        message: User's question
        language: Response language
        
    Returns:
        AI response string
    """
    # Create the chain
    chain = create_chat_chain(language)
    
    # Invoke the chain with user's message
    response = chain.invoke({
        "user_question": message
    })
    
    return response


# Example usage (for testing):
# if __name__ == "__main__":
#     response = get_chat_response("What are the best places to visit in Paris?", "en")
#     print(response)
