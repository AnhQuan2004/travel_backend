"""
LangChain chains for travel document analysis
Uses structured output with Pydantic models
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from app.config import load_google_llm
from app.models.schemas import TravelAnalysis


def create_analysis_chain(language: str = "en"):
    """
    Create a chain for structured travel document analysis
    
    How it works:
    1. User provides travel document text
    2. Prompt instructs LLM to analyze in structured format
    3. LLM generates JSON response
    4. Parser converts JSON to Pydantic model
    
    Args:
        language: Response language (en/fr)
        
    Returns:
        Runnable chain that outputs TravelAnalysis
    """
    # Load the LLM
    llm = load_google_llm()
    
    # Create Pydantic parser - forces structured output
    parser = PydanticOutputParser(pydantic_object=TravelAnalysis)
    
    # Get format instructions from parser
    format_instructions = parser.get_format_instructions()
    
    # Create prompt based on language
    if language == "fr":
        system_message = """Vous êtes un assistant de voyage IA spécialisé dans l'analyse de documents. Votre tâche est d'extraire les informations clés des documents de voyage et de les présenter de manière structurée.

Analysez le document fourni et extrayez les détails pertinents. Concentrez-vous sur les informations exploitables telles que les dates, les heures, les lieux, les numéros de confirmation et les coordonnées."""
        
        user_template = """Analysez ce document de voyage et fournissez une analyse structurée:

Document de Voyage:
{travel_text}

Contexte Additionnel:
{context}

{format_instructions}

Répondez UNIQUEMENT en JSON valide."""
    elif language == "vi":
        system_message = """Bạn là một trợ lý du lịch AI chuyên phân tích tài liệu. Nhiệm vụ của bạn là trích xuất thông tin chính từ các tài liệu du lịch và trình bày nó ở định dạng có cấu trúc.

Phân tích tài liệu được cung cấp và trích xuất các chi tiết có liên quan. Tập trung vào các thông tin có thể hành động như ngày, giờ, địa điểm, số xác nhận và chi tiết liên hệ."""
        
        user_template = """Phân tích tài liệu du lịch này và cung cấp một phân tích có cấu trúc:

Tài liệu du lịch:
{travel_text}

Bối cảnh bổ sung:
{context}

{format_instructions}

Chỉ trả lời bằng JSON hợp lệ."""
    else:
        system_message = """You are a travel AI assistant specializing in document analysis. Your task is to extract key information from travel documents and present it in a structured format.

Analyze the provided document and extract the relevant details. Focus on actionable information such as dates, times, locations, confirmation numbers, and contact details."""
        
        user_template = """Analyze this travel document and provide a structured analysis:

Travel Document:
{travel_text}

Additional Context:
{context}

{format_instructions}

Respond ONLY with valid JSON."""
    
    # Create the chat prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", user_template)
    ])
    
    # Partially fill in format instructions
    prompt = prompt.partial(format_instructions=format_instructions)
    
    # Chain: prompt → llm → parser
    chain = prompt | llm | parser
    
    return chain


def analyze_travel_document(text: str, context: str = "", language: str = "en"):
    """
    Analyze travel document text and return structured results
    
    Args:
        text: Travel document text
        context: Additional travel context
        language: Response language
        
    Returns:
        TravelAnalysis object with structured data
    """
    # Create the chain
    chain = create_analysis_chain(language)
    
    # Invoke the chain
    try:
        result = chain.invoke({
            "travel_text": text,
            "context": context if context else "No additional context provided"
        })
        return result
    except Exception as e:
        # Fallback if parsing fails
        print(f"Analysis error: {e}")
        return TravelAnalysis(
            summary=f"Analysis completed but encountered formatting issues: {str(e)[:200]}",
            key_findings=["Analysis was performed but results need manual review"],
            recommendations=["Verify all travel details with official sources"],
            next_steps=["Double-check booking confirmations", "Contact travel providers if necessary"]
        )
