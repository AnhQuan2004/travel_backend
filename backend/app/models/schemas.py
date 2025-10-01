"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from datetime import datetime


class HealthCheckResponse(BaseModel):
    """Health check endpoint response"""
    status: str
    timestamp: datetime
    message: str


class ChatRequest(BaseModel):
    """Chat/Question request model"""
    message: str = Field(..., min_length=1, max_length=1000, description="User's travel question")
    language: str = Field(default="en", description="Response language (en/fr/vi)")


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    language: str
    timestamp: datetime


class AnalysisRequest(BaseModel):
    """Travel document analysis request (for text input)"""
    text: str = Field(..., min_length=1, description="Travel document text to analyze")
    context: str = Field(default="", description="Additional context about the travel")
    language: str = Field(default="en", description="Response language")


class TravelAnalysis(BaseModel):
    """Structured travel analysis output"""
    summary: str = Field(description="Brief overview of the travel document")
    key_findings: list[str] = Field(description="List of important findings")
    recommendations: list[str] = Field(description="Travel recommendations")
    next_steps: list[str] = Field(description="Suggested next steps")


class AnalysisResponse(BaseModel):
    """Analysis response model"""
    summary: str
    key_findings: list[str]
    recommendations: list[str]
    next_steps: list[str]
    disclaimer: str
    language: str
    timestamp: datetime


class ImageAnalysisResponse(BaseModel):
    """Image analysis response"""
    extracted_text: str
    analysis: AnalysisResponse


class ResearchRequest(BaseModel):
    """Research request model"""
    query: str = Field(..., min_length=3, max_length=200, description="Travel topic to research")
    max_results: int = Field(default=5, ge=1, le=10, description="Number of results")
    language: str = Field(default="en", description="Response language")


class ResearchResult(BaseModel):
    """Single research result"""
    title: str
    url: str
    content: str
    score: float
    image_url: str | None = None


class ResearchResponse(BaseModel):
    """Research response model"""
    query: str
    response: str  # Unified synthesized response
    image_urls: list[str]  # All image URLs
    sources: list[ResearchResult]  # Keep original sources for reference
    timestamp: datetime
