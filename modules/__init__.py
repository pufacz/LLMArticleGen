"""
Moduły dla aplikacji Streamlit Article Generator
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

# Import głównych klas
from .web_scraper import WebScraper
from .llm_integration import LLMManager
from .article_generator import ArticleGenerator
from .export_manager import ExportManager

__all__ = [
    'WebScraper',
    'LLMManager', 
    'ArticleGenerator',
    'ExportManager'
]
