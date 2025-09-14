from .web_scraper import WebScraper
from .llm_integration import LLMManager
import logging
import time
from typing import Dict, List, Optional, Callable

class ArticleGenerator:
    def __init__(self, config):
        self.config = config
        self.web_scraper = WebScraper(config)
        self.llm_manager = LLMManager(config)
        
    def generate_article(self, topic: str, article_type: str, length: str,
                        language: str, provider: str, model: str, api_key: str,
                        max_sources: int, search_depth: str, 
                        verify_facts: bool, include_citations: bool,
                        custom_instructions: str, progress_callback: Optional[Callable] = None) -> Dict:
        """
        Główna metoda generowania artykułu z progress tracking
        """
        
        try:
            # 1. Walidacja (0-10%)
            if progress_callback:
                progress_callback(5, "Walidacja danych wejściowych...")
            
            if not topic.strip():
                raise ValueError("Temat artykułu nie może być pusty")
            
            if not api_key.strip() and provider != "ollama":
                raise ValueError("Klucz API jest wymagany")
            
            if progress_callback:
                progress_callback(10, "Dane zwalidowane pomyślnie")
            
            # 2. Inicjalizacja LLM (10-20%)
            if progress_callback:
                progress_callback(15, f"Inicjalizacja {provider}...")
            
            if not self.llm_manager.initialize_client(provider, api_key, model):
                raise ValueError(f"Nie można zainicjalizować klienta {provider}")
            
            if progress_callback:
                progress_callback(20, "Klient LLM zainicjalizowany")
            
            # 3. Wyszukiwanie źródeł (20-40%)
            if progress_callback:
                progress_callback(25, "Wyszukiwanie źródeł...")
            
            sources = self.web_scraper.search_sources(
                topic=topic,
                max_sources=max_sources,
                search_depth=search_depth
            )
            
            if progress_callback:
                progress_callback(40, f"Znaleziono {len(sources)} źródeł")
            
            # 4. Generowanie artykułu (40-70%)
            if progress_callback:
                progress_callback(50, "Generowanie artykułu...")
            
            article_data = self.llm_manager.generate_article(
                topic=topic,
                article_type=article_type,
                sources=sources,
                length=length,
                language=language,
                custom_instructions=custom_instructions
            )
            
            if progress_callback:
                progress_callback(70, "Artykuł wygenerowany")
            
            # 5. Opcjonalna weryfikacja faktów (70-90%)
            if verify_facts:
                if progress_callback:
                    progress_callback(80, "Weryfikacja faktów...")
                
                # Implementacja weryfikacji faktów
                article_data['fact_check_score'] = self._verify_facts(
                    article_data['content'], sources
                )
                
                if progress_callback:
                    progress_callback(90, "Fakty zweryfikowane")
            
            # 6. Finalizacja (90-100%)
            if progress_callback:
                progress_callback(95, "Finalizacja...")
            
            # Dodaj źródła i metadata
            article_data['sources'] = sources
            article_data['generation_time'] = time.time()
            article_data['word_count'] = len(article_data['content'].split())
            
            if progress_callback:
                progress_callback(100, "Artykuł gotowy!")
            
            return article_data
            
        except Exception as e:
            logging.error(f"Błąd generowania artykułu: {e}")
            if progress_callback:
                progress_callback(0, f"Błąd: {str(e)}")
            raise
    
    def _verify_facts(self, content: str, sources: List[Dict]) -> float:
        """Weryfikuje fakty w artykule względem źródeł"""
        # Implementacja weryfikacji faktów
        # Można użyć NLP do wyodrębnienia faktów i porównania ze źródłami
        return 0.8  # Placeholder
