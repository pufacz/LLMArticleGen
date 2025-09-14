import requests
from bs4 import BeautifulSoup
import wikipedia
import time
import json
import os
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
import logging

class WebScraper:
    def __init__(self, config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config.scraping.user_agent})
        self.sources_data = self._load_sources()
        
    def _load_sources(self) -> Dict:
        """Ładuje listę zaufanych źródeł"""
        sources_path = 'data/sources.json'
        if os.path.exists(sources_path):
            with open(sources_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"news": [], "academic": [], "general": []}
    
    def search_sources(self, topic: str, max_sources: int = 10, 
                      search_depth: str = "medium") -> List[Dict]:
        """Główna metoda wyszukiwania źródeł"""
        all_sources = []
        
        # Wikipedia search
        wikipedia_sources = self._search_wikipedia(topic, max_sources // 3)
        all_sources.extend(wikipedia_sources)
        
        # News search
        if search_depth in ["medium", "deep"]:
            news_sources = self._search_news(topic, max_sources // 3)
            all_sources.extend(news_sources)
        
        # Academic search
        if search_depth == "deep":
            academic_sources = self._search_academic(topic, max_sources // 3)
            all_sources.extend(academic_sources)
        
        # General web search
        general_sources = self._search_general(topic, max_sources - len(all_sources))
        all_sources.extend(general_sources)
        
        return self._rank_sources(all_sources)[:max_sources]
    
    def _search_wikipedia(self, topic: str, max_results: int) -> List[Dict]:
        """Wyszukuje w Wikipedii"""
        sources = []
        try:
            wikipedia.set_lang("pl")
            search_results = wikipedia.search(topic, results=max_results)
            
            for title in search_results[:max_results]:
                try:
                    page = wikipedia.page(title)
                    sources.append({
                        'title': page.title,
                        'url': page.url,
                        'content': page.summary[:1000],
                        'type': 'wikipedia',
                        'credibility': 0.9,
                        'date': None
                    })
                    time.sleep(self.config.scraping.rate_limit)
                except Exception as e:
                    logging.warning(f"Błąd przy pobieraniu strony Wikipedia {title}: {e}")
                    continue
                    
        except Exception as e:
            logging.error(f"Błąd wyszukiwania Wikipedia: {e}")
            
        return sources
    
    def _search_news(self, topic: str, max_results: int) -> List[Dict]:
        """Wyszukuje w serwisach informacyjnych"""
        sources = []
        news_sites = self.sources_data.get("news", [])
        
        for site in news_sites[:max_results]:
            try:
                # Przykładowe wyszukiwanie - w rzeczywistości użyj News API lub RSS
                response = self.session.get(
                    site.get('search_url', '').format(query=topic),
                    timeout=self.config.scraping.timeout
                )
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                articles = soup.find_all('article', limit=3)
                
                for article in articles:
                    title_elem = article.find(['h1', 'h2', 'h3'])
                    link_elem = article.find('a')
                    
                    if title_elem and link_elem:
                        sources.append({
                            'title': title_elem.get_text().strip(),
                            'url': urljoin(site['base_url'], link_elem.get('href')),
                            'content': self._extract_content(article.get_text()[:500]),
                            'type': 'news',
                            'credibility': site.get('credibility', 0.7),
                            'date': self._extract_date(article)
                        })
                
                time.sleep(self.config.scraping.rate_limit)
                
            except Exception as e:
                logging.warning(f"Błąd przy wyszukiwaniu w {site.get('name', 'Unknown')}: {e}")
                continue
                
        return sources
    
    def _search_academic(self, topic: str, max_results: int) -> List[Dict]:
        """Wyszukuje w bazach naukowych"""
        sources = []
        # Implementacja wyszukiwania w bazach akademickich
        # Przykład: Google Scholar, arXiv, itp.
        return sources
    
    def _search_general(self, topic: str, max_results: int) -> List[Dict]:
        """Ogólne wyszukiwanie internetowe"""
        sources = []
        general_sites = self.sources_data.get("general", [])
        
        for site in general_sites[:max_results]:
            try:
                # Implementacja ogólnego wyszukiwania
                pass
            except Exception as e:
                logging.warning(f"Błąd wyszukiwania ogólnego: {e}")
                continue
                
        return sources
    
    def _rank_sources(self, sources: List[Dict]) -> List[Dict]:
        """Ranguje źródła według wiarygodności i relevance"""
        return sorted(sources, key=lambda x: x.get('credibility', 0), reverse=True)
    
    def _extract_content(self, html_text: str) -> str:
        """Ekstraktuje czysty tekst"""
        return html_text.strip()[:1000]
    
    def _extract_date(self, element) -> Optional[str]:
        """Ekstraktuje datę z elementu"""
        # Implementacja ekstrakcji daty
        return None
