import openai
import anthropic
from groq import Groq
import yaml
import os
from typing import Dict, List, Optional, Any
import logging

class LLMManager:
    def __init__(self, config):
        self.config = config
        self.clients = {}
        self.prompts = self._load_prompts()
        
    def _load_prompts(self) -> Dict:
        """Ładuje prompty z pliku YAML"""
        with open('config/prompts.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def initialize_client(self, provider: str, api_key: str, model: str) -> bool:
        """Inicjalizuje klienta dla wybranego providera"""
        try:
            if provider == "openai":
                self.clients['openai'] = openai.OpenAI(api_key=api_key)
                self.current_provider = "openai"
                self.current_model = model
                
            elif provider == "anthropic":
                self.clients['anthropic'] = anthropic.Anthropic(api_key=api_key)
                self.current_provider = "anthropic"
                self.current_model = model
                
            elif provider == "ollama":
                # Ollama usually runs locally without API key
                try:
                    import ollama
                    self.clients['ollama'] = ollama.Client()
                except ImportError:
                    logging.error("Ollama not installed. Run: pip install ollama")
                    return False
                self.current_provider = "ollama"
                self.current_model = model
                
            elif provider == "groq":
                self.clients['groq'] = Groq(api_key=api_key)
                self.current_provider = "groq"
                self.current_model = model
                
            return True
            
        except Exception as e:
            logging.error(f"Błąd inicjalizacji klienta {provider}: {e}")
            return False
    
    def generate_article(self, topic: str, article_type: str, sources: List[Dict],
                        length: str, language: str = "polski", 
                        custom_instructions: str = "") -> Dict:
        """Generuje artykuł używając LLM"""
        
        if self.current_provider not in self.clients:
            raise ValueError("Klient LLM nie został zainicjalizowany")
        
        # Przygotuj prompt
        prompt_config = self.prompts['article_types'][article_type]
        formatted_sources = self._format_sources(sources)
        
        user_prompt = prompt_config['user_template'].format(
            topic=topic,
            length=length,
            language=language,
            sources=formatted_sources,
            custom_instructions=custom_instructions or "Brak dodatkowych instrukcji"
        )
        
        try:
            # Generuj artykuł
            if self.current_provider == "openai":
                response = self._generate_openai(prompt_config['system_prompt'], user_prompt)
            elif self.current_provider == "anthropic":
                response = self._generate_anthropic(prompt_config['system_prompt'], user_prompt)
            elif self.current_provider == "ollama":
                response = self._generate_ollama(prompt_config['system_prompt'], user_prompt)
            elif self.current_provider == "groq":
                response = self._generate_groq(prompt_config['system_prompt'], user_prompt)
            
            # Kalkuluj metryki
            quality_score = self._calculate_quality_score(response, sources)
            
            return {
                'content': response,
                'metadata': {
                    'topic': topic,
                    'type': article_type,
                    'length': length,
                    'language': language,
                    'provider': self.current_provider,
                    'model': self.current_model,
                    'sources_count': len(sources),
                    'quality_score': quality_score
                }
            }
            
        except Exception as e:
            logging.error(f"Błąd generowania artykułu: {e}")
            raise
    
    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Generuje używając OpenAI"""
        response = self.clients['openai'].chat.completions.create(
            model=self.current_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=self.config.llm.max_tokens,
            temperature=self.config.llm.temperature
        )
        return response.choices[0].message.content
    
    def _generate_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Generuje używając Anthropic Claude"""
        response = self.clients['anthropic'].messages.create(
            model=self.current_model,
            max_tokens=self.config.llm.max_tokens,
            temperature=self.config.llm.temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.content[0].text
    
    def _generate_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Generuje używając Ollama"""
        import ollama
        response = ollama.chat(
            model=self.current_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response['message']['content']
    
    def _generate_groq(self, system_prompt: str, user_prompt: str) -> str:
        """Generuje używając Groq"""
        response = self.clients['groq'].chat.completions.create(
            model=self.current_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=self.config.llm.max_tokens,
            temperature=self.config.llm.temperature
        )
        return response.choices[0].message.content
    
    def _format_sources(self, sources: List[Dict]) -> str:
        """Formatuje źródła do prompta"""
        formatted = []
        for i, source in enumerate(sources, 1):
            formatted.append(f"""
{i}. {source.get('title', 'Bez tytułu')}
   URL: {source.get('url', 'Brak URL')}
   Typ: {source.get('type', 'nieznany')}
   Treść: {source.get('content', 'Brak treści')[:500]}...
   Wiarygodność: {source.get('credibility', 0):.1f}/1.0
""")
        return "\n".join(formatted)
    
    def _calculate_quality_score(self, content: str, sources: List[Dict]) -> float:
        """Kalkuluje ocenę jakości artykułu"""
        score = 0.5  # Base score
        
        # Długość artykułu
        if len(content) > 1000:
            score += 0.1
        if len(content) > 2000:
            score += 0.1
            
        # Liczba źródeł
        if len(sources) >= 3:
            score += 0.1
        if len(sources) >= 5:
            score += 0.1
            
        # Średnia wiarygodność źródeł
        if sources:
            avg_credibility = sum(s.get('credibility', 0) for s in sources) / len(sources)
            score += avg_credibility * 0.2
        
        return min(score, 1.0)
