from dataclasses import dataclass
from typing import Dict, List
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMConfig:
    openai_models: List[str]
    anthropic_models: List[str]
    ollama_models: List[str]
    groq_models: List[str]
    max_tokens: int
    temperature: float

@dataclass
class ScrapingConfig:
    max_sources: int
    timeout: int
    rate_limit: float
    user_agent: str

@dataclass
class AppConfig:
    llm: LLMConfig
    scraping: ScrapingConfig
    cache_ttl: int
    max_history: int
    output_dir: str
    
    @classmethod
    def load(cls):
        return cls(
            llm=LLMConfig(
                openai_models=['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
                anthropic_models=['claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
                ollama_models=['llama3.1', 'mistral', 'codellama', 'gemma2'],
                groq_models=['llama3-70b-8192', 'mixtral-8x7b-32768'],
                max_tokens=4000,
                temperature=0.7
            ),
            scraping=ScrapingConfig(
                max_sources=15,
                timeout=10,
                rate_limit=1.0,
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            ),
            cache_ttl=3600,
            max_history=50,
            output_dir='outputs'
        )
