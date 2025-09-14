# 🤖 Streamlit Article Generator

Zaawansowana aplikacja do automatycznego generowania artykułów z wykorzystaniem LLM (Large Language Models).

## 🚀 Funkcjonalności

- **Integracja z wieloma LLM**: OpenAI GPT, Anthropic Claude, Ollama, Groq
- **Automatyczne wyszukiwanie źródeł**: Wikipedia, serwisy informacyjne, bazy akademickie
- **Różne typy artykułów**: eseje, posty blogowe, artykuły naukowe, poradniki
- **Analiza jakości**: ocena wiarygodności źródeł, statystyki tekstu
- **Eksport do wielu formatów**: HTML, Markdown, PDF, DOCX, TXT
- **Historia artykułów**: automatyczne zapisywanie i ładowanie poprzednich prac

## 📦 Instalacja

1. **Sklonuj repozytorium:**
```bash
git clone <repository-url>
cd streamlit_article_generator
```

2. **Zainstaluj zależności:**
```bash
pip install -r requirements.txt
```

3. **Pobierz modele NLP (opcjonalne):**
```bash
python -m spacy download pl_core_news_sm
```

4. **Skonfiguruj zmienne środowiskowe:**
```bash
cp .env.example .env
# Edytuj plik .env i dodaj swoje klucze API
```

## 🔧 Konfiguracja

### Klucze API

Uzupełnij plik `.env` swoimi kluczami API:

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### Ollama (opcjonalne)

Dla lokalnych modeli zainstaluj Ollama:
```bash
# Zainstaluj Ollama z oficjalnej strony
# Pobierz modele:
ollama pull llama3.1
ollama pull mistral
```

## 🎯 Użytkowanie

1. **Uruchom aplikację:**
```bash
streamlit run app.py
```

2. **Otwórz przeglądarkę** i przejdź do `http://localhost:8501`

3. **Skonfiguruj LLM**:
   - Wybierz providera (OpenAI, Anthropic, Ollama, Groq)
   - Wprowadź klucz API (jeśli wymagany)
   - Wybierz model

4. **Wygeneruj artykuł**:
   - Wprowadź temat
   - Wybierz typ artykułu
   - Ustaw parametry
   - Kliknij "Generuj Artykuł"

5. **Analizuj i eksportuj**:
   - Przejrzyj wygenerowany artykuł
   - Sprawdź źródła i statystyki
   - Wyeksportuj w wybranym formacie

## 📁 Struktura projektu

```
streamlit_article_generator/
├── app.py                   # Główna aplikacja Streamlit
├── config/
│   ├── settings.py          # Konfiguracja aplikacji
│   └── prompts.yaml         # Prompty dla różnych typów artykułów
├── modules/
│   ├── web_scraper.py       # Web scraping
│   ├── llm_integration.py   # Integracja z LLM
│   ├── article_generator.py # Generator artykułów
│   └── export_manager.py    # Eksport do plików
├── data/
│   ├── sources.json         # Lista zaufanych źródeł
│   └── history.json         # Historia artykułów
├── outputs/                 # Wygenerowane pliki
├── requirements.txt
├── .env                     # Zmienne środowiskowe
└── README.md
```

## 🔧 Zaawansowana konfiguracja

### Dodawanie nowych źródeł

Edytuj `data/sources.json`:
```json
{
  "news": [
    {
      "name": "Nowe Źródło",
      "base_url": "https://example.com",
      "search_url": "https://example.com/search?q={query}",
      "credibility": 0.8
    }
  ]
}
```

### Niestandardowe prompty

Modyfikuj `config/prompts.yaml` aby dostosować prompty dla różnych typów artykułów.

## 🐛 Rozwiązywanie problemów

### Błędy instalacji

```bash
# Jeśli problem z reportlab:
pip install reportlab --upgrade

# Jeśli problem z wordcloud:
pip install wordcloud --upgrade

# Jeśli problem ze spacy:
python -m spacy download pl_core_news_sm --force
```

### Błędy API

- Sprawdź poprawność kluczy API w pliku `.env`
- Upewnij się, że masz wystarczające limity w swoim koncie
- Dla Ollama sprawdź czy serwer działa: `ollama serve`

## 📈 Performance

### Optymalizacja

- Użyj cache dla powtarzających się zapytań
- Ogranicz liczbę źródeł dla szybszego działania
- Wybierz mniejsze modele dla prostszych zadań

### Limity

- Maksymalna liczba źródeł: 15
- Maksymalna długość artykułu: 3000 słów
- Cache TTL: 1 godzina

## 🤝 Wkład w projekt

1. Fork repozytorium
2. Utwórz branch dla nowej funkcjonalności
3. Commituj zmiany
4. Utwórz Pull Request

## 📄 Licencja

MIT License - szczegóły w pliku LICENSE

## 🆘 Wsparcie

W przypadku problemów:
1. Sprawdź sekcję "Rozwiązywanie problemów"
2. Przeszukaj istniejące Issues
3. Utwórz nowe Issue z opisem problemu

## 🔮 Roadmapa

- [ ] Integracja z dodatkowymi LLM
- [ ] Ulepszone wyszukiwanie akademickie
- [ ] System pluginów
- [ ] Współpraca zespołowa
- [ ] Mobile app
- [ ] API endpoints

---

Stworzono z ❤️ używając Streamlit i Python
