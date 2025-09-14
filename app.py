import streamlit as st
import os
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Import modułów
from config.settings import AppConfig
from modules.article_generator import ArticleGenerator
from modules.export_manager import ExportManager

# Konfiguracja strony
st.set_page_config(
    page_title="Generator Artykułów AI",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicjalizacja konfiguracji
@st.cache_resource
def load_config():
    return AppConfig.load()

@st.cache_resource
def load_article_generator(config):
    return ArticleGenerator(config)

@st.cache_resource
def load_export_manager(config):
    return ExportManager(config)

# Funkcje pomocnicze
def save_to_history(article_data):
    """Zapisuje artykuł do historii"""
    history_file = 'data/history.json'
    os.makedirs('data', exist_ok=True)
    
    history_entry = {
        'timestamp': datetime.now().isoformat(),
        'topic': article_data['metadata']['topic'],
        'type': article_data['metadata']['type'],
        'word_count': article_data.get('word_count', 0),
        'quality_score': article_data['metadata']['quality_score'],
        'provider': article_data['metadata']['provider']
    }
    
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = []
    
    history.insert(0, history_entry)
    history = history[:50]  # Limit do 50 ostatnich
    
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history():
    """Ładuje historię artykułów"""
    history_file = 'data/history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def main():
    # Ładowanie konfiguracji
    config = load_config()
    article_generator = load_article_generator(config)
    export_manager = load_export_manager(config)
    
    # Inicjalizacja session state
    if 'article_generated' not in st.session_state:
        st.session_state.article_generated = False
    if 'current_article' not in st.session_state:
        st.session_state.current_article = None
    if 'generation_time' not in st.session_state:
        st.session_state.generation_time = 0
    
    # Tytuł i opis
    st.title("🤖 Generator Artykułów AI")
    st.markdown("Zaawansowana aplikacja do automatycznego generowania artykułów z wykorzystaniem LLM")
    
    # Sidebar - Konfiguracja
    with st.sidebar:
        st.header("⚙️ Konfiguracja")
        
        # Sekcja LLM Provider
        st.subheader("🧠 LLM Provider")
        provider = st.selectbox(
            "Wybierz providera:",
            ["openai", "anthropic", "ollama", "groq"],
            help="Wybierz dostawcę modelu językowego"
        )
        
        # Wybór modelu w zależności od providera
        if provider == "openai":
            model = st.selectbox("Model:", config.llm.openai_models)
            api_key = st.text_input("OpenAI API Key:", type="password")
        elif provider == "anthropic":
            model = st.selectbox("Model:", config.llm.anthropic_models)
            api_key = st.text_input("Anthropic API Key:", type="password")
        elif provider == "ollama":
            model = st.selectbox("Model:", config.llm.ollama_models)
            api_key = ""  # Ollama nie wymaga API key
            st.info("Ollama działa lokalnie - nie wymaga API key")
        elif provider == "groq":
            model = st.selectbox("Model:", config.llm.groq_models)
            api_key = st.text_input("Groq API Key:", type="password")
        
        st.divider()
        
        # Zaawansowane ustawienia
        with st.expander("🔧 Ustawienia zaawansowane"):
            max_sources = st.slider(
                "Maksymalna liczba źródeł:",
                min_value=3,
                max_value=15,
                value=10
            )
            
            search_depth = st.selectbox(
                "Głębokość wyszukiwania:",
                ["shallow", "medium", "deep"],
                index=1
            )
            
            col1, col2 = st.columns(2)
            with col1:
                verify_facts = st.checkbox("Weryfikacja faktów", value=False)
            with col2:
                include_citations = st.checkbox("Cytowania", value=True)
            
            custom_instructions = st.text_area(
                "Dodatkowe instrukcje dla LLM:",
                placeholder="Np. skup się na aspekcie technicznym..."
            )
        
        st.divider()
        
        # Historia artykułów
        with st.expander("📚 Historia artykułów"):
            history = load_history()
            if history:
                for entry in history[:5]:  # Pokaż ostatnie 5
                    st.write(f"**{entry['topic']}**")
                    st.write(f"Typ: {entry['type']} | Ocena: {entry['quality_score']:.2f}")
                    st.write(f"Data: {entry['timestamp'][:16]}")
                    st.write("---")
            else:
                st.write("Brak historii")
    
    # Główny interfejs
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        topic = st.text_input(
            "🎯 Temat artykułu:",
            placeholder="Wprowadź temat artykułu...",
            help="Opisz temat, o którym ma zostać napisany artykuł"
        )
    
    with col2:
        article_type = st.selectbox(
            "📝 Typ artykułu:",
            ["esej", "blog_post", "artykul_wikipedii", "artykul_naukowy", "poradnik", "news_summary"]
        )
    
    with col3:
        language = st.selectbox(
            "🌍 Język:",
            ["polski", "angielski", "niemiecki", "francuski"]
        )
    
    # Długość artykułu
    length = st.radio(
        "📏 Długość artykułu:",
        ["krótki (500-800 słów)", "średni (800-1500 słów)", "długi (1500-3000 słów)"],
        horizontal=True
    )
    
    # Przycisk generowania
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_button = st.button(
            "🚀 Generuj Artykuł",
            type="primary",
            use_container_width=True,
            disabled=not topic.strip() or (not api_key.strip() and provider != "ollama")
        )
    
    # Proces generowania
    if generate_button:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress, message):
            progress_bar.progress(progress / 100)
            status_text.text(message)
        
        try:
            start_time = time.time()
            
            # Generowanie artykułu
            article_data = article_generator.generate_article(
                topic=topic,
                article_type=article_type,
                length=length,
                language=language,
                provider=provider,
                model=model,
                api_key=api_key,
                max_sources=max_sources,
                search_depth=search_depth,
                verify_facts=verify_facts,
                include_citations=include_citations,
                custom_instructions=custom_instructions,
                progress_callback=update_progress
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Zapisz do session state
            st.session_state.current_article = article_data
            st.session_state.article_generated = True
            st.session_state.generation_time = generation_time
            
            # Zapisz do historii
            save_to_history(article_data)
            
            # Ukryj progress bar
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Artykuł wygenerowany pomyślnie w {generation_time:.1f} sekund!")
            st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Błąd podczas generowania artykułu: {str(e)}")
    
    # Wyświetlanie wyników
    if st.session_state.article_generated and st.session_state.current_article:
        article_data = st.session_state.current_article
        
        # Tabs dla wyników
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Artykuł", "📊 Analiza", "🔗 Źródła", "💾 Eksport"])
        
        with tab1:
            # Metryki
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Liczba słów",
                    article_data.get('word_count', 0)
                )
            
            with col2:
                st.metric(
                    "Źródła",
                    len(article_data.get('sources', []))
                )
            
            with col3:
                st.metric(
                    "Czas generowania",
                    f"{st.session_state.generation_time:.1f}s"
                )
            
            with col4:
                score = article_data['metadata']['quality_score']
                st.metric(
                    "Ocena jakości",
                    f"{score:.2f}/1.0",
                    delta=f"{score-0.5:.2f}"
                )
            
            st.divider()
            
            # Główna treść artykułu
            st.markdown("### Treść artykułu:")
            st.markdown(article_data['content'])
            
            # Opcja edycji
            with st.expander("✏️ Edytuj artykuł"):
                edited_content = st.text_area(
                    "Edytuj treść:",
                    value=article_data['content'],
                    height=300
                )
                
                if st.button("💾 Zapisz zmiany"):
                    st.session_state.current_article['content'] = edited_content
                    st.success("Zmiany zapisane!")
                    st.rerun()
        
        with tab2:
            # Statystyki tekstu
            content = article_data['content']
            words = len(content.split())
            chars = len(content)
            paragraphs = len([p for p in content.split('\n') if p.strip()])
            sentences = len([s for s in content.split('.') if s.strip()])
            avg_sentence_length = words / max(sentences, 1)
            
            stats_df = pd.DataFrame({
                'Metryka': ['Słowa', 'Znaki', 'Akapity', 'Zdania', 'Średnia długość zdania'],
                'Wartość': [words, chars, paragraphs, sentences, f"{avg_sentence_length:.1f}"]
            })
            
            st.subheader("📈 Statystyki tekstu")
            st.dataframe(stats_df, use_container_width=True)
            
            # Wykresy
            col1, col2 = st.columns(2)
            
            with col1:
                # Wykres rozkładu typów źródeł
                if 'sources' in article_data and article_data['sources']:
                    source_types = [s.get('type', 'nieznany') for s in article_data['sources']]
                    type_counts = pd.Series(source_types).value_counts()
                    
                    fig_pie = px.pie(
                        values=type_counts.values,
                        names=type_counts.index,
                        title="Rozkład typów źródeł"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Wykres wiarygodności źródeł
                if 'sources' in article_data and article_data['sources']:
                    credibility_data = [s.get('credibility', 0) for s in article_data['sources']]
                    
                    fig_bar = go.Figure(data=[
                        go.Bar(x=list(range(1, len(credibility_data)+1)), y=credibility_data)
                    ])
                    fig_bar.update_layout(
                        title="Wiarygodność źródeł",
                        xaxis_title="Źródło #",
                        yaxis_title="Wiarygodność"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            # Opcjonalna chmura słów
            if WORDCLOUD_AVAILABLE and st.checkbox("☁️ Pokaż chmurę słów"):
                try:
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        colormap='viridis'
                    ).generate(content)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Błąd generowania chmury słów: {e}")
        
        with tab3:
            # Filtry źródeł
            if 'sources' in article_data and article_data['sources']:
                col1, col2 = st.columns(2)
                
                with col1:
                    source_types = list(set([s.get('type', 'nieznany') for s in article_data['sources']]))
                    selected_types = st.multiselect(
                        "Filtruj według typu:",
                        source_types,
                        default=source_types
                    )
                
                with col2:
                    min_credibility = st.slider(
                        "Minimalna wiarygodność:",
                        0.0, 1.0, 0.0, 0.1
                    )
                
                # Filtrowane źródła
                filtered_sources = [
                    s for s in article_data['sources']
                    if s.get('type', 'nieznany') in selected_types
                    and s.get('credibility', 0) >= min_credibility
                ]
                
                st.write(f"Znaleziono {len(filtered_sources)} źródeł")
                
                # Wyświetl źródła
                for i, source in enumerate(filtered_sources, 1):
                    with st.expander(f"🔗 Źródło {i}: {source.get('title', 'Bez tytułu')}"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**URL:** {source.get('url', 'Brak URL')}")
                            st.write(f"**Typ:** {source.get('type', 'nieznany')}")
                            st.write(f"**Podsumowanie:** {source.get('content', 'Brak treści')[:300]}...")
                        
                        with col2:
                            credibility = source.get('credibility', 0)
                            st.metric("Wiarygodność", f"{credibility:.1f}/1.0")
                            
                            if source.get('url'):
                                st.link_button(
                                    "Otwórz źródło",
                                    source['url'],
                                    use_container_width=True
                                )
            else:
                st.info("Brak dostępnych źródeł")
        
        with tab4:
            # Ustawienia eksportu
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⚙️ Ustawienia eksportu")
                
                export_format = st.selectbox(
                    "Format eksportu:",
                    ["html", "markdown", "pdf", "docx", "txt"]
                )
                
                include_bibliography = st.checkbox("Dołącz bibliografię", value=True)
                include_metadata = st.checkbox("Dołącz metadane", value=True)
            
            with col2:
                st.subheader("📁 Nazwa pliku")
                
                filename = st.text_input(
                    "Nazwa pliku:",
                    value=article_data['metadata']['topic'][:30].replace(' ', '_')
                )
                
                full_filename = f"{filename}.{export_format}"
                st.write(f"**Pełna nazwa:** `{full_filename}`")
            
            # Eksport
            if st.button("💾 Eksportuj artykuł", type="primary", use_container_width=True):
                try:
                    filepath = export_manager.save_article(
                        article_data=article_data,
                        filename=filename,
                        export_format=export_format,
                        include_bibliography=include_bibliography,
                        include_metadata=include_metadata
                    )
                    
                    st.success(f"✅ Artykuł wyeksportowany do: {filepath}")
                    
                    # Download button
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            label=f"⬇️ Pobierz {export_format.upper()}",
                            data=f.read(),
                            file_name=full_filename,
                            mime=f"application/{export_format}",
                            use_container_width=True
                        )
                        
                except Exception as e:
                    st.error(f"❌ Błąd eksportu: {str(e)}")
            
            # Podgląd eksportu
            with st.expander("👁️ Podgląd eksportu"):
                if export_format == "markdown":
                    preview_content = f"# {article_data['metadata']['topic']}\n\n"
                    preview_content += article_data['content'][:500] + "..."
                    st.code(preview_content, language="markdown")
                elif export_format == "html":
                    preview_content = f"<h1>{article_data['metadata']['topic']}</h1>\n"
                    preview_content += "<p>" + article_data['content'][:500] + "...</p>"
                    st.code(preview_content, language="html")
                else:
                    st.text(article_data['content'][:500] + "...")

if __name__ == "__main__":
    main()
