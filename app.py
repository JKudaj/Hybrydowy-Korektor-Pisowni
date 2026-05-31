import streamlit as st
import time
import re
import os
from model_ngram import tokenize_corpus, build_vocabulary, sent_tokenize, replace_rare_with_unk, add_sentence_boundaries, count_ngrams
from generator import BKTree
from ranker import NoisyChannelRanker 

@st.cache_data
def load_sjp_dictionary(file_path: str = "odm.txt") -> set[str]:
    sjp_words = set()
    if not os.path.exists(file_path):
        return sjp_words
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                words_in_line = line.strip().split(",")
                for w in words_in_line:
                    cleaned_word = w.strip().lower()
                    if cleaned_word and cleaned_word.isalpha():
                        sjp_words.add(cleaned_word)
    except Exception:
        pass
    return sjp_words

slownik = load_sjp_dictionary("odm.txt")

# Funkcja pobierająca książki i trenująca model – bez powtarzania operacji przy każdym kliknięciu
@st.cache_resource
def zaladuj_i_wytrenuj_system():
    import urllib.request

    slownik = load_sjp_dictionary("odm.txt")

    linki = {
        "Lalka (Prus)": "https://wolnelektury.pl/media/book/txt/lalka-tom-pierwszy.txt",
        "Przedwiosnie (Zeromski)": "https://wolnelektury.pl/media/book/txt/przedwiosnie.txt",
        "Syzyfowe prace (Zeromski)": "https://wolnelektury.pl/media/book/txt/syzyfowe-prace.txt",
        "Faraon (Prus)": "https://wolnelektury.pl/media/book/txt/faraon-tom-pierwszy.txt",
        "Rok 1984 (Orwell)": "https://wolnelektury.pl/media/book/txt/orwell-rok-1984.txt",
        "Maly Ksiaze (Saint-Exupery)": "https://wolnelektury.pl/media/book/txt/saint-exupery-maly-ksiaze.txt",
        "Ziemia obiecana (Reymont)": "https://wolnelektury.pl/media/book/txt/ziemia-obiecana-tom-pierwszy.txt",
        "W pustyni i w puszczy (Sienkiewicz)": "https://wolnelektury.pl/media/book/txt/w-pustyni-i-w-puszczy.txt"
    }
    lista_tekstow = []
    for url in linki.values():
        try:
            with urllib.request.urlopen(url) as response:
                lista_tekstow.append(response.read().decode('utf-8'))
        except:
            pass
    raw_text = "\n\n".join(lista_tekstow)
    
    # Przetwarzanie korpusu i generowanie statystyk n-gramowych
    tokens = tokenize_corpus(raw_text)
    vocab = build_vocabulary(tokens, min_count=2)
    vocab.add('<UNK>')
    vocab_size = len(vocab) + 1
    
    raw_sentences = sent_tokenize(raw_text)
    tokenized_sentences = [tokenize_corpus(s) for s in raw_sentences]
    sentences_unk = [replace_rare_with_unk(s, vocab) for s in tokenized_sentences]
    sentences_bigram = add_sentence_boundaries(sentences_unk, n=2)
    bigram_counts, unigram_counts = count_ngrams(sentences_bigram, n=2)
    
    # Filtrowanie słownika z tokenów technicznych przed wrzuceniem do drzewa BK
    czyste_slowa = {w for w in vocab if w.isalnum() and w not in ['<s>', '</s>', '<UNK>']}
    
    bk_tree = BKTree()
    for word in czyste_slowa:
        bk_tree.insert(word)
        
    # Inicjalizacja rankera oceniającego propozycje poprawek
    ranker = NoisyChannelRanker(bk_tree, bigram_counts, unigram_counts, vocab_size)
    
    return ranker, czyste_slowa

# --- INTERFEJS STRONY ---
st.set_page_config(page_title="Polski Autokorektor N-gramowy", page_icon="✍️", layout="centered")

st.title("Hybrydowy Korektor Pisowni")
st.markdown("Aplikacja wykrywa podejrzane słowa i pozwala zobaczyć **top-3 najlepsze poprawki** wyliczone przez model kanału szumowego.")

# Ładowanie i kompilacja całego systemu przy pierwszym uruchomieniu
with st.spinner("Uruchamianie aplikacji..."):
    ranker, slownik_slow = zaladuj_i_wytrenuj_system()
st.success("Aplikacja uruchomiona!")

# Panele boczne do sterowania czułością i parametrami algorytmów
st.sidebar.header("Parametry algorytmu")
max_dist = st.sidebar.slider("Maksymalna odległość edycyjna (drzewo BK)", min_value=1, max_value=3, value=2)
smoothing_k = st.sidebar.slider("Współczynnik wygładzania Add-k", min_value=0.01, max_value=1.0, value=0.1, step=0.05)

# Formularz pobierający tekst od użytkownika
with st.form("formularz_wpisywania"):
    st.subheader("Wpisz tekst do analizy:")
    tekst_uzytkownika = st.text_area("Wpisz swoje zdanie tutaj:", value="czlowiek idxie do skleup")
    
    uruchom_analize = st.form_submit_button("Uruchom")

# Logika przetwarzania i oceny tekstu po kliknięciu przycisku
if uruchom_analize:
    if tekst_uzytkownika.strip():
        # Podział tekstu z zachowaniem znaków interpunkcyjnych do wizualizacji
        slowa_i_znaki = re.findall(r'\w+|[^\w\s]+', tekst_uzytkownika)
        
        st.write("### Wizualizacja tekstu:")
        st.caption("Słowa podkreślone na czerwono mogą zawierać błędy.")
        
        html_elements = []
        bledne_slowa = []
        
        # Iteracja po tokenach w celu wykrycia słów spoza słownika treningowego
        for i, token in enumerate(slowa_i_znaki):
            if token.isalnum() and not token.isdigit():
                token_lower = token.lower()
                # Jeśli słowa nie ma w słowniku, oznaczane jest jako błąd
                if token_lower not in slownik_slow:
                    html_elements.append(f"<span style='color: #ff4b4b; font-weight: bold; text-decoration: underline wavy;'>{token}</span>")
                    bledne_slowa.append((i, token))
                else:
                    html_elements.append(token)
            else:
                html_elements.append(token)
                
        st.markdown(f"<div style='font-size: 1.25rem; line-height: 1.8; background-color: #f0f2f6; padding: 15px; border-radius: 10px; color: black;'>{' '.join(html_elements)}</div>", unsafe_allow_html=True)
        
        # Generowanie propozycji poprawek dla znalezionych błędów
        if bledne_slowa:
            st.subheader("Sugestie top-3 poprawek dla wykrytych błędów:")
            
            for idx, (pozycja, slowo) in enumerate(bledne_slowa):
                st.write(f"Słowo: **{slowo}** (pozycja {pozycja+1})")
                
                # Pobieranie słowa poprzedzającego (kontekstu do bigramu)
                poprzednie = "<s>"
                aktywne_slowa_lewa = [w.lower() for w in slowa_i_znaki[:pozycja] if w.isalnum()]
                if aktywne_slowa_lewa:
                    poprzednie = aktywne_slowa_lewa[-1]
                
                # Pobieranie listy kandydatów spełniających warunek odległości edycyjnej
                kandydaci_z_drzewa = ranker.bk_tree.search(slowo.lower(), max_dist)
                
                wyniki_kandydatów = []
                
                # Punktacja każdego kandydata na podstawie kanału szumów i bigramu
                for kandydat, dystans in kandydaci_z_drzewa:
                    try:
                        # Próba wywołania wbudowanej metody score z rankera
                        score = ranker.score_candidate(poprzednie, kandydat, slowo.lower(), k=smoothing_k)
                    except AttributeError:
                        # Logika rezerwowa w przypadku braku metody score bezpośrednio w klasie rankera
                        bigram_count = ranker.bigram_counts.get((poprzednie, kandydat), 0)
                        bigram_count = ranker.bigram_counts.get((poprzednie, kandydat), 0)
                        unigram_count = ranker.unigram_counts.get(poprzednie, 0)
                        
                        p_lang = (bigram_count + smoothing_k) / (unigram_count + smoothing_k * ranker.vocab_size)
                        p_error = ranker._get_error_probability(kandydat, slowo.lower())
                        
                        import math
                        p_lang = max(p_lang, 1e-10)
                        p_error = max(p_error, 1e-10)
                        score = math.log(p_error) + math.log(p_lang)
                    
                    wyniki_kandydatów.append((kandydat, score))
                
                # Sortowanie uzyskanych wyników i wybór top 3 najlepszych kandydatów
                wyniki_kandydatów = sorted(wyniki_kandydatów, key=lambda x: x[1], reverse=True)[:3]
                
                # Prezentacja wyników top-3 w formie czytelnych kafelków 
                if wyniki_kandydatów:
                    cols = st.columns(3)
                    for c_idx, (kandydat, score) in enumerate(wyniki_kandydatów):
                        with cols[c_idx]:
                            st.metric(label=f"Top {c_idx+1}: {kandydat}", value=f"{score:.6f}")
                else:
                    st.warning("Brak pasujących słów w drzewie BK dla podanej odległości edycyjnej.")
                st.write("---")
        else:
            st.success("Nie wykryto żadnych błędów w tekście.")
    else:
        st.warning("Proszę wpisać jakieś zdanie przed uruchomieniem analizy!")