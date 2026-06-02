import math
from rapidfuzz.distance import DamerauLevenshtein

from generator import BKTree
from model_ngram import addk_probability

# Klasa odpowiedzialna za ocenianie i rankowanie poprawnych słów za pomocą kanału szumów (Noisy Channel Model)
class NoisyChannelRanker:
    def __init__(self, bk_tree: BKTree, bigram_counts, unigram_counts, vocab_size):
        self.bk_tree = bk_tree
        self.bigram_counts = bigram_counts
        self.unigram_counts = unigram_counts
        self.vocab_size = vocab_size

    # Oblicza prawdopodobieństwo błędu (prawdopodobieństwo transformacji kandydat -> error_word)
    def _get_error_probability(self, candidate: str, error_word: str) -> float:
        dist = int(DamerauLevenshtein.distance(candidate, error_word))

        if dist == 0:
            return 0.99
        return 0.1 / (dist * 10)

    # Oblicza ostateczny wynik (score) dla kandydata na podstawie modelu języka i modelu błędu
    def score_candidate(self, prev_word: str, candidate: str, error_word: str, k: float = 0.1) -> float:
        p_error = self._get_error_probability(candidate, error_word)
        
        ngram = (prev_word, candidate)
        p_lang = addk_probability(
            ngram = ngram,
            ngram_counts = self.bigram_counts,
            context_counts = self.unigram_counts,
            vocab_size = self.vocab_size,
            k = k
        ) 
        
        # Zabezpieczenie przed logarytmowaniem zera (wartości minimalne)
        p_error = max(p_error, 1e-10)
        p_lang = max(p_lang, 1e-10)
        
        # Zwracamy sumę logarytmów (odpowiednik mnożenia prawdopodobieństw)
        return math.log(p_error) + math.log(p_lang)

    # Główna metoda poprawiająca błędy w całym zdaniu
    def correct_sentence(self, sentence: str, max_dist: int = 2, k: float = 0.1) -> str:
        words = sentence.lower().strip().split()
        if not words:
            return ""
        
        # Inicjalizacja listy poprawionych słów ze znacznikiem początku zdania
        corrected_words = ["<s>"]

        for current_word in words:
            prev_word = corrected_words[-1]

            # Szukanie potencjalnych kandydatów w drzewie BK w zadanej odległości edycyjnej
            candidates_with_dist = self.bk_tree.search(current_word, max_dist)

            # Jeśli nie znaleziono żadnych bliskich słów, pozostawiamy słowo oryginalne
            if not candidates_with_dist:
                corrected_words.append(current_word)
                continue

            best_candidate = current_word
            max_score = -float('inf')

            # Wybór najlepszego kandydata na podstawie najwyższego wyniku score
            for candidate, _ in candidates_with_dist:
                score = self.score_candidate(prev_word, candidate, current_word, k=k)
                if score > max_score:
                    max_score = score
                    best_candidate = candidate

            corrected_words.append(best_candidate)

        # Składamy zdanie w całość, odrzucając początkowy token '<s>'
        return " ".join(corrected_words[1:])

