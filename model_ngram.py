import math
import os
import re
import urllib.request
import zipfile
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

#Tokenizacja tekstu: zamiana na małe litery oraz podział na tokeny
def tokenize_corpus(text: str) -> list[str]:
    return word_tokenize(text.lower())

#Zwraca zbiór słów, które wystąpiły co najmniej min_count razy
def build_vocabulary(tokens: list[str], min_count: int = 2) -> set[str]:
    counts = Counter(tokens)
    return {word for word, count in counts.items() if count >= min_count}

#Zastępuje tokeny spoza vocabulary tokenem '<UNK>'
def replace_rare_with_unk(tokens: list[str], vocab: set[str]) -> list[str]:
    return [token if token in vocab else '<UNK>' for token in tokens]

#Dodaje n-1 tokenów '<s>' na początku i jeden '<s>' na końcu każdego zdania
def add_sentence_boundaries(sentences: list[list[str]], n: int) -> list[list[str]]:
    result = []
    for sentence in sentences:
        bounded = ['<s>'] * (n - 1) + sentence + ['</s>']
        result.append(bounded)
    return result

#Zwraca listę n-gramów z listy tokenów
def extract_ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)- n + 1)]

#Zlicza n-gramy i (n-1)-gramy w całym zbiorze zdań
def count_ngrams(sentences: list[list[str]], n: int) -> tuple[Counter, Counter]:
    ngram_counts = Counter()
    context_counts = Counter()

    for sentence in sentences:
        ngrams = extract_ngrams(sentence, n)
        ngram_counts.update(ngrams)

        contexts = extract_ngrams(sentence, n-1)
        context_counts.update(contexts)
    return ngram_counts, context_counts

#Oblicza prawdopodobieństwo MLE dla n-gramu
def mle_probability(ngram: tuple, ngram_counts: Counter, context_counts: Counter) -> float:
    context = ngram[:-1]
    if context_counts[context] == 0:
        return 0.0
    return ngram_counts[ngram] / context_counts[context]

#Oblicza prawdopodobieństwo z wygładzaniem Laplace'a (add-1)
def laplace_probability(ngram: tuple, ngram_counts: Counter, context_counts: Counter, 
                        vocab_size: int) -> float:
    context = ngram[:-1]
    prob = (ngram_counts[ngram] +1) / (context_counts[context] + vocab_size)
    return math.log(prob)

#Oblicza prawdopodobieństwo z wygładzaniem Add-k
def addk_probability(ngram: tuple, ngram_counts: Counter, context_counts: Counter,
                     vocab_size: int, k: float = 1.0) -> float:
    context = ngram[:-1]
    prob =  (ngram_counts[ngram] + k) / (context_counts[context] + k * vocab_size)
    return math.log(prob)

