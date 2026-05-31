#pip install rapidfuzz
import math
from rapidfuzz.distance import DamerauLevenshtein

class BKNode:
    #Pojedynczy wezel w drzewie BK, ktory przechowuje jedno slowo
    def __init__(self, word: str):
        self.word = word
        self.children = {}

class BKTree:
    #Główne drzewo BK zarządzające słownikiem
    def __init__(self):
        self.root = None

    def insert(self, word: str):
        #Dodaje nowe słowo ze słownika do drzewa BK
        word = word.lower().strip()
        if not word:
            return
        
        if self.root is None:
            self.root = BKNode(word)
            return
        
        current = self.root
        while True:
            #odleglosc edycyjna miedzy nowym slowem a obecnym wyrazem
            dist = int(DamerauLevenshtein.distance(word, current.word))

            if dist == 0:
                return #słowo już istnieje w drzewie, pomijamy
            
            if dist in current.children:
                current = current.children[dist]

            else:
                current.children[dist] = BKNode(word)
                break
    
    def search(self, word: str, max_dist: int) -> list:
        #Przeszukuje drzewo i zwraca listę kandydatów, których odległość od błędnego
        #słowa jest mniejsza lub równa max_dist
        word = word.lower().strip()
        if self.root is None:
            return []
        
        results = []
        #Kolejka węzłów do sprawdzenia (zaczynamy od korzenia)
        nodes_to_visit = [self.root]

        while nodes_to_visit:
            current = nodes_to_visit.pop()

            #Liczymy odległość do obecnego węzła
            dist = int(DamerauLevenshtein.distance(word, current.word))

            #Jeśli słowo mieści się w progu błędu, dodajemy je do kandydatów
            if dist <= max_dist:
                results.append((current.word, dist))

            #Sprawdzamy tylko te gałęzie, które matematycznie mogą zawierać dobre słowa
            #Pozostałe gałęzie (miliony słów) całkowicie pomijamy
            min_bound = dist - max_dist
            max_bound = dist + max_dist

            for child_dist, child_node in current.children.items():
                if min_bound <= child_dist <= max_bound:
                    nodes_to_visit.append(child_node)

        return results
    