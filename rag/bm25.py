import math
import re
from collections import Counter

_WORD = re.compile(r"\w+")


def stem(word: str) -> str:
    """Очень грубый стеммер: отрезает частые русские окончания.

    Без него "матрицы", "матрицей" и "матрица" — три разных слова,
    и поиск по конспекту почти не работает. Полноценный стеммер (pymorphy)
    лучше, но для поиска по конспектам хватает и этого.
    """
    for suf in ("ами", "ями", "ого", "его", "ому", "ему", "ыми", "ими", "ость", "ости",
                "ой", "ей", "ий", "ый", "ая", "яя", "ую", "юю", "ое", "ее", "ые", "ие", "ов", "ев",
                "ым", "им",
                "ах", "ях", "ам", "ям", "ом", "ем", "ы", "и", "а", "я", "у", "ю", "е", "о"):
        if len(word) - len(suf) >= 4 and word.endswith(suf):
            return word[: -len(suf)]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in _WORD.findall(text.lower().replace("ё", "е"))]


class BM25:
    """BM25 своими руками — чтобы было видно, откуда берётся скор.

    score(q, d) = sum_t idf(t) * tf(t,d)*(k1+1) / (tf(t,d) + k1*(1 - b + b*|d|/avgdl))
    """

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for d, dl in zip(self.docs, self.lens):
            s = 0.0
            for t in q:
                tf = d.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out

    def top(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        s = self.scores(query)
        ranked = sorted(range(len(s)), key=lambda i: -s[i])
        return [(i, s[i]) for i in ranked[:k] if s[i] > 0]
