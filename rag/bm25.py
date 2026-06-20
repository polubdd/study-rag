import math
from collections import Counter

from .text import tokenize


class BM25:
    """BM25 своими руками — чтобы было видно, откуда берётся скор.

    score(q, d) = Σ_t idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))
    """

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lens) / max(len(self.lens), 1) or 1
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = set(tokenize(query))
        out = []
        for d, dl in zip(self.docs, self.lens):
            s = 0.0
            for t in q:
                tf = d.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out

    def coverage(self, query: str) -> float:
        """Какая доля значимых слов запроса вообще встречается в корпусе."""
        q = set(tokenize(query))
        return sum(t in self.idf for t in q) / len(q) if q else 0.0
