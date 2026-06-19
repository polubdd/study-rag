"""Гибридный поиск по фрагментам конспекта.

Три ранжирования, объединённые через Reciprocal Rank Fusion:
- BM25 по стемам — точно находит термины («ранг», «ЦПТ»);
- TF-IDF по символьным n-граммам — переживает опечатки и словоформы,
  которые грубый стеммер не свёл («обращение» / «обратная»);
- эмбеддинги (если подключены) — находят синонимы и перефразировки.

RRF складывает не скоры, а ранги: 1/(k + rank). Скоры у трёх методов
в разных шкалах, а ранги сравнимы, и ничего не нужно нормировать.

Отдельно решается вопрос «в конспектах этого вообще нет». Поиск всегда
что-то вернёт, поэтому без проверки студент получит уверенный ответ не по
теме. Проверка: доля слов запроса, которые есть в корпусе, и абсолютный
скор BM25 лучшего фрагмента.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import llm
from .bm25 import BM25
from .chunk import Chunk
from .synonyms import build_map, expand
from .text import normalize, tokenize


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict


class HybridIndex:
    def __init__(self, chunks: list[Chunk], rrf_k: int = 60, use_embeddings: bool | None = None,
                 methods: tuple[str, ...] = ("bm25", "chars", "dense"), synonyms: bool = True,
                 extra_synonyms: list[list[str]] | None = None):
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.methods = methods
        self.synonyms = build_map(extra_synonyms) if synonyms else {}
        # заголовок раздела добавляю к тексту: часто термин есть только в нём
        texts = [f"{c.section}\n{c.text}" for c in chunks]
        self.bm25 = BM25(texts)
        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), preprocessor=normalize, sublinear_tf=True)
        self.X = self.tfidf.fit_transform(texts) if texts else None
        if use_embeddings is None:
            use_embeddings = llm.embeddings_enabled()
        self.E = None
        if use_embeddings and texts and "dense" in methods:
            E = np.array(llm.embed(texts), dtype=np.float32)
            self.E = E / np.linalg.norm(E, axis=1, keepdims=True)

    def _rankings(self, query: str) -> dict[str, np.ndarray]:
        dense_query = query  # эмбеддинги синонимы и так понимают, им исходный вопрос
        query = expand(query, self.synonyms)
        out = {}
        if "bm25" in self.methods:
            out["bm25"] = np.array(self.bm25.scores(query))
        if "chars" in self.methods:
            out["chars"] = cosine_similarity(self.tfidf.transform([query]), self.X)[0]
        if self.E is not None:
            q = np.array(llm.embed([dense_query])[0], dtype=np.float32)
            out["dense"] = self.E @ (q / np.linalg.norm(q))
        return out

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if not self.chunks or not query.strip():
            return []
        scores = self._rankings(query)
        fused = np.zeros(len(self.chunks))
        ranks = {}
        for name, s in scores.items():
            order = np.argsort(-s, kind="stable")
            r = np.empty(len(s), dtype=int)
            r[order] = np.arange(len(s))
            ranks[name] = r
            fused += 1 / (self.rrf_k + r + 1)
        top = np.argsort(-fused, kind="stable")[:k]
        return [Hit(self.chunks[i], round(float(fused[i]), 5), {n: int(r[i]) + 1 for n, r in ranks.items()}) for i in top]

    def _coverage(self, query: str) -> float:
        """Доля слов вопроса, которые есть в конспектах — сами или через синоним.

        Считать покрытие по расширенному запросу нельзя: «momentum» + синоним «импульс»
        дают 1 из 2, хотя вопрос полностью про то, что есть в конспекте.
        """
        toks = set(tokenize(query))
        if not toks:
            return 0.0
        covered = 0
        for t in toks:
            alts = {t} | {s for syn in self.synonyms.get(t, ()) for s in tokenize(syn)}
            covered += any(a in self.bm25.idf for a in alts)
        return covered / len(toks)

    def is_answerable(self, query: str, hits: list[Hit]) -> bool:
        """Есть ли в конспектах хоть что-то по вопросу."""
        if not hits:
            return False
        expanded = expand(query, self.synonyms)
        best_bm25 = max(self.bm25.scores(expanded))
        best_chars = float(cosine_similarity(self.tfidf.transform([expanded]), self.X)[0].max())
        cov = self._coverage(query)
        # в маленьком курсе idf низкий у всех слов, поэтому высокое покрытие запроса — достаточный сигнал
        return cov >= 0.34 and (best_bm25 >= 2.0 or best_chars >= 0.25 or cov >= 0.67)
