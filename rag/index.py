from .bm25 import BM25
from .chunk import Chunk


class Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        # заголовок раздела добавляю к тексту: часто термин есть только в нём
        self.bm25 = BM25([f"{c.section}\n{c.text}" for c in chunks])

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        return [(self.chunks[i], s) for i, s in self.bm25.top(query, k)]
