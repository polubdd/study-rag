"""Связка хранилища, поиска, ответов и карточек — то, что вызывает API."""
import threading
from pathlib import Path

from . import cards as cardgen
from . import llm
from .answer import answer
from .quiz import make_quiz
from .retrieval import HybridIndex
from .store import Store

ROOT = Path(__file__).parent.parent
DEMO_COURSE = "Демо: 1 курс"


class Service:
    def __init__(self, db_path: str | Path = ROOT / "data" / "study.db", seed_demo: bool = True):
        self.store = Store(db_path)
        self._indexes: dict[int, tuple[int, HybridIndex]] = {}
        self._lock = threading.Lock()
        if seed_demo and not self.store.courses():
            self.seed_demo()

    def seed_demo(self) -> int:
        cid = self.store.create_course(DEMO_COURSE)
        for f in sorted((ROOT / "notes").glob("*.md")):
            self.store.add_document(cid, f.name, f.read_bytes())
        self.generate_cards(cid, use_llm=False)
        return cid

    def index(self, course_id: int) -> HybridIndex:
        """Индекс курса строится лениво и пересобирается, только когда курс изменился."""
        v = self.store.version(course_id)
        with self._lock:
            cached = self._indexes.get(course_id)
            if cached and cached[0] == v:
                return cached[1]
            idx = HybridIndex(self.store.chunks(course_id), extra_synonyms=self.store.synonyms(course_id))
            self._indexes[course_id] = (v, idx)
            return idx

    def ask(self, course_id: int, question: str, k: int = 4):
        return answer(question, self.index(course_id), k)

    def search(self, course_id: int, query: str, k: int = 8):
        return [{"chunk_id": h.chunk.id, "source": h.chunk.source, "section": h.chunk.section, "text": h.chunk.text,
                 "score": h.score, "ranks": h.ranks} for h in self.index(course_id).search(query, k)]

    def generate_cards(self, course_id: int, use_llm: bool | None = None) -> dict:
        chunks = self.store.chunks(course_id)
        found = cardgen.from_chunks(chunks)
        if use_llm is None:
            use_llm = llm.enabled()
        llm_errors = 0
        if use_llm:
            for c in chunks:
                try:
                    found += cardgen.llm_questions(c)
                except Exception:
                    llm_errors += 1
        added = self.store.add_cards(course_id, found)
        return {"found": len(found), "added": added, "llm_errors": llm_errors}

    def quiz(self, course_id: int, n: int = 10):
        return make_quiz(self.store.cards(course_id), n)

    def status(self) -> dict:
        return {"llm": llm.enabled(), "embeddings": llm.embeddings_enabled()}
