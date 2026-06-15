"""Хранилище на SQLite: курсы, документы, фрагменты, карточки, история повторений.

SQLite, а не векторная БД: конспекты одного студента — это тысячи фрагментов,
индекс в памяти строится за доли секунды, а один файл легко бэкапить.
"""
import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .chunk import Chunk, split_markdown
from .ingest import to_markdown

SCHEMA = """
CREATE TABLE IF NOT EXISTS courses (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY, course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    name TEXT NOT NULL, sha1 TEXT NOT NULL, added TEXT NOT NULL, markdown TEXT NOT NULL,
    UNIQUE(course_id, sha1)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL, section TEXT NOT NULL, text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cards (
    id INTEGER PRIMARY KEY, course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
    kind TEXT NOT NULL, term TEXT, front TEXT NOT NULL, back TEXT NOT NULL,
    ease REAL NOT NULL DEFAULT 2.5, interval REAL NOT NULL DEFAULT 0, reps INTEGER NOT NULL DEFAULT 0,
    lapses INTEGER NOT NULL DEFAULT 0, due TEXT NOT NULL, suspended INTEGER NOT NULL DEFAULT 0,
    UNIQUE(course_id, front)
);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY, card_id INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    at TEXT NOT NULL, grade INTEGER NOT NULL, interval REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS ix_cards_due ON cards(course_id, due);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        # версия курса растёт при каждом изменении: по ней сбрасывается кэш поискового индекса
        self.versions: dict[int, int] = {}

    @contextmanager
    def tx(self):
        try:
            yield self.db
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def _bump(self, course_id: int):
        self.versions[course_id] = self.versions.get(course_id, 0) + 1

    # --- курсы

    def create_course(self, name: str) -> int:
        name = name.strip()
        if not name:
            raise ValueError("пустое название курса")
        with self.tx() as db:
            cur = db.execute("INSERT INTO courses(name, created) VALUES (?, ?)", (name, now()))
        return cur.lastrowid

    def course_by_name(self, name: str) -> int | None:
        r = self.db.execute("SELECT id FROM courses WHERE name = ?", (name,)).fetchone()
        return r["id"] if r else None

    def courses(self) -> list[dict]:
        rows = self.db.execute("""
            SELECT c.id, c.name, c.created,
                   (SELECT COUNT(*) FROM documents d WHERE d.course_id = c.id) AS documents,
                   (SELECT COUNT(*) FROM cards k WHERE k.course_id = c.id AND k.suspended = 0) AS cards,
                   (SELECT COUNT(*) FROM cards k WHERE k.course_id = c.id AND k.suspended = 0 AND k.due <= ?) AS due
            FROM courses c ORDER BY c.name""", (now(),)).fetchall()
        return [dict(r) for r in rows]

    def delete_course(self, course_id: int):
        with self.tx() as db:
            db.execute("DELETE FROM courses WHERE id = ?", (course_id,))
        self.versions.pop(course_id, None)

    # --- документы

    def add_document(self, course_id: int, name: str, data: bytes) -> dict:
        """Возвращает {id, chunks, duplicate}. Повторная загрузка того же файла ничего не ломает."""
        if not self.db.execute("SELECT 1 FROM courses WHERE id = ?", (course_id,)).fetchone():
            raise KeyError(course_id)
        sha1 = hashlib.sha1(data).hexdigest()
        dup = self.db.execute("SELECT id FROM documents WHERE course_id = ? AND sha1 = ?", (course_id, sha1)).fetchone()
        if dup:
            n = self.db.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (dup["id"],)).fetchone()[0]
            return {"id": dup["id"], "chunks": n, "duplicate": True}
        md = to_markdown(name, data)
        chunks = split_markdown(md, Path(name).stem)
        if not chunks:
            raise ValueError("в файле не нашлось текста (скан без текстового слоя?)")
        with self.tx() as db:
            doc_id = db.execute("INSERT INTO documents(course_id, name, sha1, added, markdown) VALUES (?,?,?,?,?)",
                                (course_id, name, sha1, now(), md)).lastrowid
            db.executemany("INSERT INTO chunks(doc_id, ord, section, text) VALUES (?,?,?,?)",
                           [(doc_id, i, c.section, c.text) for i, c in enumerate(chunks)])
        self._bump(course_id)
        return {"id": doc_id, "chunks": len(chunks), "duplicate": False}

    def documents(self, course_id: int) -> list[dict]:
        rows = self.db.execute("""
            SELECT d.id, d.name, d.added, COUNT(c.id) AS chunks
            FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id
            WHERE d.course_id = ? GROUP BY d.id ORDER BY d.added, d.id""", (course_id,)).fetchall()
        return [dict(r) for r in rows]

    def document(self, doc_id: int) -> dict | None:
        r = self.db.execute("SELECT id, course_id, name, added, markdown FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(r) if r else None

    def delete_document(self, doc_id: int):
        doc = self.document(doc_id)
        if not doc:
            raise KeyError(doc_id)
        with self.tx() as db:
            db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self._bump(doc["course_id"])

    def chunks(self, course_id: int) -> list[Chunk]:
        rows = self.db.execute("""
            SELECT c.id, c.doc_id, c.section, c.text, d.name
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            WHERE d.course_id = ? ORDER BY d.id, c.ord""", (course_id,)).fetchall()
        return [Chunk(Path(r["name"]).stem, r["section"], r["text"], r["id"], r["doc_id"]) for r in rows]

    def chunk(self, chunk_id: int) -> Chunk | None:
        r = self.db.execute("SELECT c.id, c.doc_id, c.section, c.text, d.name FROM chunks c JOIN documents d ON d.id = c.doc_id "
                            "WHERE c.id = ?", (chunk_id,)).fetchone()
        return Chunk(Path(r["name"]).stem, r["section"], r["text"], r["id"], r["doc_id"]) if r else None

    def version(self, course_id: int) -> int:
        return self.versions.get(course_id, 0)
