"""Хранилище на SQLite: курсы, документы, фрагменты, карточки, история повторений.

SQLite, а не векторная БД: конспекты одного студента — это тысячи фрагментов,
индекс в памяти строится за доли секунды, а один файл легко бэкапить.
"""
import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
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
    # даты храню строками ISO в UTC: так их можно сравнивать прямо в SQL как строки
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def week_ago() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(timespec="seconds")


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

    # --- карточки

    def add_cards(self, course_id: int, cards) -> int:
        """Новые карточки сразу доступны к повторению. Уже существующие (тот же вопрос) не трогаются:
        иначе повторная генерация сбросила бы прогресс."""
        with self.tx() as db:
            before = db.total_changes
            db.executemany(
                "INSERT OR IGNORE INTO cards(course_id, chunk_id, kind, term, front, back, due) VALUES (?,?,?,?,?,?,?)",
                [(course_id, c.chunk_id, c.kind, c.term, c.front, c.back, now()) for c in cards])
            return db.total_changes - before

    def cards(self, course_id: int, due_only: bool = False, limit: int = 1000) -> list[dict]:
        q = "SELECT * FROM cards WHERE course_id = ? AND suspended = 0"
        args: list = [course_id]
        if due_only:
            q += " AND due <= ?"
            args.append(now())
        q += " ORDER BY due, id LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.db.execute(q, args).fetchall()]

    def card(self, card_id: int) -> dict | None:
        r = self.db.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
        return dict(r) if r else None

    def review(self, card_id: int, grade: int, at=None) -> dict:
        from .srs import State, schedule
        c = self.card(card_id)
        if not c:
            raise KeyError(card_id)
        st, due = schedule(State(c["ease"], c["interval"], c["reps"], c["lapses"]), grade, at)
        with self.tx() as db:
            db.execute("UPDATE cards SET ease=?, interval=?, reps=?, lapses=?, due=? WHERE id=?",
                       (st.ease, st.interval, st.reps, st.lapses, due.isoformat(timespec="seconds"), card_id))
            db.execute("INSERT INTO reviews(card_id, at, grade, interval) VALUES (?,?,?,?)",
                       (card_id, (at.isoformat(timespec="seconds") if at else now()), grade, st.interval))
        return self.card(card_id)

    def suspend_card(self, card_id: int):
        with self.tx() as db:
            if not db.execute("UPDATE cards SET suspended = 1 WHERE id = ?", (card_id,)).rowcount:
                raise KeyError(card_id)

    def progress(self, course_id: int) -> dict:
        r = self.db.execute("""
            SELECT COUNT(*) AS total,
                   SUM(reps = 0) AS new,
                   SUM(reps > 0 AND interval < 21) AS learning,
                   SUM(interval >= 21) AS mature,
                   SUM(due <= ?) AS due
            FROM cards WHERE course_id = ? AND suspended = 0""", (now(), course_id)).fetchone()
        last = self.db.execute("""
            SELECT COUNT(*) AS n, AVG(grade > 0) AS ok FROM reviews r JOIN cards c ON c.id = r.card_id
            WHERE c.course_id = ? AND r.at >= ?""", (course_id, week_ago())).fetchone()
        return {k: int(r[k] or 0) for k in ("total", "new", "learning", "mature", "due")} | {
            "reviews_7d": int(last["n"] or 0), "recall_7d": round(float(last["ok"]), 2) if last["ok"] is not None else None}
