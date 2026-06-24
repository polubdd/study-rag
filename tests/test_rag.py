import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.answer import NOT_FOUND, answer, check_citations
from rag.cards import from_chunk, llm_questions
from rag.chunk import Chunk, split_markdown
from rag.ingest import UnsupportedFile, docx_to_markdown, guess_headings, to_markdown
from rag.quiz import make_quiz, mask
from rag.retrieval import HybridIndex
from rag.srs import AGAIN, EASY, GOOD, State, schedule
from rag.store import Store
from rag.synonyms import build_map, expand
from rag.text import sentences, tokenize

NOTES = Path(__file__).parent.parent / "notes"


@pytest.fixture(scope="module")
def course():
    s = Store()
    cid = s.create_course("демо")
    for f in sorted(NOTES.glob("*.md")):
        s.add_document(cid, f.name, f.read_bytes())
    return s, cid


@pytest.fixture(scope="module")
def index(course):
    s, cid = course
    return HybridIndex(s.chunks(cid), use_embeddings=False)


# --- текст и загрузка

def test_stemming_merges_word_forms():
    assert tokenize("матрицы")[0] == tokenize("матрицей")[0] == tokenize("матрицу")[0]
    assert tokenize("обратную")[0] == tokenize("обратная")[0]
    assert tokenize("ранг") != tokenize("рано")


def test_sentences_keep_lists_apart():
    assert sentences("Первое. Второе!\n\n- пункт\n- ещё") == ["Первое.", "Второе!", "- пункт", "- ещё"]


def test_split_keeps_heading_path():
    (c,) = split_markdown("# Линал\n## Матрицы\n### Ранг\nРанг — это число строк в ступенчатом виде.\n", "x")
    assert c.section == "Линал / Матрицы / Ранг"


def test_long_section_is_split_by_paragraphs():
    md = "# A\n" + "\n\n".join(["слово " * 60] * 4)
    assert len(split_markdown(md, "x", max_words=140)) == 2


def test_guess_headings_in_plain_text():
    md = guess_headings("ЛИНЕЙНАЯ АЛГЕБРА\n\n1.2 Ранг матрицы\nРанг — это число.\n\nОпределитель\n\nОпределитель равен нулю.")
    assert "# Линейная алгебра" in md and "### 1.2 Ранг матрицы" in md and "## Определитель" in md


def _docx(paragraphs):
    w = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    body = ""
    for style, text in paragraphs:
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        body += f"<w:p>{ppr}<w:r><w:t>{text}</w:t></w:r></w:p>"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", f"<w:document {w}><w:body>{body}</w:body></w:document>")
    return buf.getvalue()


def test_docx_headings_from_styles():
    md = docx_to_markdown(_docx([("Heading1", "Матан"), ("Heading2", "Пределы"), ("", "Предел — это число.")]))
    assert "# Матан" in md and "## Пределы" in md and "Предел — это число." in md


def test_unsupported_format():
    with pytest.raises(UnsupportedFile):
        to_markdown("photo.jpg", b"123")


def test_cp1251_text():
    assert "Ранг" in to_markdown("old.txt", "Ранг матрицы\n\nРанг — число.".encode("cp1251"))


# --- хранилище

def test_duplicate_upload_is_detected(course):
    s, cid = course
    r = s.add_document(cid, "linal-copy.md", (NOTES / "linal.md").read_bytes())
    assert r["duplicate"]


def test_delete_document_removes_chunks_and_cards():
    s = Store()
    cid = s.create_course("x")
    d = s.add_document(cid, "a.md", "# A\n\nРанг — максимальное число линейно независимых строк матрицы.".encode())
    s.add_cards(cid, [c for ch in s.chunks(cid) for c in from_chunk(ch)])
    assert s.cards(cid)
    v = s.version(cid)
    s.delete_document(d["id"])
    assert not s.chunks(cid) and not s.cards(cid) and s.version(cid) > v


# --- поиск и ответы

@pytest.mark.parametrize("q,section", [
    ("как найти обратную матрицу", "Обратная матрица"),
    ("почему нельзя KFold на временных рядах", "Кросс-валидация"),
    ("что такое полнота", "Метрики классификации"),  # только через синонимы: в конспекте «Recall»
    ("неопределённость ноль на ноль", "Правило Лопиталя"),
])
def test_search(index, q, section):
    assert section in index.search(q, 1)[0].chunk.section


def test_synonym_expansion():
    assert "recall" in expand("что такое полнота", build_map())
    assert "среднее" in expand("матожидание", build_map([["матожидание", "среднее"]]))


def test_extractive_answer_cites_sources(index):
    a = answer("что такое ранг матрицы", index, use_llm=False)
    assert a.mode == "extractive" and a.text.startswith("Ранг — максимальное число") and "[1]" in a.text


def test_extractive_covers_whole_question(index):
    a = answer("чем PCA связан с собственными векторами", index, use_llm=False)
    assert "PCA" in a.text


def test_off_topic_question(index):
    a = answer("как приготовить борщ", index, use_llm=False)
    assert a.text == NOT_FOUND and a.mode == "none"


def test_llm_answer_with_valid_citations(index):
    a = answer("что такое ранг", index, call=lambda p: "Ранг — число независимых строк [1].")
    assert a.mode == "llm"


def test_llm_answer_without_citations_is_replaced(index):
    a = answer("что такое ранг", index, call=lambda p: "Ранг — это размерность пространства.")
    assert a.mode == "extractive" and any("нет ни одной ссылки" in w for w in a.warnings)


def test_llm_bad_citation_and_failure(index):
    a = answer("что такое ранг", index, call=lambda p: "Ранг — это [9].")
    assert a.mode == "extractive"

    def boom(p):
        raise TimeoutError
    assert answer("что такое ранг", index, call=boom).mode == "extractive"
    assert check_citations(NOT_FOUND, 3) == []


# --- карточки, повторение, квиз

def test_cards_from_chunk():
    ch = Chunk("x", "Т / Формула Байеса", "P(A|B) = P(B|A) · P(A) / P(B).\n\nКлассический пример: тест на редкую болезнь.")
    cards = from_chunk(ch)
    assert [c.kind for c in cards] == ["formula"]  # «Классический пример» — не понятие
    ch = Chunk("x", "Т / Пределы", "Закон больших чисел: среднее выборки сходится к ожиданию. На этом держится PCA: матрица симметрична.")
    assert [c.front for c in from_chunk(ch)] == ["Сформулируйте: закон больших чисел"]


def test_llm_cards_parsing():
    ch = Chunk("x", "ML / Кросс-валидация", "текст", id=1)
    cards = llm_questions(ch, call=lambda p: 'Вот: [{"q": "Почему нельзя KFold на рядах?", "a": "Учится на будущем"}]')
    assert cards[0].front == "Почему нельзя KFold на рядах?" and cards[0].kind == "question"
    assert llm_questions(ch, call=lambda p: "не json") == []


def test_sm2_intervals_grow_and_reset():
    t = datetime(2026, 10, 1, tzinfo=timezone.utc)
    st, due = schedule(State(), GOOD, t)
    assert st.interval == 1
    st, _ = schedule(st, GOOD, t)
    assert st.interval == 3
    st, _ = schedule(st, GOOD, t)
    assert st.interval == pytest.approx(7.5)
    easy, _ = schedule(st, EASY, t)
    assert easy.interval > 10 and easy.ease > st.ease
    lost, due = schedule(st, AGAIN, t)
    assert lost.reps == 0 and lost.lapses == 1 and (due - t).seconds == 600 and lost.ease < st.ease


def test_regenerating_cards_keeps_progress(course):
    s, cid = course
    s.add_cards(cid, [c for ch in s.chunks(cid) for c in from_chunk(ch)])
    card = s.cards(cid)[0]
    s.review(card["id"], GOOD)
    assert s.add_cards(cid, [c for ch in s.chunks(cid) for c in from_chunk(ch)]) == 0
    assert s.card(card["id"])["reps"] == 1


def test_quiz_options(course):
    s, cid = course
    q = make_quiz(s.cards(cid), 5, seed=0)
    assert q and all(len(x["options"]) == 4 and len(set(x["options"])) == 4 and "_____" in x["prompt"] for x in q)
    assert mask("Ранг — число. Ранга нет.", "Ранг") == "_____ — число. _____ нет."


# --- API

@pytest.fixture
def client(tmp_path, monkeypatch):
    import app.main as m
    from rag.service import Service
    monkeypatch.setattr(m, "svc", Service(tmp_path / "t.db"))
    return TestClient(m.app)


def test_api_flow(client):
    courses = client.get("/api/courses").json()
    assert courses[0]["name"].startswith("Демо") and courses[0]["cards"] > 10
    cid = client.post("/api/courses", json={"name": "Физика"}).json()["id"]
    assert client.post("/api/courses", json={"name": "Физика"}).status_code == 409

    files = [("files", ("mech.md", "# Механика\n\nИмпульс — произведение массы тела на его скорость.".encode(), "text/markdown")),
             ("files", ("img.png", b"\x89PNG", "image/png"))]
    up = client.post(f"/api/courses/{cid}/documents", files=files).json()
    assert up["files"][0]["chunks"] == 1 and "error" in up["files"][1] and up["cards"]["added"] == 1

    a = client.post(f"/api/courses/{cid}/ask", json={"question": "что такое импульс"}).json()
    assert "массы" in a["text"]

    card = client.get(f"/api/courses/{cid}/cards", params={"due": True}).json()[0]
    assert client.post(f"/api/cards/{card['id']}/review", json={"grade": 2}).json()["reps"] == 1
    assert client.post(f"/api/cards/{card['id']}/review", json={"grade": 7}).status_code == 422
    assert client.get(f"/api/courses/{cid}/progress").json()["due"] == 0

    assert client.put(f"/api/courses/{cid}/synonyms", json={"groups": [["импульс", "momentum"], ["одно"]]}).json() == [["импульс", "momentum"]]
    assert "массы" in client.post(f"/api/courses/{cid}/ask", json={"question": "momentum"}).json()["text"]

    doc = client.get(f"/api/courses/{cid}/documents").json()[0]
    assert client.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert client.get(f"/api/courses/{cid}/cards").json() == []
    assert client.get("/api/courses/999/documents").status_code == 404
