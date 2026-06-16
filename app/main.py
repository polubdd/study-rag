import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag.ingest import UnsupportedFile
from rag.service import ROOT, Service

MAX_FILE = 20 * 1024 * 1024
app = FastAPI(title="study-rag", description="Вопросы по конспектам, карточки и квизы")
svc = Service(os.getenv("STUDY_DB", ROOT / "data" / "study.db"))


def _course(course_id: int) -> int:
    if not any(c["id"] == course_id for c in svc.store.courses()):
        raise HTTPException(404, "нет такого курса")
    return course_id


class CourseIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class AskIn(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    k: int = Field(4, ge=1, le=10)


class ReviewIn(BaseModel):
    grade: int = Field(ge=0, le=3)


class SynonymsIn(BaseModel):
    groups: list[list[str]]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/status")
def status():
    return svc.status()


# --- курсы

@app.get("/api/courses")
def courses():
    return svc.store.courses()


@app.post("/api/courses", status_code=201)
def create_course(body: CourseIn):
    if svc.store.course_by_name(body.name.strip()):
        raise HTTPException(409, "курс с таким названием уже есть")
    return {"id": svc.store.create_course(body.name)}


@app.delete("/api/courses/{course_id}", status_code=204)
def delete_course(course_id: int):
    svc.store.delete_course(_course(course_id))


@app.get("/api/courses/{course_id}/progress")
def progress(course_id: int):
    return svc.store.progress(_course(course_id))


@app.get("/api/courses/{course_id}/synonyms")
def get_synonyms(course_id: int):
    return svc.store.synonyms(_course(course_id))


@app.put("/api/courses/{course_id}/synonyms")
def put_synonyms(course_id: int, body: SynonymsIn):
    return svc.store.set_synonyms(_course(course_id), body.groups)


# --- документы

@app.get("/api/courses/{course_id}/documents")
def documents(course_id: int):
    return svc.store.documents(_course(course_id))


@app.post("/api/courses/{course_id}/documents")
async def upload(course_id: int, files: list[UploadFile] = File(...)):
    _course(course_id)
    results = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_FILE:
            results.append({"name": f.filename, "error": "файл больше 20 МБ"})
            continue
        try:
            r = svc.store.add_document(course_id, f.filename or "без имени.txt", data)
            results.append({"name": f.filename, **r})
        except (UnsupportedFile, ValueError) as e:
            results.append({"name": f.filename, "error": str(e)})
        except Exception:
            results.append({"name": f.filename, "error": "не удалось прочитать файл"})
    changed = any("id" in r and not r.get("duplicate") for r in results)
    cards = svc.generate_cards(course_id) if changed else None
    return {"files": results, "cards": cards}


@app.get("/api/documents/{doc_id}")
def document(doc_id: int):
    d = svc.store.document(doc_id)
    if not d:
        raise HTTPException(404)
    return d


@app.post("/api/documents/{doc_id}/reprocess")
def reprocess(doc_id: int):
    doc = svc.store.document(doc_id)
    if not doc:
        raise HTTPException(404)
    try:
        r = svc.store.reprocess(doc_id)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {**r, "cards": svc.generate_cards(doc["course_id"])}


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete_document(doc_id: int):
    try:
        svc.store.delete_document(doc_id)
    except KeyError:
        raise HTTPException(404)


# --- вопросы и поиск

@app.post("/api/courses/{course_id}/ask")
def ask(course_id: int, body: AskIn):
    return svc.ask(_course(course_id), body.question, body.k).to_dict()


@app.get("/api/courses/{course_id}/search")
def search(course_id: int, q: str, k: int = 8):
    return svc.search(_course(course_id), q, min(k, 20))


# --- карточки и квиз

@app.post("/api/courses/{course_id}/cards/generate")
def generate(course_id: int):
    return svc.generate_cards(_course(course_id))


@app.get("/api/courses/{course_id}/cards")
def cards(course_id: int, due: bool = False, limit: int = 50):
    return svc.store.cards(_course(course_id), due_only=due, limit=min(limit, 1000))


@app.post("/api/cards/{card_id}/review")
def review(card_id: int, body: ReviewIn):
    try:
        return svc.store.review(card_id, body.grade)
    except KeyError:
        raise HTTPException(404)


@app.post("/api/cards/{card_id}/suspend", status_code=204)
def suspend(card_id: int):
    try:
        svc.store.suspend_card(card_id)
    except KeyError:
        raise HTTPException(404)


@app.get("/api/courses/{course_id}/quiz")
def quiz(course_id: int, n: int = 10):
    return svc.quiz(_course(course_id), min(n, 30))
