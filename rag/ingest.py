"""Файл любого поддерживаемого формата -> markdown с заголовками.

Конспекты бывают в .md (Obsidian), .txt, .docx (Word, Google Docs) и .pdf
(лекции преподавателя). Внутри всё приводится к markdown, потому что нарезка
опирается на заголовки: у docx они берутся из стилей, у txt и pdf — угадываются.
"""
import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

SUPPORTED = {".md", ".markdown", ".txt", ".pdf", ".docx"}

_NUMBERED = re.compile(r"^(\d+(?:\.\d+){0,2})\.?\s+([А-ЯЁA-Z].{1,70})$")
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class UnsupportedFile(ValueError):
    pass


def to_markdown(name: str, data: bytes) -> str:
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED:
        raise UnsupportedFile(f"формат {ext or '?'} не поддерживается, нужен один из: {', '.join(sorted(SUPPORTED))}")
    if ext in (".md", ".markdown"):
        return _decode(data)
    if ext == ".txt":
        return guess_headings(_decode(data))
    if ext == ".docx":
        return docx_to_markdown(data)
    return guess_headings(pdf_text(data))


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    pages = [p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages]
    text = "\n\n".join(pages)
    # pypdf рвёт строки посреди предложения; склеиваю строки, если следующая начинается со строчной
    text = re.sub(r"-\n(?=[а-яёa-z])", "", text)
    return re.sub(r"(?<=[^\n.:;!?])\n(?=[а-яёa-z])", " ", text)


def docx_to_markdown(data: bytes) -> str:
    """docx — это zip с XML внутри. Заголовки берутся из стилей Heading1/Заголовок1."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = ElementTree.fromstring(z.read("word/document.xml"))
    lines = []
    for p in root.iter(_W + "p"):
        text = "".join(t.text or "" for t in p.iter(_W + "t")).strip()
        if not text:
            lines.append("")
            continue
        style = p.find(f"{_W}pPr/{_W}pStyle")
        sid = (style.get(_W + "val") if style is not None else "") or ""
        m = re.search(r"(?:heading|заголовок|Heading)\s*(\d)", sid, re.I) or re.fullmatch(r"(\d)", sid)
        if m:
            lines.append("#" * min(int(m.group(1)), 4) + " " + text)
        elif p.find(f"{_W}pPr/{_W}numPr") is not None:
            lines.append("- " + text)
        else:
            lines.append(text)
        lines.append("")
    return "\n".join(lines)


def guess_headings(text: str) -> str:
    """Заголовки в простом тексте: «2.1 Ранг матрицы», «ТЕОРИЯ ВЕРОЯТНОСТЕЙ»,
    короткая строка без точки в конце, после которой идёт абзац."""
    lines = text.replace("\r\n", "\n").split("\n")
    out = []
    for i, raw in enumerate(lines):
        line = raw.strip()
        # следующая непустая строка в пределах двух: под заголовком часто бывает пустая строка
        nxt = next((x.strip() for x in lines[i + 1:i + 3] if x.strip()), "")
        prev = lines[i - 1].strip() if i else ""
        m = _NUMBERED.match(line)
        if line.startswith("#"):
            out.append(line)
        elif m and not line.endswith((".", ",", ";")):
            level = m.group(1).count(".") + 1
            out.append("#" * min(level + 1, 4) + " " + line)
        elif (line and len(line) <= 60 and line.isupper() and sum(c.isalpha() for c in line) >= 4):
            out.append("# " + line.capitalize())
        elif (line and len(line.split()) <= 7 and not prev and nxt and line[0].isupper()
              and not line.endswith((".", ",", ";", ":", "?", "!")) and not re.search(r"[=<>]", line)):
            out.append("## " + line)
        else:
            out.append(raw)
    return "\n".join(out)
