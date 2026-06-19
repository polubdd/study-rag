import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,4})\s+(.*)")


@dataclass
class Chunk:
    source: str
    section: str
    text: str
    id: int | None = None
    doc_id: int | None = None


def split_markdown(text: str, source: str, max_words: int = 140, min_words: int = 12) -> list[Chunk]:
    """Режет конспект по заголовкам, а длинные разделы — по абзацам.

    Резать по фиксированному числу символов плохо: определение разрывается
    посередине, и в выдачу попадает половина мысли. Слишком короткие куски
    (одна строка под заголовком) склеиваются со следующим абзацем того же раздела.
    """
    chunks: list[Chunk] = []
    path: list[str] = []
    buf: list[str] = []

    def flush():
        para = "\n".join(buf).strip()
        buf.clear()
        if not para:
            return
        section = " / ".join(path) or source
        parts, cur = [p.strip() for p in re.split(r"\n\s*\n", para) if p.strip()], []
        for p in parts:
            if cur and len(" ".join(cur + [p]).split()) > max_words and len(" ".join(cur).split()) >= min_words:
                chunks.append(Chunk(source, section, "\n\n".join(cur)))
                cur = []
            cur.append(p)
        if cur:
            chunks.append(Chunk(source, section, "\n\n".join(cur)))

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip().strip("#").strip()]
        else:
            buf.append(line)
    flush()
    return chunks
