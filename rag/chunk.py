import re
from dataclasses import dataclass
from pathlib import Path

_HEADING = re.compile(r"^(#{1,3})\s+(.*)")


@dataclass
class Chunk:
    source: str
    section: str
    text: str


def split_markdown(text: str, source: str, max_words: int = 120) -> list[Chunk]:
    """Режет конспект по заголовкам, а длинные разделы — по абзацам.

    Резать по фиксированному числу символов плохо: определение разрывается
    посередине, и в выдачу попадает половина мысли.
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
        words, part = para.split("\n\n"), []
        for p in words:
            if part and len(" ".join(part + [p]).split()) > max_words:
                chunks.append(Chunk(source, section, "\n\n".join(part)))
                part = []
            part.append(p)
        if part:
            chunks.append(Chunk(source, section, "\n\n".join(part)))

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    return chunks


def load_dir(folder: str | Path) -> list[Chunk]:
    out = []
    for f in sorted(Path(folder).glob("**/*.md")):
        out += split_markdown(f.read_text(encoding="utf-8"), f.stem)
    return out
