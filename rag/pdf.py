"""PDF -> markdown с формулами, по позициям и шрифтам символов (PyMuPDF).

Простое извлечение текста ломает конспекты в двух местах:
- слова слипаются («называетсяманхэттенской»), и поиск их не находит;
- формулы рассыпаются: знак суммы из математического шрифта CMEX
  превращается в букву «P», индексы и степени съезжают в строку
  («(ξi −ηi)2»), пределы суммы уходят на отдельные строки.

Здесь строки собираются заново по геометрии:
1. каждая строка текста — это кластер фрагментов с общей базовой линией;
2. фрагменты мелким кеглем выше базовой линии — верхний индекс ^{…},
   ниже — нижний _{…}; два мелких фрагмента друг над другом — дробь;
3. символы шрифтов TeX (CMEX, MSBM, CMSY) переводятся в Юникод;
4. крупный кегль — заголовки, номера страниц и колонтитулы выбрасываются,
   переносы слов склеиваются.
"""
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

# cmex10: большие операторы и скобки. Куски составных скобок (матрицы) выбрасываю
CMEX = {
    "P": "∑", "X": "∑", "Q": "∏", "Y": "∏", "R": "∫", "Z": "∫", "H": "∮", "I": "∮",
    "S": "⋃", "[": "⋃", "T": "⋂", "\\": "⋂", "L": "⊕", "M": "⊕", "N": "⊗", "O": "⊗",
    "p": "√", "q": "√", "r": "√", "s": "√", "t": "√",
    "\x00": "(", "\x01": ")", "\x02": "[", "\x03": "]", "\x08": "{", "\x09": "}", "\x0a": "⟨", "\x0b": "⟩",
    "\x0c": "|", "\x0d": "‖", "\x10": "(", "\x11": ")", "\x12": "(", "\x13": ")", "\x14": "[", "\x15": "]",
    "\x1a": "{", "\x1b": "}", "\x1c": "⟨", "\x1d": "⟩", " ": "(", "!": ")", "\"": "[", "#": "]",
    "(": "{", ")": "}", "*": "⟨", "+": "⟩", "D": "⟨", "E": "⟩", "n": "{", "o": "}",
    "": "(", "": ")", "": "[", "": "]", "": "{", "": "}",
}
# вертикальные куски составного корня («u», «v», «w» в cmex) и скобок
CMEX_DROP = {"u", "v", "w", "", "", "", "", "", "", "", "", "", "", "",
             "", "", "", ""}
MSBM = {"R": "ℝ", "C": "ℂ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "K": "𝕂", "k": "𝕜", "P": "ℙ", "F": "𝔽", "E": "𝔼"}
SCRIPT = {"B": "ℬ", "E": "ℰ", "F": "ℱ", "H": "ℋ", "I": "ℐ", "L": "ℒ", "M": "ℳ", "R": "ℛ"}


def _script(c: str) -> str:
    """CMSY: заглавные латинские — рукописные (𝒫, 𝒜)."""
    if c in SCRIPT:
        return SCRIPT[c]
    if "A" <= c <= "Z":
        return chr(0x1D49C + ord(c) - ord("A"))
    return c


def map_font_text(text: str, font: str) -> str:
    f = font.upper()
    if f.startswith("CMEX"):
        return "".join("" if c in CMEX_DROP else CMEX.get(c, c) for c in text)
    if f.startswith("MSBM"):
        return "".join(MSBM.get(c, c) for c in text)
    if f.startswith("CMSY"):
        return "".join(_script(c) for c in text)
    return text


def fix_negations(text: str) -> str:
    """В PDF перечёркивание идёт ДО символа («̸ =»), а Юникод ждёт его после: «≠»."""
    text = re.sub("̸\\s*(\\S)", lambda m: m.group(1) + "̸", text)
    return unicodedata.normalize("NFC", text)


@dataclass
class Span:
    text: str
    x0: float
    x1: float
    y0: float
    y1: float
    base: float  # базовая линия
    size: float
    bold: bool
    big_op: bool = False

    @property
    def yc(self):
        return (self.y0 + self.y1) / 2


@dataclass
class Row:
    base: float
    size: float
    items: list = field(default_factory=list)  # (x0, kind, span) kind: main | sup | sub
    x0: float = 1e9
    x1: float = 0.0

    def add(self, kind: str, s: Span):
        self.items.append((s.x0, kind, s))
        self.x0, self.x1 = min(self.x0, s.x0), max(self.x1, s.x1)

    def main_spans(self):
        return [s for _, k, s in self.items if k == "main"]


def _spans(page) -> list[Span]:
    out = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                raw = s["text"]
                if not raw.strip():
                    continue
                font = s["font"]
                text = map_font_text(raw, font)
                if not text.strip():
                    continue
                x0, y0, x1, y1 = s["bbox"]
                bold = bool(s["flags"] & 16) or "BX" in font.upper() or "BOLD" in font.upper()
                out.append(Span(text, x0, x1, y0, y1, s["origin"][1], s["size"], bold,
                                big_op=font.upper().startswith("CMEX") and text.strip() in "∑∏∫∮⋃⋂⊕⊗"))
    return out


def _rows(spans: list[Span], body: float) -> list[Row]:
    small = [s for s in spans if s.size < 0.85 * body]
    normal = sorted((s for s in spans if s.size >= 0.85 * body), key=lambda s: s.base)
    rows: list[Row] = []
    for s in normal:
        # большие операторы стоят на своей базовой линии, поэтому их привязываю по центру
        key = s.yc + 0.3 * body if s.big_op else s.base
        row = next((r for r in rows if abs(r.base - key) < 0.4 * max(body, r.size)), None)
        if row is None:
            row = Row(s.base if not s.big_op else key, s.size)
            rows.append(row)
        row.add("main", s)
        if not s.big_op:
            row.size = max(row.size, s.size) if row.items else s.size
    for s in small:
        best = min(rows, key=lambda r: abs(r.base - s.base), default=None)
        if best is None or abs(best.base - s.base) > 1.3 * body:
            r = Row(s.base, s.size)
            r.add("main", s)
            rows.append(r)
            continue
        d = s.base - best.base
        kind = "sup" if d < -0.12 * body else "sub" if d > 0.08 * body else "main"
        best.add(kind, s)
    return sorted(rows, key=lambda r: r.base)


def _row_text(row: Row) -> str:
    items = sorted(row.items, key=lambda t: (t[0], t[2].base))
    # мелкие фрагменты, стоящие друг над другом в одной позиции, — дробь (показатель 1/p)
    merged: list[tuple[str, Span]] = []
    for x0, kind, s in items:
        if merged and kind in ("sup", "sub") and merged[-1][0] in ("sup", "sub"):
            pk, ps = merged[-1]
            overlap = min(ps.x1, s.x1) - max(ps.x0, s.x0)
            if overlap > 0.5 * min(ps.x1 - ps.x0, s.x1 - s.x0) and pk == kind:
                top, bottom = sorted([ps, s], key=lambda z: z.base)
                frac = Span(f"{top.text.strip()}/{bottom.text.strip()}", min(ps.x0, s.x0), max(ps.x1, s.x1),
                            top.y0, bottom.y1, ps.base, ps.size, False)
                merged[-1] = (kind, frac)
                continue
        merged.append((kind, s))

    parts: list[str] = []
    prev: Span | None = None
    prev_kind = None
    for kind, s in merged:
        t = s.text if kind == "main" else s.text.strip()
        if kind in ("sup", "sub"):
            mark = "^" if kind == "sup" else "_"
            if prev_kind == kind and parts and parts[-1].endswith("}"):
                parts[-1] = parts[-1][:-1] + t + "}"  # продолжение того же индекса
            else:
                parts.append(f"{mark}{{{t}}}")
        else:
            if prev is not None and parts and not parts[-1].endswith(" ") and not t.startswith(" "):
                gap = s.x0 - prev.x1
                if gap > 0.18 * s.size:
                    parts.append(" ")
            parts.append(t)
        prev, prev_kind = s, kind
    return re.sub(r"[ \t]+", " ", "".join(parts)).strip()


def pdf_to_markdown(data: bytes) -> str:
    import pymupdf

    doc = pymupdf.open(stream=data, filetype="pdf")
    pages = [_spans(p) for p in doc]
    sizes = Counter(round(s.size, 1) for page in pages for s in page for _ in s.text)
    if not sizes:
        return ""
    body = sizes.most_common(1)[0][0]
    heading_sizes = sorted({round(s.size) for page in pages for s in page if s.size >= 1.15 * body}, reverse=True)
    level = {sz: min(i + 1, 3) for i, sz in enumerate(heading_sizes)}

    page_rows = []
    for p, spans in zip(doc, pages):
        h = p.rect.height
        rows = []
        for r in _rows(spans, body):
            text = _row_text(r)
            if not text:
                continue
            # номер страницы вверху/внизу листа
            if re.fullmatch(r"\d{1,4}", text) and (r.base < 0.1 * h or r.base > 0.88 * h):
                continue
            rows.append((r, text))
        page_rows.append(rows)

    # колонтитулы: верхняя строка капсом («2 ГЛАВА 24. ЛИНЕЙНОЕ ПРОСТРАНСТВО») или строка,
    # которая без цифр повторяется на трети страниц и больше
    def skeleton(t):
        return re.sub(r"[\d\s.]+", " ", t).strip()

    counts = Counter(skeleton(t) for rows in page_rows for _, t in rows)
    many = max(3, len(page_rows) // 3)

    def is_running(r, t, page_h):
        letters = [c for c in t if c.isalpha()]
        top = r.base < 0.09 * page_h
        caps = letters and sum(c.isupper() for c in letters) / len(letters) > 0.8
        return (top and caps) or (len(page_rows) >= 3 and counts[skeleton(t)] >= many and len(t) < 80)

    page_rows = [[(r, t) for r, t in rows if not is_running(r, t, p.rect.height)] for rows, p in zip(page_rows, doc)]

    out: list[str] = []
    para: list[str] = []

    def flush():
        if para:
            out.append(" ".join(para))
            out.append("")
            para.clear()

    pending_heading: tuple[int, list[str]] | None = None
    for rows in page_rows:
        prev_row = None
        for r, text in rows:
            mains = sorted(r.main_spans(), key=lambda s: s.x0)
            main_size = statistics.median([s.size for s in mains]) if mains else body
            if main_size >= 1.15 * body and len(text) <= 2:
                continue  # декоративная буквица «α» на титуле лекции
            if main_size >= 1.15 * body and len(text) < 120:
                lvl = level.get(round(main_size), 3)
                flush()
                if pending_heading and pending_heading[0] == lvl:
                    # «24.1» + «Метрическое пространство» — один заголовок; «про-» + «странствах» — одно слово
                    if pending_heading[1][-1].endswith("-"):
                        pending_heading[1][-1] = pending_heading[1][-1][:-1] + text
                    else:
                        pending_heading[1].append(text)
                else:
                    if pending_heading:
                        out += ["#" * pending_heading[0] + " " + " ".join(pending_heading[1]), ""]
                    pending_heading = (lvl, [text])
                prev_row = r
                continue
            if pending_heading:
                out += ["#" * pending_heading[0] + " " + " ".join(pending_heading[1]), ""]
                pending_heading = None

            # новый абзац: большой отступ по вертикали, красная строка относительно предыдущей строки
            # или начало пункта («1.», «D2.», «Теорема»). Сравнивать отступ с левым краем страницы нельзя:
            # продолжение пункта списка тоже сдвинуто вправо
            sentence_ended = bool(para) and para[-1].rstrip().endswith((".", ":", ";", "!", "?", "◀"))
            new_par = (prev_row is None or (r.base - prev_row.base) > 1.9 * body
                       or (r.x0 > prev_row.x0 + 0.8 * body and sentence_ended))
            if para and para[-1].endswith("-") and text[:1].islower():
                new_par = False  # перенос слова сильнее любых отступов
            if re.match(r"^(\d+\.|\d+\)|[A-ZА-Я]\d+\.|[a-zа-я]\)|Пример|Теорема|Лемма|Определение|Следствие|Замечание|"
                        r"Утверждение|Доказательство|Nota bene|▶)", text):
                new_par = True
            if new_par:
                flush()
            lead_spans = []
            for s in mains:
                if not s.bold:
                    break
                lead_spans.append(s.text.strip())
            lead = " ".join(lead_spans)
            if lead and not para and len(lead) > 2:
                m = re.match(r"\s*" + r"\s*".join(map(re.escape, lead.split())), text)
                if m:
                    text = f"**{lead}**" + text[m.end():]
            if para and para[-1].endswith("-") and text[:1].islower():
                para[-1] = para[-1][:-1] + text  # перенос «метри-кой»
            else:
                para.append(text)
            prev_row = r
    flush()
    if pending_heading:
        out += ["#" * pending_heading[0] + " " + " ".join(pending_heading[1]), ""]
    return fix_negations("\n".join(out)).strip() + "\n"
