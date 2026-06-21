"""Карточки для повторения из конспекта.

Три вида карточек, по тому, как обычно пишут конспекты:
- определение:   «Ранг — максимальное число линейно независимых строк…»
                 -> «Что такое ранг?»
- формулировка:  «Центральная предельная теорема: сумма большого числа…»
                 -> «Сформулируйте: центральная предельная теорема»
- формула:       под заголовком «Формула Байеса» строка «P(A|B) = …»
                 -> «Запишите формулу: формула Байеса»

Правила ловят то, что выделено структурой текста. LLM (если подключена)
дописывает вопросы на понимание: «почему нельзя KFold на временных рядах».
"""
import json
import re
from dataclasses import dataclass

from . import llm
from .chunk import Chunk
from .text import sentences

DEF = re.compile(r"^(?P<term>[^—–:=()\[\]]{2,60}?)\s+[—–]\s+(?:это\s+)?(?P<body>.{12,})$")
STATEMENT = re.compile(r"^(?P<term>[А-ЯЁA-Z][^:=]{3,60}?):\s+(?P<body>.{20,})$")
FORMULA = re.compile(r"[=≈≤≥]|\bdet\b|∑|∫")
# Лекции чаще пишут определения так:
#   «Метрическим пространством M называется множество…»  -> термин до глагола, в творительном падеже
#   «Отображение ρ называется расстоянием.»               -> термин после глагола
CALLED = re.compile(r"^(?P<left>.{3,90}?)\s+(?:называется|называют|называются)\s+(?P<right>.{3,})$")
# «Классический пример: …», «Что помогает: …» — двоеточие есть, а понятия нет
GENERIC = {"пример", "классический пример", "что помогает", "итог", "вывод", "важно", "замечание", "например", "идея",
           "итоговая метрика", "популярные реализации", "плюсы", "минусы", "пояснение", "доказательство"}


@dataclass
class Card:
    kind: str  # definition | statement | formula | question
    term: str
    front: str
    back: str
    chunk_id: int | None = None


# без -ло/-ла/-ли: на них кончаются и «правило», «число», «сила»
VERBISH = re.compile(r"(ся|сь|ть|ет|ит|ут|ют|ат|ят)$")
ADVERB_PREFIX = re.compile(r"^(геометрически|алгебраически|формально|интуитивно|грубо говоря|по сути|иначе говоря)\s+", re.I)


def _clean_term(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip(" .,:;")
    return ADVERB_PREFIX.sub("", t)


def _is_concept(term: str) -> bool:
    """«Закон больших чисел» — понятие, «На этом держится PCA» — нет.

    Понятие — именная группа: без служебных слов и без глаголов. Глагол ловлю
    по окончанию, поэтому прилагательные на -ый/-ая проходят, а «держится» нет.
    """
    from .text import STOP
    ws = [w.lower() for w in re.findall(r"[А-ЯЁа-яёA-Za-z]+", term)]
    if not ws or term.lower() in GENERIC:
        return False
    # «Обратное неверно: …», «Следовательно: …» — связки рассуждения, а не понятия
    filler = {"её", "ее", "его", "нельзя", "можно", "нужно", "проще", "всего", "верно", "неверно", "обратное",
              "справедливо", "следовательно", "аналогично", "заметим", "тогда", "поэтому", "отсюда", "значит"}
    return not any(w in STOP or w in filler for w in ws) \
        and not any(len(w) > 4 and VERBISH.search(w) and not w.endswith(("ость", "ность")) for w in ws if re.match("[а-яё]", w))


def _lower_first(t: str) -> str:
    # «Ранг» -> «ранг», но «SVD», «PCA» и «Байеса» в середине не трогаю
    return t if len(t) > 1 and t[:2].isupper() else t[:1].lower() + t[1:]


INSTRUMENTAL = re.compile(r"(ым|им|ом|ем|ой|ей|ою|ею|ами|ями|ыми|ими)$")
# прилагательное в творительном падеже или наречие при нём («положительно определенной»)
ADJ = re.compile(r"(ым|им|ыми|ими|[нквчшщжл]ой|[нкчшщж]ей|о)$")
# «Эта метрика называется…», «Полученное пространство называется…» — без контекста вопрос бессмыслен
DEICTIC = {"эта", "этот", "это", "эти", "такая", "такой", "такое", "такие", "полученное", "полученный", "полученная",
           "полученные", "данное", "данный", "данная", "он", "она", "оно", "они", "последнее", "последний"}
STARTERS = {"пусть", "тогда", "если", "где", "здесь", "то", "при", "для", "итак", "значит", "откуда", "заметим"}


def _strip_label(s: str) -> tuple[str, str | None]:
    """«**Nota bene** Отображение ρ…» -> «Отображение ρ…»: жирная подпись абзаца — не часть определения.
    Но «**Метрическим пространством** M называется…» — жирным выделен сам термин, его оставляю."""
    m = re.match(r"^\s*\*\*(.+?)\*\*\s*(.*)$", s)
    if not m:
        return s.replace("**", ""), None
    lead, rest = m.group(1).strip(), m.group(2)
    # подпись: дальше идёт слово с заглавной («Отображение…») или пояснение в скобках. Одна заглавная
    # буква («M называется») — это переменная из определения, а не новое предложение
    if re.match(r"^(\[|[A-ZА-ЯЁ][a-zа-яё]{2,})", rest):
        return rest.replace("**", ""), None
    return (lead + " " + rest).replace("**", ""), lead


def _called_card(s: str, bold_term: str | None, chunk_id) -> Card | None:
    m = CALLED.match(s)
    if not m:
        return None
    left, right = m.group("left").strip(), m.group("right").strip().rstrip(".")
    words = left.split()
    if not words or words[0].lower() in DEICTIC | STARTERS:
        return None
    if INSTRUMENTAL.search(words[0].lower()) and len(words) <= 6 and not FORMULA.search(left):
        # «Метрическим пространством M называется…»: термин слева, в творительном падеже
        term = _clean_term(bold_term or left)
        if all(ADJ.search(w.lower()) for w in term.split()):
            # «Унитарным называется оператор…» -> «Какой оператор называется унитарным?»
            noun = re.findall(r"[А-ЯЁа-яё]+", right)[:1]
            if not noun:
                return None
            n = noun[0].lower()
            which = "Какая" if n.endswith(("а", "я", "ь")) else "Какое" if n.endswith(("о", "е")) else \
                "Какие" if n.endswith(("ы", "и")) else "Какой"
            return Card("definition", term, f"{which} {n} называется {term.lower()}?", s, chunk_id)
        return Card("definition", term, f"Что называется {_lower_first(term)}?", s, chunk_id)
    # «Отображение ρ называется расстоянием», «Форма называется положительно определенной, если…»: термин справа
    term = _clean_term(re.split(r",|:|\s(?:если|когда|при|где)\s", right)[0])
    if not (1 <= len(term.split()) <= 5) or FORMULA.search(term) or MATHY.search(term):
        return None
    adjective_only = all(ADJ.search(w.lower()) for w in term.split())
    if (re.search(r",?\s+если\s", right) or adjective_only) and len(words) <= 6:
        # условное определение или одно прилагательное: «Что называется изотропным?» без подлежащего
        # не имеет смысла, поэтому вопрос строится вокруг него — «Когда элемент x называется изотропным?»
        return Card("definition", term, f"Когда {_lower_first(left)} называется {term}?", s, chunk_id)
    if adjective_only:
        return None
    if INSTRUMENTAL.search(term.split()[-1].lower()):  # «называется расстоянием»: название в творительном падеже
        return Card("definition", term, f"Что называется {_lower_first(term)}?", s, chunk_id)
    return None


# «∥f∥_{∗}», «φ», одиночная буква-переменная: такой «термин» — обозначение, а не понятие
MATHY = re.compile(r"[∥^_{}=→∈⊂αβγδεζηθλμνξπρστφχψωΩΦΨΣΛΔΓ]|(?:^|\s)[A-Za-zА-Яа-я](?:\s|$|[,:¯])")


FORMULA_SECTION = re.compile(r"формул|теорем|неравенств|закон|правил|тождеств|равенств|лемм", re.I)


def from_chunk(chunk: Chunk) -> list[Card]:
    out: list[Card] = []
    leaf = re.sub(r"^[\d.]+\s*", "", chunk.section.split(" / ")[-1])
    formula_done = False
    for i, raw in enumerate(sentences(chunk.text)):
        s, bold_term = _strip_label(raw.lstrip("-• "))
        s = s.strip()
        card = _called_card(s, bold_term, chunk.id)
        if card:
            out.append(card)
            continue
        m = DEF.match(s)
        if (m and len(m.group("term").split()) <= 6 and len(m.group("body").split()) >= 4
                and not FORMULA.search(m.group("term")) and not MATHY.search(m.group("term"))
                and m.group("term").split()[0].lower() not in STARTERS | DEICTIC):
            term = _clean_term(m.group("term"))
            out.append(Card("definition", term, f"Что такое {_lower_first(term)}?", s, chunk.id))
            continue
        m = STATEMENT.match(s)
        if (m and len(m.group("term").split()) <= 6 and _is_concept(_clean_term(m.group("term")))
                and not MATHY.search(m.group("term"))):
            term = _clean_term(m.group("term"))
            out.append(Card("statement", term, f"Сформулируйте: {_lower_first(term)}", s, chunk.id))
            continue
        # формула под заголовком «Формула Байеса», «Неравенство Шварца». Под заголовком «24.1 Метрические
        # пространства» первая формула — случайная строка из доказательства, вопрос по ней бессмыслен
        if (not formula_done and FORMULA_SECTION.search(leaf) and FORMULA.search(s)
                and re.search(r"[A-Za-zА-Яа-я]\s*\(?.*[=≤≥]", s) and len(s) < 160):
            out.append(Card("formula", leaf, f"Запишите формулу: {_lower_first(leaf)}", s, chunk.id))
            formula_done = True
    return out


def from_chunks(chunks: list[Chunk]) -> list[Card]:
    seen, out = set(), []
    for c in chunks:
        for card in from_chunk(c):
            key = card.front.lower()
            if key not in seen:
                seen.add(key)
                out.append(card)
    return out


LLM_PROMPT = """Ты преподаватель. По фрагменту конспекта составь до 2 вопросов НА ПОНИМАНИЕ (не «дай определение»),
ответ на которые есть во фрагменте. Верни ТОЛЬКО JSON: [{{"q": "вопрос", "a": "ответ из фрагмента"}}]

Раздел: {section}
{text}"""


def llm_questions(chunk: Chunk, call=None) -> list[Card]:
    raw = (call or llm.chat)(LLM_PROMPT.format(section=chunk.section, text=chunk.text))
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items[:2]:
        q, a = str(it.get("q", "")).strip(), str(it.get("a", "")).strip()
        if len(q) > 8 and len(a) > 3:
            out.append(Card("question", chunk.section.split(" / ")[-1], q, a, chunk.id))
    return out
