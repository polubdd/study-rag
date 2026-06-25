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


def from_chunk(chunk: Chunk) -> list[Card]:
    out: list[Card] = []
    leaf = chunk.section.split(" / ")[-1]
    formula_done = False
    for i, s in enumerate(sentences(chunk.text)):
        s = s.lstrip("-•* ").strip()
        m = DEF.match(s)
        if m and len(m.group("term").split()) <= 6 and len(m.group("body").split()) >= 4 and not FORMULA.search(m.group("term")):
            term = _clean_term(m.group("term"))
            out.append(Card("definition", term, f"Что такое {_lower_first(term)}?", s, chunk.id))
            continue
        m = STATEMENT.match(s)
        if m and len(m.group("term").split()) <= 6 and _is_concept(_clean_term(m.group("term"))):
            term = _clean_term(m.group("term"))
            out.append(Card("statement", term, f"Сформулируйте: {_lower_first(term)}", s, chunk.id))
            continue
        if not formula_done and FORMULA.search(s) and re.search(r"[A-Za-zА-Яа-я]\s*\(?.*[=]", s) and len(s) < 160:
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
