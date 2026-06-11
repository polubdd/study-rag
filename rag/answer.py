"""Ответ на вопрос по найденным фрагментам.

Два режима:
- экстрактивный (без LLM): из найденных фрагментов выбираются предложения,
  которые лучше всего отвечают на вопрос. Ничего не выдумывается, каждое
  предложение — дословно из конспекта, со ссылкой на источник;
- генеративный (LLM): модель пишет ответ своими словами, но обязана ставить
  [n] после каждого утверждения. Ссылки проверяются: несуществующий номер
  или ответ вообще без ссылок — и вместо него показывается экстрактивный.
"""
import math
import re
from dataclasses import asdict, dataclass, field

from . import llm
from .retrieval import Hit, HybridIndex
from .synonyms import expand
from .text import sentences, tokenize

DEFINITION = re.compile(r"\s[—–-]\s|\bэто\b|\bназывает|\bназывают|\bопределяется")
WHAT_IS = re.compile(r"^\s*(что так(ое|ая|ой)|что есть|что называ|как называ|определени|дай(те)? определение|кто такой)", re.I)

PROMPT = """Ты помогаешь студенту готовиться к экзамену. Ответь на вопрос ТОЛЬКО по фрагментам его конспекта.
Правила:
- после каждого утверждения ставь номер фрагмента в квадратных скобках, например [2];
- если во фрагментах нет ответа, ответь ровно: В конспектах этого нет.
- коротко, 2–5 предложений, без вступлений.

{context}

Вопрос: {question}"""

NOT_FOUND = "В конспектах этого нет."


@dataclass
class Answer:
    question: str
    text: str
    mode: str  # extractive | llm | none
    sources: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _sources(hits: list[Hit]) -> list[dict]:
    return [{"n": i + 1, "chunk_id": h.chunk.id, "source": h.chunk.source, "section": h.chunk.section,
             "text": h.chunk.text, "score": h.score, "ranks": h.ranks} for i, h in enumerate(hits)]


def _definition_weight(sentence: str, q: set[str], pos: int) -> float:
    """Насколько предложение — определение ИСКОМОГО термина.

    «Метрическим пространством M называется множество…» определяет метрическое пространство,
    а «Полученное метрическое пространство называется пространством Гильберта» — другое понятие,
    хотя слова вопроса есть в обоих. Определяемый термин берётся тем же разбором, что и для карточек.
    """
    from .cards import CALLED, DEF, _called_card, _strip_label
    plain, bold = _strip_label(sentence)
    card = _called_card(plain.strip(), bold, None)
    if card is None and CALLED.match(plain.strip()):
        return 0.6  # «Полученное пространство называется…» — называют что-то другое
    term = card.term if card else (DEF.match(plain.strip()).group("term") if DEF.match(plain.strip()) else None)
    content = q - {"называ", "называет", "определен", "определени"}
    if term:
        t = set(tokenize(term)) | set(tokenize(card.front if card else ""))
        # «что такое ранг матрицы» -> «Ранг — максимальное число строк матрицы»: термин содержит часть
        # вопроса, остальное есть в самом определении
        if content and t & content and content - t <= set(tokenize(plain)):
            return 3.0  # определение именно того, о чём спросили
        return 0.6  # определение, но чего-то другого
    return 1.3 if pos == 0 and DEFINITION.search(sentence) else 1.0


def extractive(question: str, hits: list[Hit], idf: dict[str, float], max_sentences: int = 3,
               synonyms: dict | None = None) -> str:
    """Жадный выбор предложений по покрытию вопроса.

    Каждое следующее предложение берётся за НОВЫЕ слова вопроса, которые
    ещё не покрыты. Без этого на «чем PCA связан с собственными векторами»
    выбирались два определения собственного вектора, а предложение про PCA
    проигрывало, хотя только оно и отвечает на вопрос.
    """
    q_orig = set(tokenize(question))
    # синонимы помогают найти предложение («momentum» -> «импульс»), но определение проверяю по исходному вопросу
    q = q_orig | set(tokenize(expand(question, synonyms))) if synonyms else q_orig
    wants_definition = bool(WHAT_IS.search(question))
    cands = []
    by_rank: dict[int, list[str]] = {}
    for rank, h in enumerate(hits):
        section = set(tokenize(h.chunk.section))
        by_rank[rank] = sentences(h.chunk.text)
        for pos, s in enumerate(by_rank[rank]):
            toks = set(tokenize(s))
            if not toks:
                continue
            # множитель качества: короткие предложения, первые в разделе, из лучших фрагментов
            quality = 1 / math.sqrt(len(toks) + 1) / (1 + 0.25 * rank) / (1 + 0.1 * pos)
            if wants_definition:
                quality *= _definition_weight(s, q_orig, pos)
            cands.append({"rank": rank, "pos": pos, "s": s, "toks": toks & q, "section": section & q, "quality": quality})

    def gain(c, covered):
        new = sum(idf.get(t, 1.0) for t in c["toks"] - covered)
        # заголовок раздела, где стоит термин, — сигнал, что предложение про него. Засчитываю его
        # в паре с совпадением в тексте или для первого предложения раздела: под заголовком
        # «Формула Байеса» первым идёт P(A|B) = ..., где слов вопроса нет вовсе
        ctx = 0.5 * sum(idf.get(t, 1.0) for t in c["section"] - covered) if new or c["pos"] == 0 else 0
        return (new + ctx) * c["quality"]

    # порог остановки — по доле вопроса, которую закрывает новое предложение, а не по качеству:
    # иначе позиционный штраф отсекал единственное предложение про PCA (пятое в разделе)
    total = sum(idf.get(t, 0) for t in q) or 1.0
    picked, covered = [], set()
    while len(picked) < max_sentences:
        best = max(cands, key=lambda c: gain(c, covered), default=None)
        if not best or gain(best, covered) <= 0:
            break
        if picked:
            new_share = sum(idf.get(t, 0) for t in best["toks"] - covered) / total
            # из другого фрагмента добираю, только если оно закрывает заметную часть вопроса
            need = 0.2 if any(p["rank"] == best["rank"] for p in picked) else 0.45
            if new_share < need:
                break
        picked.append(best)
        covered |= best["toks"] | (best["section"] if best["pos"] == 0 else set())
        cands.remove(best)
    picked.sort(key=lambda c: (c["rank"], c["pos"]))  # в порядке источников и текста — так читается связно
    parts = []
    for c in picked:
        text = c["s"]
        if text.rstrip().endswith(":"):
            # «…обладающее следующими свойствами (аксиомами):» без самих аксиом бесполезно —
            # добираю следующие короткие строки (пункты, формулы) того же фрагмента
            tail = []
            for nxt in by_rank[c["rank"]][c["pos"] + 1:c["pos"] + 6]:
                if len(nxt) > 160 or any(p["rank"] == c["rank"] and p["s"] == nxt for p in picked):
                    break
                list_item = re.match(r"^([A-ZА-Я]?\d+[.)]|[-•]|\(?[a-zа-я]\))\s", nxt)
                formula = re.search(r"[=≤≥<>∑∫∈⇔→]", nxt) and len(re.findall(r"[а-яё]{4,}", nxt.lower())) <= 2
                if not (list_item or formula):
                    break  # дальше уже обычный текст — продолжение раскрыто
                tail.append(nxt)
            text = "\n".join([text] + tail)
        parts.append(f"{text} [{c['rank'] + 1}]")
    return "\n".join(parts) if any("\n" in p for p in parts) else " ".join(parts)


def check_citations(text: str, n_sources: int) -> list[str]:
    problems = []
    cited = [int(x) for x in re.findall(r"\[(\d+)\]", text)]
    if text.strip() == NOT_FOUND:
        return problems
    if not cited:
        problems.append("в ответе нет ни одной ссылки на конспект")
    bad = sorted({c for c in cited if not 1 <= c <= n_sources})
    if bad:
        problems.append(f"ссылки на несуществующие фрагменты: {bad}")
    return problems


def answer(question: str, index: HybridIndex, k: int = 4, use_llm: bool | None = None, call=None) -> Answer:
    hits = index.search(question, k)
    if not index.is_answerable(question, hits):
        return Answer(question, NOT_FOUND, "none", _sources(hits[:2]),
                      ["похожих фрагментов почти нет — возможно, этой темы нет в загруженных конспектах"])
    sources = _sources(hits)
    # предложения ищу по вопросу вместе с синонимами: на «momentum» ответ в конспекте написан про «импульс»
    fallback = extractive(question, hits, index.bm25.idf, synonyms=index.synonyms)
    if use_llm is None:
        use_llm = llm.enabled() or call is not None
    if not use_llm:
        return Answer(question, fallback or NOT_FOUND, "extractive" if fallback else "none", sources)

    context = "\n\n".join(f"[{s['n']}] ({s['section']})\n{s['text']}" for s in sources)
    try:
        text = (call or llm.chat)(PROMPT.format(context=context, question=question)).strip()
    except Exception as e:
        return Answer(question, fallback or NOT_FOUND, "extractive", sources, [f"LLM недоступна ({type(e).__name__}), ответ собран из конспекта"])
    problems = check_citations(text, len(sources))
    if problems:
        return Answer(question, fallback or NOT_FOUND, "extractive", sources,
                      [f"ответ LLM отклонён: {p}" for p in problems])
    return Answer(question, text, "llm", sources)
