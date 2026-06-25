"""Квиз: определение с пропуском и четыре варианта термина.

Неправильные варианты берутся из того же курса и, по возможности, того же
вида карточек: «ранг / определитель / след матрицы» путать интереснее,
чем «ранг / формула Байеса».
"""
import random
import re


def mask(text: str, term: str) -> str:
    """Прячет термин в определении, включая его падежные формы («Ранг», «ранга»)."""
    words = term.split()
    stems = [re.escape(w[:max(3, len(w) - 2)]) for w in words]
    pattern = r"\b" + r"\s+".join(s + r"[а-яёa-z]*" for s in stems) + r"\b"
    masked, n = re.subn(pattern, "_____", text, flags=re.I)
    return masked if n else "_____ — " + text.split("—", 1)[-1].strip()


def make_quiz(cards: list[dict], n: int = 10, seed: int | None = None) -> list[dict]:
    rng = random.Random(seed)
    pool = [c for c in cards if c["kind"] in ("definition", "statement") and c.get("term")]
    terms = sorted({c["term"] for c in pool})
    if len(terms) < 4:
        return []
    rng.shuffle(pool)
    out = []
    for c in pool[:n]:
        same_kind = [t for t in {x["term"] for x in pool if x["kind"] == c["kind"]} if t != c["term"]]
        others = [t for t in terms if t != c["term"] and t not in same_kind]
        rng.shuffle(same_kind)
        rng.shuffle(others)
        options = [c["term"]] + (same_kind + others)[:3]
        rng.shuffle(options)
        out.append({"card_id": c["id"], "prompt": mask(c["back"], c["term"]), "options": options,
                    "answer": options.index(c["term"])})
    return out
