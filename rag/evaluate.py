"""python -m rag.evaluate — качество поиска на наборе вопросов.

Набор вопросов (eval/questions.json) написан своими словами, а не фразами из
конспекта: «модель запомнила трейн, а на тесте плохо» вместо «переобучение».
Иначе любой поиск выглядит идеально.

Метрики: recall@1, recall@3 (нужный раздел в первой / первых трёх позициях)
и MRR (средняя обратная позиция нужного раздела).
"""
import json
import sys
from pathlib import Path

from .retrieval import HybridIndex
from .store import Store

ROOT = Path(__file__).parent.parent

CONFIGS = {
    "BM25": (("bm25",), False),
    "символьный TF-IDF": (("chars",), False),
    "гибрид (RRF)": (("bm25", "chars"), False),
    "гибрид + синонимы": (("bm25", "chars"), True),
}


def evaluate(chunks, questions, methods, synonyms: bool = True, k: int = 5) -> dict:
    idx = HybridIndex(chunks, methods=methods, use_embeddings=False, synonyms=synonyms)
    r1 = r3 = rr = 0.0
    misses = []
    for item in questions:
        hits = idx.search(item["q"], k)
        pos = next((i for i, h in enumerate(hits) if item["section"].lower() in h.chunk.section.lower()), None)
        r1 += pos == 0
        r3 += pos is not None and pos < 3
        rr += 1 / (pos + 1) if pos is not None else 0
        if pos != 0:
            misses.append((item["q"], hits[0].chunk.section.split(" / ")[-1] if hits else "—"))
    n = len(questions)
    return {"recall@1": round(r1 / n, 3), "recall@3": round(r3 / n, 3), "mrr": round(rr / n, 3), "misses": misses}


def main(notes=ROOT / "notes", sets=("questions", "holdout")):
    store = Store()
    cid = store.create_course("eval")
    for f in sorted(Path(notes).glob("*.md")):
        store.add_document(cid, f.name, f.read_bytes())
    chunks = store.chunks(cid)
    results = {}
    for name in sets:
        questions = json.loads((ROOT / "eval" / f"{name}.json").read_text(encoding="utf-8"))
        print(f"\n{name}: {len(questions)} вопросов, {len(chunks)} фрагментов")
        print(f"{'метод':22} recall@1  recall@3   MRR")
        for cfg, (methods, syn) in CONFIGS.items():
            r = evaluate(chunks, questions, methods, syn)
            results[(name, cfg)] = r
            print(f"{cfg:22} {r['recall@1']:>8.3f}  {r['recall@3']:>8.3f}  {r['mrr']:.3f}")
        if "-v" in sys.argv:
            for q, got in results[(name, "гибрид + синонимы")]["misses"]:
                print(f"  промах: «{q}» -> {got}")
    return results


if __name__ == "__main__":
    main()
