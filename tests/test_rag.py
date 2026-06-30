from rag import Index, load_dir, split_markdown
from rag.bm25 import BM25, tokenize
from rag.llm import build_prompt


def test_split_keeps_heading_path():
    md = "# Линал\n## Матрицы\n### Ранг\nРанг — это ...\n"
    (c,) = split_markdown(md, "x")
    assert c.section == "Линал / Матрицы / Ранг"


def test_long_section_is_split_by_paragraphs():
    md = "# A\n" + "\n\n".join(["слово " * 50] * 4)
    chunks = split_markdown(md, "x", max_words=120)
    assert len(chunks) == 2
    assert all(len(c.text.split()) <= 120 for c in chunks)


def test_stemming_matches_word_forms():
    assert tokenize("матрицы")[0] == tokenize("матрицей")[0] == tokenize("матрица")[0]
    assert tokenize("обратную")[0] == tokenize("обратная")[0]


def test_bm25_prefers_rare_terms():
    bm = BM25(["кошка сидит", "собака сидит", "кошка кошка спит"])
    assert bm.top("кошка", 1)[0][0] == 2


def test_search_on_notes():
    idx = Index(load_dir("notes"))
    cases = {
        "как найти обратную матрицу": "Обратная матрица",
        "формула Байеса": "Формула Байеса",
        "почему нельзя KFold на временных рядах": "Кросс-валидация",
    }
    for q, expected in cases.items():
        top_chunk, _ = idx.search(q, 1)[0]
        assert expected in top_chunk.section, q


def test_prompt_numbers_sources():
    idx = Index(load_dir("notes"))
    prompt = build_prompt("что такое ранг", [c for c, _ in idx.search("ранг", 2)])
    assert "[1]" in prompt and "[2]" in prompt
