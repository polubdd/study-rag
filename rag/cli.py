import argparse

from .chunk import load_dir
from .index import Index
from .llm import ask


def main():
    p = argparse.ArgumentParser(description="Вопросы по конспектам")
    p.add_argument("question")
    p.add_argument("--notes", default="notes")
    p.add_argument("-k", type=int, default=3)
    args = p.parse_args()

    index = Index(load_dir(args.notes))
    hits = index.search(args.question, args.k)
    if not hits:
        print("В конспектах ничего не нашлось.")
        return

    answer = ask(args.question, [c for c, _ in hits])
    if answer:
        print(answer, "\n")
    print("Источники:")
    for i, (c, s) in enumerate(hits, 1):
        print(f"[{i}] {c.source} › {c.section}  (score {s:.2f})")
        if not answer:
            print("    " + c.text.replace("\n", "\n    ")[:400], "\n")


if __name__ == "__main__":
    main()
