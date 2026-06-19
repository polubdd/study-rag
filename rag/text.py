"""Нормализация русского текста для поиска."""
import re

_WORD = re.compile(r"[а-яёa-z0-9]+(?:[-'][а-яёa-z0-9]+)*")
_SENT = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯЁA-Z0-9«\"(])")

STOP = set("""а без более бы был была были было быть в вам вас весь во вот все всего всех вы где да даже для до его ее если есть
еще же за здесь и из или им их к как ко когда кто ли либо мне может мы на над надо наш не него нее нет ни них но ну о об однако
он она они оно от очень по под при с со так также такой там те тем то того тоже той только том ты у уже хотя чего чей чем что
чтобы чье чья эта эти это этот я как какой какая какие каков почему зачем""".split())

# окончания от длинных к коротким: первое подходящее и отрезается
_SUFFIXES = sorted(set("""иями ями ами ием иям иях ого его ому ему ыми ими ией ией ость ости остью ться тся ешь ишь ете ите ует
уют ают яют ал ала али ало ил ила или ило ия ие ию ии ой ей ий ый ая яя ую юю ое ее ые ие ов ев ах ях ам ям ом ем ть ет ит ут ют
ат ят а я у ю ы и е о ь""".split()), key=len, reverse=True)


def normalize(text: str) -> str:
    return str(text).lower().replace("ё", "е")


def stem(word: str) -> str:
    """Грубый стеммер: «матрицы», «матрицей», «матрицу» -> «матриц».

    Для поиска по конспектам этого хватает, а pymorphy тянет словари на 20 МБ.
    Минимальная длина основы 3 символа: иначе «ранг» и «рано» склеятся.
    """
    if len(word) <= 4 or not word.isalpha():
        return word
    for s in _SUFFIXES:
        if word.endswith(s) and len(word) - len(s) >= 3:
            return word[: -len(s)]
    return word


def words(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in words(text) if w not in STOP]


def sentences(text: str) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n|\n(?=[-•*]\s)", text):
        para = " ".join(para.split())
        if para:
            out += [s.strip() for s in _SENT.split(para) if s.strip()]
    return out
