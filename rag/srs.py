"""Интервальное повторение по SM-2 (алгоритм SuperMemo, на нём же построен Anki).

Оценка ответа: 0 — не вспомнил, 1 — тяжело, 2 — нормально, 3 — легко.
Каждая карточка хранит «лёгкость» (ease): интервал до следующего показа
растёт геометрически, а лёгкость решает, насколько быстро. Забытая
карточка возвращается через 10 минут и теряет лёгкость.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

AGAIN, HARD, GOOD, EASY = 0, 1, 2, 3
MIN_EASE = 1.3
RELEARN = timedelta(minutes=10)


@dataclass
class State:
    ease: float = 2.5
    interval: float = 0.0  # дни
    reps: int = 0
    lapses: int = 0


def schedule(st: State, grade: int, now: datetime | None = None) -> tuple[State, datetime]:
    if grade not in (AGAIN, HARD, GOOD, EASY):
        raise ValueError("оценка 0..3")
    now = now or datetime.now(timezone.utc)
    if grade == AGAIN:
        new = State(max(MIN_EASE, st.ease - 0.2), 0.0, 0, st.lapses + 1)
        return new, now + RELEARN

    ease = max(MIN_EASE, st.ease + {HARD: -0.15, GOOD: 0.0, EASY: 0.15}[grade])
    if st.reps == 0:
        interval = {HARD: 1, GOOD: 1, EASY: 3}[grade]
    elif st.reps == 1:
        interval = {HARD: 2, GOOD: 3, EASY: 5}[grade]
    else:
        mult = {HARD: 1.2, GOOD: ease, EASY: ease * 1.3}[grade]
        interval = max(st.interval + 1, st.interval * mult)
    new = State(ease, float(interval), st.reps + 1, st.lapses)
    return new, now + timedelta(days=interval)
