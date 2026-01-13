import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

STATE_DIR = ".p4_state"
BOOST_PATH = os.path.join(STATE_DIR, "boost_counts.json")
os.makedirs(STATE_DIR, exist_ok=True)


def to_digits(x, n=4) -> Tuple[int, ...]:
    if isinstance(x, (list, tuple, np.ndarray)):
        digs = [int(v) for v in x]
    else:
        s = "".join(ch for ch in str(x) if ch.isdigit())
        digs = [int(ch) for ch in s]
    if len(digs) > n:
        digs = digs[-n:]
    elif len(digs) < n:
        digs = [0] * (n - len(digs)) + digs
    return tuple(digs)


def digits_to_str(t: Tuple[int, ...]) -> str:
    return "".join(str(int(d)) for d in t)


def multiset_overlap(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    ca, cb = Counter(a), Counter(b)
    return sum(min(ca[k], cb[k]) for k in ca.keys() | cb.keys())


def greedy_multiset_mapping(a: Tuple[int, ...], b: Tuple[int, ...]):
    ca, cb = Counter(a), Counter(b)
    pairs = []
    for d in range(10):
        m = min(ca[d], cb[d])
        if m:
            pairs.extend((d, d) for _ in range(m))
            ca[d] -= m
            cb[d] -= m
    rem_a, rem_b = [], []
    for d in range(10):
        if ca[d] > 0:
            rem_a.extend([d] * ca[d])
        if cb[d] > 0:
            rem_b.extend([d] * cb[d])
    rem_a.sort()
    rem_b.sort()
    pairs.extend(zip(rem_a, rem_b))
    return pairs


def load_boosts() -> Dict[str, float]:
    if os.path.exists(BOOST_PATH):
        try:
            with open(BOOST_PATH, "r") as f:
                raw = json.load(f)
            return {k: float(v) for k, v in raw.items()}
        except Exception:
            return {}
    return {}


def save_boosts(boosts: Dict[str, float]) -> None:
    try:
        with open(BOOST_PATH, "w") as f:
            json.dump(boosts, f)
    except Exception:
        pass
