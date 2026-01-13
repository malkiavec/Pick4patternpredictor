from collections import Counter
from typing import List, Tuple, Dict, Optional
import json
import os

def to_digits(x, n=4) -> Tuple[int, ...]:
    if isinstance(x, (list, tuple)):
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

def greedy_multiset_mapping(a: Tuple[int, ...], b: Tuple[int, ...]) -> List[Tuple[int, int]]:
    ca, cb = Counter(a), Counter(b)
    pairs: List[Tuple[int, int]] = []
    # Match equal digits first
    for d in range(10):
        m = min(ca[d], cb[d])
        if m:
            pairs.extend((d, d) for _ in range(m))
            ca[d] -= m
            cb[d] -= m
    # Pair remaining in sorted order (positionless)
    rem_a, rem_b = [], []
    for d in range(10):
        if ca[d] > 0: rem_a.extend([d] * ca[d])
        if cb[d] > 0: rem_b.extend([d] * cb[d])
    rem_a.sort()
    rem_b.sort()
    pairs.extend(zip(rem_a, rem_b))
    return pairs

def ensure_state_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_json(path: str, data: Dict):
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass
