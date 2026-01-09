```python name=streamlit_app.py url=https://github.com/malkiavec/Pick4patternpredictor/blob/8bfda0fe8523898cf7e7cf5bb28fa4508588ae59/streamlit_app.py
# app.py
# Streamlit app for Pick 4 with variable shift-pattern prediction, backtesting, and live correctness tracking.
# Run: streamlit run app.py

import io
import os
import json
import math
from collections import Counter
from typing import Iterable, Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import requests

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except Exception:
    HAS_PLOTTING = False

# --------------------------------
# App config
# --------------------------------
st.set_page_config(page_title="Pick 4 Pattern Predictor", layout="wide")
st.title("🔢 Pick 4 Pattern Predictor")
st.caption("Trains on history, predicts using digit-shift patterns (positionless), backtests 3-of-4 success, and tracks live correctness.")

# --------------------------------
# Persistence
# --------------------------------
STATE_DIR = ".p4_state"
os.makedirs(STATE_DIR, exist_ok=True)
PRED_LOG_PATH = os.path.join(STATE_DIR, "pred_log.jsonl")
BOOST_PATH = os.path.join(STATE_DIR, "boost_counts.json")

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

def append_pred_log(record: Dict) -> None:
    try:
        with open(PRED_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass

def read_pred_log() -> pd.DataFrame:
    rows = []
    if os.path.exists(PRED_LOG_PATH):
        with open(PRED_LOG_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    if not rows:
        return pd.DataFrame(columns=["timestamp", "seed", "preds", "actual", "hit", "mode"])
    df = pd.DataFrame(rows)
    if "hit" in df.columns:
        df["hit"] = df["hit"].astype(int)
    return df

# --------------------------------
# Core utils
# --------------------------------
def digits_only(s: str) -> List[int]:
    return [int(ch) for ch in s if ch.isdigit()]

def to_tuple(x, n_digits: Optional[int] = None) -> Tuple[int, ...]:
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        digs = [int(v) for v in x]
    else:
        digs = digits_only(str(x))
    if n_digits is not None:
        if len(digs) > n_digits:
            digs = digs[-n_digits:]
        elif len(digs) < n_digits:
            digs = [0] * (n_digits - len(digs)) + digs
    return tuple(digs)

def tuple_to_str(t: Tuple[int, ...]) -> str:
    return "".join(str(int(d)) for d in t)

def greedy_multiset_mapping(a: Tuple[int, ...], b: Tuple[int, ...]) -> List[Tuple[int, int]]:
    ca, cb = Counter(a), Counter(b)
    pairs: List[Tuple[int, int]] = []
    # Exact matches first
    for d in range(10):
        m = min(ca[d], cb[d])
        if m:
            pairs.extend((d, d) for _ in range(m))
            ca[d] -= m
            cb[d] -= m
    # Leftovers matched by sorted order
    rem_a, rem_b = [], []
    for d in range(10):
        if ca[d] > 0: rem_a.extend([d] * ca[d])
        if cb[d] > 0: rem_b.extend([d] * cb[d])
    rem_a.sort()
    rem_b.sort()
    pairs.extend(zip(rem_a, rem_b))
    return pairs

def extract_digit_transitions(draws: List[Tuple[int, ...]], lag: int) -> Counter:
    trans: Counter = Counter()
    if not draws or lag <= 0:
        return trans
    for i in range(len(draws) - lag):
        a, b = draws[i], draws[i + lag]
        for x, y in greedy_multiset_mapping(a, b):
            trans[(x, y)] += 1
    return trans

def normalize_matrix(cnt: Counter, alpha: float = 0.5, boosts: Optional[Dict[str, float]] = None, boost_weight: float = 1.0) -> Dict[Tuple[int, int], float]:
    totals = Counter()
    for (x, y), c in cnt.items():
        totals[x] += c
    if boosts:
        for k, v in boosts.items():
            try:
                xs, ys = k.split("->")
                x, y = int(xs), int(ys)
                cnt[(x, y)] += float(v) * boost_weight
                totals[x] += float(v) * boost_weight
            except Exception:
                continue
    probs: Dict[Tuple[int, int], float] = {}
    for x in range(10):
        row_total = totals[x]
        denom = row_total + alpha * 10
        if denom <= 0:
            for y in range(10):
                probs[(x, y)] = 1.0 / 10.0
            continue
        for y in range(10):
            c = cnt.get((x, y), 0.0)
            probs[(x, y)] = (c + alpha) / denom
    return probs

def transition_matrix_df(cnt: Counter) -> pd.DataFrame:
    mat = np.zeros((10, 10), dtype=float)
    row_totals = np.zeros(10, dtype=float)
    for (x, y), c in cnt.items():
        mat[x, y] += c
        row_totals[x] += c
    for x in range(10):
        s = row_totals[x]
        if s > 0:
            mat[x] /= s
        else:
            mat[x][:] = 0.1
    return pd.DataFrame(mat, index=[f"{i}" for i in range(10)], columns=[f"{j}" for j in range(10)])

def apply_positionless_transitions(seed: Tuple[int, ...], probs: Dict[Tuple[int, int], float], top_k: int = 3) -> List[Tuple[int, ...]]:
    choices_per_digit: List[List[int]] = []
    for v in seed:
        row = [(y, probs.get((v, y), 0.0)) for y in range(10)]
        row.sort(key=lambda t: t[1], reverse=True)
        top = [y for y, _ in row[: max(1, top_k)]]
        if v not in top:
            top = [v] + top[:-1]
        choices_per_digit.append(top)
    # Cartesian product -> dedupe
    cands = set()
    def dfs(i: int, cur: List[int]):
        if i == len(choices_per_digit):
            cands.add(tuple(cur))
            return
        for y in choices_per_digit[i]:
            cur.append(y)
            dfs(i + 1, cur)
            cur.pop()
    dfs(0, [])
    return list(cands)

# ---------------- Pattern rules (variable, positionless) ----------------
def mod10(x: int) -> int:
    return x % 10

def all_shift_patterns(n_digits: int, shift_set: List[int], limit: int = 1000) -> List[Tuple[int, ...]]:
    from itertools import product
    patterns = list(product(shift_set, repeat=n_digits))
    if len(patterns) > limit:
        np.random.seed(42)
        idx = np.random.choice(len(patterns), size=limit, replace=False)
        patterns = [patterns[i] for i in sorted(idx)]
    return [tuple(int(s) for s in p) for p in patterns]

def strict_pattern_family(n_digits: int = 4) -> List[Tuple[int, ...]]:
    # Exactly one digit with ±5, one with ±2, one with 0, one with ±3 (for Pick 4)
    from itertools import product, permutations
    buckets = [[-5, 5], [-2, 2], [0], [-3, 3]]
    base_choices = list(product(*buckets))  # tuples like (±5, ±2, 0, ±3)
    pats = set()
    for choice in base_choices:
        for perm in permutations(choice, n_digits):
            pats.add(tuple(perm))
    return list(pats)

def apply_pattern(seed: Tuple[int, ...], pattern: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(mod10(d + s) for d, s in zip(seed, pattern))

def multiset_overlap(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    ca, cb = Counter(a), Counter(b)
    return sum(min(ca[k], cb[k]) for k in ca.keys() | cb.keys())

def pos_matches(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    return sum(1 for i in range(len(a)) if a[i] == b[i])

def pattern_from_pair(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(mod10(b[i] - a[i]) for i in range(len(a)))

def learn_pattern_multisets(draws: List[Tuple[int, ...]], lags: List[int], top_m: int = 200) -> List[Tuple[int, ...]]:
    pat_counts = Counter()
    for lag in lags:
        for i in range(len(draws) - lag):
            a, b = draws[i], draws[i + lag]
            p = pattern_from_pair(a, b)
            pat_counts[p] += 1
    if not pat_counts:
        return []
    return [p for p, _ in pat_counts.most_common(top_m)]

def pattern_prior_from_counts(patterns: List[Tuple[int, ...]], draws: List[Tuple[int, ...]], lags: List[int], alpha_prior: float = 0.1) -> Dict[Tuple[int, ...], float]:
    pat_counts = Counter()
    for lag in lags:
        for i in range(len(draws) - lag):
            a, b = draws[i], draws[i + lag]
            p = pattern_from_pair(a, b)
            pat_counts[p] += 1
    total = sum(pat_counts.values())
    K = max(1, len(patterns))
    priors = {}
    for p in patterns:
        c = pat_counts.get(p, 0)
        priors[p] = (c + alpha_prior) / (total + alpha_prior * K) if total + alpha_prior * K > 0 else 1.0 / K
    z = sum(priors.values()) or 1.0
    for k in priors:
        priors[k] /= z
    return priors

def score_prediction_against_actual(pred: Tuple[int, ...], actual: Tuple[int, ...], bonus_pos: float = 0.25) -> float:
    base = multiset_overlap(pred, actual)
    bonus = bonus_pos * pos_matches(pred, actual)
    return float(base) + float(bonus)

# --------------------------------
# Cached stages
# --------------------------------
@st.cache_data(show_spinner=False)
def parse_draws_from_df(df: pd.DataFrame, n_digits: int, hist_col: Optional[str]) -> List[Tuple[int, ...]]:
    if hist_col and hist_col in df.columns:
        col = df[hist_col]
    else:
        col = df.iloc[:, 0]
    out: List[Tuple[int, ...]] = []
    for val in col.dropna():
        t = to_tuple(val, n_digits=n_digits)
        if len(t) == n_digits and all(0 <= d <= 9 for d in t):
            out.append(t)
    return out

@st.cache_data(show_spinner=False)
def compute_transitions(draws: List[Tuple[int, ...]], recent_window: int, max_lag: int, lag_weights: Iterable[float]) -> Counter:
    windowed = draws[-recent_window:] if recent_window > 0 else draws
    all_trans = Counter()
    for lag, w in zip(range(1, max_lag + 1), lag_weights):
        if w <= 0:
            continue
        trans = extract_digit_transitions(windowed, lag)
        if w != 1.0:
            trans = Counter({k: v * w for k, v in trans.items()})
        all_trans.update(trans)
    return all_trans

# --------------------------------
# NY Lottery API fetch helpers
# --------------------------------
@st.cache_data(show_spinner=False)
def fetch_ny_draws(url: str, limit: int = 1000, timeout: int = 10) -> pd.DataFrame:
    """
    Fetch data from the NY State data API endpoint (Socrata style).
    Tries to handle common response shapes: dict with 'data' + 'meta', list of dicts, or list of lists.
    Returns a pandas DataFrame.
    """
    params = {"$limit": limit, "$order": "draw_date DESC"}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    # Socrata JSON shape: dict with 'data' (list of rows) and 'meta.view.columns' for column names
    if isinstance(data, dict) and "data" in data:
        rows = data.get("data", [])
        cols = []
        try:
            cols = [c.get("name") for c in data.get("meta", {}).get("view", {}).get("columns", [])]
        except Exception:
            cols = []
        if cols and isinstance(rows, list) and rows and isinstance(rows[0], list):
            return pd.DataFrame(rows, columns=cols)
        elif isinstance(rows, list) and rows and isinstance(rows[0], dict):
            return pd.DataFrame(rows)
        else:
            return pd.DataFrame(rows)
    # If it's a list of dicts (common)
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            return pd.DataFrame(data)
        elif data and isinstance(data[0], list):
            # no column names available
            return pd.DataFrame(data)
    # Fallback: return empty DataFrame
    return pd.DataFrame()

def detect_draw_column(df: pd.DataFrame, n_digits: int = 4) -> Optional[str]:
    """
    Heuristic: choose the column most likely to contain the draw string (e.g., '1234' or '1-2-3-4').
    """
    if df is None or df.empty:
        return None
    best_col = None
    best_score = -1.0
    for col in df.columns:
        series = df[col].dropna().astype(str).head(1000)
        if series.empty:
            continue
        matches = 0
        total = 0
        for v in series:
            total += 1
            digs = digits_only(v)
            if len(digs) == n_digits:
                matches += 1
        score = matches / max(1, total)
        if score > best_score:
            best_score = score
            best_col = col
    # require at least one match to be confident; otherwise return first column
    if best_score <= 0:
        return df.columns[0]
    return best_col

# --------------------------------
# Sidebar controls (NY API instead of CSV upload)
# --------------------------------
st.sidebar.header("History / Data source")
st.sidebar.write("This build fetches Pick 4 history from the NY State Lottery dataset (Socrata API).")
ny_api = st.sidebar.text_input("NY Lottery API URL", "https://data.ny.gov/api/v3/views/hsys-3def/query.json")
ny_limit = st.sidebar.number_input("Number of draws to fetch", min_value=50, max_value=10000, value=1000, step=50)
fetch_button = st.sidebar.button("Fetch draws from NY API")

st.sidebar.header("Training window")
n_digits = 4  # Pick 4 primarily
recent_window = st.sidebar.slider("Recent window (draws); 0 = all", 0, 5000, 500, 50)
max_lag = st.sidebar.slider("Max skip (lag)", 1, 10, 3, 1)

st.sidebar.subheader("Lag weighting")
weight_scheme = st.sidebar.selectbox("Weights", ["Uniform", "Linear decay", "Exponential decay"], index=0)
if weight_scheme == "Uniform":
    lag_weights = [1.0] * max_lag
elif weight_scheme == "Linear decay":
    lag_weights = [max(0.1, (max_lag - i) / max_lag) for i in range(max_lag)]
else:
    decay = st.sidebar.slider("Exponential decay factor", 0.5, 0.99, 0.85, 0.01)
    lag_weights = [decay**i for i in range(max_lag)]

st.sidebar.header("Model smoothing")
alpha = st.sidebar.slider("Smoothing α (Laplace=1.0)", 0.0, 2.0, 0.5, 0.1)

st.sidebar.header("Pattern rules")
pattern_mode = st.sidebar.selectbox("Pattern source", ["Hand-crafted", "Strict family (±5, ±2, 0, ±3)", "Learned from history"], index=2)
shift_set_str = st.sidebar.text_input("Shift set (for Hand-crafted)", "0,1,-1,2,-2,3,-3,5,-5")
try:
    SHIFT_SET = [int(x.strip()) for x in shift_set_str.split(",") if x.strip() != ""]
except Exception:
    SHIFT_SET = [0, 1, -1, 2, -2, 3, -3, 5, -5]
pattern_limit = st.sidebar.slider("Max patterns to consider", 50, 5000, 500, 50)
bonus_pos = st.sidebar.slider("Exact-position bonus per match", 0.0, 1.0, 0.25, 0.05)
success_k = st.sidebar.select_slider("Success threshold (overlap)", options=[1, 2, 3, 4], value=3)

st.sidebar.header("Prediction size")
num_preds = st.sidebar.select_slider("Number of predictions", options=[10, 20, 30, 50, 100], value=30)

st.sidebar.header("Feedback (optional)")
enable_boosts = st.sidebar.checkbox("Enable feedback boosts from live correct hits", value=False)
boost_weight = st.sidebar.slider("Boost weight (pseudo-count scale)", 0.0, 5.0, 1.0, 0.1)

# --------------------------------
# Load data (from NY API)
# --------------------------------
draws: List[Tuple[int, ...]] = []
df = pd.DataFrame()
if fetch_button:
    try:
        df = fetch_ny_draws(ny_api, limit=int(ny_limit))
        if df.empty:
            st.error("Fetched empty dataset from NY API. Verify the URL or increase the limit.")
        else:
            # detect the best column that looks like a 4-digit draw
            chosen_col = detect_draw_column(df, n_digits=n_digits)
            draws = parse_draws_from_df(df, n_digits=n_digits, hist_col=chosen_col)
            st.success(f"Fetched {len(df)} rows. Using column: {chosen_col}")
    except Exception as e:
        st.error(f"Failed to fetch NY draws: {e}")

if not draws:
    st.info("Fetch Pick 4 history from the NY API (use the controls in the sidebar) to begin.")
    st.stop()

st.subheader("Data")
st.write(f"Total parsed draws: {len(draws)} | Unique: {len(set(draws))}")
if len(draws) < 50:
    st.warning("Few draws detected; estimates may be noisy.")

# --------------------------------
# Train transitions
# --------------------------------
st.subheader("Training and transitions")
base_trans = compute_transitions(draws, recent_window=recent_window, max_lag=max_lag, lag_weights=lag_weights)
df_mat = transition_matrix_df(base_trans)

colA, colB = st.columns([1, 1])
with colA:
    st.markdown("Top outgoing transitions per digit (from → to, P(y|x))")
    top_rows = []
    for x in range(10):
        row = [(y, df_mat.iloc[x, y]) for y in range(10)]
        row.sort(key=lambda t: t[1], reverse=True)
        for y, p in row[:3]:
            top_rows.append({"from": x, "to": y, "prob": round(float(p), 4)})
    st.dataframe(pd.DataFrame(top_rows), use_container_width=True)
with colB:
    if HAS_PLOTTING:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df_mat, annot=False, cmap="Blues", cbar=True, ax=ax)
        ax.set_title("Weighted digit transition heatmap")
        st.pyplot(fig)

boosts = load_boosts() if enable_boosts else {}
probs = normalize_matrix(base_trans.copy(), alpha=alpha, boosts=boosts, boost_weight=boost_weight)

# --------------------------------
# Pattern preparation
# --------------------------------
def build_patterns(history: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    used_lags = [lag for lag, w in zip(range(1, max_lag + 1), lag_weights) if w > 0]
    if pattern_mode == "Hand-crafted":
        return all_shift_patterns(n_digits, SHIFT_SET, limit=pattern_limit)
    elif pattern_mode == "Strict family (±5, ±2, 0, ±3)":
        pats = strict_pattern_family(n_digits)
        # Subsample if too large
        if len(pats) > pattern_limit:
            np.random.seed(42)
            idx = np.random.choice(len(pats), size=pattern_limit, replace=False)
            pats = [pats[i] for i in sorted(idx)]
        return pats
    else:
        learned = learn_pattern_multisets(history, used_lags, top_m=pattern_limit)
        # Fallback if history too small
        if not learned:
            return all_shift_patterns(n_digits, [0, 1, -1, 2, -2, 3, -3, 5, -5], limit=min(500, pattern_limit))
        return learned

def pattern_priors(patterns: List[Tuple[int, ...]], history: List[Tuple[int, ...]]) -> Dict[Tuple[int, ...], float]:
    # Use lag=1 for prior unless you want to mix lags
    return pattern_prior_from_counts(patterns, history, [1], alpha_prior=0.2)

# --------------------------------
# Predict next (pattern-based, positionless scoring)
# --------------------------------
st.subheader("Predictions (pattern-based, positionless)")

history_used = draws[-recent_window:] if recent_window > 0 else draws
patterns = build_patterns(history_used)
priors = pattern_priors(patterns, history_used)
seed = draws[-1]

scored = []
for p in patterns:
    cand = apply_pattern(seed, p)
    prior = priors.get(p, 1.0 / max(1, len(patterns)))
    # Likelihood tie-breaker based on digit transitions (positionless mapping)
    like = 0.0
    for x, y in greedy_multiset_mapping(seed, cand):
        like += math.log(max(probs.get((x, y), 1e-12), 1e-12))
    total = math.log(prior + 1e-12) + like
    scored.append((cand, total, p))

scored.sort(key=lambda t: t[1], reverse=True)
predictions = [tuple_to_str(c) for c, _, _ in scored[:num_preds]]
st.write(predictions)
st.caption(f"Scored by pattern prior from history and transition likelihood. Success defined as overlap ≥ {success_k} (positionless), with bonus for exact positions.")

st.download_button("Download predictions (CSV)", pd.DataFrame({"prediction": predictions}).to_csv(index=False).encode(
