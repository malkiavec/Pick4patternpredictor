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
     
def annotate_preds(preds: List[Tuple[int, ...]], actual: Tuple[int, ...], bonus_pos: float = 0.25) -> List[Dict]:
    rows = []
    for p in preds:
        rows.append({
            "pred": tuple_to_str(p),
            "overlap": multiset_overlap(p, actual),
            "pos_matches": pos_matches(p, actual),
            "score": score_prediction_against_actual(p, actual, bonus_pos=bonus_pos),
        })
    rows.sort(key=lambda r: (r["overlap"], r["pos_matches"], r["score"]), reverse=True)
    return rows

    

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
# Sidebar controls
# --------------------------------
st.sidebar.header("History")
n_digits = 4  # Pick 4 primarily
hist_file = st.sidebar.file_uploader("Upload history CSV (one column with draws like 4728)", type=["csv"])
hist_col = st.sidebar.text_input("Draw column name (optional)")

if st.sidebar.checkbox("Use sample data", value=False):
    sample = pd.DataFrame({"draw": ["1234","3601","9870","7412","2583","1235","3691","0246","1357","2468","3579","4680","5791","6913","8035"]})
    hist_file = io.BytesIO(sample.to_csv(index=False).encode("utf-8"))
    hist_col = "draw"

st.sidebar.header("Training window")
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
# Load data
# --------------------------------
draws: List[Tuple[int, ...]] = []
if hist_file is not None:
    try:
        df = pd.read_csv(hist_file)
        draws = parse_draws_from_df(df, n_digits=n_digits, hist_col=hist_col)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

if not draws:
    st.info("Upload your history CSV or use sample data to begin.")
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

st.download_button("Download predictions (CSV)", pd.DataFrame({"prediction": predictions}).to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")

# Remember predictions for live correctness
if st.button("Remember these predictions for next draw"):
    append_pred_log(
        {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "seed": tuple_to_str(seed),
            "preds": predictions,
            "actual": None,
            "hit": 0,
            "mode": {
                "recent_window": recent_window,
                "max_lag": max_lag,
                "lag_weights": list(lag_weights),
                "alpha": alpha,
                "pattern_mode": pattern_mode,
                "shift_set": SHIFT_SET,
                "pattern_limit": pattern_limit,
                "bonus_pos": bonus_pos,
                "success_k": int(success_k),
                "num_preds": int(num_preds),
                "enable_boosts": enable_boosts,
                "boost_weight": boost_weight,
            },
        }
    )
    st.success("Saved. When you load history that includes the next draw, the app will check hit/miss.")

# --------------------------------
# Live correctness tracking
# --------------------------------
st.subheader("Live correctness tracking")
log_df = read_pred_log()
if log_df.empty:
    st.info("No saved predictions yet. Click 'Remember these predictions' after generating predictions.")
else:
    series_str = [tuple_to_str(t) for t in draws]
    updates = []
    for idx, row in log_df.iterrows():
        if row.get("actual"):
            continue
        seed_str = row.get("seed")
        if seed_str in series_str:
            pos = series_str.index(seed_str)
            if pos + 1 < len(series_str):
                actual = series_str[pos + 1]
                preds_list = row.get("preds", [])
                # Success by overlap ≥ success_k
                actual_t = to_tuple(actual, n_digits=n_digits)
                overlaps = [multiset_overlap(to_tuple(p, n_digits=n_digits), actual_t) for p in preds_list]
                best_overlap = max(overlaps) if overlaps else 0
                success = int(best_overlap >= int(success_k))
                log_df.at[idx, "actual"] = actual
                log_df.at[idx, "hit"] = success
                updates.append((idx, actual, success))

                # Optional: update boosts for any correct digits (positionlessly) from top-1 prediction
                if enable_boosts and preds_list:
                    top1 = to_tuple(preds_list[0], n_digits=n_digits)
                    pairs = greedy_multiset_mapping(to_tuple(seed_str, n_digits=n_digits), top1)
                    if multiset_overlap(top1, actual_t) >= int(success_k):
                        b = load_boosts()
                        for x, y in pairs:
                            key = f"{x}->{y}"
                            b[key] = b.get(key, 0.0) + 1.0
                        save_boosts(b)
# -----------------------------
# Manual feedback: mark a prediction as correct (positionless)
# -----------------------------
st.subheader("Manual feedback (teach the app a correct hit)")

with st.expander("Mark a past prediction as correct (positionless order)"):
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        seed_input = st.text_input("Seed (previous draw)", value=tuple_to_str(draws[-2]) if len(draws) >= 2 else "")
        actual_input = st.text_input("Actual next draw", value=tuple_to_str(draws[-1]) if len(draws) >= 1 else "")
        threshold_choice = st.selectbox("Success threshold (overlap)", options=[3, 4], index=1)  # default 4-of-4
    with col_m2:
        mode_choice = st.selectbox("Which predictions to validate", ["Paste predictions", "Use current predictions"], index=1)
        pasted_preds = st.text_area("Predictions (comma-separated like 4728,6913,8035)", value="")

    # Prepare prediction list
    preds_for_check: List[str] = []
    if mode_choice == "Use current predictions":
        # Use the predictions we just generated above
        preds_for_check = predictions  # from the Predictions section
    else:
        if pasted_preds.strip():
            preds_for_check = [p.strip() for p in pasted_preds.split(",") if p.strip()]

    if st.button("Mark success (positionless)"):
        try:
            seed_t = to_tuple(seed_input, n_digits=4)
            actual_t = to_tuple(actual_input, n_digits=4)
            if not preds_for_check:
                st.warning("No predictions to check. Paste predictions or select 'Use current predictions'.")
            else:
                overlaps = [multiset_overlap(to_tuple(p, n_digits=4), actual_t) for p in preds_for_check]
                best_overlap = max(overlaps) if overlaps else 0
                success = int(best_overlap >= int(threshold_choice))
                # Write a record to live log
                append_pred_log(
                    {
                        "timestamp": pd.Timestamp.utcnow().isoformat(),
                        "seed": tuple_to_str(seed_t),
                        "preds": preds_for_check,
                        "actual": tuple_to_str(actual_t),
                        "hit": success,
                        "mode": {
                            "note": "manual_feedback",
                            "success_k": int(threshold_choice),
                        },
                    }
                )
                # Optional: add boosts if success (reinforce top-1 that achieved best overlap)
                if enable_boosts and success:
                    # Choose the prediction with best overlap; if many tie, use first
                    best_idx = int(np.argmax(overlaps))
                    top_pred_t = to_tuple(preds_for_check[best_idx], n_digits=4)
                    pairs = greedy_multiset_mapping(seed_t, top_pred_t)
                    b = load_boosts()
                    for x, y in pairs:
                        key = f"{x}->{y}"
                        b[key] = b.get(key, 0.0) + 1.0
                    save_boosts(b)
                if success:
                    st.success(f"Recorded success: best overlap = {best_overlap} (threshold {threshold_choice}).")
                else:
                    st.info(f"Recorded miss: best overlap = {best_overlap} (threshold {threshold_choice}).")
        except Exception as e:
            st.error(f"Could not record feedback: {e}")

    if updates:
        # Rewrite JSONL
        try:
            with open(PRED_LOG_PATH, "w") as f:
                for _, r in log_df.iterrows():
                    f.write(json.dumps({k: (v if not isinstance(v, (np.ndarray,)) else v.tolist()) for k, v in r.to_dict().items()}) + "\n")
        except Exception:
            pass

    resolved = log_df.dropna(subset=["actual"])
    if resolved.empty:
        st.info("No matches yet between saved predictions and a subsequent actual draw in this upload.")
    else:
        total = len(resolved)
        hits = int(resolved["hit"].sum())
        st.metric(f"Live success rate (overlap ≥ {success_k})", f"{(hits/total):.1%}", help=f"{hits}/{total}")
        st.dataframe(resolved.tail(20), use_container_width=True)
        if enable_boosts:
            st.caption("Feedback boosts applied as pseudo-counts after successful top-1 hits.")

# --------------------------------
# Backtest (pattern-based, overlap threshold)
# --------------------------------
st.dataframe(bt[["t","seed","actual","best_pred","best_overlap","best_pos_matches","success"]].tail(25))
st.dataframe(bt[["t","seed","actual","best_pred","best_overlap","best_pos_matches","success"]].tail(25))

# Optional: pick a row to inspect
row_idx = st.number_input("Inspect backtest row index", min_value=int(bt["t"].min()), max_value=int(bt["t"].max()), value=int(bt["t"].max()))
row = bt.loc[bt["t"] == row_idx].iloc[0]
with st.expander(f"Predictions for t={row_idx} (actual {row['actual']})"):
    st.write(row["preds"])

st.subheader("Backtest (pattern-based)")

min_hist = st.number_input("Min history before testing", min_value=20, max_value=2000, value=80, step=10)
run_bt = st.button("Run backtest")

@st.cache_data(show_spinner=True)
def backtest_patterns(draws: List[Tuple[int, ...]],
                      recent_window: int,
                      max_lag: int,
                      lag_weights: List[float],
                      alpha: float,
                      pattern_mode: str,
                      shift_set: List[int],
                      pattern_limit: int,
                      num_preds: int,
                      bonus_pos: float,
                      success_k: int,
                      min_hist: int) -> pd.DataFrame:
    rows = []
    for t in range(int(min_hist), len(draws)):
        history = draws[max(0, t - recent_window):t] if recent_window > 0 else draws[:t]
        if len(history) < 2:
            continue
        # Fit transitions
        trans = Counter()
        for lag, w in zip(range(1, max_lag + 1), lag_weights):
            if w <= 0: continue
            c = extract_digit_transitions(history, lag)
            if w != 1.0: c = Counter({k: v * w for k, v in c.items()})
            trans.update(c)
        probs_bt = normalize_matrix(trans.copy(), alpha=alpha)

        # Build patterns and priors from history up to t
        if pattern_mode == "Hand-crafted":
            pats = all_shift_patterns(n_digits=4, shift_set=shift_set, limit=pattern_limit)
        elif pattern_mode == "Strict family (±5, ±2, 0, ±3)":
            pats = strict_pattern_family(n_digits=4)
            if len(pats) > pattern_limit:
                np.random.seed(42)
                idx = np.random.choice(len(pats), size=pattern_limit, replace=False)
                pats = [pats[i] for i in sorted(idx)]
        else:
            used_lags = [lag for lag, w in zip(range(1, max_lag + 1), lag_weights) if w > 0]
            pats = learn_pattern_multisets(history, used_lags, top_m=pattern_limit)
            if not pats:
                pats = all_shift_patterns(n_digits=4, shift_set=[0,1,-1,2,-2,3,-3,5,-5], limit=min(500, pattern_limit))
        priors_bt = pattern_prior_from_counts(pats, history, [1], alpha_prior=0.2)

        seed_bt = history[-1]
        scored_bt = []
        for p in pats:
            cand = apply_pattern(seed_bt, p)
            prior = priors_bt.get(p, 1.0 / max(1, len(pats)))
            like = 0.0
            for x, y in greedy_multiset_mapping(seed_bt, cand):
                like += math.log(max(probs_bt.get((x, y), 1e-12), 1e-12))
            score = math.log(prior + 1e-12) + like
            scored_bt.append((cand, score, p))
        scored_bt.sort(key=lambda r: r[1], reverse=True)
        preds = [c for c, _, _ in scored_bt[:num_preds]]
        actual = draws[t]

        annot = annotate_preds(pred, actual, bonus_pos=bonus_pos)
best = annot[0] if annot else {"pred": None, "overlap": 0, "pos_matches": 0, "score": 0}
rows.append({
    "t": t,
    "seed": tuple_to_str(seed_bt),
    "actual": tuple_to_str(actual),
    "best_overlap": int(best["overlap"]),
    "best_pos_matches": int(best["pos_matches"]),
    "success": int(best["overlap"] >= int(success_k)),
    "best_pred": best["pred"],
    "best_score": float(best["score"]),
    "pred": [a["pred"] for a in annot],  # keep full list if you still want it})
    return pd.DataFrame(rows)
    if run_bt:bt = backtest_patterns(draws, recent_window, max_lag, list(lag_weights), alpha,
pattern_mode, SHIFT_SET, pattern_limit, num_preds, bonus_pos, int(success_k), int(min_hist))
    if bt.empty:
        st.info("No backtest rows. Increase history or adjust settings.")
    else:
        total = len(bt)
        successes = int(bt["success"].sum())
        st.metric(f"Backtest success (overlap ≥ {success_k})", f"{(successes/total):.1%}", help=f"{successes}/{total}")
        bt["rolling_success"] = bt["success"].rolling(50, min_periods=1).mean()
        st.line_chart(bt.set_index("t")[["rolling_success"]])
        st.dataframe(bt.tail(25), use_container_width=True)
        st.download_button("Download backtest (CSV)", bt.to_csv(index=False).encode("utf-8"), "backtest_pattern.csv", "text/csv")

# --------------------------------
# Notes
# --------------------------------
with st.expander("How it works and tips"):
    st.write(
        "- Training: fits digit transition probabilities from your history with lag weighting and Laplace smoothing.\n"
        "- Patterns: generates or learns shift patterns (per-digit ±k modulo 10). The pattern can vary each draw.\n"
        "- Positionless scoring: success if at least K digits match regardless of order (default K=3). Exact-position matches add a small bonus for ranking.\n"
        "- Learned patterns: uses the most frequent positional shift patterns seen in history to set priors. This helps capture runs where digits move by ±1 for a while (e.g., 1234 → ... → 1235).\n"
        "- Feedback boosts (optional): after live successes, adds small pseudo-counts to the mapped digit transitions to slightly favor recent winning moves.\n"
        "- Guidance: If you believe the key mix is one ±5, one ±2, one 0, one ±3, try 'Strict family'. If patterns drift a lot, use 'Learned from history' or Hand-crafted with {0,±1,±2,±3,±5}."
    )
