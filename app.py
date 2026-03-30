# ───────────────────────────────────────────────────────────────────────────────
# Question Difficulty Ranker — Adaptive Comparative Judgement (ACJ)
# Backend: Supabase  |  Hosting: Streamlit Community Cloud
#
# Supports multiple item sets. Add / remove / disable sets in ITEM_SETS below.
#
# Secrets required (Streamlit Cloud → App Settings → Secrets):
#
# [supabase]
# url = "https://xxxx.supabase.co"
# key = "your-anon-key"
#
# Database migration: run migrate.sql in the Supabase SQL editor before
# deploying this version. It adds an `item_set` column to all three tables
# and tags existing rows as belonging to the first set.
# ───────────────────────────────────────────────────────────────────────────────

import itertools
import os
import random
import datetime
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from supabase import create_client, Client


# ─── Item‑set configuration ──────────────────────────────────────────────────
# Each key is a unique slug stored in the database. To add a new set:
# 1. Create a folder in the repo with the question images.
# 2. Add an entry below pointing "images_dir" at that folder.
# 3. Commit — Streamlit auto-redeploys.
# To disable judging on a set (while keeping its results visible) set
# "enabled": False.

ITEM_SETS = {
    "figures": {
        "label": "Figures",
        "images_dir": "images",
        "enabled": True,
        "description": "Original spatial reasoning question set",
    },
    "folding-and-cutting": {
        "label": "Folding & Cutting",
        "images_dir": "images_fc",
        "enabled": True,
        "description": "Folding and cutting question set",
    },
    "layering": {
        "label": "Layering",
        "images_dir": "images_ly",
        "enabled": True,
        "description": "Layering composite images question set",
    },
    "balancing-mobiles": {
        "label": "Balancing Mobiles",
        "images_dir": "images_bm",
        "enabled": True,
        "description": "Balancing mobiles question set",
    },
}

DEFAULT_TARGET = 100          # default personal goal per judge
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
WARMUP_COUNT = 3              # calibration pairs at the start of a session

TABLE_COMPARISONS = "comparisons"
TABLE_FLAGS = "flags"
TABLE_ENEMY_PAIRS = "enemy_pairs"
TABLE_EXCLUDED_ITEMS = "excluded_items"
AUTO_EXCLUDE_FLAG_THRESHOLD = 3


# ─── Supabase connection ─────────────────────────────────────────────────────
@st.cache_resource
def get_client() -> Client:
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["key"],
    )


# ─── Data access — comparisons ───────────────────────────────────────────────
@st.cache_data(ttl=20)
def load_comparisons(item_set: str) -> pd.DataFrame:
    client = get_client()
    response = (
        client.table(TABLE_COMPARISONS)
        .select("*")
        .eq("item_set", item_set)
        .limit(10000)
        .execute()
    )
    data = response.data
    if not data:
        return pd.DataFrame(
            columns=["id", "judge_name", "winner_id", "loser_id", "created_at", "item_set"]
        )
    return pd.DataFrame(data)


def save_comparison(item_set: str, judge_name: str, winner_id: str, loser_id: str) -> "str | None":
    """Insert a comparison. Returns the row ID on success, None on failure."""
    # ── Duplicate guard ──────────────────────────────────────────────────
    client = get_client()
    existing = (
        client.table(TABLE_COMPARISONS)
        .select("id")
        .eq("item_set", item_set)
        .eq("judge_name", judge_name)
        .eq("winner_id", winner_id)
        .eq("loser_id", loser_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        # Already recorded — silently skip
        load_comparisons.clear()
        return existing.data[0]["id"]

    # Also check the reverse direction (same pair, different winner)
    existing_rev = (
        client.table(TABLE_COMPARISONS)
        .select("id")
        .eq("item_set", item_set)
        .eq("judge_name", judge_name)
        .eq("winner_id", loser_id)
        .eq("loser_id", winner_id)
        .limit(1)
        .execute()
    )
    if existing_rev.data:
        load_comparisons.clear()
        return existing_rev.data[0]["id"]

    row_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow().isoformat()
    response = client.table(TABLE_COMPARISONS).insert({
        "id": row_id,
        "judge_name": judge_name,
        "winner_id": winner_id,
        "loser_id": loser_id,
        "created_at": timestamp,
        "item_set": item_set,
    }).execute()
    if hasattr(response, "error") and response.error:
        st.error(f"Failed to save comparison: {response.error}")
        return None
    load_comparisons.clear()
    return row_id


def delete_comparison(row_id: str) -> bool:
    """Delete a single comparison by ID (used for undo)."""
    client = get_client()
    response = (
        client.table(TABLE_COMPARISONS)
        .delete()
        .eq("id", row_id)
        .execute()
    )
    if hasattr(response, "error") and response.error:
        st.error(f"Failed to undo: {response.error}")
        return False
    load_comparisons.clear()
    return True


def count_judge_comparisons(item_set: str, judge_name: str) -> int:
    df = load_comparisons(item_set)
    if df.empty:
        return 0
    return int((df["judge_name"] == judge_name).sum())


# ─── Data access — flags ─────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_flags(item_set: str) -> pd.DataFrame:
    client = get_client()
    response = (
        client.table(TABLE_FLAGS)
        .select("*")
        .eq("item_set", item_set)
        .limit(10000)
        .execute()
    )
    data = response.data
    if not data:
        return pd.DataFrame(
            columns=["id", "judge_name", "item_id", "created_at", "item_set"]
        )
    return pd.DataFrame(data)


def save_flag(item_set: str, judge_name: str, item_id: str) -> bool:
    # ── Duplicate guard ──────────────────────────────────────────────────
    client = get_client()
    existing = (
        client.table(TABLE_FLAGS)
        .select("id")
        .eq("item_set", item_set)
        .eq("judge_name", judge_name)
        .eq("item_id", item_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        return False

    row_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow().isoformat()
    response = client.table(TABLE_FLAGS).insert({
        "id": row_id,
        "judge_name": judge_name,
        "item_id": item_id,
        "created_at": timestamp,
        "item_set": item_set,
    }).execute()
    if hasattr(response, "error") and response.error:
        st.error(f"Failed to save flag: {response.error}")
        return False
    load_flags.clear()
    return True


# ─── Data access — enemy pairs ───────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_enemy_pairs(item_set: str) -> pd.DataFrame:
    client = get_client()
    response = (
        client.table(TABLE_ENEMY_PAIRS)
        .select("*")
        .eq("item_set", item_set)
        .limit(10000)
        .execute()
    )
    data = response.data
    if not data:
        return pd.DataFrame(
            columns=["id", "item_a", "item_b", "reported_by", "created_at", "item_set"]
        )
    return pd.DataFrame(data)


def save_enemy_pair(item_set: str, judge_name: str, item_x: str, item_y: str) -> bool:
    item_a, item_b = sorted([item_x, item_y])
    client = get_client()
    existing = (
        client.table(TABLE_ENEMY_PAIRS)
        .select("id")
        .eq("item_set", item_set)
        .eq("item_a", item_a)
        .eq("item_b", item_b)
        .limit(1)
        .execute()
    )
    if existing.data:
        return False

    row_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow().isoformat()
    response = client.table(TABLE_ENEMY_PAIRS).insert({
        "id": row_id,
        "item_a": item_a,
        "item_b": item_b,
        "reported_by": judge_name,
        "created_at": timestamp,
        "item_set": item_set,
    }).execute()
    if hasattr(response, "error") and response.error:
        st.error(f"Failed to save enemy pair: {response.error}")
        return False
    load_enemy_pairs.clear()
    return True


# ─── Data access — excluded items ────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_excluded_items(item_set: str) -> pd.DataFrame:
    """Load manually excluded items from the excluded_items table."""
    client = get_client()
    response = (
        client.table(TABLE_EXCLUDED_ITEMS)
        .select("*")
        .eq("item_set", item_set)
        .limit(10000)
        .execute()
    )
    data = response.data
    if not data:
        return pd.DataFrame(
            columns=["id", "item_id", "item_set", "excluded_by", "reason", "created_at"]
        )
    return pd.DataFrame(data)


def save_excluded_item(item_set: str, item_id: str, excluded_by: str, reason: str = "") -> bool:
    """Manually exclude an item. Returns True on success."""
    client = get_client()
    existing = (
        client.table(TABLE_EXCLUDED_ITEMS)
        .select("id")
        .eq("item_set", item_set)
        .eq("item_id", item_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        return False  # already excluded

    row_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow().isoformat()
    response = client.table(TABLE_EXCLUDED_ITEMS).insert({
        "id": row_id,
        "item_id": item_id,
        "item_set": item_set,
        "excluded_by": excluded_by,
        "reason": reason,
        "created_at": timestamp,
    }).execute()
    if hasattr(response, "error") and response.error:
        st.error(f"Failed to exclude item: {response.error}")
        return False
    load_excluded_items.clear()
    return True


def remove_excluded_item(item_set: str, item_id: str) -> bool:
    """Re-include a manually excluded item. Returns True on success."""
    client = get_client()
    response = (
        client.table(TABLE_EXCLUDED_ITEMS)
        .delete()
        .eq("item_set", item_set)
        .eq("item_id", item_id)
        .execute()
    )
    if hasattr(response, "error") and response.error:
        st.error(f"Failed to re-include item: {response.error}")
        return False
    load_excluded_items.clear()
    return True


def get_excluded_item_ids(item_set: str) -> set:
    """
    Get the full set of excluded item IDs for an item set.
    Combines:
    - Auto-excluded: items flagged by >= AUTO_EXCLUDE_FLAG_THRESHOLD different judges
      (unless overridden via an "override_include" entry in excluded_items)
    - Manually excluded: items in the excluded_items table (reason != "override_include")
    """
    excluded = set()
    overrides = set()

    # Manual exclusions and overrides
    manual_df = load_excluded_items(item_set)
    if not manual_df.empty:
        overrides = set(
            manual_df[manual_df["reason"] == "override_include"]["item_id"].tolist()
        )
        manual_excluded = set(
            manual_df[manual_df["reason"] != "override_include"]["item_id"].tolist()
        )
        excluded |= manual_excluded

    # Auto-exclude from flags (minus overrides)
    flags_df = load_flags(item_set)
    if not flags_df.empty:
        flag_counts = flags_df.groupby("item_id")["judge_name"].nunique()
        auto_excluded = set(flag_counts[flag_counts >= AUTO_EXCLUDE_FLAG_THRESHOLD].index)
        excluded |= (auto_excluded - overrides)

    return excluded


# ─── Question loading & metadata ─────────────────────────────────────────────
@st.cache_data
def load_question_ids(images_dir: str) -> list:
    if not os.path.exists(images_dir):
        return []
    return sorted(
        p.stem
        for p in Path(images_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_image_path(images_dir: str, question_id: str):
    for ext in IMAGE_EXTENSIONS:
        path = Path(images_dir) / f"{question_id}{ext}"
        if path.exists():
            return str(path)
    return None


def parse_item_metadata(question_id: str) -> dict:
    """
    Parse metadata from the filename stem. Handles three conventions:
    Figures:         item_1.01_EASY_ROTATED → batch=1, difficulty=EASY, orientation=ROTATED
    Folding & Cutting: item_1dtf.01_corner_2cuts → fold_type=1dtf, cut_position=corner, cuts=2
    Layering:        lc_001_typ-typ_color_s5 → prefix=lc, item_number=001, layer_type=typ-typ, style=color, shapes=5
    """
    meta = {
        "batch": "—",
        "difficulty": "—",
        "orientation": "—",
        "fold_type": "—",
        "cut_position": "—",
        "cuts": "—",
        "layer_prefix": "—",
        "layer_type": "—",
        "style": "—",
        "shapes": "—",
        "bm_batch": "—",
        "bm_branches": "—",
        "bm_difficulty": "—",
        "item_number": question_id,
    }
    parts = question_id.split("_")

    # ── Balancing Mobiles format: item{N}bm.{num}_{branches}_{difficulty}
    #    or item{N}bm.{num}_diff_shape_{branches}_{difficulty} (batch 3)
    #    parts[0] = "item1bm.01", so split on "." to get prefix and item number
    import re
    prefix_dot = parts[0].split(".")
    bm_match = re.match(r'^item(\d)bm$', prefix_dot[0]) if prefix_dot else None
    if bm_match:
        meta["bm_batch"] = bm_match.group(1)
        meta["item_number"] = prefix_dot[1] if len(prefix_dot) > 1 else "—"
        # batch 3 has extra "diff_shape" tokens: parts = [item3bm.01, diff, shape, 3, 9]
        if len(parts) >= 5 and parts[1] == "diff" and parts[2] == "shape":
            meta["bm_branches"] = parts[3]
            meta["bm_difficulty"] = parts[4]
        else:
            meta["bm_branches"] = parts[1] if len(parts) > 1 else "—"
            meta["bm_difficulty"] = parts[2] if len(parts) > 2 else "—"
        return meta

    # ── Layering format: {prefix}_{number}_{type}_{style}_{sN} ────────
    if len(parts) == 5 and parts[0] in ("lc", "lr", "lu"):
        meta["layer_prefix"] = parts[0]
        meta["item_number"] = parts[1]
        meta["layer_type"] = parts[2]
        meta["style"] = parts[3]
        meta["shapes"] = parts[4].replace("s", "") if parts[4].startswith("s") else parts[4]
        return meta

    # ── Figures / Folding & Cutting (item_ prefix) ────────────────────
    if len(parts) < 3 or parts[0] != "item":
        return meta

    num_str = parts[1]  # e.g. "1.01" or "1dtf.01"

    # Detect Folding & Cutting format (contains letters in the prefix)
    prefix = num_str.split(".")[0] if "." in num_str else num_str
    if any(c.isalpha() for c in prefix):
        # Folding & Cutting: item_{fold_type}.{num}_{cut_position}_{N}cuts
        meta["fold_type"] = prefix
        meta["item_number"] = num_str
        meta["cut_position"] = parts[2] if len(parts) > 2 else "—"
        if len(parts) > 3 and parts[3].endswith("cuts"):
            meta["cuts"] = parts[3].replace("cuts", "")
    else:
        # Figures: item_{number}_{DIFFICULTY}_{ORIENTATION}
        meta["item_number"] = num_str
        try:
            meta["batch"] = str(int(float(num_str)))
        except ValueError:
            pass
        meta["difficulty"] = parts[2] if len(parts) > 2 else "—"
        meta["orientation"] = parts[3] if len(parts) > 3 else "—"

    return meta


# ─── Bradley‑Terry ranking ───────────────────────────────────────────────────
def compute_rankings(item_set: str, question_ids: list) -> tuple:
    """Returns (scores_dict, comp_counts_dict)."""
    df = load_comparisons(item_set)
    comp_counts = {q: 0 for q in question_ids}
    if df.empty:
        return {q: 0.0 for q in question_ids}, comp_counts

    q_set = set(question_ids)
    df = df[df["winner_id"].isin(q_set) & df["loser_id"].isin(q_set)]

    # Vectorised comparison counts
    winner_counts = df["winner_id"].value_counts()
    loser_counts = df["loser_id"].value_counts()
    for q in question_ids:
        comp_counts[q] = int(winner_counts.get(q, 0) + loser_counts.get(q, 0))

    try:
        import choix

        q_list = sorted(question_ids)
        idx = {q: i for i, q in enumerate(q_list)}
        winners = df["winner_id"].map(idx)
        losers = df["loser_id"].map(idx)
        data = list(zip(winners.tolist(), losers.tolist()))
        if len(data) < 2:
            raise ValueError("Not enough data for BT model.")
        params = choix.ilsr_pairwise(len(q_list), data, alpha=0.01)
        scores = {q_list[i]: float(params[i]) for i in range(len(q_list))}
    except Exception:
        # Fallback: normalised win rate
        wins = df["winner_id"].value_counts()
        totals = pd.concat([df["winner_id"], df["loser_id"]]).value_counts()
        scores = {
            q: float(wins.get(q, 0)) / max(int(totals.get(q, 1)), 1)
            for q in question_ids
        }

    return scores, comp_counts


# ─── Bootstrap confidence intervals ──────────────────────────────────────────
@st.cache_data(ttl=600)
def compute_bootstrap_cis(
    _df: pd.DataFrame, question_ids: tuple, n_resamples: int = 500
) -> dict:
    """
    Bootstrap 95 % CIs for each item's rank.
    Returns: question_id → (rank_lo, rank_median, rank_hi)
    """
    import choix
    import numpy as np

    df = _df
    active = set(df["winner_id"]) | set(df["loser_id"])
    q_list = [q for q in question_ids if q in active]

    if len(q_list) < 2 or df.empty:
        return {}

    idx = {q: i for i, q in enumerate(q_list)}
    winners = df["winner_id"].map(idx)
    losers = df["loser_id"].map(idx)
    mask = winners.notna() & losers.notna()
    data = list(zip(winners[mask].astype(int).tolist(), losers[mask].astype(int).tolist()))
    if len(data) < 2:
        return {}

    rank_lists = {q: [] for q in q_list}
    rng = np.random.default_rng(42)

    for _ in range(n_resamples):
        indices = rng.integers(0, len(data), len(data))
        sample = [data[i] for i in indices]
        try:
            params = choix.ilsr_pairwise(len(q_list), sample, alpha=0.01)
            order = sorted(range(len(q_list)), key=lambda i: -params[i])
            for rank, qi in enumerate(order, start=1):
                rank_lists[q_list[qi]].append(rank)
        except Exception:
            continue

    cis = {}
    for q in q_list:
        ranks = sorted(rank_lists[q])
        if ranks:
            n = len(ranks)
            lo = ranks[max(0, int(0.025 * n))]
            hi = ranks[min(n - 1, int(0.975 * n))]
            med = ranks[n // 2]
            cis[q] = (lo, med, hi)
    return cis


# ─── Rank stability score ────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def compute_stability_score(
    _df: pd.DataFrame, question_ids: tuple
) -> "float | None":
    """Split‑half Spearman ρ as a ranking stability indicator."""
    import choix
    import numpy as np
    from scipy import stats as scipy_stats

    df = _df
    active = set(df["winner_id"]) | set(df["loser_id"])
    q_list = [q for q in question_ids if q in active]
    if len(q_list) < 4 or len(df) < 20:
        return None

    idx = {q: i for i, q in enumerate(q_list)}
    winners = df["winner_id"].map(idx)
    losers = df["loser_id"].map(idx)
    mask = winners.notna() & losers.notna()
    data = list(zip(winners[mask].astype(int).tolist(), losers[mask].astype(int).tolist()))
    if len(data) < 20:
        return None

    rng = np.random.default_rng(99)
    shuffled = list(data)
    rng.shuffle(shuffled)
    mid = len(shuffled) // 2

    try:
        params_a = choix.ilsr_pairwise(len(q_list), shuffled[:mid], alpha=0.01)
        params_b = choix.ilsr_pairwise(len(q_list), shuffled[mid:], alpha=0.01)
        corr, _ = scipy_stats.spearmanr(params_a, params_b)
        return float(corr)
    except Exception:
        return None


# ─── Adaptive pair selection ──────────────────────────────────────────────────
def _build_valid_pool(
    question_ids: list, excluded: set
) -> list:
    """
    Generate all valid candidate pairs using itertools.combinations,
    minus any pairs in the excluded set.
    """
    pool = [
        (a, b)
        for a, b in itertools.combinations(question_ids, 2)
        if frozenset([a, b]) not in excluded
    ]
    return pool


def select_next_pair(
    item_set: str,
    question_ids: list,
    judge_name: str,
    scores: dict,
) -> tuple:
    """
    Pick the most informative unjudged pair for this judge.
    Left/right order is randomised to avoid position bias.
    """
    df = load_comparisons(item_set)
    judge_df = df[df["judge_name"] == judge_name] if not df.empty else pd.DataFrame()

    judged = set()
    if not judge_df.empty:
        judged = {
            frozenset([r.winner_id, r.loser_id])
            for r in judge_df.itertuples()
        }

    enemy_df = load_enemy_pairs(item_set)
    enemies = set()
    if not enemy_df.empty:
        enemies = {
            frozenset([r.item_a, r.item_b])
            for r in enemy_df.itertuples()
        }

    excluded = judged | enemies
    pool = _build_valid_pool(question_ids, excluded)

    if not pool:
        # All pairs exhausted — pick a random pair anyway
        i, j = random.sample(range(len(question_ids)), 2)
        chosen = (question_ids[i], question_ids[j])
    else:
        # CI‑weighted selection when bootstrap data is available
        cis = compute_bootstrap_cis(df, tuple(question_ids))
        if cis:
            ci_widths = {q: (cis[q][2] - cis[q][0]) for q in cis}
            max_width = max(ci_widths.values(), default=10)
            weights = [
                ci_widths.get(a, max_width) + ci_widths.get(b, max_width)
                for a, b in pool
            ]
            total = sum(weights)
            chosen = (
                random.choices(pool, weights=weights, k=1)[0]
                if total > 0
                else random.choice(pool)
            )
        elif scores:
            # Fall back to score‑proximity (most uncertain region)
            pool.sort(
                key=lambda p: abs(scores.get(p[0], 0.0) - scores.get(p[1], 0.0))
            )
            top_n = max(1, len(pool) // 10)
            chosen = random.choice(pool[:top_n])
        else:
            chosen = random.choice(pool)

    # Randomise left/right
    return chosen if random.random() < 0.5 else (chosen[1], chosen[0])


def select_warmup_pair(
    item_set: str,
    question_ids: list,
    judge_name: str,
) -> "tuple | None":
    """
    For calibration: pick one very‑hard item and one very‑easy item
    based on existing scores from other judges.
    Returns None if not enough data.
    """
    df = load_comparisons(item_set)
    if df.empty or len(question_ids) < 2:
        return None

    scores, _ = compute_rankings(item_set, question_ids)
    if not scores or all(v == 0.0 for v in scores.values()):
        return None

    # Avoid questions already compared by this judge
    judge_df = df[df["judge_name"] == judge_name]
    compared = set()
    if not judge_df.empty:
        compared = (
            set(judge_df["winner_id"]) | set(judge_df["loser_id"])
        )

    available = [q for q in question_ids if q not in compared]
    if len(available) < 2:
        return None

    available_scores = {q: scores.get(q, 0.0) for q in available}
    easy = min(available_scores, key=available_scores.get)
    hard = max(available_scores, key=available_scores.get)
    if easy == hard:
        return None
    return (easy, hard) if random.random() < 0.5 else (hard, easy)


# ─── Page: Judging ────────────────────────────────────────────────────────────
def page_judging(item_set: str, cfg: dict, question_ids: list):
    images_dir = cfg["images_dir"]

    # ── Filter out excluded items ─────────────────────────────────────────
    excluded_ids = get_excluded_item_ids(item_set)
    active_question_ids = [q for q in question_ids if q not in excluded_ids]

    if not active_question_ids:
        st.error(
            f"No active questions in this set. "
            "All questions are either missing or excluded."
        )
        return

    if not question_ids:
        st.error(
            f"No images found in `{images_dir}/`. "
            "Add question images and restart."
        )
        return

    # ── Login ─────────────────────────────────────────────────────────────
    if not st.session_state.get("judge_name"):
        st.markdown("### Welcome")
        st.markdown(
            "You'll be shown pairs of questions. "
            "For each pair, click **the button under whichever question you "
            "think is harder**."
        )
        st.markdown("Enter your name or judge ID to begin.")
        name = st.text_input("Your name / judge ID")
        if st.button("Start judging", type="primary") and name.strip():
            st.session_state["judge_name"] = name.strip()
            st.session_state.pop("current_pair", None)
            st.session_state.pop("last_comparison_id", None)
            st.session_state.setdefault("personal_target", DEFAULT_TARGET)
            st.session_state["warmup_done"] = 0
            st.rerun()
        return

    judge = st.session_state["judge_name"]
    done = count_judge_comparisons(item_set, judge)
    target = st.session_state.get("personal_target", DEFAULT_TARGET)

    # ── Sidebar status ────────────────────────────────────────────────────
    st.sidebar.markdown(f"**Judge:** {judge}")
    st.sidebar.markdown(f"**Item set:** {cfg['label']}")

    new_target = st.sidebar.number_input(
        "Your personal target",
        min_value=10,
        max_value=5000,
        value=target,
        step=10,
        key="target_input",
    )
    if new_target != target:
        st.session_state["personal_target"] = new_target
        target = new_target

    st.sidebar.progress(min(done / max(target, 1), 1.0))
    st.sidebar.markdown(f"**{done}** / **{target}** comparisons")

    if done >= target:
        st.sidebar.success("Target reached! You can stop or keep going.")

    if st.sidebar.button("Switch judge / set"):
        st.session_state.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"{len(active_question_ids)} active questions ({len(excluded_ids)} excluded)"
    )

    # ── Main‑area progress bar ────────────────────────────────────────────
    prog_col, count_col = st.columns([4, 1])
    with prog_col:
        st.progress(min(done / max(target, 1), 1.0))
    with count_col:
        st.markdown(f"**{done} / {target}**")

    # ── Select next pair (warm‑up or adaptive) ────────────────────────────
    if "current_pair" not in st.session_state:
        warmup_done = st.session_state.get("warmup_done", 0)
        if warmup_done < WARMUP_COUNT:
            pair = select_warmup_pair(item_set, active_question_ids, judge)
            if pair:
                st.session_state["current_pair"] = pair
                st.session_state["is_warmup"] = True
            else:
                # Not enough data for warm‑up — go straight to adaptive
                st.session_state["warmup_done"] = WARMUP_COUNT
                scores = (
                    compute_rankings(item_set, active_question_ids)[0]
                    if done >= 20
                    else {}
                )
                st.session_state["current_pair"] = select_next_pair(
                    item_set, active_question_ids, judge, scores
                )
                st.session_state["is_warmup"] = False
        else:
            scores = (
                compute_rankings(item_set, active_question_ids)[0]
                if done >= 20
                else {}
            )
            st.session_state["current_pair"] = select_next_pair(
                item_set, active_question_ids, judge, scores
            )
            st.session_state["is_warmup"] = False

    q_left, q_right = st.session_state["current_pair"]
    is_warmup = st.session_state.get("is_warmup", False)

    # ── Load flags for this judge ─────────────────────────────────────────
    flags_df = load_flags(item_set)
    flagged_by_judge = set()
    if not flags_df.empty:
        flagged_by_judge = set(
            flags_df[flags_df["judge_name"] == judge]["item_id"].tolist()
        )

    # ── Comparison UI ─────────────────────────────────────────────────────
    if is_warmup:
        st.info(
            "**Calibration round** — these pairs span a wide difficulty range "
            "to help you get a feel for the set before adaptive selection begins."
        )

    st.subheader("Which question is more difficult?")
    st.caption("Click the button **below** the question you think is harder.")
    st.markdown("---")

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        if q_left in flagged_by_judge:
            st.caption("Already flagged")
        else:
            _, flag_col_l = st.columns([4, 1])
            with flag_col_l:
                if st.button("Flag", key="flag_left", help="Flag this question as incorrect"):
                    if save_flag(item_set, judge, q_left):
                        st.toast(f"Flagged: {q_left}", icon="🚩")
                        st.rerun()

        img_path = get_image_path(images_dir, q_left)
        if img_path:
            st.image(img_path, use_container_width=True)
        else:
            st.warning(f"Image not found: {q_left}")

        if st.button("This one is harder", key="btn_left", use_container_width=True, type="primary"):
            row_id = save_comparison(item_set, judge, q_left, q_right)
            if row_id:
                st.session_state["last_comparison_id"] = row_id
                st.session_state.pop("current_pair", None)
                if is_warmup:
                    st.session_state["warmup_done"] = (
                        st.session_state.get("warmup_done", 0) + 1
                    )
                st.rerun()

    with col_r:
        if q_right in flagged_by_judge:
            st.caption("Already flagged")
        else:
            _, flag_col_r = st.columns([4, 1])
            with flag_col_r:
                if st.button("Flag", key="flag_right", help="Flag this question as incorrect"):
                    if save_flag(item_set, judge, q_right):
                        st.toast(f"Flagged: {q_right}", icon="🚩")
                        st.rerun()

        img_path = get_image_path(images_dir, q_right)
        if img_path:
            st.image(img_path, use_container_width=True)
        else:
            st.warning(f"Image not found: {q_right}")

        if st.button("This one is harder", key="btn_right", use_container_width=True, type="primary"):
            row_id = save_comparison(item_set, judge, q_right, q_left)
            if row_id:
                st.session_state["last_comparison_id"] = row_id
                st.session_state.pop("current_pair", None)
                if is_warmup:
                    st.session_state["warmup_done"] = (
                        st.session_state.get("warmup_done", 0) + 1
                    )
                st.rerun()

    st.markdown("---")

    skip_col, enemy_col, undo_col = st.columns(3)
    with skip_col:
        if st.button("Too close to call — skip", key="btn_skip",
                      use_container_width=True,
                      help="Move on without recording a judgement."):
            st.session_state.pop("current_pair", None)
            st.rerun()

    with enemy_col:
        if st.button("Mark as enemies", key="btn_enemy",
                      use_container_width=True,
                      help="Mark these two items as too similar to meaningfully compare."):
            if save_enemy_pair(item_set, judge, q_left, q_right):
                st.session_state.pop("current_pair", None)
                st.toast(f"Enemy pair marked: {q_left} ↔ {q_right}", icon="🔗")
                st.rerun()
            else:
                st.warning("Enemy pair already marked.")

    with undo_col:
        last_id = st.session_state.get("last_comparison_id")
        if last_id:
            if st.button("Undo last comparison", key="btn_undo",
                          use_container_width=True):
                if delete_comparison(last_id):
                    st.toast("Last comparison undone", icon="↩️")
                    st.session_state.pop("last_comparison_id", None)
                    st.session_state.pop("current_pair", None)
                    st.rerun()
        else:
            st.button("Undo last comparison", key="btn_undo",
                       use_container_width=True, disabled=True)


# ─── Page: Results ────────────────────────────────────────────────────────────
def page_results(item_set: str, cfg: dict, question_ids: list):
    st.title(f"Difficulty Rankings — {cfg['label']}")
    images_dir = cfg["images_dir"]

    # ── Filter out excluded items ─────────────────────────────────────────
    excluded_ids = get_excluded_item_ids(item_set)
    active_question_ids = [q for q in question_ids if q not in excluded_ids]

    if not active_question_ids:
        st.warning("No active questions in this set.")
        return

    if not question_ids:
        st.warning("No questions loaded.")
        return

    df = load_comparisons(item_set)
    if df.empty:
        st.info("No comparisons recorded yet.")
        return

    scores, comp_counts = compute_rankings(item_set, active_question_ids)
    if all(v == 0.0 for v in scores.values()):
        st.info("Not enough comparisons yet — complete a few more rounds first.")
        return

    # ── Stability score ───────────────────────────────────────────────────
    stability = compute_stability_score(df, tuple(active_question_ids))
    if stability is not None:
        if stability >= 0.85:
            emoji, verdict = "🟢", "High — positions are well settled."
        elif stability >= 0.70:
            emoji, verdict = "🟡", "Moderate — middle ranks still uncertain."
        else:
            emoji, verdict = "🔴", "Low — more comparisons needed."
        st.metric(
            label="Ranking Stability (split‑half Spearman ρ)",
            value=f"{stability:.2f}",
            help=(
                "Compares BT scores from two random halves of all comparisons. "
                "≥ 0.85 = high · 0.70–0.85 = moderate · < 0.70 = low."
            ),
        )
        st.caption(f"{emoji} {verdict}")

    # ── Bootstrap CIs ─────────────────────────────────────────────────────
    with st.spinner("Computing bootstrap confidence intervals..."):
        cis = compute_bootstrap_cis(df, tuple(active_question_ids))

    # ── Rankings table with metadata ──────────────────────────────────────
    sorted_qs = sorted(scores, key=lambda q: scores[q], reverse=True)
    rank_map = {q: i + 1 for i, q in enumerate(sorted_qs)}

    def ci_str(q: str) -> str:
        if q not in cis:
            return "—"
        lo, _med, hi = cis[q]
        r = rank_map[q]
        margin = max(r - lo, hi - r)
        return f"±{margin}" if margin > 0 else "< ±1"

    rows = []
    for rank, q in enumerate(sorted_qs, 1):
        meta = parse_item_metadata(q)
        row = {
            "Rank": rank,
            "Question ID": q,
            "Difficulty Score": round(scores[q], 4),
            "95% CI": ci_str(q),
            "Comparisons": comp_counts.get(q, 0),
        }
        # Add set-appropriate metadata columns
        if meta["bm_batch"] != "—":
            row["Batch"] = meta["bm_batch"]
            row["Branches"] = meta["bm_branches"]
            row["Design Difficulty"] = meta["bm_difficulty"]
        elif meta["layer_prefix"] != "—":
            row["Prefix"] = meta["layer_prefix"]
            row["Layer Type"] = meta["layer_type"]
            row["Style"] = meta["style"]
            row["Shapes"] = meta["shapes"]
        elif meta["fold_type"] != "—":
            row["Fold Type"] = meta["fold_type"]
            row["Cut Position"] = meta["cut_position"]
            row["Cuts"] = meta["cuts"]
        else:
            row["Batch"] = meta["batch"]
            row["Design Difficulty"] = meta["difficulty"]
            row["Orientation"] = meta["orientation"]
        rows.append(row)

    results_df = pd.DataFrame(rows)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    total = len(df)
    avg_per_item = (total * 2) / max(len(active_question_ids), 1)
    st.caption(
        f"Total comparisons: **{total}** · "
        f"Avg per question: **{avg_per_item:.1f}** · "
        f"Questions with 0 comparisons: "
        f"**{sum(1 for q in active_question_ids if comp_counts.get(q, 0) == 0)}**"
    )

    st.download_button(
        label="Download rankings as CSV",
        data=results_df.to_csv(index=False),
        file_name=f"difficulty_rankings_{item_set}.csv",
        mime="text/csv",
    )

    # ── Priority judging list ─────────────────────────────────────────────
    if cis:
        priority = sorted(
            [(q, cis[q][2] - cis[q][0]) for q in cis if cis[q][2] > cis[q][0]],
            key=lambda x: -x[1],
        )[:10]
        if priority:
            with st.expander("Priority judging list — questions with widest CIs"):
                priority_df = pd.DataFrame({
                    "Question ID": [q for q, _ in priority],
                    "Current Rank": [rank_map.get(q, "—") for q, _ in priority],
                    "CI Width (ranks)": [w for _, w in priority],
                    "Comparisons": [comp_counts.get(q, 0) for q, _ in priority],
                })
                st.dataframe(priority_df, use_container_width=True, hide_index=True)
                st.caption(
                    "These questions have the most uncertain rank positions. "
                    "Prioritising them will tighten the overall ranking fastest."
                )

    # ── Per‑judge breakdown ───────────────────────────────────────────────
    st.subheader("Comparisons per Judge")
    judge_counts = (
        df.groupby("judge_name")
        .size()
        .reset_index(name="Comparisons")
        .rename(columns={"judge_name": "Judge"})
        .sort_values("Comparisons", ascending=False)
    )
    st.dataframe(judge_counts, use_container_width=True, hide_index=True)

    with st.expander("Download raw comparison data"):
        st.download_button(
            label="Download all comparisons as CSV",
            data=df.to_csv(index=False),
            file_name=f"raw_comparisons_{item_set}.csv",
            mime="text/csv",
        )

    # ── Excluded items summary ────────────────────────────────────────────
    if excluded_ids:
        st.subheader(f"Excluded Items ({len(excluded_ids)})")
        st.caption(
            "These items are excluded from pair selection and rankings. "
            "Review and manage exclusions on the Review Flags page."
        )
        excluded_list = sorted(excluded_ids)
        excluded_df = pd.DataFrame({"Item ID": excluded_list})
        st.dataframe(excluded_df, use_container_width=True, hide_index=True)

    # ── Flagged items ──────────────────────────────────────────────────────
    flags_df = load_flags(item_set)
    if not flags_df.empty:
        st.subheader("Flagged Items")
        flag_summary = (
            flags_df
            .groupby("item_id")
            .agg({
                "judge_name": ["count", lambda x: ", ".join(sorted(set(x)))]
            })
            .reset_index()
        )
        flag_summary.columns = ["item_id", "flag_count", "flagged_by"]
        flag_summary = (
            flag_summary
            .rename(columns={
                "item_id": "Item ID",
                "flag_count": "Times Flagged",
                "flagged_by": "Flagged By",
            })
            .sort_values("Times Flagged", ascending=False)
        )
        st.dataframe(flag_summary, use_container_width=True, hide_index=True)

        with st.expander("Download raw flag data"):
            st.download_button(
                label="Download all flags as CSV",
                data=flags_df.to_csv(index=False),
                file_name=f"raw_flags_{item_set}.csv",
                mime="text/csv",
            )

    # ── Enemy pairs ───────────────────────────────────────────────────────
    st.subheader("Enemy Pairs (too similar to compare)")
    enemy_df = load_enemy_pairs(item_set)
    if enemy_df.empty:
        st.info("No enemy pairs have been logged yet.")
    else:
        enemy_display = (
            enemy_df[["item_a", "item_b", "reported_by", "created_at"]]
            .rename(columns={
                "item_a": "Item A",
                "item_b": "Item B",
                "reported_by": "Reported By",
                "created_at": "Logged At",
            })
            .sort_values("Logged At", ascending=False)
        )
        st.caption(
            f"{len(enemy_df)} pair(s) permanently excluded from judging."
        )
        st.dataframe(enemy_display, use_container_width=True, hide_index=True)

        with st.expander("Download enemy pairs data"):
            st.download_button(
                label="Download enemy pairs as CSV",
                data=enemy_df.to_csv(index=False),
                file_name=f"enemy_pairs_{item_set}.csv",
                mime="text/csv",
            )


# ─── Page: Review Flags ───────────────────────────────────────────────────────
def page_review_flags(item_set: str, cfg: dict, question_ids: list):
    st.title(f"Review Flags — {cfg['label']}")
    images_dir = cfg["images_dir"]

    flags_df = load_flags(item_set)
    excluded_ids = get_excluded_item_ids(item_set)
    manual_df = load_excluded_items(item_set)
    manual_excluded_ids = set(
        manual_df[manual_df["reason"] != "override_include"]["item_id"].tolist()
    ) if not manual_df.empty else set()
    override_included_ids = set(
        manual_df[manual_df["reason"] == "override_include"]["item_id"].tolist()
    ) if not manual_df.empty else set()

    # Determine auto-excluded items (flagged by 3+ judges)
    auto_excluded_ids = set()
    flag_counts_by_item = {}
    flagged_by_map = {}
    if not flags_df.empty:
        flag_counts_by_item = flags_df.groupby("item_id")["judge_name"].nunique().to_dict()
        flagged_by_map = flags_df.groupby("item_id")["judge_name"].apply(
            lambda x: ", ".join(sorted(set(x)))
        ).to_dict()
        auto_excluded_ids = {
            item for item, count in flag_counts_by_item.items()
            if count >= AUTO_EXCLUDE_FLAG_THRESHOLD
        }

    # Items that hit the threshold but have been overridden
    overridden_ids = auto_excluded_ids & override_included_ids
    # Items actually excluded by auto-threshold (not overridden)
    active_auto_excluded = auto_excluded_ids - override_included_ids

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Flagged", len(flag_counts_by_item))
    with col2:
        st.metric(f"Auto-Excluded ({AUTO_EXCLUDE_FLAG_THRESHOLD}+ flags)", len(active_auto_excluded))
    with col3:
        st.metric("Manually Excluded", len(manual_excluded_ids - auto_excluded_ids))
    with col4:
        st.metric("Total Excluded", len(excluded_ids))

    if not flag_counts_by_item and not manual_excluded_ids:
        st.info("No items have been flagged or excluded yet.")
        return

    st.markdown("---")

    # ── Auto-excluded items (3+ flags) ────────────────────────────────────
    if auto_excluded_ids:
        st.subheader(f"Auto-Excluded — flagged by {AUTO_EXCLUDE_FLAG_THRESHOLD}+ judges ({len(auto_excluded_ids)})")
        st.caption(
            "These items were automatically excluded because multiple judges flagged them. "
            "You can override and re-include them if the flag was unwarranted."
        )
        for item_id in sorted(auto_excluded_ids):
            is_overridden = item_id in overridden_ids
            with st.container(border=True):
                img_col, info_col, action_col = st.columns([2, 2, 1])
                with img_col:
                    img_path = get_image_path(images_dir, item_id)
                    if img_path:
                        st.image(img_path, use_container_width=True)
                    else:
                        st.warning(f"Image not found: {item_id}")
                with info_col:
                    st.markdown(f"**{item_id}**")
                    st.caption(f"Flagged {flag_counts_by_item.get(item_id, 0)} time(s)")
                    st.caption(f"By: {flagged_by_map.get(item_id, '—')}")
                    if is_overridden:
                        st.success("Overridden — currently included")
                    else:
                        st.warning("Currently excluded")
                with action_col:
                    if is_overridden:
                        if st.button("Re-exclude", key=f"reexclude_{item_id}",
                                     use_container_width=True):
                            remove_excluded_item(item_set, item_id)
                            st.toast(f"Re-excluded: {item_id}", icon="🚫")
                            st.rerun()
                    else:
                        if st.button("Re-include", key=f"override_{item_id}",
                                     use_container_width=True, type="primary"):
                            save_excluded_item(item_set, item_id, "admin", "override_include")
                            st.toast(f"Override: {item_id} re-included", icon="✅")
                            st.rerun()

    st.markdown("---")

    # ── Single-flag items (need review) ───────────────────────────────────
    single_flag_items = {
        item for item, count in flag_counts_by_item.items()
        if count < AUTO_EXCLUDE_FLAG_THRESHOLD
    }
    if single_flag_items:
        st.subheader(f"Needs Review — below auto-exclude threshold ({len(single_flag_items)})")
        st.caption(
            f"These items were flagged fewer than {AUTO_EXCLUDE_FLAG_THRESHOLD} times. "
            "Review the image and decide whether to exclude it from the ranking."
        )
        for item_id in sorted(single_flag_items):
            with st.container(border=True):
                img_col, info_col, action_col = st.columns([2, 2, 1])
                with img_col:
                    img_path = get_image_path(images_dir, item_id)
                    if img_path:
                        st.image(img_path, use_container_width=True)
                    else:
                        st.warning(f"Image not found: {item_id}")
                with info_col:
                    st.markdown(f"**{item_id}**")
                    st.caption(f"Flagged by: {flagged_by_map.get(item_id, '—')}")
                    if item_id in manual_excluded_ids:
                        st.success("Currently excluded")
                    else:
                        st.caption("Currently active")
                with action_col:
                    if item_id in manual_excluded_ids:
                        if st.button("Re-include", key=f"include_{item_id}",
                                     use_container_width=True):
                            if remove_excluded_item(item_set, item_id):
                                st.toast(f"Re-included: {item_id}", icon="✅")
                                st.rerun()
                    else:
                        if st.button("Exclude", key=f"exclude_{item_id}",
                                     use_container_width=True, type="primary"):
                            if save_excluded_item(item_set, item_id, "admin", "Confirmed flag"):
                                st.toast(f"Excluded: {item_id}", icon="🚫")
                                st.rerun()

    st.markdown("---")

    # ── Manually excluded (not flagged) ───────────────────────────────────
    manual_only = manual_excluded_ids - set(flag_counts_by_item.keys())
    if manual_only:
        st.subheader(f"Manually Excluded — not flagged ({len(manual_only)})")
        st.caption("These items were manually excluded without being flagged by any judge.")
        for item_id in sorted(manual_only):
            with st.container(border=True):
                img_col, info_col, action_col = st.columns([2, 2, 1])
                with img_col:
                    img_path = get_image_path(images_dir, item_id)
                    if img_path:
                        st.image(img_path, use_container_width=True)
                    else:
                        st.warning(f"Image not found: {item_id}")
                with info_col:
                    st.markdown(f"**{item_id}**")
                    # Show who excluded it and when
                    row = manual_df[manual_df["item_id"] == item_id].iloc[0] if not manual_df.empty else None
                    if row is not None:
                        st.caption(f"Excluded by: {row.get('excluded_by', '—')}")
                        st.caption(f"Reason: {row.get('reason', '—')}")
                with action_col:
                    if st.button("Re-include", key=f"include_manual_{item_id}",
                                 use_container_width=True):
                        if remove_excluded_item(item_set, item_id):
                            st.toast(f"Re-included: {item_id}", icon="✅")
                            st.rerun()

    # ── Bulk exclude by item ID ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("Manually Exclude an Item")
    st.caption("Enter an item ID to exclude it from the ranking, even if it hasn't been flagged.")
    manual_item = st.text_input("Item ID to exclude", key="manual_exclude_input")
    manual_reason = st.text_input("Reason (optional)", key="manual_exclude_reason")
    if st.button("Exclude item", key="btn_manual_exclude", type="primary"):
        if manual_item.strip():
            item_id = manual_item.strip()
            if item_id in set(question_ids):
                if save_excluded_item(item_set, item_id, "admin", manual_reason.strip()):
                    st.toast(f"Excluded: {item_id}", icon="🚫")
                    st.rerun()
                else:
                    st.warning("Item is already excluded.")
            else:
                st.error(f"Item ID '{item_id}' not found in this item set.")
        else:
            st.warning("Please enter an item ID.")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Difficulty Ranker",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        get_client()
    except Exception as e:
        st.error(
            "Could not connect to Supabase. "
            f"Check your secrets are configured correctly. Error: {e}"
        )
        st.stop()

    st.sidebar.title("📊 Difficulty Ranker")

    # ── Item‑set selection ────────────────────────────────────────────────
    if "item_set" not in st.session_state:
        st.title("Select an Item Set")
        st.markdown("Choose a question set to judge or view results for.")

        # One tile per item set, laid out in columns
        set_items = list(ITEM_SETS.items())
        cols = st.columns(min(len(set_items), 3), gap="large")

        for i, (slug, cfg) in enumerate(set_items):
            df = load_comparisons(slug)
            total_comps = len(df)
            judge_names = (
                sorted(df["judge_name"].unique().tolist())
                if not df.empty
                else []
            )
            q_ids = load_question_ids(cfg["images_dir"])
            excluded = get_excluded_item_ids(slug)
            active_count = len(q_ids) - len(excluded)
            status = "Open" if cfg["enabled"] else "Closed"

            with cols[i % len(cols)]:
                with st.container(border=True):
                    st.subheader(cfg["label"])
                    st.caption(cfg.get("description", ""))
                    stat_l, stat_r = st.columns(2)
                    with stat_l:
                        st.metric("Active Questions", active_count)
                        if excluded:
                            st.caption(f"({len(excluded)} excluded)")
                    with stat_r:
                        st.metric("Comparisons", total_comps)
                    st.metric("Judges", len(judge_names) if judge_names else 0)
                    if judge_names:
                        st.caption(", ".join(judge_names))

                    st.markdown("---")
                    if cfg["enabled"]:
                        if st.button(
                            "Start judging",
                            key=f"select_{slug}",
                            type="primary",
                            use_container_width=True,
                        ):
                            st.session_state["item_set"] = slug
                            st.rerun()

                    if st.button(
                        "View results",
                        key=f"results_{slug}",
                        use_container_width=True,
                    ):
                        st.session_state["item_set"] = slug
                        st.session_state["_go_to_results"] = True
                        st.rerun()

                    if not cfg["enabled"]:
                        st.caption("Judging closed for this set")
        return

    # ── Active session ────────────────────────────────────────────────────
    item_set = st.session_state["item_set"]
    cfg = ITEM_SETS[item_set]
    question_ids = load_question_ids(cfg["images_dir"])

    go_to_results = st.session_state.pop("_go_to_results", False)
    default_page = "Results" if go_to_results else "Judge Comparisons"
    pages = ["Judge Comparisons", "Results", "Review Flags"] if cfg["enabled"] else ["Results", "Review Flags"]
    page = st.sidebar.radio(
        "Navigation",
        pages,
        index=pages.index(default_page) if default_page in pages else 0,
    )

    if st.sidebar.button("← Change item set"):
        kept = st.session_state.get("personal_target", DEFAULT_TARGET)
        st.session_state.clear()
        st.session_state["personal_target"] = kept
        st.rerun()

    if page == "Judge Comparisons":
        page_judging(item_set, cfg, question_ids)
    elif page == "Results":
        page_results(item_set, cfg, question_ids)
    else:
        page_review_flags(item_set, cfg, question_ids)


if __name__ == "__main__":
    main()
