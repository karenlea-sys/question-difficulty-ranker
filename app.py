# ───────────────────────────────────────────────────────────────────────────────
#  Question Difficulty Ranker  —  Adaptive Comparative Judgement (ACJ)
#  Backend: Supabase   |   Hosting: Streamlit Community Cloud
#
#  Secrets required (paste into Streamlit Cloud → App Settings → Secrets):
#
#  [supabase]
#  url = "https://xxxx.supabase.co"
#  key = "your-anon-key"
#
#  Supabase tables (run once in the Supabase SQL editor):
#
#  create table comparisons (
#    id         text primary key,
#    judge_name text not null,
#    winner_id  text not null,
#    loser_id   text not null,
#    created_at text not null
#  );
#
#  create table flags (
#    id         text primary key,
#    judge_name text not null,
#    item_id    text not null,
#    created_at text not null
#  );
#
#  create table enemy_pairs (
#    id          text primary key,
#    item_a      text not null,
#    item_b      text not null,
#    reported_by text not null,
#    created_at  text not null
#  );
# ───────────────────────────────────────────────────────────────────────────────
import os
import random
import datetime
import uuid
from pathlib import Path
import pandas as pd
import streamlit as st
from supabase import create_client, Client
# ─── Configuration ───────────────────────────────────────────────────────────────────────────
IMAGES_DIR = "images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
# Comparisons per judge: 4 judges × 500 = 2,000 total ≈ 8 per question (250 Qs)
TARGET_PER_JUDGE = 500
TABLE_NAME = "comparisons"
FLAGS_TABLE_NAME = "flags"
ENEMY_PAIRS_TABLE_NAME = "enemy_pairs"
# ─── Supabase connection ───────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_client() -> Client:
    """Create and cache the Supabase client. Reused across all sessions."""
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["key"],
    )
# ─── Data access — comparisons ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=20)
def load_comparisons() -> pd.DataFrame:
    """
    Read all comparisons from Supabase.
    Cached for 20 seconds to reduce API calls — manually invalidated after writes.
    """
    client = get_client()
    response = client.table(TABLE_NAME).select("*").execute()
    data = response.data
    if not data:
        return pd.DataFrame(
            columns=["id", "judge_name", "winner_id", "loser_id", "created_at"]
        )
    return pd.DataFrame(data)
def save_comparison(judge_name: str, winner_id: str, loser_id: str):
    """Insert a comparison row and immediately invalidate the read cache."""
    client = get_client()
    row_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow().isoformat()
    response = client.table(TABLE_NAME).insert(
        {
            "id": row_id,
            "judge_name": judge_name,
            "winner_id": winner_id,
            "loser_id": loser_id,
            "created_at": timestamp,
        }
    ).execute()
    if hasattr(response, "error") and response.error:
        st.error(f"Failed to save comparison: {response.error}")
        return False
    # Invalidate cached data so the next read reflects this write
    load_comparisons.clear()
    return True
def count_judge_comparisons(judge_name: str) -> int:
    df = load_comparisons()
    if df.empty:
        return 0
    return int((df["judge_name"] == judge_name).sum())
# ─── Data access — flags ─────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_flags() -> pd.DataFrame:
    """
    Read all flags from Supabase.
    Cached for 30 seconds — manually invalidated after writes.
    """
    client = get_client()
    response = client.table(FLAGS_TABLE_NAME).select("*").execute()
    data = response.data
    if not data:
        return pd.DataFrame(
            columns=["id", "judge_name", "item_id", "created_at"]
        )
    return pd.DataFrame(data)
def save_flag(judge_name: str, item_id: str) -> bool:
    """
    Flag an item as potentially incorrect.
    Returns True if saved, False if this judge has already flagged this item.
    Each (judge_name, item_id) pair is stored at most once.
    """
    df = load_flags()
    if not df.empty:
        already = ((df["judge_name"] == judge_name) & (df["item_id"] == item_id)).any()
        if already:
            return False
    client = get_client()
    row_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow().isoformat()
    response = client.table(FLAGS_TABLE_NAME).insert(
        {
            "id": row_id,
            "judge_name": judge_name,
            "item_id": item_id,
            "created_at": timestamp,
        }
    ).execute()
    if hasattr(response, "error") and response.error:
        st.error(f"Failed to save flag: {response.error}")
        return False
    load_flags.clear()
    return True
# ─── Data access — enemy pairs ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_enemy_pairs() -> pd.DataFrame:
    """
    Read all enemy pairs from Supabase.
    Enemy pairs are stored in canonical (alphabetical) order so (A,B) == (B,A).
    Cached for 60 seconds — manually invalidated after writes.
    """
    client = get_client()
    response = client.table(ENEMY_PAIRS_TABLE_NAME).select("*").execute()
    data = response.data
    if not data:
        return pd.DataFrame(
            columns=["id", "item_a", "item_b", "reported_by", "created_at"]
        )
    return pd.DataFrame(data)
def save_enemy_pair(judge_name: str, item_x: str, item_y: str) -> bool:
    """
    Record that two items are too similar to be compared.
    Stored in canonical alphabetical order so duplicates can be detected.
    Returns True if saved, False if this pair is already logged.
    """
    item_a, item_b = sorted([item_x, item_y])
    df = load_enemy_pairs()
    if not df.empty:
        already = ((df["item_a"] == item_a) & (df["item_b"] == item_b)).any()
        if already:
            return False
    client = get_client()
    row_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow().isoformat()
    response = client.table(ENEMY_PAIRS_TABLE_NAME).insert(
        {
            "id": row_id,
            "item_a": item_a,
            "item_b": item_b,
            "reported_by": judge_name,
            "created_at": timestamp,
        }
    ).execute()
    if hasattr(response, "error") and response.error:
        st.error(f"Failed to save enemy pair: {response.error}")
        return False
    load_enemy_pairs.clear()
    return True
# ─── Question loading ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_question_ids() -> list:
    """Scan the images folder; filename stem = question ID."""
    if not os.path.exists(IMAGES_DIR):
        return []
    return sorted(
        p.stem
        for p in Path(IMAGES_DIR).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
def get_image_path(question_id: str):
    for ext in IMAGE_EXTENSIONS:
        path = Path(IMAGES_DIR) / f"{question_id}{ext}"
        if path.exists():
            return str(path)
    return None
# ─── Bradley-Terry ranking ────────────────────────────────────────────────────────────────────────
def compute_rankings(question_ids: list) -> tuple:
    """
    Returns (scores_dict, comp_counts_dict).
    Uses the choix library (Bradley-Terry ILSR) when available;
    falls back to normalised win-rate if not.
    """
    df = load_comparisons()
    comp_counts = {q: 0 for q in question_ids}
    if df.empty:
        return {q: 0.0 for q in question_ids}, comp_counts
    q_set = set(question_ids)
    df = df[df["winner_id"].isin(q_set) & df["loser_id"].isin(q_set)]
    for q in question_ids:
        comp_counts[q] = int(
            ((df["winner_id"] == q) | (df["loser_id"] == q)).sum()
        )
    try:
        import choix
        q_list = sorted(question_ids)
        idx = {q: i for i, q in enumerate(q_list)}
        data = [
            (idx[r.winner_id], idx[r.loser_id])
            for r in df.itertuples()
            if r.winner_id in idx and r.loser_id in idx
        ]
        if len(data) < 2:
            raise ValueError("Not enough data for BT model yet.")
        params = choix.ilsr_pairwise(len(q_list), data, alpha=0.01)
        scores = {q_list[i]: float(params[i]) for i in range(len(q_list))}
    except Exception:
        # Fallback: normalised win rate
        wins = {q: 0 for q in question_ids}
        total = {q: 0 for q in question_ids}
        for r in df.itertuples():
            if r.winner_id in wins:
                wins[r.winner_id] += 1
                total[r.winner_id] += 1
            if r.loser_id in wins:
                total[r.loser_id] += 1
        scores = {q: wins[q] / max(total[q], 1) for q in question_ids}
    return scores, comp_counts
# ─── Adaptive pair selection ──────────────────────────────────────────────────────────────────────
def select_next_pair(question_ids: list, judge_name: str, scores: dict) -> tuple:
    """
    Pick the most informative unjudged pair for this judge.
    Samples up to 2,000 candidate pairs, then:
      - If BT scores exist: picks from the top 10% most informative
        (closest scores — greatest uncertainty to resolve).
      - Otherwise: picks randomly.
    Left/right order is randomised to avoid position bias.
    """
    df = load_comparisons()
    judge_df = df[df["judge_name"] == judge_name] if not df.empty else pd.DataFrame()
    judged = set()
    if not judge_df.empty:
        judged = {
            frozenset([r.winner_id, r.loser_id])
            for r in judge_df.itertuples()
        }
    # Exclude enemy pairs globally — these are too similar to compare for any judge
    enemy_df = load_enemy_pairs()
    enemies = set()
    if not enemy_df.empty:
        enemies = {
            frozenset([r.item_a, r.item_b])
            for r in enemy_df.itertuples()
        }
    excluded = judged | enemies
    n = len(question_ids)
    target_pool = min(2000, n * (n - 1) // 2)
    pool = []
    attempts = 0
    while len(pool) < target_pool and attempts < 15000:
        i, j = random.sample(range(n), 2)
        pair = frozenset([question_ids[i], question_ids[j]])
        if pair not in excluded:
            pool.append((question_ids[i], question_ids[j]))
        attempts += 1
    if not pool:
        i, j = random.sample(range(n), 2)
        chosen = (question_ids[i], question_ids[j])
    elif scores:
        pool.sort(key=lambda p: abs(scores.get(p[0], 0.0) - scores.get(p[1], 0.0)))
        top_n = max(1, len(pool) // 10)
        chosen = random.choice(pool[:top_n])
    else:
        chosen = random.choice(pool)
    return chosen if random.random() < 0.5 else (chosen[1], chosen[0])
# ─── Page: Judging ────────────────────────────────────────────────────────────────────────────────
def page_judging(question_ids: list):
    st.title("Question Difficulty Ranking")
    if not question_ids:
        st.error(
            f"No images found in the `{IMAGES_DIR}/` folder. "
            "Add question images named by their ID (e.g. `Q001.png`) and restart."
        )
        return
    # ── Login ──────────────────────────────────────────────────────────────────────────────────────
    if not st.session_state.get("judge_name"):
        st.markdown("### Welcome")
        st.markdown(
            "You'll be shown pairs of questions. "
            "For each pair, click **the button under whichever question you think is harder**."
        )
        st.markdown("Enter your name or judge ID to begin.")
        name = st.text_input("Your name / judge ID")
        if st.button("Start judging", type="primary") and name.strip():
            st.session_state["judge_name"] = name.strip()
            st.session_state.pop("current_pair", None)
            st.rerun()
        return
    judge = st.session_state["judge_name"]
    done = count_judge_comparisons(judge)
    # ── Sidebar status ────────────────────────────────────────────────────────────────────────────
    st.sidebar.markdown(f"**Judge:** {judge}")
    st.sidebar.markdown(f"**Comparisons made:** {done} / {TARGET_PER_JUDGE}")
    st.sidebar.progress(min(done / TARGET_PER_JUDGE, 1.0))
    if done >= TARGET_PER_JUDGE:
        st.sidebar.success("🎉 Target reached! You can stop or keep going.")
    if st.sidebar.button("Switch judge"):
        st.session_state.clear()
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"{len(question_ids)} questions loaded. "
        f"Target: {TARGET_PER_JUDGE} comparisons per judge."
    )
    # ── Select next pair ──────────────────────────────────────────────────────────────────────────
    if "current_pair" not in st.session_state:
        scores, _ = compute_rankings(question_ids) if done >= 20 else ({}, {})
        st.session_state["current_pair"] = select_next_pair(
            question_ids, judge, scores
        )
    q_left, q_right = st.session_state["current_pair"]
    # ── Load flags for this judge (to show already-flagged state) ─────────────────────────────────
    flags_df = load_flags()
    flagged_by_judge = set()
    if not flags_df.empty:
        flagged_by_judge = set(
            flags_df[flags_df["judge_name"] == judge]["item_id"].tolist()
        )
    # ── Comparison UI ─────────────────────────────────────────────────────────────────────────────
    st.subheader("Which question is more difficult?")
    st.caption("Click the button **below** the question you think is harder.")
    st.markdown("---")
    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        # Flag button — small, top-right of column, above the image
        if q_left in flagged_by_judge:
            st.caption("🚩 Already flagged")
        else:
            _, flag_col_l = st.columns([4, 1])
            with flag_col_l:
                if st.button("🚩 Flag", key="flag_left", help="Flag this question as incorrect"):
                    saved = save_flag(judge, q_left)
                    if saved:
                        st.toast(f"Flagged: {q_left}", icon="🚩")
                    st.rerun()
        img_path = get_image_path(q_left)
        if img_path:
            st.image(img_path, use_container_width=True)
        else:
            st.warning(f"Image not found: {q_left}")
        if st.button(
            "This one is harder ↑",
            key="btn_left",
            use_container_width=True,
            type="primary",
        ):
            if save_comparison(judge, q_left, q_right):
                st.session_state.pop("current_pair", None)
                st.rerun()
    with col_r:
        # Flag button — small, top-right of column, above the image
        if q_right in flagged_by_judge:
            st.caption("🚩 Already flagged")
        else:
            _, flag_col_r = st.columns([4, 1])
            with flag_col_r:
                if st.button("🚩 Flag", key="flag_right", help="Flag this question as incorrect"):
                    saved = save_flag(judge, q_right)
                    if saved:
                        st.toast(f"Flagged: {q_right}", icon="🚩")
                    st.rerun()
        img_path = get_image_path(q_right)
        if img_path:
            st.image(img_path, use_container_width=True)
        else:
            st.warning(f"Image not found: {q_right}")
        if st.button(
            "This one is harder ↑",
            key="btn_right",
            use_container_width=True,
            type="primary",
        ):
            if save_comparison(judge, q_right, q_left):
                st.session_state.pop("current_pair", None)
                st.rerun()
    st.markdown("---")
    skip_col, enemy_col = st.columns(2)
    with skip_col:
        if st.button(
            "Too close to call — skip",
            key="btn_skip",
            use_container_width=True,
            help="Move on without recording a judgement. No data saved.",
        ):
            st.session_state.pop("current_pair", None)
            st.rerun()
    with enemy_col:
        if st.button(
            "⚠️ Very similar — shouldn't appear together",
            key="btn_enemy",
            use_container_width=True,
            help="Log these two items as too similar to compare. They won't be paired again for any judge.",
        ):
            saved = save_enemy_pair(judge, q_left, q_right)
            if saved:
                st.toast(
                    f"Logged: {q_left} & {q_right} won't be paired again",
                    icon="⚠️",
                )
            else:
                st.toast("Already logged as too similar to pair", icon="ℹ️")
            st.session_state.pop("current_pair", None)
            st.rerun()
# ─── Page: Results ────────────────────────────────────────────────────────────────────────────────
def page_results(question_ids: list):
    st.title("Difficulty Rankings")
    if not question_ids:
        st.warning("No questions loaded.")
        return
    df = load_comparisons()
    if df.empty:
        st.info(
            "No comparisons recorded yet. "
            "Come back once judges have completed some decisions."
        )
        return
    scores, comp_counts = compute_rankings(question_ids)
    if all(v == 0.0 for v in scores.values()):
        st.info("Not enough comparisons yet — complete a few more rounds first.")
        return
    # ── Rankings table ────────────────────────────────────────────────────────────────────────────
    sorted_qs = sorted(scores, key=lambda q: scores[q], reverse=True)
    results_df = pd.DataFrame(
        {
            "Rank": range(1, len(sorted_qs) + 1),
            "Question ID": sorted_qs,
            "Difficulty Score": [round(scores[q], 4) for q in sorted_qs],
            "Comparisons": [comp_counts.get(q, 0) for q in sorted_qs],
        }
    )
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    total = len(df)
    avg_per_item = (total * 2) / max(len(question_ids), 1)
    st.caption(
        f"Total comparisons: **{total}**  ·  "
        f"Avg per question: **{avg_per_item:.1f}**  ·  "
        f"Questions with 0 comparisons: "
        f"**{sum(1 for q in question_ids if comp_counts.get(q, 0) == 0)}**"
    )
    st.download_button(
        label="⬇ Download rankings as CSV",
        data=results_df.to_csv(index=False),
        file_name="difficulty_rankings.csv",
        mime="text/csv",
    )
    # ── Per-judge breakdown ───────────────────────────────────────────────────────────────────────
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
            label="⬇ Download all comparisons as CSV",
            data=df.to_csv(index=False),
            file_name="raw_comparisons.csv",
            mime="text/csv",
        )
    # ── Flagged items ─────────────────────────────────────────────────────────────────────────────
    st.subheader("🚩 Flagged Items")
    flags_df = load_flags()
    if flags_df.empty:
        st.info("No items have been flagged yet.")
    else:
        flag_summary = (
            flags_df.groupby("item_id")
            .agg(
                flag_count=("judge_name", "count"),
                flagged_by=("judge_name", lambda x: ", ".join(sorted(set(x)))),
            )
            .reset_index()
            .rename(
                columns={
                    "item_id": "Item ID",
                    "flag_count": "Times Flagged",
                    "flagged_by": "Flagged By",
                }
            )
            .sort_values("Times Flagged", ascending=False)
        )
        st.dataframe(flag_summary, use_container_width=True, hide_index=True)
        with st.expander("Download raw flag data"):
            st.download_button(
                label="⬇ Download all flags as CSV",
                data=flags_df.to_csv(index=False),
                file_name="raw_flags.csv",
                mime="text/csv",
            )
    # ── Enemy pairs ───────────────────────────────────────────────────────────────────────────────
    st.subheader("⚠️ Enemy Pairs (too similar to compare)")
    enemy_df = load_enemy_pairs()
    if enemy_df.empty:
        st.info("No enemy pairs have been logged yet.")
    else:
        enemy_display = (
            enemy_df[["item_a", "item_b", "reported_by", "created_at"]]
            .rename(
                columns={
                    "item_a": "Item A",
                    "item_b": "Item B",
                    "reported_by": "Reported By",
                    "created_at": "Logged At",
                }
            )
            .sort_values("Logged At", ascending=False)
        )
        st.caption(
            f"{len(enemy_df)} pair(s) permanently excluded from judging for all judges."
        )
        st.dataframe(enemy_display, use_container_width=True, hide_index=True)
        with st.expander("Download enemy pairs data"):
            st.download_button(
                label="⬇ Download enemy pairs as CSV",
                data=enemy_df.to_csv(index=False),
                file_name="enemy_pairs.csv",
                mime="text/csv",
            )
# ─── Main ─────────────────────────────────────────────────────────────────────────────────────────
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
            "Check that your secrets are configured correctly in Streamlit Cloud. "
            f"Error: {e}"
        )
        st.stop()
    question_ids = load_question_ids()
    st.sidebar.title("📊 Difficulty Ranker")
    page = st.sidebar.radio("Navigation", ["Judge Comparisons", "Results"])
    if page == "Judge Comparisons":
        page_judging(question_ids)
    else:
        page_results(question_ids)
if __name__ == "__main__":
    main()
