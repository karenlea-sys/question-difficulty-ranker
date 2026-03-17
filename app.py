# ─────────────────────────────────────────────────────────────────────────────
#  Question Difficulty Ranker  —  Adaptive Comparative Judgement (ACJ)
#  Backend: Google Sheets   |   Hosting: Streamlit Community Cloud
#
#  See DEPLOY.txt for full setup instructions.
# ─────────────────────────────────────────────────────────────────────────────

import os
import random
import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# ─── Configuration ────────────────────────────────────────────────────────────

IMAGES_DIR = "images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# Comparisons per judge: 4 judges × 500 = 2,000 total ≈ 8 per question (250 Qs)
TARGET_PER_JUDGE = 500

SHEET_HEADERS = ["id", "judge_name", "winner_id", "loser_id", "created_at"]

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


# ─── Google Sheets connection ──────────────────────────────────────────────────

@st.cache_resource
def get_worksheet():
    """
    Authenticate with the Google service account stored in Streamlit secrets
    and return the first worksheet of the configured spreadsheet.
    Cached as a resource so the connection is reused across sessions.
    """
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=SCOPES
    )
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(st.secrets["google_sheets"]["spreadsheet_id"])
    return spreadsheet.sheet1


def init_sheet():
    """Add header row if the sheet is empty."""
    ws = get_worksheet()
    existing = ws.row_values(1)
    if existing != SHEET_HEADERS:
        ws.clear()
        ws.append_row(SHEET_HEADERS)


# ─── Data access ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=20)
def load_comparisons() -> pd.DataFrame:
    """
    Read all comparisons from Google Sheets.
    Cached for 20 seconds to reduce API calls — safe because we manually
    invalidate the cache immediately after writing.
    """
    ws = get_worksheet()
    records = ws.get_all_records()
    if not records:
        return pd.DataFrame(columns=SHEET_HEADERS)
    return pd.DataFrame(records)


def save_comparison(judge_name: str, winner_id: str, loser_id: str):
    """Append a comparison row and immediately invalidate the read cache."""
    ws = get_worksheet()
    row_id = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    timestamp = datetime.datetime.utcnow().isoformat()
    ws.append_row([row_id, judge_name, winner_id, loser_id, timestamp])
    # Invalidate cached data so the next read reflects this write
    load_comparisons.clear()


def count_judge_comparisons(judge_name: str) -> int:
    df = load_comparisons()
    if df.empty:
        return 0
    return int((df["judge_name"] == judge_name).sum())


# ─── Question loading ──────────────────────────────────────────────────────────

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


# ─── Bradley-Terry ranking ─────────────────────────────────────────────────────

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


# ─── Adaptive pair selection ────────────────────────────────────────────────────

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

    n = len(question_ids)
    target_pool = min(2000, n * (n - 1) // 2)
    pool = []
    attempts = 0

    while len(pool) < target_pool and attempts < 15000:
        i, j = random.sample(range(n), 2)
        pair = frozenset([question_ids[i], question_ids[j]])
        if pair not in judged:
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


# ─── Page: Judging ──────────────────────────────────────────────────────────────

def page_judging(question_ids: list):
    st.title("Question Difficulty Ranking")

    if not question_ids:
        st.error(
            f"No images found in the `{IMAGES_DIR}/` folder. "
            "Add question images named by their ID (e.g. `Q001.png`) and restart."
        )
        return

    # ── Login ──────────────────────────────────────────────────────────────────
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

    # ── Sidebar status ─────────────────────────────────────────────────────────
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

    # ── Select next pair ───────────────────────────────────────────────────────
    if "current_pair" not in st.session_state:
        scores, _ = compute_rankings(question_ids) if done >= 20 else ({}, {})
        st.session_state["current_pair"] = select_next_pair(
            question_ids, judge, scores
        )

    q_left, q_right = st.session_state["current_pair"]

    # ── Comparison UI ──────────────────────────────────────────────────────────
    st.subheader("Which question is more difficult?")
    st.caption("Click the button **below** the question you think is harder.")
    st.markdown("---")

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
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
            save_comparison(judge, q_left, q_right)
            st.session_state.pop("current_pair", None)
            st.rerun()

    with col_r:
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
            save_comparison(judge, q_right, q_left)
            st.session_state.pop("current_pair", None)
            st.rerun()

    st.markdown("---")
    if st.button("Skip this pair (too close to call)"):
        st.session_state.pop("current_pair", None)
        st.rerun()


# ─── Page: Results ──────────────────────────────────────────────────────────────

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

    # ── Rankings table ─────────────────────────────────────────────────────────
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

    # ── Per-judge breakdown ────────────────────────────────────────────────────
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


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Difficulty Ranker",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        init_sheet()
    except Exception as e:
        st.error(
            "Could not connect to Google Sheets. "
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
