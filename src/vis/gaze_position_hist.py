"""
Compute per-(user, task, query, source) normalized gaze/mouse-position histograms.

For each valid sample (centre_idx >= 0 and assigned query_id >= 0) the
relative position within the response is centre_idx / response_length.  A
100-bin histogram over [0, 1] is built for every (user_id, task_id, query_id,
source) group and normalized to a probability distribution.  Average
distributions are plotted (one 2x2 figure for gaze, one for mouse) and the
full table is saved to output/user_gazing_hist.csv.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from worker_filter import BAD_WORKERS

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EXCLUDE_BAD_WORKERS = True   # set False to include all workers

_qc = "filtered" if EXCLUDE_BAD_WORKERS else "all"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
USER_BEHAVIOR_DIR = os.path.join(PROJECT_ROOT, "user_behavior")
QUERY_LOGS_PATH   = os.path.join(PROJECT_ROOT, "query_logs_table.csv")
_DATA_DIR = os.path.join(PROJECT_ROOT, "output", "data")
_POS_DIR  = os.path.join(PROJECT_ROOT, "output", "position")
os.makedirs(_DATA_DIR, exist_ok=True)
os.makedirs(_POS_DIR,  exist_ok=True)
GAZE_OUTPUT_PATH  = os.path.join(_DATA_DIR, f"user_gazing_hist_{_qc}.csv")
MOUSE_OUTPUT_PATH = os.path.join(_DATA_DIR, f"user_mouse_hist_{_qc}.csv")
GAZE_PLOT_PATH        = os.path.join(_POS_DIR, f"gaze_position_hist_{_qc}.png")
MOUSE_PLOT_PATH       = os.path.join(_POS_DIR, f"mouse_position_hist_{_qc}.png")
GAZE_LEN_PLOT_PATH    = os.path.join(_POS_DIR, f"gaze_length_category_hist_{_qc}.png")
MOUSE_LEN_PLOT_PATH   = os.path.join(_POS_DIR, f"mouse_length_category_hist_{_qc}.png")

N_BINS = 100
BIN_EDGES = np.linspace(0, 1, N_BINS + 1)
BIN_COLS = [f"bin_{i}" for i in range(N_BINS)]

# ---------------------------------------------------------------------------
# File type configuration  (filename, source_label, response_column)
# ---------------------------------------------------------------------------
FILE_CONFIGS = [
    # gaze
    ("rel_gaze_query_id_assigned.csv",     "gaze_pointwise",      "llm_response_1"),
    ("rel_gaze_one_query_id_assigned.csv", "gaze_pairwise_left",  "llm_response_1"),
    ("rel_gaze_two_query_id_assigned.csv", "gaze_pairwise_right", "llm_response_2"),
    # mouse
    ("rel_mouse_query_id_assigned.csv",       "mouse_pointwise",      "llm_response_1"),
    ("rel_mouse_left_query_id_assigned.csv",  "mouse_pairwise_left",  "llm_response_1"),
    ("rel_mouse_right_query_id_assigned.csv", "mouse_pairwise_right", "llm_response_2"),
]

# ---------------------------------------------------------------------------
# Load query logs and pre-compute response lengths
# ---------------------------------------------------------------------------
logs = pd.read_csv(QUERY_LOGS_PATH)
logs["resp1_len"] = logs["llm_response_1"].str.len().fillna(0).astype(int)
logs["resp2_len"] = logs["llm_response_2"].str.len().fillna(0).astype(int)
logs_indexed = logs.set_index(["query_ID", "user_id", "task_id"])


def get_response_length(query_id, user_id, task_id, response_col):
    key = (query_id, user_id, task_id)
    if key not in logs_indexed.index:
        return None
    col = "resp1_len" if response_col == "llm_response_1" else "resp2_len"
    val = logs_indexed.loc[key, col]
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    return int(val)


# ---------------------------------------------------------------------------
# Process each file
# ---------------------------------------------------------------------------
rows = []
violations = 0

for filename, source, response_col in FILE_CONFIGS:
    pattern = os.path.join(USER_BEHAVIOR_DIR, "*", "*", filename)
    for filepath in sorted(glob.glob(pattern)):
        parts = filepath.split(os.sep)
        task_id = int(parts[-2])
        user_id = parts[-3]
        if EXCLUDE_BAD_WORKERS and user_id in BAD_WORKERS:
            continue

        df = pd.read_csv(filepath)

        valid = df[(df["centre_idx"] >= 0) & (df["query_id"] >= 0)].copy()
        if valid.empty:
            continue

        for qid, grp in valid.groupby("query_id"):
            resp_len = get_response_length(int(qid), user_id, task_id, response_col)
            if not resp_len:
                continue

            positions = grp["centre_idx"].values / resp_len

            over = (positions > 1).sum()
            if over > 0:
                violations += over
                print(
                    f"WARNING: {over} samples exceed 1.0 in {filepath} query {qid} "
                    f"(max={positions.max():.4f}, resp_len={resp_len})"
                )
                positions = np.clip(positions, 0, 1)

            counts, _ = np.histogram(positions, bins=BIN_EDGES)
            total = counts.sum()
            probs = counts / total if total > 0 else counts.astype(float)

            row = {
                "user_id": user_id,
                "task_id": task_id,
                "query_id": int(qid),
                "source": source,
                "response_length": resp_len,
            }
            row.update(dict(zip(BIN_COLS, probs)))
            rows.append(row)

print(f"Total samples with centre_idx/response_length > 1: {violations}")
print(f"Total (user, task, query, source) groups: {len(rows)}")

# ---------------------------------------------------------------------------
# Build and save output dataframes
# ---------------------------------------------------------------------------
out_df = pd.DataFrame(
    rows, columns=["user_id", "task_id", "query_id", "source", "response_length"] + BIN_COLS
)

# Assign length_category using tertiles so each category has ~equal record counts
out_df["length_category"] = pd.qcut(
    out_df["response_length"], q=3, labels=["short", "medium", "long"]
)
out_df = out_df.drop(columns=["response_length"])

# Reorder: metadata columns first, then bins
meta_cols = ["user_id", "task_id", "query_id", "source", "length_category"]
out_df = out_df[meta_cols + BIN_COLS]

counts = out_df["length_category"].value_counts().sort_index()
print(f"length_category counts: {counts.to_dict()}")

gaze_df  = out_df[out_df["source"].str.startswith("gaze")].reset_index(drop=True)
mouse_df = out_df[out_df["source"].str.startswith("mouse")].reset_index(drop=True)

gaze_df.to_csv(GAZE_OUTPUT_PATH, index=False)
print(f"Saved gaze histogram dataframe to {GAZE_OUTPUT_PATH}")
mouse_df.to_csv(MOUSE_OUTPUT_PATH, index=False)
print(f"Saved mouse histogram dataframe to {MOUSE_OUTPUT_PATH}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
bin_centers = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2

AVG_CURVE_STYLES = [
    ("Overall",          None,                {"color": "black",      "lw": 2.5, "ls": "-"}),
    ("pointwise",        "{p}_pointwise",     {"color": "tab:blue",   "lw": 1.5, "ls": "--"}),
    ("pairwise (left)",  "{p}_pairwise_left", {"color": "tab:orange", "lw": 1.5, "ls": "-."}),
    ("pairwise (right)", "{p}_pairwise_right",{"color": "tab:green",  "lw": 1.5, "ls": ":"}),
]

LEN_CAT_STYLES = {
    "short":  {"color": "tab:blue",   "lw": 1.5, "ls": "-"},
    "medium": {"color": "tab:orange", "lw": 1.5, "ls": "--"},
    "long":   {"color": "tab:green",  "lw": 1.5, "ls": "-."},
}


def plot_avg_histograms(df, prefix, suptitle, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, source_tpl, style in AVG_CURVE_STYLES:
        subset = df if source_tpl is None else df[df["source"] == source_tpl.format(p=prefix)]
        if subset.empty:
            continue
        avg_probs = subset[BIN_COLS].mean().values
        ax.plot(bin_centers, avg_probs, label=f"{label} (n={len(subset)})", **style)
    ax.set_xlabel("Relative position (centre_idx / response_length)")
    ax.set_ylabel("Average probability")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.set_title(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved plot to {save_path}")


def plot_length_category_histograms(df, suptitle, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for cat, style in LEN_CAT_STYLES.items():
        subset = df[df["length_category"] == cat]
        if subset.empty:
            continue
        avg_probs = subset[BIN_COLS].mean().values
        ax.plot(bin_centers, avg_probs, label=f"{cat} (n={len(subset)})", **style)
    ax.set_xlabel("Relative position (centre_idx / response_length)")
    ax.set_ylabel("Average probability")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.set_title(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved plot to {save_path}")


# ---------------------------------------------------------------------------
# Plot gaze and mouse figures
# ---------------------------------------------------------------------------
plot_avg_histograms(gaze_df,  "gaze",  "Average gaze position distributions",  GAZE_PLOT_PATH)
plot_avg_histograms(mouse_df, "mouse", "Average mouse position distributions", MOUSE_PLOT_PATH)

plot_length_category_histograms(
    gaze_df,  "Gaze position distributions by response length category",  GAZE_LEN_PLOT_PATH
)
plot_length_category_histograms(
    mouse_df, "Mouse position distributions by response length category", MOUSE_LEN_PLOT_PATH
)
