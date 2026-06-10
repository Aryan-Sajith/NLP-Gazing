"""
Correlation between per-(user, task, query) gaze and mouse time-interpolated
position trajectories.

For each (user_id, task_id, query_id, side) that has both a gaze and a mouse
row in the time-interpolated CSVs, compute Pearson and Spearman correlations
between the two 100-point relative-position vectors over normalized time.

Outputs:
  output/gaze_mouse_correlation_per_query.csv  — one row per matched pair
  output/gaze_mouse_correlation_summary.csv    — aggregate stats by segment
  output/gaze_mouse_correlation.png            — 2x2 visualization
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, gaussian_kde
from worker_filter import BAD_WORKERS

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EXCLUDE_BAD_WORKERS = True  # set False to include all workers
_qc = "filtered" if EXCLUDE_BAD_WORKERS else "all"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR        = os.path.join(PROJECT_ROOT, "output", "data")
_CORR_DIR        = os.path.join(PROJECT_ROOT, "output", "correlation")
_MAIN_PAPER_DIR  = os.path.join(PROJECT_ROOT, "output", "main_paper_images")
os.makedirs(_CORR_DIR,       exist_ok=True)
os.makedirs(_MAIN_PAPER_DIR, exist_ok=True)
GAZE_PATH     = os.path.join(_DATA_DIR, f"user_gazing_time_interp_{_qc}.csv")
MOUSE_PATH    = os.path.join(_DATA_DIR, f"user_mouse_time_interp_{_qc}.csv")
PER_QUERY_OUT = os.path.join(_CORR_DIR, f"gaze_mouse_correlation_per_query_{_qc}.csv")
SUMMARY_OUT   = os.path.join(_CORR_DIR, f"gaze_mouse_correlation_summary_{_qc}.csv")
PLOT_OVERALL   = os.path.join(_CORR_DIR, f"gaze_mouse_correlation_overall_{_qc}.png")
PLOT_BY_LENGTH = os.path.join(_CORR_DIR, f"gaze_mouse_correlation_by_length_{_qc}.png")
PLOT_BY_SIDE   = os.path.join(_CORR_DIR, f"gaze_mouse_correlation_by_side_{_qc}.png")
PLOT_PER_USER  = os.path.join(_CORR_DIR, f"gaze_mouse_correlation_per_user_{_qc}.png")

N_POINTS = 100
POS_COLS = [f"pos_{i}" for i in range(N_POINTS)]
SIG_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Load and align gaze and mouse time-interpolated tables
# ---------------------------------------------------------------------------
gaze  = pd.read_csv(GAZE_PATH)
mouse = pd.read_csv(MOUSE_PATH)
if EXCLUDE_BAD_WORKERS:
    gaze  = gaze[~gaze["user_id"].isin(BAD_WORKERS)]
    mouse = mouse[~mouse["user_id"].isin(BAD_WORKERS)]


def _strip_modality(src: str) -> str:
    for prefix in ("gaze_", "mouse_"):
        if src.startswith(prefix):
            return src[len(prefix):]
    return src


gaze["side"]  = gaze["source"].map(_strip_modality)
mouse["side"] = mouse["source"].map(_strip_modality)

JOIN_KEYS = ["user_id", "task_id", "query_id", "side"]
GAZE_COLS  = [f"gaze_{c}"  for c in POS_COLS]
MOUSE_COLS = [f"mouse_{c}" for c in POS_COLS]

g_renamed = gaze.rename(columns=dict(zip(POS_COLS, GAZE_COLS)))[JOIN_KEYS + ["length_category"] + GAZE_COLS]
m_renamed = mouse.rename(columns=dict(zip(POS_COLS, MOUSE_COLS)))[JOIN_KEYS + MOUSE_COLS]

merged = g_renamed.merge(m_renamed, on=JOIN_KEYS, how="inner")
print(f"Gaze rows: {len(gaze)}  |  Mouse rows: {len(mouse)}  |  Merged pairs: {len(merged)}")

# ---------------------------------------------------------------------------
# Compute Pearson / Spearman per merged pair
# ---------------------------------------------------------------------------
G_mat = merged[GAZE_COLS].to_numpy()
M_mat = merged[MOUSE_COLS].to_numpy()
meta  = merged[JOIN_KEYS + ["length_category"]].to_dict(orient="records")

results = []
skipped_zero_var = 0
for i, row in enumerate(meta):
    g_vec, m_vec = G_mat[i], M_mat[i]
    if np.ptp(g_vec) < 1e-10 or np.ptp(m_vec) < 1e-10:
        skipped_zero_var += 1
        continue
    pr, pp = pearsonr(g_vec, m_vec)
    sr, sp = spearmanr(g_vec, m_vec)
    results.append({
        **row,
        "pearson_r":  pr,
        "pearson_p":  pp,
        "spearman_r": sr,
        "spearman_p": sp,
    })

print(f"Skipped (zero variance in gaze or mouse vector): {skipped_zero_var}")

results_df = pd.DataFrame(results)
results_df.to_csv(PER_QUERY_OUT, index=False)
print(f"Saved per-query correlations to {PER_QUERY_OUT}")

# ---------------------------------------------------------------------------
# Aggregate summary by segment
# ---------------------------------------------------------------------------
def _agg(df: pd.DataFrame, segment_type: str, segment_value) -> dict:
    n = len(df)
    sig = int((df["pearson_p"] < SIG_THRESHOLD).sum())
    return {
        "segment_type":  segment_type,
        "segment_value": segment_value,
        "n":             n,
        "pearson_mean":   df["pearson_r"].mean(),
        "pearson_median": df["pearson_r"].median(),
        "pearson_std":    df["pearson_r"].std(),
        "pct_significant_pearson": 100.0 * sig / n if n else 0.0,
        "spearman_mean":   df["spearman_r"].mean(),
        "spearman_median": df["spearman_r"].median(),
    }


summary_rows = [_agg(results_df, "overall", "all")]
for cat in ["short", "medium", "long"]:
    sub = results_df[results_df["length_category"] == cat]
    if not sub.empty:
        summary_rows.append(_agg(sub, "length_category", cat))
for side in ["pointwise", "pairwise_left", "pairwise_right"]:
    sub = results_df[results_df["side"] == side]
    if not sub.empty:
        summary_rows.append(_agg(sub, "side", side))
for user in sorted(results_df["user_id"].unique()):
    sub = results_df[results_df["user_id"] == user]
    if not sub.empty:
        summary_rows.append(_agg(sub, "user_id", user))

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_OUT, index=False)
print(f"Saved summary to {SUMMARY_OUT}")

overall = summary_rows[0]
print("")
print("=== Overall ===")
print(f"  n queries:            {overall['n']}")
print(f"  Pearson r  mean:      {overall['pearson_mean']:.3f}   median: {overall['pearson_median']:.3f}   std: {overall['pearson_std']:.3f}")
print(f"  Spearman r mean:      {overall['spearman_mean']:.3f}   median: {overall['spearman_median']:.3f}")
print(f"  % significant Pearson (p<{SIG_THRESHOLD}): {overall['pct_significant_pearson']:.1f}%")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def _kde_curve(ax, values: np.ndarray, label: str, *, color: str, ls: str = "-", lw: float = 1.5) -> None:
    """Plot a Gaussian KDE for `values` if it has enough variance, else skip."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) < 5 or values.std() == 0:
        return
    xs = np.linspace(-1.0, 1.0, 400)
    kde = gaussian_kde(values)
    ax.plot(xs, kde(xs), color=color, ls=ls, lw=lw,
            label=f"{label} (n={len(values)})")


LEN_STYLES = {
    "short":  {"color": "tab:blue",   "ls": "-"},
    "medium": {"color": "tab:orange", "ls": "--"},
    "long":   {"color": "tab:green",  "ls": "-."},
}

SIDE_STYLES = {
    "pointwise":      {"color": "tab:blue",   "ls": "-"},
    "pairwise_left":  {"color": "tab:orange", "ls": "--"},
    "pairwise_right": {"color": "tab:green",  "ls": "-."},
}

# Overall histogram of Pearson r
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(results_df["pearson_r"], bins=40, color="steelblue", edgecolor="black", alpha=0.85)
ax.axvline(0, color="gray", lw=0.8, ls="--")
ax.axvline(results_df["pearson_r"].mean(), color="red", lw=1.6,
           label=f"mean = {results_df['pearson_r'].mean():.3f}")
ax.set_xlabel("Pearson r  (gaze vs mouse position-over-time)", fontsize=14)
ax.set_ylabel("Number of queries", fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.set_xlim(-1, 1)
ax.legend(fontsize=13)
fig.tight_layout()
fig.savefig(PLOT_OVERALL, dpi=150)
plt.close(fig)
print(f"Saved overall plot to {PLOT_OVERALL}")

# KDE by length category
fig, ax = plt.subplots(figsize=(8, 5))
for cat, style in LEN_STYLES.items():
    vals = results_df[results_df["length_category"] == cat]["pearson_r"].values
    _kde_curve(ax, vals, cat, **style)
ax.axvline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("Pearson r", fontsize=18, labelpad=8)
ax.set_ylabel("Density", fontsize=18)
ax.tick_params(axis='both', labelsize=16)
ax.set_xlim(-1, 1)
ax.legend(fontsize=16)
fig.tight_layout()
fig.savefig(PLOT_BY_LENGTH, dpi=150)
plt.close(fig)
shutil.copy(PLOT_BY_LENGTH, _MAIN_PAPER_DIR)
print(f"Saved by-length plot to {PLOT_BY_LENGTH}")

# KDE by side / task type
fig, ax = plt.subplots(figsize=(7, 5))
for side, style in SIDE_STYLES.items():
    vals = results_df[results_df["side"] == side]["pearson_r"].values
    _kde_curve(ax, vals, side, **style)
ax.axvline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("Pearson r", fontsize=14)
ax.set_ylabel("Density", fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.set_xlim(-1, 1)
ax.legend(fontsize=13)
fig.tight_layout()
fig.savefig(PLOT_BY_SIDE, dpi=150)
plt.close(fig)
print(f"Saved by-side plot to {PLOT_BY_SIDE}")

# Histogram of per-user mean Pearson r (diversity)
fig, ax = plt.subplots(figsize=(7, 5))
per_user_means = results_df.groupby("user_id")["pearson_r"].mean()
ax.hist(per_user_means, bins=25, color="seagreen", edgecolor="black", alpha=0.85)
ax.axvline(0, color="gray", lw=0.8, ls="--")
ax.axvline(per_user_means.mean(), color="red", lw=1.6,
           label=f"mean of means = {per_user_means.mean():.3f}")
ax.set_xlabel("Per-user mean Pearson r", fontsize=14)
ax.set_ylabel("Number of users", fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.set_xlim(-1, 1)
ax.legend(fontsize=13)
fig.tight_layout()
fig.savefig(PLOT_PER_USER, dpi=150)
plt.close(fig)
print(f"Saved per-user plot to {PLOT_PER_USER}")
