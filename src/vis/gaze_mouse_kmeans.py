"""
BisectingKMeans clustering (k=6, bisecting_strategy=largest_cluster) on
per-(user, task, query) position histograms.

Hyperparameters at the top of the file control which data to cluster:
  MODALITY   — "gaze" or "mouse"
  SIDE       — "left" or "right"
  DATA_TYPE  — "pairwise" (user_{gaze/mouse}_hist.csv, bin_N columns)
               "time_interp" (user_{gaze/mouse}_time_interp.csv, pos_N columns)

Each row's 100-bin/pos histogram is used as the feature vector.
The 6 cluster centroids are plotted as bar charts in a single 2x3 figure.
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import BisectingKMeans, KMeans
from scipy.cluster.hierarchy import dendrogram
from worker_filter import BAD_WORKERS

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MODALITY  = "gaze"        # "gaze" or "mouse"
SIDE      = "left"        # "left" or "right"
# DATA_TYPE = "histogram"    # "histogram" or "time_interp"
DATA_TYPE = "time_interp"    # "histogram" or "time_interp"

RANDOM_STATE = 42
EXCLUDE_BAD_WORKERS = True   # set False to include all workers
_qc = "filtered" if EXCLUDE_BAD_WORKERS else "all"
N_GROUPS = 2        # number of shape-similarity groups for centroid/sample plots
MIN_CLUSTER_SAMPLES = 10  # clusters smaller than this are excluded from group plots
# Manual overrides: cluster 1-indexed label -> group 0-indexed.  e.g. {4: 1} moves C4 to Group 2.
# Fully specified: C6,C7,C8,C9 -> Group 1 (0); all others -> Group 2 (1).
CENTROID_GROUP_OVERRIDES: dict = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 10: 1,
    6: 0, 7: 0, 8: 0, 9: 0,
}
# Manual overrides for sample plot: sample 1-indexed label -> group 0-indexed.
SAMPLE_GROUP_OVERRIDES: dict = {3: 0}

# histogram needs a lower k to avoid tiny outlier clusters; time_interp handles k=10 fine
N_CLUSTERS = 6 if DATA_TYPE == "histogram" else 10

# ---------------------------------------------------------------------------
# Derived paths and source label
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DATA_DIR       = os.path.join(PROJECT_ROOT, "output", "data")
_KMEANS_DIR     = os.path.join(PROJECT_ROOT, "output", "kmeans")
_MAIN_PAPER_DIR = os.path.join(PROJECT_ROOT, "output", "main_paper_images")
os.makedirs(_KMEANS_DIR,     exist_ok=True)
os.makedirs(_MAIN_PAPER_DIR, exist_ok=True)

if DATA_TYPE == "time_interp":
    _modality_stem = "gazing" if MODALITY == "gaze" else "mouse"
    _INPUT_FILE = f"user_{_modality_stem}_time_interp_{_qc}.csv"
else:
    _INPUT_FILE = f"user_gazing_hist_{_qc}.csv" if MODALITY == "gaze" else f"user_mouse_hist_{_qc}.csv"
INPUT_PATH = os.path.join(_DATA_DIR, _INPUT_FILE)

SOURCE_LABEL = f"{MODALITY}_pairwise_{SIDE}"
_stem = f"{SOURCE_LABEL}_kmeans_{DATA_TYPE}_{_qc}"
OUTPUT_PATH        = os.path.join(_KMEANS_DIR, f"{_stem}.png")
OUTPUT_PATH_SAMPLE = os.path.join(_KMEANS_DIR, f"{_stem}_samples.png")

N_BINS = 100
_COL_PREFIX = "pos" if DATA_TYPE == "time_interp" else "bin"
BIN_COLS = [f"{_COL_PREFIX}_{i}" for i in range(N_BINS)]
BIN_CENTERS = (np.linspace(0, 1, N_BINS + 1)[:-1] + np.linspace(0, 1, N_BINS + 1)[1:]) / 2

# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_PATH)
if EXCLUDE_BAD_WORKERS:
    df = df[~df["user_id"].isin(BAD_WORKERS)]
subset = df[df["source"] == SOURCE_LABEL].reset_index(drop=True)
print(f"Rows for {SOURCE_LABEL}: {len(subset)}")

X = subset[BIN_COLS].values

# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------
kmeans = BisectingKMeans(
    n_clusters=N_CLUSTERS, random_state=RANDOM_STATE,
    bisecting_strategy="largest_cluster", n_init=10,
)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_
cluster_counts = np.bincount(labels, minlength=N_CLUSTERS)

# ---------------------------------------------------------------------------
# Shared colormap (one color per cluster, consistent across all figures)
# ---------------------------------------------------------------------------
_cmap = plt.get_cmap("tab10" if N_CLUSTERS <= 10 else "tab20")
CLUSTER_COLORS = [_cmap(i / max(N_CLUSTERS - 1, 1)) for i in range(N_CLUSTERS)]


def _ax_labels(ax):
    if DATA_TYPE == "time_interp":
        ax.set_xlabel("Normalized time", fontsize=18, labelpad=8)
        ax.set_ylabel("Average relative position", fontsize=18)
        ax.set_ylim(0, 1)
    else:
        ax.set_xlabel("Relative position", fontsize=18, labelpad=8)
        ax.set_ylabel("Average probability", fontsize=18)
        ax.set_ylim(bottom=0)   # auto-scale top; histogram probs are ~0–0.05
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xlim(0, 1)


# ---------------------------------------------------------------------------
# Figure 1 — cluster centroids grouped by shape (N_GROUPS panels)
# ---------------------------------------------------------------------------
_valid_idxs = np.where(cluster_counts >= MIN_CLUSTER_SAMPLES)[0]
_skipped = np.where(cluster_counts < MIN_CLUSTER_SAMPLES)[0]
if len(_skipped):
    print(f"Excluding {len(_skipped)} tiny cluster(s) from group plot: "
          + ", ".join(f"C{i+1}(n={cluster_counts[i]})" for i in _skipped))

_n_groups = min(N_GROUPS, len(_valid_idxs))   # can't have more groups than valid clusters
_centroid_group_labels = KMeans(
    n_clusters=_n_groups, random_state=RANDOM_STATE, n_init=10
).fit_predict(centroids[_valid_idxs])

for _c1idx, _tgt_grp in CENTROID_GROUP_OVERRIDES.items():
    _pos = np.where(_valid_idxs == _c1idx - 1)[0]
    if len(_pos):
        _centroid_group_labels[_pos[0]] = _tgt_grp

_sharey = DATA_TYPE == "time_interp"   # histogram groups have very different y ranges
fig1, axes1 = plt.subplots(1, _n_groups, figsize=(6 * _n_groups, 5),
                            sharey=_sharey, constrained_layout=True)
for g, ax in enumerate(axes1):
    member_positions = np.where(_centroid_group_labels == g)[0]
    member_idxs = _valid_idxs[member_positions]
    for idx in member_idxs:
        ax.plot(BIN_CENTERS, centroids[idx], color=CLUSTER_COLORS[idx], lw=1.5,
                label=f"C{idx + 1} (n={cluster_counts[idx]})")
    _ax_labels(ax)
    ax.legend(fontsize=16)
    ax.set_title(f"Group {g + 1}  ({len(member_idxs)} clusters)", fontsize=20)

fig1.savefig(OUTPUT_PATH, dpi=150)
shutil.copy(OUTPUT_PATH, _MAIN_PAPER_DIR)
print(f"Saved centroids plot to {OUTPUT_PATH}")

# ---------------------------------------------------------------------------
# Figure 2 — random sample rows grouped by shape (N_GROUPS panels)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(RANDOM_STATE)
sample_idx = rng.choice(len(subset), size=N_CLUSTERS, replace=False)
sample_rows = X[sample_idx]

_n_sample_groups = min(N_GROUPS, N_CLUSTERS)
_sample_group_labels = KMeans(
    n_clusters=_n_sample_groups, random_state=RANDOM_STATE, n_init=10
).fit_predict(sample_rows)

for _s1idx, _tgt_grp in SAMPLE_GROUP_OVERRIDES.items():
    _pos = _s1idx - 1
    if 0 <= _pos < len(_sample_group_labels):
        _sample_group_labels[_pos] = _tgt_grp

fig2, axes2 = plt.subplots(1, _n_sample_groups, figsize=(6 * _n_sample_groups, 5),
                            sharey=_sharey, constrained_layout=True)
for g, ax in enumerate(axes2):
    member_idxs = np.where(_sample_group_labels == g)[0]
    for idx in member_idxs:
        row_i = sample_idx[idx]
        ax.plot(BIN_CENTERS, sample_rows[idx], color=CLUSTER_COLORS[idx], lw=1.2, alpha=0.75,
                label=f"Sample {idx + 1} (row {row_i})")
    _ax_labels(ax)
    ax.legend(fontsize=16)
    ax.set_title(f"Group {g + 1}  ({len(member_idxs)} samples)", fontsize=20)

fig2.savefig(OUTPUT_PATH_SAMPLE, dpi=150)
shutil.copy(OUTPUT_PATH_SAMPLE, _MAIN_PAPER_DIR)
print(f"Saved samples plot to {OUTPUT_PATH_SAMPLE}")

# ---------------------------------------------------------------------------
# Figure 3 — BisectingKMeans hierarchy dendrogram
# ---------------------------------------------------------------------------
def _build_linkage(bisection_tree, n_clusters):
    """Convert _bisecting_tree to a scipy-compatible linkage matrix.

    The root node has score=0 (not computed by sklearn), and indices are
    cleared after fitting. We use the sum of descendant leaf scores as the
    merge height — guaranteed monotonically increasing from leaves to root.
    """
    internal_nodes = []
    leaf_score_sum = {}   # id(node) -> sum of leaf scores in subtree
    leaf_count     = {}   # id(node) -> number of leaf nodes in subtree

    def _traverse(node):
        if node.left is None:  # leaf
            leaf_score_sum[id(node)] = node.score
            leaf_count[id(node)]     = 1
            return
        _traverse(node.left)
        _traverse(node.right)
        leaf_score_sum[id(node)] = (
            leaf_score_sum[id(node.left)] + leaf_score_sum[id(node.right)]
        )
        leaf_count[id(node)] = (
            leaf_count[id(node.left)] + leaf_count[id(node.right)]
        )
        internal_nodes.append(node)

    _traverse(bisection_tree)
    # Sort ascending so each node's children appear before it in the matrix.
    internal_nodes.sort(key=lambda n: leaf_score_sum[id(n)])

    node_to_id = {}

    def _assign_leaf_ids(node):
        if node.left is None:
            node_to_id[id(node)] = node.label
        else:
            _assign_leaf_ids(node.left)
            _assign_leaf_ids(node.right)

    _assign_leaf_ids(bisection_tree)
    for i, node in enumerate(internal_nodes):
        node_to_id[id(node)] = n_clusters + i

    Z = []
    for node in internal_nodes:
        Z.append([
            float(node_to_id[id(node.left)]),
            float(node_to_id[id(node.right)]),
            float(leaf_score_sum[id(node)]),
            float(leaf_count[id(node)]),
        ])
    return np.array(Z)


Z = _build_linkage(kmeans._bisecting_tree, N_CLUSTERS)
leaf_labels = [f"C{i+1} (n={cluster_counts[i]})" for i in range(N_CLUSTERS)]

fig3, ax3 = plt.subplots(figsize=(max(14, N_CLUSTERS), 6), constrained_layout=True)
dendrogram(Z, ax=ax3, labels=leaf_labels, leaf_rotation=90, leaf_font_size=8)
ax3.set_title(
    f"BisectingKMeans hierarchy — {SOURCE_LABEL}  (k={N_CLUSTERS})",
    fontsize=12,
)
ax3.set_ylabel("Cluster inertia at split", fontsize=10)

OUTPUT_PATH_DENDRO = os.path.join(_KMEANS_DIR, f"{_stem}_dendrogram.png")
fig3.savefig(OUTPUT_PATH_DENDRO, dpi=150)
print(f"Saved dendrogram to {OUTPUT_PATH_DENDRO}")

# ---------------------------------------------------------------------------
# Figure 4 — cluster centers layer by layer (BFS order, top → bottom)
# ---------------------------------------------------------------------------
# node.center in sklearn 1.4.x is stored in a locally-centred coordinate
# space (relative to each bisection step), not in the original data space.
# We compute the true centroid by averaging X rows whose final cluster label
# belongs to the node's subtree.
from collections import defaultdict, deque

def _subtree_leaf_labels(node):
    if node.left is None:
        return {node.label}
    return _subtree_leaf_labels(node.left) | _subtree_leaf_labels(node.right)

def _true_center(node, X, labels):
    mask = np.isin(labels, list(_subtree_leaf_labels(node)))
    return X[mask].mean(axis=0)

_node_by_layer: dict = defaultdict(list)  # depth -> [(node, par_depth, par_pos)]
_bfs_q: deque = deque([(kmeans._bisecting_tree, 0, None, None)])  # type: ignore[attr-defined]
while _bfs_q:
    _node, _depth, _par_depth, _par_pos = _bfs_q.popleft()
    _pos = len(_node_by_layer[_depth])
    _node_by_layer[_depth].append((_node, _par_depth, _par_pos))
    if _node.left:
        _bfs_q.append((_node.left,  _depth + 1, _depth, _pos))
        _bfs_q.append((_node.right, _depth + 1, _depth, _pos))

_n_layers = len(_node_by_layer)

fig4, axes4 = plt.subplots(
    _n_layers, 1,
    figsize=(9, _n_layers * 3),
    squeeze=False,
    constrained_layout=True,
)

for _depth in range(_n_layers):
    _ax = axes4[_depth, 0]
    _nodes = _node_by_layer[_depth]
    _layer_cmap = plt.get_cmap("tab10" if len(_nodes) <= 10 else "tab20")

    for _pos, (_node, _par_depth, _par_pos) in enumerate(_nodes):
        _center = _true_center(_node, X, labels)
        _n_samples = sum(cluster_counts[lbl] for lbl in _subtree_leaf_labels(_node))
        _leaf_tag  = f" [C{_node.label + 1}]" if _node.left is None else ""
        _node_color = _layer_cmap(_pos / max(len(_nodes) - 1, 1))
        _ax.plot(BIN_CENTERS, _center, color=_node_color, lw=1.5,
                 label=f"Node {_pos}{_leaf_tag} (n={_n_samples})")

    _ax_labels(_ax)
    _ax.tick_params(labelsize=8)
    _ax.set_title(
        f"Layer {_depth}  ({len(_nodes)} node{'s' if len(_nodes) > 1 else ''})",
        fontsize=10,
    )
    _ax.legend(fontsize=7, ncol=min(4, len(_nodes)))

fig4.suptitle(
    f"BisectingKMeans centers by layer — {SOURCE_LABEL}  (k={N_CLUSTERS})",
    fontsize=12,
)
OUTPUT_PATH_LAYERS = os.path.join(_KMEANS_DIR, f"{_stem}_layers.png")
fig4.savefig(OUTPUT_PATH_LAYERS, dpi=150)
print(f"Saved layer-by-layer centers to {OUTPUT_PATH_LAYERS}")

plt.show()
