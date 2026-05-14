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
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import BisectingKMeans
from scipy.cluster.hierarchy import dendrogram

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MODALITY  = "gaze"        # "gaze" or "mouse"
SIDE      = "left"        # "left" or "right"
#DATA_TYPE = "histogram"    # "histogram" or "time_interp"
DATA_TYPE = "time_interp"    # "histogram" or "time_interp"

#N_CLUSTERS = 20
N_CLUSTERS = 10
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Derived paths and source label
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if DATA_TYPE == "time_interp":
    _modality_stem = "gazing" if MODALITY == "gaze" else "mouse"
    _INPUT_FILE = f"user_{_modality_stem}_time_interp.csv"
else:
    _INPUT_FILE = "user_gazing_hist.csv" if MODALITY == "gaze" else "user_mouse_hist.csv"
INPUT_PATH = os.path.join(PROJECT_ROOT, "output", _INPUT_FILE)

SOURCE_LABEL = f"{MODALITY}_pairwise_{SIDE}"
OUTPUT_PATH        = os.path.join(PROJECT_ROOT, "output", f"{SOURCE_LABEL}_kmeans.png")
OUTPUT_PATH_SAMPLE = os.path.join(PROJECT_ROOT, "output", f"{SOURCE_LABEL}_kmeans_samples.png")

N_BINS = 100
_COL_PREFIX = "pos" if DATA_TYPE == "time_interp" else "bin"
BIN_COLS = [f"{_COL_PREFIX}_{i}" for i in range(N_BINS)]
BIN_CENTERS = (np.linspace(0, 1, N_BINS + 1)[:-1] + np.linspace(0, 1, N_BINS + 1)[1:]) / 2

# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_PATH)
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
# Shared plot helpers
# ---------------------------------------------------------------------------
def _make_figure(n_panels):
    n_cols = math.ceil(math.sqrt(n_panels))
    n_rows = math.ceil(n_panels / n_cols)
    label_fs = max(6, 10 - n_cols)
    title_fs = max(7, 11 - n_cols)
    tick_fs  = max(5,  9 - n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 6, n_rows * 4),
        sharey=False,
        constrained_layout=True,
    )
    return fig, axes, n_cols, n_rows, label_fs, title_fs, tick_fs


def _style_ax(ax, title, label_fs, title_fs, tick_fs):
    ax.set_title(title, fontsize=title_fs)
    if DATA_TYPE == "time_interp":
        ax.set_xlabel("Normalized time", fontsize=label_fs)
        ax.set_ylabel("Relative position\n(centre_idx / response_length)", fontsize=label_fs)
    else:
        ax.set_xlabel("Relative position\n(centre_idx / response_length)", fontsize=label_fs)
        ax.set_ylabel("Average probability", fontsize=label_fs)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=tick_fs)


# ---------------------------------------------------------------------------
# Figure 1 — cluster centroids
# ---------------------------------------------------------------------------
fig1, axes1, n_cols, n_rows, label_fs, title_fs, tick_fs = _make_figure(N_CLUSTERS)
bar_width = 1 / N_BINS

for idx, ax in enumerate(axes1.flat):
    if idx < N_CLUSTERS:
        ax.bar(BIN_CENTERS, centroids[idx], width=bar_width, align="center",
               edgecolor="none", alpha=0.8)
        _style_ax(ax, f"Cluster {idx + 1}  (n={cluster_counts[idx]})", label_fs, title_fs, tick_fs)
    else:
        ax.set_visible(False)

fig1.suptitle(
    f"BisectingKMeans (k={N_CLUSTERS}, largest_cluster) — {SOURCE_LABEL}  "
    f"(total n={len(subset)})",
    fontsize=max(9, 13 - n_cols),
)
fig1.savefig(OUTPUT_PATH, dpi=150)
print(f"Saved centroids plot to {OUTPUT_PATH}")

# ---------------------------------------------------------------------------
# Figure 2 — random sample of N_CLUSTERS rows
# ---------------------------------------------------------------------------
rng = np.random.default_rng(RANDOM_STATE)
sample_idx = rng.choice(len(subset), size=N_CLUSTERS, replace=False)
sample_rows = X[sample_idx]

fig2, axes2, n_cols2, _, label_fs2, title_fs2, tick_fs2 = _make_figure(N_CLUSTERS)

for idx, ax in enumerate(axes2.flat):
    if idx < N_CLUSTERS:
        row_i = sample_idx[idx]
        ax.bar(BIN_CENTERS, sample_rows[idx], width=bar_width, align="center",
               edgecolor="none", alpha=0.8)
        _style_ax(ax, f"Sample {idx + 1}  (row {row_i})", label_fs2, title_fs2, tick_fs2)
    else:
        ax.set_visible(False)

fig2.suptitle(
    f"Random samples (n={N_CLUSTERS}) — {SOURCE_LABEL}  (total n={len(subset)})",
    fontsize=max(9, 13 - n_cols2),
)
fig2.savefig(OUTPUT_PATH_SAMPLE, dpi=150)
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

OUTPUT_PATH_DENDRO = os.path.join(
    PROJECT_ROOT, "output", f"{SOURCE_LABEL}_kmeans_dendrogram.png"
)
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

_n_layers       = len(_node_by_layer)
_max_per_layer  = max(len(v) for v in _node_by_layer.values())

fig4, axes4 = plt.subplots(
    _n_layers, _max_per_layer,
    figsize=(max(_max_per_layer * 5, 8), _n_layers * 3.5),
    squeeze=False,
    constrained_layout=True,
)
for _row in axes4:
    for _ax in _row:
        _ax.set_visible(False)

for _depth in range(_n_layers):
    _nodes = _node_by_layer[_depth]
    _n     = len(_nodes)
    _start = (_max_per_layer - _n) // 2  # centre nodes within the row

    for _pos, (_node, _par_depth, _par_pos) in enumerate(_nodes):
        _ax = axes4[_depth, _start + _pos]
        _ax.set_visible(True)
        _center = _true_center(_node, X, labels)
        _ax.bar(BIN_CENTERS, _center, width=bar_width,
                align="center", edgecolor="none", alpha=0.8)
        _ax.set_xlim(0, 1)
        _ax.set_ylim(0, 1)
        _ax.tick_params(labelsize=6)
        if DATA_TYPE == "time_interp":
            _ax.set_xlabel("Norm. time", fontsize=7)
            _ax.set_ylabel("Rel. position", fontsize=7)
        else:
            _ax.set_xlabel("Rel. position", fontsize=7)
            _ax.set_ylabel("Avg. prob.", fontsize=7)

        _n_samples  = sum(cluster_counts[lbl] for lbl in _subtree_leaf_labels(_node))
        _leaf_tag   = f"  [C{_node.label + 1}]" if _node.left is None else ""
        _parent_tag = "(root)" if _par_depth is None else f"↑ Layer {_par_depth}, Node {_par_pos}"
        _ax.set_title(
            f"Layer {_depth}, Node {_pos}{_leaf_tag}  (n={_n_samples})\n{_parent_tag}",
            fontsize=8,
        )

fig4.suptitle(
    f"BisectingKMeans centers by layer — {SOURCE_LABEL}  (k={N_CLUSTERS})",
    fontsize=12,
)
OUTPUT_PATH_LAYERS = os.path.join(
    PROJECT_ROOT, "output", f"{SOURCE_LABEL}_kmeans_layers.png"
)
fig4.savefig(OUTPUT_PATH_LAYERS, dpi=150)
print(f"Saved layer-by-layer centers to {OUTPUT_PATH_LAYERS}")

plt.show()
