"""
Pymovements-based interpolated gaze heatmap for pairwise responses.

Reads output/user_gazing_hist.csv and query_logs_table.csv, randomly picks one
(user_id, task_id, query_id) pair from the selected LENGTH_CATEGORY that has both
gaze_pairwise_left and gaze_pairwise_right data.

The average histogram for each source is converted to a 2-D weighted point cloud
at character-grid positions; pymovements pm.plotting.heatmap then renders it with
Gaussian interpolation.  The response text is shown as a background stimulus.

Layout
------
  Row 0  (tall): pm.plotting.heatmap panels — left response | right response
  Row 1 (short): average histogram bar charts for direct comparison
"""

import os
import csv
import random
import shutil
from worker_filter import BAD_WORKERS
import tempfile
import numpy as np
import polars as pl
import pymovements as pm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties


def _measure_char_dims(font_size: int, family: str = 'monospace', dpi: int = 100) -> tuple[int, int]:
    """Return (char_width_px, line_height_px) as rendered by matplotlib at the given DPI."""
    fig, ax = plt.subplots(figsize=(6, 2), dpi=dpi)
    fp = FontProperties(size=font_size, family=family)
    t = ax.text(0, 0.5, 'M' * 20, fontproperties=fp, transform=ax.transData)
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    char_w  = max(1, round(bb.width / 20))
    line_h  = max(1, round(bb.height * 1.4))   # add ~40% leading
    plt.close(fig)
    return char_w, line_h

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QUERY_LOGS_PATH = os.path.join(PROJECT_ROOT, "query_logs_table.csv")

N_BINS    = 100
BIN_COLS  = [f"bin_{i}" for i in range(N_BINS)]
BIN_EDGES = np.linspace(0, 1, N_BINS + 1)

# ---------------------------------------------------------------------------
# Style constants  (mirrors the reference trajectory visualiser)
# ---------------------------------------------------------------------------
WRAP       = 60
FONT_SIZE  = 14
BG         = 'white'
PANBG      = '#f5f5f5'
TXTCLR     = '#222222'
BORDERS    = ['#0077b6', '#023e8a']

# Character-grid coordinate space for pymovements
CHAR_SCALE, LINE_SCALE = _measure_char_dims(FONT_SIZE)  # width / line-height in px
N_TOTAL     = 8000  # total weighted gaze samples per panel
JITTER_STD  = 1  # Gaussian position jitter (char units) for smooth interpolation

RANDOM_SEED = 81  # set to int for reproducible pair selection
EXCLUDE_BAD_WORKERS = True   # set False to include all workers
_qc = "filtered" if EXCLUDE_BAD_WORKERS else "all"

# ---------------------------------------------------------------------------
# *** Hyperparameters ***
# ---------------------------------------------------------------------------
LENGTH_CATEGORY = 'long'   # one of: 'short' | 'medium' | 'long'
MODALITY        = 'mouse'     # one of: 'gaze'  | 'mouse'
# ---------------------------------------------------------------------------

assert LENGTH_CATEGORY in ('short', 'medium', 'long'), \
    f"LENGTH_CATEGORY must be 'short', 'medium', or 'long', got '{LENGTH_CATEGORY}'"
assert MODALITY in ('gaze', 'mouse'), \
    f"MODALITY must be 'gaze' or 'mouse', got '{MODALITY}'"

# Derived from MODALITY
_DATA_DIR       = os.path.join(PROJECT_ROOT, "output", "data")
_HEAT_DIR       = os.path.join(PROJECT_ROOT, "output", "heatmap")
_MAIN_PAPER_DIR = os.path.join(PROJECT_ROOT, "output", "main_paper_images")
os.makedirs(_HEAT_DIR,       exist_ok=True)
os.makedirs(_MAIN_PAPER_DIR, exist_ok=True)
_HIST_FILE = f'user_gazing_hist_{_qc}.csv' if MODALITY == 'gaze' else f'user_mouse_hist_{_qc}.csv'
HIST_PATH  = os.path.join(_DATA_DIR, _HIST_FILE)
SRC_LEFT   = f'{MODALITY}_pairwise_left'
SRC_RIGHT  = f'{MODALITY}_pairwise_right'

OUTPUT_PATH = os.path.join(_HEAT_DIR,
                           f"{MODALITY}_pairwise_{LENGTH_CATEGORY}_pm_heatmap_{_qc}.png")


# ---------------------------------------------------------------------------
# Load gaze histogram CSV
# ---------------------------------------------------------------------------
hist_rows: list[dict] = []
with open(HIST_PATH) as f:
    for row in csv.DictReader(f):
        hist_rows.append(row)

if EXCLUDE_BAD_WORKERS:
    hist_rows = [r for r in hist_rows if r['user_id'] not in BAD_WORKERS]

medium_left  = [r for r in hist_rows
                if r['length_category'] == LENGTH_CATEGORY
                and r['source'] == SRC_LEFT]
medium_right = [r for r in hist_rows
                if r['length_category'] == LENGTH_CATEGORY
                and r['source'] == SRC_RIGHT]

print(f"{LENGTH_CATEGORY} {SRC_LEFT}  rows: {len(medium_left)}")
print(f"{LENGTH_CATEGORY} {SRC_RIGHT} rows: {len(medium_right)}")


def avg_hist(rows: list[dict]) -> np.ndarray:
    arr = np.array([[float(r[c]) for c in BIN_COLS] for r in rows])
    return arr.mean(axis=0)


avg_left  = avg_hist(medium_left)
avg_right = avg_hist(medium_right)

# ---------------------------------------------------------------------------
# Pick a random pair present in both sources
# ---------------------------------------------------------------------------
left_keys  = {(r['user_id'], r['task_id'], r['query_id']) for r in medium_left}
right_keys = {(r['user_id'], r['task_id'], r['query_id']) for r in medium_right}
pair_keys  = sorted(left_keys & right_keys)
print(f"Pairs with both left+right: {len(pair_keys)}")

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
user_id, task_id, query_id = random.choice(pair_keys)
print(f"Selected pair: user={user_id}  task={task_id}  query={query_id}")

# ---------------------------------------------------------------------------
# Retrieve response texts from query logs
# ---------------------------------------------------------------------------
query_data: dict[str, dict] = {}
with open(QUERY_LOGS_PATH) as f:
    for row in csv.DictReader(f):
        query_data[row['query_ID']] = row

qrow  = query_data[query_id]
text1 = qrow['llm_response_1']
text2 = qrow['llm_response_2']
ts    = qrow['query_timestamp']

print(f"Response 1: {len(text1)} chars  |  Response 2: {len(text2)} chars")


# ---------------------------------------------------------------------------
# Text layout helper  (identical to reference code)
# ---------------------------------------------------------------------------
def build_map(text: str, wrap_width: int = WRAP):
    char_to_pos: dict[int, tuple[int, int]] = {}
    display_lines = ['']
    col, line_num = 0, 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '\n':
            char_to_pos[i] = (col, line_num)
            display_lines.append('')
            line_num += 1
            col = 0
            i += 1
        elif ch in ' \t\r':
            char_to_pos[i] = (col, line_num)
            display_lines[-1] += ch
            col += 1
            i += 1
        else:
            j = i
            while j < len(text) and text[j] not in ' \t\r\n':
                j += 1
            word_len = j - i
            if col > 0 and col + word_len > wrap_width:
                display_lines.append('')
                line_num += 1
                col = 0
            for k in range(word_len):
                char_to_pos[i + k] = (col + k, line_num)
                display_lines[-1] += text[i + k]
            col += word_len
            i = j
    while display_lines and display_lines[-1].strip() == '':
        display_lines.pop()
    return display_lines, char_to_pos


# ---------------------------------------------------------------------------
# Render response text to a PNG  (used as heatmap background stimulus)
# ---------------------------------------------------------------------------
def render_text_background(display_lines: list[str], n_lines: int,
                           filepath: str, dpi: int = 100) -> None:
    """Save monospace text on background to PNG for use as pm stimulus.

    The image is sized to exactly WRAP*CHAR_SCALE × n_lines*CHAR_SCALE pixels
    so that it maps 1-to-1 onto the pymovements pixel coordinate space and
    text characters align with their heatmap positions.
    """
    fig_w = WRAP * CHAR_SCALE / dpi        # exact pixel match in x
    fig_h = n_lines * LINE_SCALE / dpi    # exact pixel match in y
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, WRAP)
    ax.set_ylim(n_lines, 0)   # y=0 at top, y=n_lines at bottom
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)   # no margins
    fp = FontProperties(size=FONT_SIZE, family='monospace')
    for ln, line in enumerate(display_lines):
        ax.text(0, ln + 0.5, line, fontproperties=fp,
                color=TXTCLR, va='center', ha='left', clip_on=True)
    fig.savefig(filepath, dpi=dpi, facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Build weighted gaze point cloud from average histogram
# ---------------------------------------------------------------------------
def make_gaze_object(text: str, c2p: dict, avg_probs: np.ndarray,
                     n_lines: int, rng: np.random.Generator,
                     jitter_x: float = JITTER_STD,
                     jitter_y: float = JITTER_STD) -> pm.Gaze:
    """
    Convert average 1-D histogram to a 2-D weighted gaze point cloud and wrap
    it in a pm.Gaze object in the character-grid pixel coordinate space.
    """
    resp_len = max(len(text) - 1, 1)
    xs_list, ys_list = [], []

    for ci, (col, ln) in c2p.items():
        rel_pos = ci / resp_len
        bin_idx = min(int(rel_pos * N_BINS), N_BINS - 1)
        weight  = avg_probs[bin_idx]
        n_samp  = max(round(weight * N_TOTAL), 0)
        if n_samp == 0:
            continue
        x_base = col * CHAR_SCALE + CHAR_SCALE // 2
        y_base = ln  * LINE_SCALE + LINE_SCALE // 2
        xs_list.append(rng.normal(x_base, jitter_x, n_samp))
        ys_list.append(rng.normal(y_base, jitter_y, n_samp))

    xs = np.clip(np.concatenate(xs_list), 0, WRAP * CHAR_SCALE - 1)
    ys = np.clip(np.concatenate(ys_list), 0, n_lines * LINE_SCALE - 1)

    df = pl.DataFrame({'x_pix': xs.tolist(), 'y_pix': ys.tolist()})
    exp = pm.Experiment(
        screen_width_px=WRAP * CHAR_SCALE,
        screen_height_px=n_lines * LINE_SCALE,
        screen_width_cm=40, screen_height_cm=30,
        distance_cm=60,
        origin='upper left',
        sampling_rate=1000,     # samples/s; heatmap divides count by this → [s]
    )
    return pm.Gaze(data=df, pixel_columns=['x_pix', 'y_pix'], experiment=exp)


# ---------------------------------------------------------------------------
# Average histogram bar-chart panel
# ---------------------------------------------------------------------------
def draw_hist_panel(ax, avg_probs: np.ndarray, border: str,
                    source_label: str, n_samples: int) -> None:
    bin_centers = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
    ax.set_facecolor(PANBG)
    for sp in ax.spines.values():
        sp.set_edgecolor(border); sp.set_linewidth(1.5)
    ax.tick_params(colors='black', labelsize=7)
    ax.bar(bin_centers, avg_probs, width=1 / N_BINS, align='center',
           color=border, edgecolor='none', alpha=0.75)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Relative position  (char_idx / response_length)',
                  color='black', fontsize=8)
    ax.set_ylabel('Average probability', color='black', fontsize=8)
    ax.set_title(f'[{source_label}]  avg histogram  (n={n_samples} {LENGTH_CATEGORY} samples)',
                 color=border, fontsize=9, pad=4)
    ax.yaxis.grid(True, color='#cccccc', linewidth=0.5, zorder=0)
    ax.xaxis.grid(True, color='#cccccc', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Compute layout dimensions
# ---------------------------------------------------------------------------
display_lines1, c2p1 = build_map(text1)
display_lines2, c2p2 = build_map(text2)
n_lines1 = len(display_lines1)
n_lines2 = len(display_lines2)
n_lines  = max(n_lines1, n_lines2)

screen_w = WRAP   * CHAR_SCALE
screen_h = n_lines * CHAR_SCALE

# Figure height: heatmap panel height is proportional to text line count;
# bar chart row is fixed.
heat_h   = max(n_lines * 0.22, 6)   # inches
fig_h    = heat_h + 0.5

# Anisotropic jitter: scale per-axis so Gaussian spread is isotropic in display space.
# Each panel occupies fig_width * col_frac inches wide and fig_h * row_frac inches tall.
# GridSpec: left=0.03, right=0.97, wspace=0.06 (fraction of avg col width), 2 cols
# → col_width_frac = 0.94 / (2 + 0.06); GridSpec: top=0.97, bottom=0.05 → row_frac=0.92
_ax_w_in   = 18 * 0.94 / (2 + 0.06)
_ax_h_in   = fig_h * 0.92
_char_w_in = _ax_w_in / WRAP          # display inches per character column
_char_h_in = _ax_h_in / n_lines       # display inches per text line
# Normalise to y; shrink x jitter when characters appear wider than tall
jitter_x = JITTER_STD * (_char_h_in / _char_w_in)
jitter_y = JITTER_STD

rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Render text background images to temp files
# ---------------------------------------------------------------------------
tmp_dir   = tempfile.mkdtemp()
bg_path1  = os.path.join(tmp_dir, 'bg_left.png')
bg_path2  = os.path.join(tmp_dir, 'bg_right.png')
render_text_background(display_lines1, n_lines, bg_path1)
render_text_background(display_lines2, n_lines, bg_path2)

# ---------------------------------------------------------------------------
# Build pm.Gaze objects
# ---------------------------------------------------------------------------
gaze_left  = make_gaze_object(text1, c2p1, avg_left,  n_lines, rng, jitter_x, jitter_y)
gaze_right = make_gaze_object(text2, c2p2, avg_right, n_lines, rng, jitter_x, jitter_y)

# ---------------------------------------------------------------------------
# Assemble figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(18, fig_h), facecolor=BG)

gs = gridspec.GridSpec(
    1, 3, figure=fig,
    left=0.03, right=0.97, top=0.97, bottom=0.05,
    wspace=0.04,
    width_ratios=[1, 0.05, 1],
)

ax_heat_left  = fig.add_subplot(gs[0, 0])
ax_cbar       = fig.add_subplot(gs[0, 1])
ax_heat_right = fig.add_subplot(gs[0, 2])

# ── pymovements heatmaps (colorbars disabled; shared scale added manually) ──
pm.plotting.heatmap(
    gaze=gaze_left,
    position_column='pixel',
    gridsize=[WRAP, n_lines],
    cmap='jet',
    interpolation='gaussian',
    origin='upper',
    show_cbar=False,
    title='',
    xlabel='Character column',
    ylabel='Text line',
    show=False,
    add_stimulus=True,
    path_to_image_stimulus=bg_path1,
    stimulus_origin='upper',
    alpha=0.70,
    ax=ax_heat_left,
)

pm.plotting.heatmap(
    gaze=gaze_right,
    position_column='pixel',
    gridsize=[WRAP, n_lines],
    cmap='jet',
    interpolation='gaussian',
    origin='upper',
    show_cbar=False,
    title='',
    xlabel='Character column',
    ylabel='Text line',
    show=False,
    add_stimulus=True,
    path_to_image_stimulus=bg_path2,
    stimulus_origin='upper',
    alpha=0.70,
    ax=ax_heat_right,
)

# Shared colour scale: heatmap image is the last imshow artist in each axes
img_left  = ax_heat_left.get_images()[-1]
img_right = ax_heat_right.get_images()[-1]
shared_vmax = max(float(img_left.get_array().max()),
                  float(img_right.get_array().max()))
for img in (img_left, img_right):
    img.set_clim(0, shared_vmax)

# Style heatmap panel borders
for ax, border in [(ax_heat_left, BORDERS[0]), (ax_heat_right, BORDERS[1])]:
    for sp in ax.spines.values():
        sp.set_edgecolor(border); sp.set_linewidth(2.0)
    ax.tick_params(colors='black', labelsize=12)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)

# Single shared colorbar in the centre column
cbar = fig.colorbar(img_right, cax=ax_cbar)
cbar.ax.yaxis.set_label_position('left')
cbar.set_label(f'Avg fixation weight [a.u.]  ({LENGTH_CATEGORY})', fontsize=13, labelpad=6)
cbar.ax.tick_params(labelsize=12)

# Nudge colorbar slightly left so it sits visually centred between the panels
fig.canvas.draw()
_pos = ax_cbar.get_position()
ax_cbar.set_position([_pos.x0 - 0.01, _pos.y0, _pos.width, _pos.height])

plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig)
shutil.copy(OUTPUT_PATH, _MAIN_PAPER_DIR)

# Clean up temp files
shutil.rmtree(tmp_dir, ignore_errors=True)

print(f'Saved → {OUTPUT_PATH}')
