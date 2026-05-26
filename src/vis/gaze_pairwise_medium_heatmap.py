"""
Average gaze heatmap for medium-length pairwise responses.

Reads output/user_gazing_hist.csv and query_logs_table.csv, randomly picks
one (user_id, task_id, query_id) pair from the medium length_category that
has both gaze_pairwise_left and gaze_pairwise_right data, then visualises
the average histogram for medium/gaze_pairwise_{left,right} as a character-
level heatmap overlaid on the selected response texts.

Layout
------
  Row 0 (tall): left-response text panel  |  right-response text panel
                heatmap colour = avg fixation probability at each char position
  Row 1 (short): avg histogram bar chart  |  avg histogram bar chart
"""

import os
import csv
import random
from worker_filter import BAD_WORKERS
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
RANDOM_SEED = None   # set to an int for reproducible picks
EXCLUDE_BAD_WORKERS = True   # set False to include all workers
_qc = "filtered" if EXCLUDE_BAD_WORKERS else "all"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QUERY_LOGS_PATH = os.path.join(PROJECT_ROOT, "query_logs_table.csv")
_DATA_DIR       = os.path.join(PROJECT_ROOT, "output", "data")
_HEAT_DIR       = os.path.join(PROJECT_ROOT, "output", "heatmap")
os.makedirs(_HEAT_DIR, exist_ok=True)
GAZE_HIST_PATH  = os.path.join(_DATA_DIR, f"user_gazing_hist_{_qc}.csv")
OUTPUT_PATH     = os.path.join(_HEAT_DIR, f"gaze_pairwise_medium_heatmap_{_qc}.png")

N_BINS    = 100
BIN_COLS  = [f"bin_{i}" for i in range(N_BINS)]
BIN_EDGES = np.linspace(0, 1, N_BINS + 1)

# ---------------------------------------------------------------------------
# Style constants  (mirrors the reference trajectory visualiser)
# ---------------------------------------------------------------------------
WRAP      = 60
FONT_SIZE = 8
BG        = '#0b0b18'
PANBG     = '#10101e'
TXTCLR    = '#6b6b8a'
BORDERS   = ['#00b4d8', '#4cc9f0']
X0, Y0    = 0.02, 0.97


# ---------------------------------------------------------------------------
# Load gaze histogram CSV
# ---------------------------------------------------------------------------
hist_rows: list[dict] = []
with open(GAZE_HIST_PATH) as f:
    for row in csv.DictReader(f):
        hist_rows.append(row)

if EXCLUDE_BAD_WORKERS:
    hist_rows = [r for r in hist_rows if r['user_id'] not in BAD_WORKERS]

medium_left  = [r for r in hist_rows
                if r['length_category'] == 'medium' and r['source'] == 'gaze_pairwise_left']
medium_right = [r for r in hist_rows
                if r['length_category'] == 'medium' and r['source'] == 'gaze_pairwise_right']

print(f"Medium gaze_pairwise_left  rows: {len(medium_left)}")
print(f"Medium gaze_pairwise_right rows: {len(medium_right)}")


def avg_hist(rows: list[dict]) -> np.ndarray:
    arr = np.array([[float(r[c]) for c in BIN_COLS] for r in rows])
    return arr.mean(axis=0)


avg_left  = avg_hist(medium_left)
avg_right = avg_hist(medium_right)

# ---------------------------------------------------------------------------
# Find (user_id, task_id, query_id) pairs present in BOTH sources
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
# Load query logs and retrieve the selected response texts
# ---------------------------------------------------------------------------
query_data: dict[str, dict] = {}
with open(QUERY_LOGS_PATH) as f:
    for row in csv.DictReader(f):
        query_data[row['query_ID']] = row

qrow  = query_data[query_id]
text1 = qrow['llm_response_1']
text2 = qrow['llm_response_2']
ts    = qrow['query_timestamp']

print(f"Response 1 length: {len(text1)} chars")
print(f"Response 2 length: {len(text2)} chars")


# ---------------------------------------------------------------------------
# Text layout helpers  (identical to reference code)
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


def measure_char_dims(fig, ax):
    fp = FontProperties(size=FONT_SIZE, family='monospace')
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    ref = ax.text(0, 0, 'M', fontproperties=fp, transform=ax.transAxes, alpha=0)
    bbox = ref.get_window_extent(renderer=renderer)
    p0 = inv.transform((bbox.x0, bbox.y0))
    p1 = inv.transform((bbox.x1, bbox.y1))
    char_w = p1[0] - p0[0]
    char_h = p1[1] - p0[1]
    ref.remove()
    return char_w, char_h, char_h * 1.4, fp


# ---------------------------------------------------------------------------
# Text-panel heatmap drawing
# ---------------------------------------------------------------------------
def draw_heatmap_panel(fig, ax, text: str, avg_probs: np.ndarray,
                       border: str, source_label: str, n_samples: int):
    """Render response text with per-character heatmap based on avg_probs."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(PANBG)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(border)
        sp.set_linewidth(1.8)

    display_lines, c2p = build_map(text)
    char_w, char_h, line_h, fp = measure_char_dims(fig, ax)

    # Render text
    for ln, line in enumerate(display_lines):
        y = Y0 - ln * line_h
        if y < 0:
            break
        ax.text(X0, y, line, fontproperties=fp,
                transform=ax.transAxes,
                color=TXTCLR, va='top', ha='left', zorder=2)

    # Character-level heatmap: colour = avg fixation prob at that relative pos
    resp_len = max(len(text) - 1, 1)
    hcmap = plt.colormaps['YlOrRd']
    vmax  = avg_probs.max()

    for ci, (char_col, ln) in c2p.items():
        rel_pos   = ci / resp_len
        bin_idx   = min(int(rel_pos * N_BINS), N_BINS - 1)
        intensity = avg_probs[bin_idx] / vmax if vmax > 0 else 0
        if intensity < 0.02:
            continue
        rect = mpatches.FancyBboxPatch(
            (X0 + char_col * char_w, Y0 - ln * line_h - char_h),
            char_w, char_h,
            boxstyle='square,pad=0',
            facecolor=hcmap(intensity),
            alpha=0.45 * intensity + 0.05,
            linewidth=0, zorder=1,
        )
        ax.add_patch(rect)

    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=Normalize(0, vmax))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.009, pad=0.004)
    cb.set_label('Avg fixation probability (medium responses)', color='white', fontsize=7)
    cb.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=6)
    cb.outline.set_edgecolor(border)

    short = text[:65].replace('\n', ' ')
    ax.set_title(
        f'[{source_label}]  query {query_id}  ·  user {user_id}  ·  {ts}\n'
        f'"{short}…"  ·  {len(text)} chars  ·  heatmap = avg over {n_samples} medium samples',
        color='white', fontsize=8.5, pad=5, loc='left',
    )


# ---------------------------------------------------------------------------
# Average histogram bar-chart panel
# ---------------------------------------------------------------------------
def draw_hist_panel(ax, avg_probs: np.ndarray, border: str, source_label: str,
                    n_samples: int):
    bin_centers = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
    ax.set_facecolor(PANBG)
    for sp in ax.spines.values():
        sp.set_edgecolor(border); sp.set_linewidth(1.5)
    ax.tick_params(colors='white', labelsize=7)

    bars = ax.bar(bin_centers, avg_probs,
                  width=1 / N_BINS, align='center',
                  color=border, edgecolor='none', alpha=0.75)

    ax.set_xlim(0, 1)
    ax.set_xlabel('Relative position  (char_idx / response_length)',
                  color='white', fontsize=8)
    ax.set_ylabel('Average probability', color='white', fontsize=8)
    ax.set_title(f'[{source_label}]  avg histogram  (n={n_samples} medium samples)',
                 color=border, fontsize=9, pad=4)
    ax.yaxis.grid(True, color='#222244', linewidth=0.5, zorder=0)
    ax.xaxis.grid(True, color='#222244', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Assemble figure
# ---------------------------------------------------------------------------
_, c2p1 = build_map(text1)
_, c2p2 = build_map(text2)
n_lines1 = max((ln for _, ln in c2p1.values()), default=0) + 1
n_lines2 = max((ln for _, ln in c2p2.values()), default=0) + 1
text_rows = max(n_lines1, n_lines2)

# Row heights: text panel proportional to line count, bar chart fixed fraction
text_h  = max(text_rows * 0.17, 10)
bar_h   = 3
fig_h   = text_h + bar_h + 1.5

fig = plt.figure(figsize=(18, fig_h), facecolor=BG)
fig.suptitle(
    'Average Gaze Heatmap  —  Medium Response Length  ·  '
    'gaze_pairwise_left  |  gaze_pairwise_right\n'
    f'Randomly selected: user={user_id}  task={task_id}  query={query_id}  ·  '
    f'{len(medium_left)} left samples  /  {len(medium_right)} right samples',
    color='white', fontsize=13, fontweight='bold', y=1.0,
)

gs = gridspec.GridSpec(
    2, 2, figure=fig,
    left=0.01, right=0.99, top=0.94, bottom=0.04,
    hspace=0.18, wspace=0.04,
    height_ratios=[text_h, bar_h],
)
fig.canvas.draw()

ax_text_left  = fig.add_subplot(gs[0, 0])
ax_text_right = fig.add_subplot(gs[0, 1])
ax_hist_left  = fig.add_subplot(gs[1, 0])
ax_hist_right = fig.add_subplot(gs[1, 1])

draw_heatmap_panel(fig, ax_text_left,  text1, avg_left,  BORDERS[0],
                   'gaze_pairwise_left',  len(medium_left))
draw_heatmap_panel(fig, ax_text_right, text2, avg_right, BORDERS[1],
                   'gaze_pairwise_right', len(medium_right))

draw_hist_panel(ax_hist_left,  avg_left,  BORDERS[0], 'gaze_pairwise_left',  len(medium_left))
draw_hist_panel(ax_hist_right, avg_right, BORDERS[1], 'gaze_pairwise_right', len(medium_right))

plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f'Saved → {OUTPUT_PATH}')
