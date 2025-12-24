import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict

STANDARDS = {
    'min_total_rows': 2000,
    'min_text_rows': 500,
    'min_duration': 2.5,
    'min_text_pct': 10.0
}

print(f"Quality Standards: Min {STANDARDS['min_total_rows']} rows, {STANDARDS['min_text_rows']} text rows, "
      f"{STANDARDS['min_duration']}min, {STANDARDS['min_text_pct']}% text\n")

def read_file_stats(file_path):
    """Extract text_rows, total_rows, duration from CSV file"""
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', header=None)
        total_rows = len(df)
        text_rows = df[2].notna().sum() if 2 in df.columns else 0
        
        if 4 in df.columns:
            timestamps = pd.to_numeric(df[4], errors='coerce').dropna()
            duration_min = (timestamps.max() - timestamps.min()) / 1000 / 60 if len(timestamps) > 0 else 0
        else:
            duration_min = 0
        
        return text_rows, total_rows, duration_min
    except:
        return None

user_data = defaultdict(lambda: {'tasks': [], 'total_text': 0, 'total_rows': 0, 'total_duration': 0, 'task_count': 0})

for user_dir in Path("user_behavior").iterdir():
    if not user_dir.is_dir() or user_dir.name.startswith('.'):
        continue
    
    for task_dir in user_dir.iterdir():
        if not task_dir.is_dir():
            continue
        
        all_stats = []
        for csv_file in task_dir.glob("rel_*.csv"):
            stats = read_file_stats(csv_file)
            if stats:
                all_stats.append(stats)
        
        if all_stats:
            avg_text = sum(s[0] for s in all_stats) / len(all_stats)
            avg_total = sum(s[1] for s in all_stats) / len(all_stats)
            avg_duration = sum(s[2] for s in all_stats) / len(all_stats)
            text_pct = (avg_text / avg_total * 100) if avg_total > 0 else 0
            
            user_id = user_dir.name
            user_data[user_id]['tasks'].append(text_pct)
            user_data[user_id]['total_text'] += avg_text
            user_data[user_id]['total_rows'] += avg_total
            user_data[user_id]['total_duration'] += avg_duration
            user_data[user_id]['task_count'] += 1

low_quality = []
for user_id, data in user_data.items():
    if data['task_count'] == 0:
        continue
    
    avg_total = data['total_rows'] / data['task_count']
    avg_text = data['total_text'] / data['task_count']
    avg_duration = data['total_duration'] / data['task_count']
    avg_text_pct = sum(data['tasks']) / len(data['tasks'])
    
    issues = []
    if avg_total < STANDARDS['min_total_rows']:
        issues.append(f"total={avg_total:.0f}")
    if avg_text < STANDARDS['min_text_rows']:
        issues.append(f"text={avg_text:.0f}")
    if avg_duration < STANDARDS['min_duration']:
        issues.append(f"duration={avg_duration:.1f}min")
    if avg_text_pct < STANDARDS['min_text_pct']:
        issues.append(f"text%={avg_text_pct:.1f}%")
    
    if issues:
        low_quality.append({
            'user_id': user_id,
            'tasks': data['task_count'],
            'avg_text_pct': avg_text_pct,
            'avg_total_rows': avg_total,
            'avg_text_rows': avg_text,
            'avg_duration': avg_duration,
            'issues': ', '.join(issues)
        })

low_quality.sort(key=lambda x: len(x['issues']), reverse=True)

print(f"Total users: {len(user_data)} | Low quality: {len(low_quality)} | Good quality: {len(user_data) - len(low_quality)}\n")

if low_quality:
    print("="*80)
    print("LOW QUALITY WORKERS")
    print("="*80)
    for u in low_quality:
        print(f"\n{u['user_id']} ({u['tasks']} tasks) | Rows: {u['avg_total_rows']:.0f} | Text: {u['avg_text_rows']:.0f} ({u['avg_text_pct']:.1f}%) | Duration: {u['avg_duration']:.1f}min")
        print(f"  Issues: {u['issues']}")

with open("low_quality_workers.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=['user_id', 'tasks', 'avg_text_pct', 'avg_total_rows', 'avg_text_rows', 'avg_duration', 'issues'])
    writer.writeheader()
    writer.writerows(low_quality)

# Write comprehensive text report
with open("worker_quality_report.txt", "w") as f:
    f.write("="*80 + "\n")
    f.write("WORKER QUALITY REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Quality Standards: Min {STANDARDS['min_total_rows']} rows, {STANDARDS['min_text_rows']} text rows, "
            f"{STANDARDS['min_duration']}min, {STANDARDS['min_text_pct']}% text\n\n")
    f.write(f"Total users: {len(user_data)} | Low quality: {len(low_quality)} | Good quality: {len(user_data) - len(low_quality)}\n\n")
    
    f.write("="*80 + "\n")
    f.write("ALL WORKERS\n")
    f.write("="*80 + "\n\n")
    
    # Sort all users by task count
    all_users = []
    for user_id, data in user_data.items():
        if data['task_count'] == 0:
            continue
        avg_total = data['total_rows'] / data['task_count']
        avg_text = data['total_text'] / data['task_count']
        avg_duration = data['total_duration'] / data['task_count']
        avg_text_pct = sum(data['tasks']) / len(data['tasks'])
        all_users.append({
            'user_id': user_id,
            'tasks': data['task_count'],
            'avg_total': avg_total,
            'avg_text': avg_text,
            'avg_text_pct': avg_text_pct,
            'avg_duration': avg_duration
        })
    
    all_users.sort(key=lambda x: x['tasks'], reverse=True)
    
    for u in all_users:
        f.write(f"{u['user_id']} ({u['tasks']} tasks) | Rows: {u['avg_total']:.0f} | Text: {u['avg_text']:.0f} ({u['avg_text_pct']:.1f}%) | Duration: {u['avg_duration']:.1f}min\n")
    
    if low_quality:
        f.write(f"\n{'='*80}\n")
        f.write("LOW QUALITY WORKERS (WITH ISSUES)\n")
        f.write("="*80 + "\n\n")
        
        for u in low_quality:
            f.write(f"{u['user_id']} ({u['tasks']} tasks) | Rows: {u['avg_total_rows']:.0f} | Text: {u['avg_text_rows']:.0f} ({u['avg_text_pct']:.1f}%) | Duration: {u['avg_duration']:.1f}min\n")
            f.write(f"  Issues: {u['issues']}\n\n")

print(f"\n{'='*80}")
print("Saved to: low_quality_workers.csv & worker_quality_report.txt")
