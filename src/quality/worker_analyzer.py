"""Analyze worker quality based on behavioral metrics"""

import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from src.config.constants import QUALITY_STANDARDS


class WorkerAnalyzer:
    """Analyzes worker data quality based on behavioral metrics"""
    
    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.standards = QUALITY_STANDARDS
    
    def analyze_all_workers(self):
        """Analyze all workers and generate quality reports"""
        print(f"Quality Standards: Min {self.standards['min_total_rows']} rows, "
              f"{self.standards['min_text_rows']} text rows, "
              f"{self.standards['min_duration_minutes']}min, "
              f"{self.standards['min_text_percentage']}% text\n")
        
        # Collect stats for all users
        user_stats = self._collect_user_stats()
        
        # Identify low quality workers
        low_quality = self._identify_low_quality_workers(user_stats)
        
        # Print summary
        self._print_summary(user_stats, low_quality)
        
        # Write reports
        self._write_csv_report(low_quality)
        self._write_text_report(user_stats, low_quality)
    
    def _collect_user_stats(self) -> Dict:
        """Collect statistics for all users"""
        user_data = defaultdict(lambda: {
            'tasks': [],
            'total_text': 0,
            'total_rows': 0,
            'total_duration': 0,
            'task_count': 0
        })
        
        for user_dir in self.data_dir.iterdir():
            if not user_dir.is_dir() or user_dir.name.startswith('.'):
                continue
            
            user_id = user_dir.name
            
            for task_dir in user_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                
                # Collect stats from all CSV files in task
                all_stats = []
                for csv_file in task_dir.glob("rel_*.csv"):
                    stats = self._read_file_stats(csv_file)
                    if stats:
                        all_stats.append(stats)
                
                if all_stats:
                    avg_text = sum(s[0] for s in all_stats) / len(all_stats)
                    avg_total = sum(s[1] for s in all_stats) / len(all_stats)
                    avg_duration = sum(s[2] for s in all_stats) / len(all_stats)
                    text_pct = (avg_text / avg_total * 100) if avg_total > 0 else 0
                    
                    user_data[user_id]['tasks'].append(text_pct)
                    user_data[user_id]['total_text'] += avg_text
                    user_data[user_id]['total_rows'] += avg_total
                    user_data[user_id]['total_duration'] += avg_duration
                    user_data[user_id]['task_count'] += 1
        
        return user_data
    
    def _read_file_stats(self, file_path: Path) -> Optional[Tuple[int, int, float]]:
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
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    
    def _identify_low_quality_workers(self, user_data: Dict) -> List[Dict]:
        """Identify workers who don't meet quality standards"""
        low_quality = []
        
        for user_id, data in user_data.items():
            if data['task_count'] == 0:
                continue
            
            avg_total = data['total_rows'] / data['task_count']
            avg_text = data['total_text'] / data['task_count']
            avg_duration = data['total_duration'] / data['task_count']
            avg_text_pct = sum(data['tasks']) / len(data['tasks'])
            
            issues = []
            if avg_total < self.standards['min_total_rows']:
                issues.append(f"total={avg_total:.0f}")
            if avg_text < self.standards['min_text_rows']:
                issues.append(f"text={avg_text:.0f}")
            if avg_duration < self.standards['min_duration_minutes']:
                issues.append(f"duration={avg_duration:.1f}min")
            if avg_text_pct < self.standards['min_text_percentage']:
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
        return low_quality
    
    def _print_summary(self, user_data: Dict, low_quality: List[Dict]):
        """Print summary to console"""
        total_users = len(user_data)
        low_quality_count = len(low_quality)
        good_quality_count = total_users - low_quality_count
        
        print(f"Total users: {total_users} | Low quality: {low_quality_count} | "
              f"Good quality: {good_quality_count}\n")
        
        if low_quality:
            print("=" * 80)
            print("LOW QUALITY WORKERS")
            print("=" * 80)
            
            for u in low_quality:
                print(f"\n{u['user_id']} ({u['tasks']} tasks) | "
                      f"Rows: {u['avg_total_rows']:.0f} | "
                      f"Text: {u['avg_text_rows']:.0f} ({u['avg_text_pct']:.1f}%) | "
                      f"Duration: {u['avg_duration']:.1f}min")
                print(f"  Issues: {u['issues']}")
    
    def _write_csv_report(self, low_quality: List[Dict]):
        """Write CSV report of low quality workers"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / "low_quality_workers.csv"
        
        try:
            with open(output_file, "w", newline="") as f:
                fieldnames = ['user_id', 'tasks', 'avg_text_pct', 'avg_total_rows',
                            'avg_text_rows', 'avg_duration', 'issues']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(low_quality)
            
            print(f"\n{'=' * 80}")
            print(f"CSV report saved to: {output_file}")
            
        except Exception as e:
            print(f"Error writing CSV report: {e}")
    
    def _write_text_report(self, user_data: Dict, low_quality: List[Dict]):
        """Write comprehensive text report"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / "worker_quality_report.txt"
        
        try:
            with open(output_file, "w") as f:
                f.write("=" * 80 + "\n")
                f.write("WORKER QUALITY REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Quality Standards: Min {self.standards['min_total_rows']} rows, "
                       f"{self.standards['min_text_rows']} text rows, "
                       f"{self.standards['min_duration_minutes']}min, "
                       f"{self.standards['min_text_percentage']}% text\n\n")
                
                total = len(user_data)
                low = len(low_quality)
                f.write(f"Total users: {total} | Low quality: {low} | "
                       f"Good quality: {total - low}\n\n")
                
                # Write all workers section
                f.write("=" * 80 + "\n")
                f.write("ALL WORKERS\n")
                f.write("=" * 80 + "\n\n")
                
                all_users = []
                for user_id, data in user_data.items():
                    if data['task_count'] == 0:
                        continue
                    
                    all_users.append({
                        'user_id': user_id,
                        'tasks': data['task_count'],
                        'avg_total': data['total_rows'] / data['task_count'],
                        'avg_text': data['total_text'] / data['task_count'],
                        'avg_text_pct': sum(data['tasks']) / len(data['tasks']),
                        'avg_duration': data['total_duration'] / data['task_count']
                    })
                
                all_users.sort(key=lambda x: x['tasks'], reverse=True)
                
                for u in all_users:
                    f.write(f"{u['user_id']} ({u['tasks']} tasks) | "
                           f"Rows: {u['avg_total']:.0f} | "
                           f"Text: {u['avg_text']:.0f} ({u['avg_text_pct']:.1f}%) | "
                           f"Duration: {u['avg_duration']:.1f}min\n")
                
                # Write low quality section
                if low_quality:
                    f.write(f"\n{'=' * 80}\n")
                    f.write("LOW QUALITY WORKERS (WITH ISSUES)\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for u in low_quality:
                        f.write(f"{u['user_id']} ({u['tasks']} tasks) | "
                               f"Rows: {u['avg_total_rows']:.0f} | "
                               f"Text: {u['avg_text_rows']:.0f} ({u['avg_text_pct']:.1f}%) | "
                               f"Duration: {u['avg_duration']:.1f}min\n")
                        f.write(f"  Issues: {u['issues']}\n\n")
            
            print(f"Text report saved to: {output_file}")
            
        except Exception as e:
            print(f"Error writing text report: {e}")
