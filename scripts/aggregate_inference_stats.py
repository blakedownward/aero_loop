#!/usr/bin/env python3
"""
Aggregate inference log statistics by model version.

Scans all inference_log.jsonl files in data/raw/[batch] directories,
aggregates statistics grouped by model version, and outputs a CSV table.
"""

import os
import json
import csv
from collections import defaultdict
from pathlib import Path


def _repo_root() -> str:
    """Get the repository root directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    # Script is in scripts/, so go up one level to get repo root
    return os.path.dirname(here)


RAW_PATH = os.path.join(_repo_root(), 'data', 'raw')


def is_negative(filename: str) -> bool:
    """
    Determine if a sample is negative based on filename.
    
    Args:
        filename: The filename from the inference log entry
        
    Returns:
        True if negative (filename starts with "000000"), False if aircraft
    """
    if not filename:
        return False
    return filename.startswith("000000")


def process_inference_log(batch_path: str) -> list:
    """
    Process a single inference_log.jsonl file and return entries.
    
    Args:
        batch_path: Path to the batch directory
        
    Returns:
        List of parsed entries (dicts) from the inference log
    """
    inference_log_path = os.path.join(batch_path, 'inference_log.jsonl')
    
    if not os.path.isfile(inference_log_path):
        return []
    
    entries = []
    try:
        with open(inference_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading {inference_log_path}: {e}")
        return []
    
    return entries


def aggregate_stats():
    """
    Scan all batch directories, process inference logs, and aggregate by model.
    
    Returns:
        Dictionary mapping model_version to statistics dict
    """
    if not os.path.isdir(RAW_PATH):
        print(f"Error: {RAW_PATH} does not exist")
        return {}
    
    # Dictionary to aggregate stats by model_version
    # Structure: {model_version: {total: 0, aircraft: 0, negative: 0, negative_deleted: 0}}
    stats_by_model = defaultdict(lambda: {
        'total_samples': 0,
        'aircraft_samples': 0,
        'total_negative_samples': 0,
        'negative_deleted': 0
    })
    
    # Scan all batch directories
    batch_dirs = [d for d in os.listdir(RAW_PATH) 
                  if os.path.isdir(os.path.join(RAW_PATH, d))]
    
    print(f"Found {len(batch_dirs)} batch directories")
    
    for batch_dir in sorted(batch_dirs):
        batch_path = os.path.join(RAW_PATH, batch_dir)
        entries = process_inference_log(batch_path)
        
        if not entries:
            continue
        
        print(f"Processing {batch_dir}: {len(entries)} entries")
        
        for entry in entries:
            model_version = entry.get('model_version', 'unknown')
            filename = entry.get('file', '')
            status = entry.get('status', 'unknown')
            
            # Update total samples
            stats_by_model[model_version]['total_samples'] += 1
            
            # Classify as aircraft or negative
            if is_negative(filename):
                # Negative sample
                stats_by_model[model_version]['total_negative_samples'] += 1
                if status == 'deleted':
                    stats_by_model[model_version]['negative_deleted'] += 1
            else:
                # Aircraft sample
                stats_by_model[model_version]['aircraft_samples'] += 1
    
    return dict(stats_by_model)


def output_csv(stats_by_model: dict, output_path: str = None):
    """
    Output aggregated statistics to CSV file.
    
    Args:
        stats_by_model: Dictionary mapping model_version to statistics
        output_path: Path to output CSV file (default: repo root)
    """
    if not stats_by_model:
        print("No statistics to output")
        return
    
    # Default to data folder if not specified
    if output_path is None:
        output_path = os.path.join(_repo_root(), 'data', 'inference_stats_by_model.csv')
    
    # Sort by model version for consistent output
    sorted_models = sorted(stats_by_model.keys())
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['model', 'total_samples', 'aircraft_samples', 
                        'total_negative_samples', 'negative_deleted'])
        
        # Write data rows
        for model in sorted_models:
            stats = stats_by_model[model]
            writer.writerow([
                model,
                stats['total_samples'],
                stats['aircraft_samples'],
                stats['total_negative_samples'],
                stats['negative_deleted']
            ])
    
    print(f"\nOutput written to: {output_path}")
    print(f"Processed {len(sorted_models)} model(s)")


def main():
    """Main entry point."""
    print("Aggregating inference log statistics by model...")
    print(f"Scanning: {RAW_PATH}\n")
    
    stats_by_model = aggregate_stats()
    
    if not stats_by_model:
        print("No data found. Exiting.")
        return
    
    # Print summary
    print("\n" + "="*60)
    print("Summary by Model:")
    print("="*60)
    for model in sorted(stats_by_model.keys()):
        stats = stats_by_model[model]
        print(f"\nModel: {model}")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Aircraft samples: {stats['aircraft_samples']}")
        print(f"  Total negative samples: {stats['total_negative_samples']}")
        print(f"  Negative deleted: {stats['negative_deleted']}")
    
    # Output CSV
    output_csv(stats_by_model)


if __name__ == '__main__':
    main()


