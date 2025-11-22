"""Remove files from labels.csv to allow reprocessing.

This is useful when you need to reprocess files that were already processed.
Simply edit the files_to_remove list below with (batch, filename) tuples.
"""

import os
import csv
import sys

_repo_root = os.path.dirname(os.path.abspath(__file__))
LABELS_CSV = os.path.join(_repo_root, 'data', 'processed', 'labels.csv')

# Files to remove from labels.csv
# Format: (batch_name, filename)
# Example: ('2025-11-19_07-59', '000000_2025-11-19_08-04.wav')
files_to_remove = [
    # Add files here that you want to reprocess
    # ('batch_name', 'filename.wav'),
]

if not files_to_remove:
    print("No files specified to remove.")
    print("Edit the files_to_remove list in this script to specify files.")
    sys.exit(0)

if os.path.isfile(LABELS_CSV):
    # Read all rows
    rows = []
    removed_count = 0
    with open(LABELS_CSV, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            # Check if this row should be removed
            batch = row.get('batch', '')
            filename = row.get('filename', '')
            if (batch, filename) not in files_to_remove:
                rows.append(row)
            else:
                print(f"Removing: {batch} - {filename}")
                removed_count += 1
    
    # Write back without removed rows
    with open(LABELS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nRemoved {removed_count} entries from labels.csv")
    print(f"Kept {len(rows)} entries")
    
    if removed_count < len(files_to_remove):
        print(f"\nWarning: {len(files_to_remove) - removed_count} files were not found in labels.csv")
else:
    print("labels.csv not found")

