"""
Migration script to reorganize old result format to new batch-size aware format.

Old format: results/{dataset}/{model}/noise_{noise}/{optimizer}/run_*.csv
New format: results/{dataset}/{model}/noise_{noise}/{optimizer}/batch_size_128/run_*.csv

This script:
1. Scans old directory structure (without batch_size subdirectories)
2. Moves files to new structure with batch_size_128/ subdirectory (default)
3. Handles both results/ and runs/ directories
4. Preserves all file metadata and content
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple
import json


def find_old_format_files(base_dir: str, file_pattern: str = "run_*") -> List[Path]:
    """
    Find files in old format (without batch_size subdirectories).
    
    Old structure:
        results/{dataset}/{model}/noise_{noise}/{optimizer}/run_*.csv
        results/{dataset}/{model}/noise_{noise}/{optimizer}/run_*.json
    
    Returns: List of paths to old-format files
    """
    old_files = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Directory not found: {base_dir}")
        return old_files
    
    # Pattern: base_dir/{dataset}/{model}/noise_{noise}/{optimizer}/run_*
    for dataset_dir in base_path.glob("*"):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("batch_size"):
            continue
        
        for model_dir in dataset_dir.glob("*"):
            if not model_dir.is_dir() or model_dir.name.startswith("batch_size"):
                continue
            
            for noise_dir in model_dir.glob("noise_*"):
                if not noise_dir.is_dir() or noise_dir.name.startswith("batch_size"):
                    continue
                
                for optimizer_dir in noise_dir.glob("*"):
                    if not optimizer_dir.is_dir() or optimizer_dir.name.startswith("batch_size"):
                        continue
                    
                    # Check if this has old-format run files (no batch_size subdirs)
                    has_old_files = False
                    for file_path in optimizer_dir.glob("run_*"):
                        if file_path.is_file():
                            has_old_files = True
                            old_files.append(file_path)
                    
                    # Also check for aggregated files in old format
                    for file_path in optimizer_dir.glob("aggregated*"):
                        if file_path.is_file():
                            has_old_files = True
                            old_files.append(file_path)
                    
                    for file_path in optimizer_dir.glob("test_acc_*"):
                        if file_path.is_file():
                            has_old_files = True
                            old_files.append(file_path)
    
    return old_files


def get_new_format_path(old_path: Path, base_dir: str, batch_size: int = 128) -> Path:
    """
    Convert old format path to new format path with batch_size subdirectory.
    
    Old: results/CIFAR10/simplecnn/noise_0.0/Adam/run_1_metrics.csv
    New: results/CIFAR10/simplecnn/noise_0.0/Adam/batch_size_128/run_1_metrics.csv
    """
    # Get relative path from base_dir
    rel_path = old_path.relative_to(base_dir)
    
    # Insert batch_size directory before the filename
    parts = list(rel_path.parts)
    
    # parts: ('CIFAR10', 'simplecnn', 'noise_0.0', 'Adam', 'run_1_metrics.csv')
    # Modify to: ('CIFAR10', 'simplecnn', 'noise_0.0', 'Adam', 'batch_size_128', 'run_1_metrics.csv')
    
    filename = parts[-1]
    directory_parts = parts[:-1]
    
    new_parts = list(directory_parts) + [f"batch_size_{batch_size}", filename]
    new_path = Path(base_dir) / Path(*new_parts)
    
    return new_path


def migrate_results(base_dir: str, batch_size: int = 128, dry_run: bool = False) -> Tuple[int, int]:
    """
    Migrate files from old format to new format.
    
    Args:
        base_dir: Base directory (results/ or runs/)
        batch_size: Default batch size for new format (default 128)
        dry_run: If True, show what would be done without actually doing it
    
    Returns:
        Tuple of (files_moved, errors)
    """
    print(f"\n{'='*80}")
    print(f"Migrating: {base_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Dry run: {dry_run}")
    print(f"{'='*80}\n")
    
    old_files = find_old_format_files(base_dir)
    
    if not old_files:
        print(f"No old-format files found in {base_dir}")
        return 0, 0
    
    print(f"Found {len(old_files)} old-format files:\n")
    
    files_moved = 0
    errors = 0
    
    for old_path in old_files:
        new_path = get_new_format_path(old_path, base_dir, batch_size)
        
        # Create new directory if needed
        new_dir = new_path.parent
        
        print(f"Old: {old_path.relative_to(base_dir)}")
        print(f"New: {new_path.relative_to(base_dir)}")
        
        try:
            if not dry_run:
                # Create batch_size directory
                new_dir.mkdir(parents=True, exist_ok=True)
                
                # Move file
                shutil.move(str(old_path), str(new_path))
                print(f"✓ Moved successfully\n")
            else:
                print(f"[DRY RUN] Would move\n")
            
            files_moved += 1
        
        except Exception as e:
            print(f"✗ Error: {e}\n")
            errors += 1
    
    return files_moved, errors


def cleanup_empty_directories(base_dir: str, dry_run: bool = False) -> int:
    """
    Remove empty optimizer directories (after files have been moved).
    
    Args:
        base_dir: Base directory (results/ or runs/)
        dry_run: If True, show what would be done without actually doing it
    
    Returns:
        Number of directories removed
    """
    print(f"\n{'='*80}")
    print(f"Cleaning up empty directories in: {base_dir}")
    print(f"Dry run: {dry_run}")
    print(f"{'='*80}\n")
    
    removed = 0
    base_path = Path(base_dir)
    
    # Walk from deepest to shallowest to remove empty dirs
    for dirpath in sorted(base_path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not dirpath.is_dir():
            continue
        
        # Skip batch_size directories (they should have files)
        if "batch_size" in dirpath.name:
            continue
        
        try:
            # Check if directory is empty
            contents = list(dirpath.iterdir())
            if not contents:
                print(f"Removing empty directory: {dirpath.relative_to(base_dir)}")
                if not dry_run:
                    dirpath.rmdir()
                removed += 1
        except OSError:
            pass
    
    return removed


def generate_migration_report(results_dir: str, runs_dir: str) -> dict:
    """
    Generate a report of what would be migrated.
    
    Returns:
        Dictionary with statistics
    """
    report = {
        "results_old_files": len(find_old_format_files(results_dir)),
        "runs_old_files": len(find_old_format_files(runs_dir)),
    }
    
    return report


def main():
    """Main migration workflow."""
    
    # Directories to migrate
    RESULTS_DIR = "results"
    RUNS_DIR = "runs"
    DEFAULT_BATCH_SIZE = 128
    
    print("\n" + "="*80)
    print("SR-Adam Results Format Migration Tool")
    print("Old format: {base}/{dataset}/{model}/noise_{noise}/{optimizer}/run_*.csv")
    print("New format: {base}/{dataset}/{model}/noise_{noise}/{optimizer}/batch_size_128/run_*.csv")
    print("="*80)
    
    # Generate report
    print("\n[1] Analyzing existing files...")
    report = generate_migration_report(RESULTS_DIR, RUNS_DIR)
    print(f"    Results directory: {report['results_old_files']} old-format files found")
    print(f"    Runs directory: {report['runs_old_files']} old-format files found")
    
    total_files = report['results_old_files'] + report['runs_old_files']
    if total_files == 0:
        print("\n✓ No old-format files found. Your results are already in the new format!")
        return
    
    # Dry run
    print("\n[2] Performing dry run (preview mode)...")
    migrate_results(RESULTS_DIR, DEFAULT_BATCH_SIZE, dry_run=True)
    migrate_results(RUNS_DIR, DEFAULT_BATCH_SIZE, dry_run=True)
    
    # Ask for confirmation
    print("\n" + "="*80)
    response = input(f"\nProceed with migration of {total_files} files? (yes/no): ").strip().lower()
    
    if response not in ["yes", "y"]:
        print("Migration cancelled.")
        return
    
    # Actual migration
    print("\n[3] Performing actual migration...")
    results_moved, results_errors = migrate_results(RESULTS_DIR, DEFAULT_BATCH_SIZE, dry_run=False)
    runs_moved, runs_errors = migrate_results(RUNS_DIR, DEFAULT_BATCH_SIZE, dry_run=False)
    
    # Cleanup
    print("\n[4] Cleaning up empty directories...")
    results_cleaned = cleanup_empty_directories(RESULTS_DIR, dry_run=False)
    runs_cleaned = cleanup_empty_directories(RUNS_DIR, dry_run=False)
    
    # Summary
    print("\n" + "="*80)
    print("MIGRATION SUMMARY")
    print("="*80)
    print(f"\nResults directory:")
    print(f"  Files moved: {results_moved}")
    print(f"  Errors: {results_errors}")
    print(f"  Empty dirs cleaned: {results_cleaned}")
    
    print(f"\nRuns directory:")
    print(f"  Files moved: {runs_moved}")
    print(f"  Errors: {runs_errors}")
    print(f"  Empty dirs cleaned: {runs_cleaned}")
    
    print(f"\nTotal:")
    print(f"  Files moved: {results_moved + runs_moved}")
    print(f"  Errors: {results_errors + runs_errors}")
    print(f"  Empty dirs cleaned: {results_cleaned + runs_cleaned}")
    
    if results_errors + runs_errors == 0:
        print("\n✓ Migration completed successfully!")
    else:
        print(f"\n⚠ Migration completed with {results_errors + runs_errors} errors")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
