"""
Move all model checkpoint files (.pt) from results/ to runs/ directory.
Preserves the directory structure and prevents uploading large files to GitHub.
"""

import os
import shutil
from pathlib import Path

def move_models_to_runs(source_root='results', target_root='runs'):
    """
    Move all *.pt files from results/ to runs/ while preserving directory structure.
    
    Example:
        results/CIFAR10/simplecnn/noise_0.0/Adam/run_1_best.pt
        -> runs/CIFAR10/simplecnn/noise_0.0/Adam/run_1_best.pt
    """
    source_path = Path(source_root)
    target_path = Path(target_root)
    
    if not source_path.exists():
        print(f"Source directory '{source_root}' does not exist.")
        return
    
    # Find all .pt files
    pt_files = list(source_path.rglob('*.pt'))
    
    if not pt_files:
        print(f"No .pt files found in '{source_root}'.")
        return
    
    print(f"Found {len(pt_files)} model checkpoint files.")
    moved_count = 0
    
    for pt_file in pt_files:
        # Calculate relative path
        relative = pt_file.relative_to(source_path)
        target_file = target_path / relative
        
        # Create target directory
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Move file
        try:
            shutil.move(str(pt_file), str(target_file))
            moved_count += 1
            print(f"Moved: {relative}")
        except Exception as e:
            print(f"Failed to move {relative}: {e}")
    
    print(f"\n✓ Successfully moved {moved_count}/{len(pt_files)} files to '{target_root}/'")
    print(f"✓ Add '{target_root}/' to .gitignore to exclude from version control")


if __name__ == "__main__":
    move_models_to_runs()
