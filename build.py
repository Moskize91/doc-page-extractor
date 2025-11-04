#!/usr/bin/env python3
"""Cross-platform build script for doc-page-extractor"""

import shutil
import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).parent
    dist_dir = project_root / "dist"

    # 1. Clean old build artifacts
    print("Cleaning old build artifacts...")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
        print("✓ Removed dist/ directory")
    else:
        print("✓ dist/ directory does not exist")

    # 2. Build the package
    print("\nBuilding package...")
    result = subprocess.run(["poetry", "build"], cwd=project_root)

    if result.returncode == 0:
        print("\n✓ Build completed successfully!")
        if dist_dir.exists():
            print("\nBuilt files:")
            for file in dist_dir.iterdir():
                print(f"  - {file.name}")
    else:
        print("\n✗ Build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
