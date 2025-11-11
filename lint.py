#!/usr/bin/env python3
"""Cross-platform lint script for doc-page-extractor"""

import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).parent

    # Lint only the source code directory
    targets = [
        "doc_page_extractor",
    ]

    print("Running pylint...")
    print(f"Checking: {', '.join(targets)}")
    print()

    result = subprocess.run(["poetry", "run", "pylint"] + targets, cwd=project_root)

    if result.returncode == 0:
        print("\nLint passed!")
    else:
        print("\nLint failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
