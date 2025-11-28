import time
from pathlib import Path

from doc_page_extractor import create_page_extractor


_REVISION = "9f30c71f441d010e5429c532364a86705536c53a"

def main() -> None:
    project_root = Path(__file__).parent
    extractor = create_page_extractor(
        model_path=project_root / "models-cache",
        local_only=False,
    )
    begin_at = time.time()
    extractor.download_models(_REVISION)
    print(f"Models downloaded cost {time.time() - begin_at:.2f} seconds.")

if __name__ == "__main__":
    main()