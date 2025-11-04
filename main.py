from pathlib import Path
from PIL import Image
from doc_page_extractor import PageExtractor

def main() -> None:
    project_root = Path(__file__).parent
    image_dir_path = project_root / "tests" / "images"
    extractor = PageExtractor()

    for _, layouts in extractor.extract(
        image=Image.open(image_dir_path / "lost-citation.png"),
        size="gundam",
        stages=2,
    ):
        print("Layouts:")
        for layout in layouts:
            print(f"  Ref: {layout.ref}, Det: {layout.det}, Text: {layout.text}")

if __name__ == "__main__":
    main()