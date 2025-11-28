import time
from pathlib import Path

from doc_page_extractor import plot, create_page_extractor, ExtractionContext

_ABORT_TIMEOUT = 9999.0  # seconds


def main() -> None:
    project_root = Path(__file__).parent
    image_dir_path = project_root / "tests" / "images"
    image_name = "double_column.png"
    extractor = create_page_extractor(
        model_path=project_root / "models-cache",
        local_only=False,
    )
    begin_at = time.time()
    extractor.load_models()
    print(f"Models loaded in {time.time() - begin_at:.2f} seconds.")

    plot_dir = project_root / "plot"
    plot_dir.mkdir(exist_ok=True)
    name_stem = Path(image_name).stem
    name_suffix = Path(image_name).suffix
    begin_at = time.time()

    def check_aborted() -> bool:
        if time.time() - begin_at > _ABORT_TIMEOUT:
            print("Aborted extraction due to timeout.")
            return True
        return False

    print("Starting extraction...")
    for i, get_pair in enumerate(
        extractor.extract(
            image_path=image_dir_path / image_name,
            size="gundam",
            stages=2,
            context=ExtractionContext(check_aborted=check_aborted),
        )
    ):
        print("Layouts:")
        image, layouts = get_pair()
        for layout in layouts:
            print(f"  Ref: {layout.ref}, Det: {layout.det}, Text: {layout.text}")
        image = plot(image.copy(), layouts)
        output_path = plot_dir / f"{name_stem}_{i}{name_suffix}"
        image.save(output_path)

    print(f"Extraction cost {time.time() - begin_at:.2f} seconds.")

if __name__ == "__main__":
    main()
