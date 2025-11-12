from pathlib import Path
import time

from PIL import Image

from doc_page_extractor import AbortContext, PageExtractor, plot

_ABORT_TIMEOUT = 5.0 # seconds

def main() -> None:
    project_root = Path(__file__).parent
    image_dir_path = project_root / "tests" / "images"
    image_name = "double_column.png"
    extractor = PageExtractor(
        model_path=project_root / "models-cache",
        local_only=False,
    )
    plot_dir = project_root / "plot"
    plot_dir.mkdir(exist_ok=True)
    name_stem = Path(image_name).stem
    name_suffix = Path(image_name).suffix
    current_time = time.time()

    def check_aborted() -> bool:
        if time.time() - current_time > _ABORT_TIMEOUT:
            print("Aborted extraction due to timeout.")
            return True
        return False

    for i, (image, layouts) in enumerate(
        extractor.extract(
            image=Image.open(image_dir_path / image_name),
            size="gundam",
            stages=2,
            aborted_context=AbortContext(check_aborted=check_aborted),
        )
    ):
        print("Layouts:")
        for layout in layouts:
            print(f"  Ref: {layout.ref}, Det: {layout.det}, Text: {layout.text}")
        image = plot(image.copy(), layouts)
        output_path = plot_dir / f"{name_stem}_{i}{name_suffix}"
        image.save(output_path)


if __name__ == "__main__":
    main()
