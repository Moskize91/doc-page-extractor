from pathlib import Path
from PIL import Image
from doc_page_extractor import plot, PageExtractor

def main() -> None:
    project_root = Path(__file__).parent
    image_dir_path = project_root / "tests" / "images"
    image_name = "lost-citation.png"
    extractor = PageExtractor()

    # 创建 plot 文件夹
    plot_dir = project_root / "plot"
    plot_dir.mkdir(exist_ok=True)

    # 获取文件名和扩展名
    name_stem = Path(image_name).stem  # 不带扩展名的文件名
    name_suffix = Path(image_name).suffix  # 扩展名

    for i, (image, layouts) in enumerate(extractor.extract(
        image=Image.open(image_dir_path / image_name),
        size="gundam",
        stages=2,
    )):
        print("Layouts:")
        for layout in layouts:
            print(f"  Ref: {layout.ref}, Det: {layout.det}, Text: {layout.text}")
        image = plot(image.copy(), layouts)
        
        # 保存图片到 plot 文件夹
        output_path = plot_dir / f"{name_stem}_{i}{name_suffix}"
        image.save(output_path)

if __name__ == "__main__":
    main()