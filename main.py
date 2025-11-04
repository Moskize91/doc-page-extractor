from PIL import Image
from doc_page_extractor import PageExtractor

def main() -> None:
    extractor = PageExtractor()
    for _, layouts in extractor.extract(
        image=Image.open("C:\\Users\\i\\codes\\github.com\\moskize91\\doc-page-extractor\\tests\\images\\page3.png"),
        size="gundam",
        stages=2,
    ):
        print("Layouts:")
        for layout in layouts:
            print(f"  Ref: {layout.ref}, Det: {layout.det}, Text: {layout.text}")

if __name__ == "__main__":
    main()