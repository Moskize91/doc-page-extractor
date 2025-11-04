from PIL import Image
from doc_page_extractor import PageExtractor

def main() -> None:
    extractor = PageExtractor()
    extractor.extract(
        image=Image.open("C:\\Users\\i\\Downloads\\ocr\\source.png"),
        size="gundam",
    )

if __name__ == "__main__":
    main()