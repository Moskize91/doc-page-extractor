from PIL import Image
from doc_page_extractor.model import DeepSeekOCRModel

def main() -> None:
    model = DeepSeekOCRModel()
    result = model.generate(
        prompt="<image>\n<|grounding|>Convert the document to markdown.",
        image=Image.open("C:\\Users\\i\\Downloads\\ocr\\source.png"),
        size="gundam",
    )
    print("========Result========")
    print(result)

if __name__ == "__main__":
    main()