import os

from PIL import Image
from doc_page_extractor import plot, DocExtractor


def main():
  project_path = os.path.dirname(__file__)
  model_path = os.path.join(project_path, "model")
  plot_path = os.path.join(project_path, "plot")
  image_path = os.path.join(project_path, "images", "page1.png")
  os.makedirs(model_path, exist_ok=True)
  os.makedirs(plot_path, exist_ok=True)

  extractor = DocExtractor(model_path, "cpu")

  with Image.open(image_path) as image:
    layouts = extractor.extract(image, "ch")
    plot_image = image.copy()
    plot(plot_image, layouts)
    plot_image.save(os.path.join(plot_path, "output.png"))

    for layout in layouts:
      print("\n", layout.cls)
      for fragment in layout.fragments:
        print(fragment.rect, fragment.text)

if __name__ == "__main__":
  main()