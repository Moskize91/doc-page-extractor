import os

from PIL import Image
from doc_page_extractor import plot, clip, DocExtractor, TableLayout, FormulaLayout


def main():
  project_path = os.path.dirname(__file__)
  model_path = os.path.join(project_path, "model")
  plot_path = os.path.join(project_path, "output")
  image_path = os.path.join(project_path, "tests", "images", "table.png")
  os.makedirs(model_path, exist_ok=True)
  os.makedirs(plot_path, exist_ok=True)

  extractor = DocExtractor(
    model_dir_path=model_path,
    device="cpu",
  )

  with Image.open(image_path) as image:
    result = extractor.extract(image)
    plot_image: Image.Image
    if result.adjusted_image is None:
      plot_image = image.copy()
    else:
      plot_image = result.adjusted_image

    plot(plot_image, result.layouts)
    clip_image = clip(result, result.layouts[0], 120.0, 240.0)
    plot_image.save(os.path.join(plot_path, "plot.png"))
    clip_image.save(os.path.join(plot_path, "clip.png"))

    for layout in result.layouts:
      print("\n", layout.cls)
      if isinstance(layout, FormulaLayout):
        print("LaTeX:", layout.latex)
      elif isinstance(layout, TableLayout):
        parsed = layout.parsed
        if parsed is not None:
          print("Table:", parsed[1].name, parsed[0])
        else:
          print("Table:")
      else:
        for fragment in layout.fragments:
          print(fragment.text, fragment.rect.wrapper)

if __name__ == "__main__":
  main()