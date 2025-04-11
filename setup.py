from setuptools import setup, find_packages

setup(
  name="doc-page-extractor",
  version="0.0.10",
  author="Tao Zeyu",
  author_email="i@taozeyu.com",
  url="https://github.com/Moskize91/doc-page-extractor",
  description="doc page extractor can identify text and format in images and return structured data.",
  packages=find_packages(),
  long_description=open("./README.md", encoding="utf8").read(),
  long_description_content_type="text/markdown",
  install_requires=[
    "opencv-python>=4.11.0,<5.0",
    "pillow>=10.3,<11.0",
    "pyclipper>=1.2.0,<2.0",
    "numpy>=1.24.0,<2.0",
    "shapely>=2.0.0,<3.0",
    "transformers>=4.42.4,<5.0",
    "doclayout_yolo>=0.0.3",
    "pix2tex>=0.1.4,<=0.2.0",
    "struct-eqtable>=0.3.3,<0.4",
    "accelerate>=1.6.0,<2.0",
  ],
)