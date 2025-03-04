import os
import re

def ensure_dir(path: str) -> str:
  path = os.path.abspath(path)
  os.makedirs(path, exist_ok=True)
  return path

def is_space_text(text: str) -> bool:
  return re.match(r"^\s*$", text)