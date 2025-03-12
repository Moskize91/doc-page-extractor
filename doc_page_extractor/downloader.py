import os
import requests
from pathlib import Path


def download(url: str, file_path: Path):
  response = requests.get(url, stream=True, timeout=60)
  if response.status_code != 200:
    raise FileNotFoundError(f"Failed to download file from {url}: {response.status_code}")
  try:
    with open(file_path, "wb") as file:
      file.write(response.content)
  except Exception as e:
    if os.path.exists(file_path):
      os.remove(file_path)
    raise e
