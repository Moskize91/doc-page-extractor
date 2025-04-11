import os

def path_from_root(*args: str) -> str:
  path = os.path.join(__file__, "..", "..")
  path = os.path.join(path, *args)
  path = os.path.abspath(path)
  return path