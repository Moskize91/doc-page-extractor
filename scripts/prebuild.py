from subprocess import run
from pathlib import Path


def prebuild(setup_kwargs):
  shell_path = Path(__file__).parent / "sync-struct-eqtable.sh"
  run(["bash", str(shell_path)], check=True)
  return setup_kwargs