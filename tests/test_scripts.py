import os
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.ocr_sample import _local_model_path


class TestScripts(unittest.TestCase):
    def test_local_model_path_empty_env_falls_back_to_default_cache(self):
        project_root = Path("project")

        with patch.dict(os.environ, {"LOCAL_MODEL_PATH": "   "}):
            path = _local_model_path(project_root, "LOCAL_MODEL_PATH")

        self.assertEqual(path, project_root / "models-cache")


if __name__ == "__main__":
    unittest.main()
