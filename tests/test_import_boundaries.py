import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class TestImportBoundaries(unittest.TestCase):
    def test_vendor_public_api_does_not_import_local_runtime(self):
        project_root = Path(__file__).resolve().parents[1]
        code = textwrap.dedent(
            """
            import builtins

            blocked = {"torch", "transformers", "huggingface_hub", "readerwriterlock"}
            real_import = builtins.__import__

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if level == 0 and name.split(".", 1)[0] in blocked:
                    raise ImportError(f"blocked optional local runtime: {name}")
                return real_import(name, globals, locals, fromlist, level)

            builtins.__import__ = guarded_import

            from doc_page_extractor import (
                AbortError,
                DeepSeekOCRVendorConfig,
                ExtractionAbortedError,
                TokenLimitError,
                UnlimitedOCRVendorConfig,
                create_deepseek_ocr_vendor_page_extractor,
                create_unlimited_ocr_vendor_page_extractor,
            )

            assert AbortError
            assert ExtractionAbortedError
            assert TokenLimitError
            create_deepseek_ocr_vendor_page_extractor(
                DeepSeekOCRVendorConfig(
                    base_url="https://example.test",
                    api_key="key",
                    model="deepseek-ocr",
                )
            )
            create_unlimited_ocr_vendor_page_extractor(
                UnlimitedOCRVendorConfig(ak="ak", sk="sk")
            )
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)

        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
