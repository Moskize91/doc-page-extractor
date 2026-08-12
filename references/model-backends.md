# Model Backends

## Default Backend

`create_page_extractor()` constructs `DeepSeekOCRHugginfaceModel`. This backend downloads and loads `deepseek-ai/DeepSeek-OCR` through Hugging Face and requires CUDA at runtime.

The model cache path is optional in API terms, but production deployments should provide one. With `local_only=True`, `model_path` is required and must contain the Hugging Face cache structure for DeepSeek-OCR.

## Development And Remote Backends

`create_page_extractor_with_model(model)` is the supported way to replace local CUDA inference. The injected object must implement the `DeepSeekOCRModel` protocol:

```python
download(revision)
load()
unload()
generate(prompt, image_path, output_path, size, context, device_number) -> str
```

`download()`, `load()`, and `unload()` may be no-ops for fixture or remote backends. `generate()` must return DeepSeek-OCR-compatible text using `<|ref|>` and `<|det|>` tags so the existing parser can produce `Layout` values.

When implementing a remote backend:

- Upload or encode the `image_path` produced by `extractor.py`.
- Send the exact `prompt` argument unless the task intentionally changes prompting.
- Return only the OCR response text expected by the parser.
- If usage metadata is available, update `context.input_tokens` and `context.output_tokens`.
- Convert provider token/billing failures into `TokenLimitError` when that is the closest semantic match.

## CUDA Path Rules

- `model.py` is the only module that should import torch for CUDA model loading.
- `check_env()` warning behavior is not a complete runtime guard; `_ensure_models()` is where no-CUDA becomes a hard failure.
- Avoid model downloads during ordinary imports, tests, lint, or package build.
- `download.py` and `main.py` are manual scripts for real-backend work. They should not become required for macOS development.

## Compatibility Notes

Downstream projects may rely on `create_page_extractor_with_model()` to keep their own process CPU-only. Treat that function and the `DeepSeekOCRModel.generate()` signature as public compatibility surface.
