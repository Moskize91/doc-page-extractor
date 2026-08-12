# Architecture And Module Boundaries

## Runtime Shape

`doc-page-extractor` is a small library that converts a document page image into layout records. A layout record combines a DeepSeek-OCR reference label, a pixel bounding box, and optional text.

The public package exports lazy symbols from `doc_page_extractor/__init__.py`. Keep imports lazy when possible so lightweight consumers can import the package without immediately loading model code.

## Source Responsibilities

- `types.py` defines public protocols and data structures. Changes here affect downstream libraries.
- `extractor.py` owns the high-level extraction loop: save a page image, call `DeepSeekOCRModel.generate`, parse the response, yield `Layout` values, and optionally redact already-seen regions for another stage.
- `model.py` owns the default Hugging Face DeepSeek-OCR backend. This is the CUDA path and should stay isolated from pure parsing/post-processing code.
- `parser.py` parses `<|ref|>` and `<|det|>` tags and scales normalized coordinates into image pixels.
- `redacter.py` computes a page-like fill color and redacts regions between extraction stages.
- `plot.py` draws debug overlays on extracted layouts.
- `extraction_context.py` provides abort and token-limit bookkeeping for generation.
- `injection.py` patches the downloaded model object at runtime so the package can add stopping criteria without editing Hugging Face cache files.

## Data Flow

`PageExtractor.extract()` receives a `PIL.Image`, writes a temporary `raw-N.png`, and calls:

```python
model.generate(prompt, image_path, output_path, size, context, device_number)
```

The returned string must use DeepSeek-OCR-compatible tags. `parse_ocr_response()` emits ordered `TEXT`, `REF`, and `DET` items. `_PageExtractorImpls._parse_response()` pairs refs and boxes with following text and yields `Layout` values.

When `stages > 1`, the extractor redacts the top two-thirds of the page plus detected lower text blocks before the next model call. Keep this behavior inside `extractor.py`; model backends should not know about staged redaction.

## Boundary Rules

- Do not add model-provider logic to `parser.py`, `redacter.py`, or `plot.py`.
- Do not make `types.py` import heavyweight runtime dependencies beyond what is already needed for public type contracts.
- Preserve `create_page_extractor_with_model()` as the stable test and macOS development seam.
- Treat protected/private access by downstream projects as compatibility pressure: renaming private fields inside `extractor.py` can still break real users.
