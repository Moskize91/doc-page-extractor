# GLM-OCR SDK contract fixture

`sdk-rich-response.json` was captured from a real localhost SDK request for an
original synthetic 1800 x 2600 page (no third-party/private source document).
Only `json_result` and empty `usage` are retained; duplicate Markdown/layout
aliases, request IDs, timestamps, server/model paths, and image data are omitted.
Text, region labels, ordering, polygons, and boxes are otherwise unmodified.

- SDK source: `zai-org/GLM-OCR` at `cef4d0ea120d1741f5cefe8985eee45f6c8eff1d`
  (declares 0.1.5).
- MLX-VLM 0.7.2, MLX 0.32.2, Transformers 5.17.0, Python 3.12.13.
- Recognition model: `mlx-community/GLM-OCR-bf16`; CPU layout model:
  `PaddlePaddle/PP-DocLayoutV3_safetensors`.
- Request: `POST /glmocr/parse`, JSON `images` with one PNG data URL.
- MaaS disabled; CPU layout; local recognition URL; one worker; model files
  cached locally. Non-loopback Python DNS/socket connections were blocked.
- `bbox_2d` and polygon points are normalized to 0–1000, not pixel coordinates.
- SDK formatter switches `enable_merge_text_blocks` and
  `enable_merge_formula_numbers` were both false.
- Layout task mappings retained text, titles, captions, tables, formulas,
  images/charts, footnotes, headers/footers, page numbers, asides, and references
  instead of using the SDK's default `abandon` mapping.

The response contains 23 regions. The synthetic equation was missed by layout
detection, so this fixture intentionally does not claim real formula recognition;
formula mapping has separate synthetic unit coverage. Returned furniture layouts
are retained in `OCRPageResult.layouts` but ignored by the existing structured
page builder. Footnotes remain structured content.
