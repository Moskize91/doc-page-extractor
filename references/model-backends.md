# 模型后端

## 本地 Hugging Face 后端

`create_deepseek_ocr_page_extractor()` 会根据 `ocr_model` 创建本地
DeepSeek OCR 后端。`deepseek-ocr` 使用 `DeepSeekOCRHuggingFaceModel`，
`deepseek-ocr2` 使用 `DeepSeekOCR2HuggingFaceModel`。
`create_unlimited_ocr_page_extractor()` 创建本地 Unlimited OCR 后端，
使用 `UnlimitedOCRHuggingFaceModel`。本地后端通过 Hugging Face 下载并
加载模型，运行时需要 CUDA。

从 API 角度看，模型缓存路径是可选的；生产部署应显式提供。使用
`local_only=True` 时，必须提供 `model_path`，且其中需要包含对应模型的
Hugging Face 缓存结构。

## 开发后端与远程后端

后端中心是 `OCRAdapter`。新后端应实现：

```python
extract_page(prompt, image_path, output_path, size, context, device_number) -> OCRPageResult
```

Adapter 协议还要求实现 `download()`、`load()` 和 `allows_multi_stage`，
让 `PageExtractor` 能用显式接口处理生命周期和多阶段能力。
`DeepSeekOCRVendorAdapter` 解析 OCR 1 的 `<|ref|>` / `<|det|>` 输出；
`DeepSeekOCR2VendorAdapter` 解析 OCR 2 的行块输出；
`UnlimitedOCRVendorAdapter` 直接把 `parse_result_url` 的 JSON 映射成项目
统一布局；`UnlimitedModelOCRAdapter` 解析本地 Transformers 输出的检测标签。

本地 `.env` 不再使用单个互斥后端字段，而是同时保存多个后端配置：

- `DEEPSEEK_OCR_*`：DeepSeek OCR Vendor。
- `DEEPSEEK_OCR2_*`：DeepSeek OCR 2 Vendor。
- `UNLIMITED_OCR_*`：Unlimited OCR。
- `DEEPSEEK_LOCAL_MODEL_PATH` 和 `DEEPSEEK_LOCAL_ONLY`：DeepSeek 本地 Hugging Face 路径。
- `UNLIMITED_LOCAL_MODEL_PATH` 和 `UNLIMITED_LOCAL_ONLY`：Unlimited OCR 本地 Hugging Face 路径。

这只是当前本地调试脚本的约定；库代码本身不会自动读取 `.env`，也不会在普通 adapter 构造或导入路径读取环境变量。

实现远程后端时：

- 上传或编码 `extractor.py` 生成的 `image_path`。
- 除非任务明确要改 prompt，否则传递原始 `prompt` 参数。
- DeepSeek OCR Vendor 只返回解析器期望的 OCR 响应文本。
- DeepSeek OCR 2 Vendor 在 adapter 内把行块输出归一为统一布局。
- 如果供应商返回 usage 信息，更新 `context.input_tokens` 和 `context.output_tokens`。
- Unlimited OCR Vendor adapter 需要处理异步 submit/query/download 流程，
  并把 `parse_result_url` JSON 映射成统一布局。
- 本地 sample 使用 `scripts/ocr_sample.py`，它是开发验证脚本，不是生产后端抽象。

## CUDA 路径规则

- `model.py` 是唯一应该为了 CUDA 模型加载而 import torch 的模块。
- `check_env()` 的 warning 不是完整运行时保护；无 CUDA 的硬失败发生在 `_ensure_models()`。
- 普通 import、测试、lint 或包构建过程中不得下载模型。
- `download.py` 和 `main.py` 是真实后端手动脚本，不应成为 macOS 开发的必要步骤。
