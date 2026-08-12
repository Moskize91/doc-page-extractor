# 模型后端

## 默认后端

`create_page_extractor()` 会创建 `DeepSeekOCRHugginfaceModel`。这个后端通过 Hugging Face 下载并加载 `deepseek-ai/DeepSeek-OCR`，运行时需要 CUDA。

从 API 角度看，模型缓存路径是可选的；生产部署应显式提供。使用 `local_only=True` 时，必须提供 `model_path`，且其中需要包含 DeepSeek-OCR 的 Hugging Face 缓存结构。

## 开发后端与远程后端

`create_page_extractor_with_model(model)` 是替换本地 CUDA 推理的受支持方式。注入对象必须实现 `DeepSeekOCRModel` 协议：

```python
download(revision)
load()
unload()
generate(prompt, image_path, output_path, size, context, device_number) -> str
```

对 fixture 或远程后端来说，`download()`、`load()` 和 `unload()` 可以是空实现。`generate()` 必须返回包含 `<|ref|>` 和 `<|det|>` 标签的 DeepSeek-OCR 兼容文本，这样现有解析器才能产出 `Layout`。

本地 `.env` 约定用 `DOC_PAGE_EXTRACTOR_BACKEND` 做互斥选择：

- `fixture` 使用固定 OCR 响应或 fixture 文件。
- `vendor` 使用 OpenAI-compatible 远程 OCR 后端。
- `local` 使用 `create_page_extractor()` 和本地 Hugging Face 模型缓存。

这只是当前脚本和后续开发适配器的约定；库代码本身不会自动读取 `.env`。

实现远程后端时：

- 上传或编码 `extractor.py` 生成的 `image_path`。
- 除非任务明确要改 prompt，否则传递原始 `prompt` 参数。
- 只返回解析器期望的 OCR 响应文本。
- 如果供应商返回 usage 信息，更新 `context.input_tokens` 和 `context.output_tokens`。
- 当供应商的 token/计费失败语义接近 token 限制时，转换为 `TokenLimitError`。

## CUDA 路径规则

- `model.py` 是唯一应该为了 CUDA 模型加载而 import torch 的模块。
- `check_env()` 的 warning 不是完整运行时保护；无 CUDA 的硬失败发生在 `_ensure_models()`。
- 普通 import、测试、lint 或包构建过程中不得下载模型。
- `download.py` 和 `main.py` 是真实后端手动脚本，不应成为 macOS 开发的必要步骤。

## 兼容性说明

下游项目可能依赖 `create_page_extractor_with_model()` 来保持自身进程 CPU-only。应把这个函数和 `DeepSeekOCRModel.generate()` 的签名视作公开兼容面。
