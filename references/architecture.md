# 架构与模块边界

## 运行时形态

`doc-page-extractor` 是一个小型库，用来把文档页面图片转换为带坐标的布局记录。一个布局记录包含引用类型、像素坐标框、可选文本，以及后端提供的可选元数据。

公开包通过 `doc_page_extractor/__init__.py` 延迟导出符号。尽量保持延迟导入，让轻量消费者可以 import 包而不立即加载模型相关代码。

## 源码职责

- `types.py` 定义公开协议和数据结构。这里的变更会影响下游库。
- `extractor.py` 负责高层抽取循环：保存页面图片、调用 `OCRAdapter.extract_page`、产出 `Layout`，并在多阶段抽取时涂抹已识别区域。
- `adapters/` 存放后端适配器。DeepSeek 本地 CUDA、DeepSeek OpenAI-compatible Vendor、百度云 OCR 都应在这里转换成统一布局。
- `model.py` 负责 Hugging Face DeepSeek-OCR 本地 CUDA 实现。这是 DeepSeek local adapter 的实现细节，应和纯解析/后处理代码保持隔离。
- `parser.py` 解析 DeepSeek `<|ref|>` 和 `<|det|>` 标签，并把归一化坐标缩放成图片像素坐标。它不负责解析百度云 JSON。
- `redacter.py` 计算接近纸张背景的填充色，并在阶段之间涂抹区域。
- `plot.py` 在抽取结果上绘制调试标注。
- `extraction_context.py` 提供生成过程中的中断和 token 限制统计。
- `injection.py` 在运行时 patch 下载得到的模型对象，让本包无需修改 Hugging Face 缓存文件也能注入 stopping criteria。

## 数据流

`PageExtractor.extract()` 接收 `PIL.Image`，写入临时 `raw-N.png`，然后调用：

```python
adapter.extract_page(prompt, image_path, output_path, size, context, device_number)
```

adapter 返回 `OCRPageResult`，其中包含统一的 `Layout` 列表。DeepSeek adapter 可以先得到 DeepSeek-OCR 兼容标签字符串，再用 `parse_ocr_response()` 转换成布局；百度云 adapter 则直接把 `parse_result_url` JSON 映射成布局。

当 `stages > 1` 时，抽取器会在下一次模型调用前涂抹页面上方三分之二，以及识别到的下方文字块。这个行为应保留在 `extractor.py` 内；模型后端不应该感知阶段涂抹策略。

## 边界规则

- 不要把模型供应商逻辑加进 `parser.py`、`redacter.py` 或 `plot.py`。
- 不要让 `types.py` 引入超出公开类型契约所需的重量级运行时依赖。
- 保持 `create_page_extractor_with_model()` 作为稳定的兼容入口；新后端优先实现 `OCRAdapter`。
- 下游项目可能会访问受保护/私有字段。即使字段名在 Python 层面是私有的，重命名 `extractor.py` 内部字段也可能破坏真实用户。
