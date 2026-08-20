# macOS 本地开发

## 默认环境

使用 Poetry 和项目内虚拟环境：

```shell
pipx install poetry==2.1.3
PYTHON_BIN="$(pyenv which python3 2>/dev/null || command -v python3)"
"$PYTHON_BIN" -m venv .venv
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
poetry install --only dev
```

这个环境刻意避开 CUDA PyTorch 和模型下载。它足够用于解析器测试、包导入检查、lint，以及使用开发模型的测试。

## 本地环境文件

把 `.env.template` 复制为 `.env`，填写本机私有配置：

```shell
cp .env.template .env
```

`.env` 会被 git 忽略。它可以包含私有 Vendor API 配置或本地模型路径。本库当前不会自动读取 `.env`；只有脚本或手动命令需要这些值时才 source：

```shell
set -a && source .env && set +a
```

`.env.template` 不得包含密钥。

VGE/Conductor 新建 worktree 时，ignored 的 `.env` 不会自动随 git 带过去。`setup` 会在缺失时从 `.env.template` 创建 `.env`。

1. 如果 worktree 里已经有 `.env`，保留它。
2. 否则复制 `.env.template`，并提示手动填写私有配置。

Conductor 不会从仓库外的隐藏路径复制私有配置。需要真实远程 OCR 凭据时，应在对应 worktree 的 `.env` 中明确填写；`.env` 不属于仓库，不应提交或写入文档。

## 后端配置

`.env` 现在同时保存多个后端的私有配置，脚本或开发适配器按自己的 `--adapter` 参数读取对应字段：

- `DPE_DEEPSEEK_OCR_*`：DeepSeek OCR Vendor。
- `DPE_DEEPSEEK_OCR2_*`：DeepSeek OCR 2 Vendor。
- `DPE_UNLIMITED_OCR_*`：Unlimited OCR。
- `DPE_DEEPSEEK_LOCAL_MODEL_PATH` 和 `DPE_DEEPSEEK_LOCAL_ONLY`：DeepSeek 本地 Hugging Face 路径。

## 验证命令

macOS 默认检查：

```shell
poetry run python test.py
poetry run pylint --disable=import-error doc_page_extractor
```

除非任务明确要求真实后端实验，否则不要在 macOS 上运行 `main.py`、`download.py` 或 `PageExtractor.load_models()`。

## OCR Sample

填写 `.env` 中的私有配置后，可以运行：

```shell
poetry run python scripts/ocr_sample.py --adapter deepseek-ocr-vendor --image tests/images/friendly-title.png
poetry run python scripts/ocr_sample.py --adapter deepseek-ocr2-vendor --image tests/images/friendly-title.png
poetry run python scripts/ocr_sample.py --adapter unlimited-ocr --image tests/images/friendly-title.png
```

该脚本默认读取 `tests/images/friendly-title.png`，调用指定 OCR adapter。成功输出会包含图片路径、layout 数量、前几个 layout 的 `ref`、稳定 `kind`、供应商原始 `type`、文本预览和耗时。可以用 `--image path/to/image.png` 指定其他图片。

新代码应优先依赖 `Layout.kind` 判断跨供应商稳定语义。`Layout.ref` 是兼容旧 DeepSeek 风格 API 的字段；`Layout.type` 和 `Layout.raw` 保留供应商原始数据，不应当作稳定跨供应商契约。需要结构化页面结果时，使用 `extract_page_results()` 读取 `OCRPageResult.structured`；只需要旧式扁平 layout 列表时，继续使用 `extract()`。

## 开发后端模式

需要在无 CUDA 环境测试完整抽取循环时，新后端优先实现 `OCRAdapter`，并通过 `create_page_extractor_with_adapter()` 或专用工厂函数接入。远程 OCR adapter 可直接使用：

```python
from doc_page_extractor import (
    DeepSeekOCR2VendorConfig,
    DeepSeekOCRVendorConfig,
    UnlimitedOCRConfig,
    create_deepseek_ocr2_vendor_page_extractor,
    create_deepseek_ocr_vendor_page_extractor,
    create_unlimited_ocr_page_extractor,
)

deepseek_ocr = create_deepseek_ocr_vendor_page_extractor(
    DeepSeekOCRVendorConfig.from_env()
)
deepseek_ocr2 = create_deepseek_ocr2_vendor_page_extractor(
    DeepSeekOCR2VendorConfig.from_env()
)
unlimited_ocr = create_unlimited_ocr_page_extractor(UnlimitedOCRConfig.from_env())
```

兼容旧 DeepSeek 模型协议或编写极小 fixture 时，也可以使用 `create_page_extractor_with_model()`：

```python
from pathlib import Path

from doc_page_extractor import create_page_extractor_with_model


class FixtureOCRModel:
    def download(self, revision: str | None) -> None:
        pass

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def generate(self, prompt, image_path: Path, output_path: Path, size, context, device_number) -> str:
        return "<|ref|>sample<|/ref|><|det|>[[100, 100, 500, 200]]<|/det|>hello"


extractor = create_page_extractor_with_model(FixtureOCRModel())
```

这会覆盖图片保存、响应解析、布局构造、阶段涂抹；如果 fixture 更新了 `context`，也能覆盖 token 统计。

## macOS 不能验证的内容

macOS 检查不能验证 CUDA 可用性、显存行为、Hugging Face remote code 兼容性、`flash_attn` 或多 GPU 设备路由。这些需要 Linux/NVIDIA 环境。
