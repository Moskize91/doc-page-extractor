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

- `DOC_PAGE_EXTRACTOR_DEEPSEEK_VENDOR_*`：DeepSeek OpenAI-compatible Vendor。
- `DOC_PAGE_EXTRACTOR_BAIDU_*`：百度云 Unlimited-OCR。
- `DOC_PAGE_EXTRACTOR_MODEL_PATH` 和 `DOC_PAGE_EXTRACTOR_LOCAL_ONLY`：DeepSeek 本地 Hugging Face 路径。

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
poetry run python scripts/ocr_sample.py --adapter deepseek-vendor --image tests/images/friendly-title.png
poetry run python scripts/ocr_sample.py --adapter baidu --image tests/images/friendly-title.png
poetry run python scripts/ocr_sample.py --adapter both --image tests/images/friendly-title.png
```

该脚本默认读取 `tests/images/friendly-title.png`，分别调用 DeepSeek Vendor、百度云 OCR，或两者同时运行。成功输出会包含图片路径、layout 数量、前几个 layout 摘要、文本预览和耗时。可以用 `--image path/to/image.png` 指定其他图片。

## 开发后端模式

需要在无 CUDA 环境测试完整抽取循环时，新后端优先实现 `OCRAdapter`，并通过 `create_page_extractor_with_adapter()` 或专用工厂函数接入。DeepSeek Vendor 和百度云可直接使用：

```python
from doc_page_extractor import (
    BaiduCloudOCRConfig,
    DeepSeekVendorOCRConfig,
    create_baidu_page_extractor,
    create_deepseek_vendor_page_extractor,
)

deepseek = create_deepseek_vendor_page_extractor(DeepSeekVendorOCRConfig.from_env())
baidu = create_baidu_page_extractor(BaiduCloudOCRConfig.from_env())
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
