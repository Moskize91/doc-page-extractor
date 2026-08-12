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

## 验证命令

macOS 默认检查：

```shell
poetry run python test.py
poetry run pylint --disable=import-error doc_page_extractor
```

除非任务明确要求真实后端实验，否则不要在 macOS 上运行 `main.py`、`download.py` 或 `PageExtractor.load_models()`。

## 开发后端模式

需要在无 CUDA 环境测试完整抽取循环时，使用 `create_page_extractor_with_model()`：

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
