# Agent 工作流

本文件是 Agent 进入 `doc-page-extractor` 后的项目入口，只记录项目事实、工作区边界和按条件读取的文档路由。

## 工作区边界

- `doc_page_extractor/` 是发布包源码。`extractor.py` 负责页面抽取流程，`model.py` 负责 Hugging Face local CUDA 后端，`parser.py`、`redacter.py`、`plot.py` 是可在 macOS 上测试的纯 Python 后处理。
- `tests/` 存放轻量单元测试，默认不得要求 CUDA、下载模型或访问 Hugging Face。
- `main.py` 和 `download.py` 是手动验证脚本，会触发真实模型路径；在 macOS 或普通 Agent 任务中不要默认运行。
- `models-cache/`、`plot/`、`.venv/`、`.env` 是本地状态或生成产物，不要提交。
- `.env.template` 是可提交的本地配置模板；`.env` 是私有配置，Agent 可以读取但不得把其中密钥写入文档、日志或提交。

## 仅在触发条件满足时读取

- 修改抽取流程、数据类型、解析器、涂抹或可视化逻辑时，阅读[架构与模块边界](references/architecture.md)。
- 设置 macOS 本地环境、运行测试、调 lint、写不依赖 CUDA 的测试或使用 `.env` 时，阅读[macOS 本地开发](references/local-development.md)。
- 修改 Hugging Face 模型加载、模型下载、local/Vendor/mock 后端或 CUDA 相关行为时，阅读[模型后端](references/model-backends.md)。
- 修改构建、版本、发布、依赖声明或 PyPI 文档时，阅读[发布流程](docs/RELEASE.md) 和 [macOS 本地开发](references/local-development.md)。

不要一次性读取所有 reference。先根据任务选择最小相关文档，再回到代码确认事实。

## 项目默认规则

- macOS 是默认开发环境。不要把“本地不能跑 CUDA 模型”当成阻塞；优先通过 `create_page_extractor_with_adapter()` 注入开发 adapter 来测试抽取流程。
- 真实 local Hugging Face 后端需要 CUDA PyTorch、NVIDIA GPU 和模型缓存，只在专门的 GPU 环境验证。
- 新测试默认应可通过 `poetry run python test.py` 在无 CUDA 环境运行。
- 不要让普通 import 路径下载模型、加载 torch CUDA 或访问网络。模型下载只能发生在明确的下载/真实模型验证命令中。
- 文档任务不得修改运行代码，除非用户明确要求或文档链接/配置必须配套调整。
