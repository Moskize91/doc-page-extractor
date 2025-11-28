# Development Guide

## Setup

Setup Python env
```shell
python -m venv .venv
. ./.venv/bin/activate
```

Install dependencies:

```shell
poetry install
```

Install CUDA 12.1 version of torch and torchvision (Linux/Windows, GPU support):

```shell
poetry run pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

## Development Workflow

### Run Tests

```shell
poetry run python test.py
```

### Run Lint

Check code quality with pylint:

```shell
python lint.py
```

Or directly:

```shell
poetry run pylint doc_page_extractor
```

### macOS Development Setup

For macOS developers, PyTorch CUDA version is not compatible. Use the following steps:

```shell
# Install only main dependencies first (skip dev group to avoid CUDA installation)
poetry install

# Install PyTorch CPU version
poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Build Package

Clean old builds and create distribution files:

```shell
python build.py
```

## Before Submitting PR

Make sure all checks pass:

```shell
poetry run python test.py
python lint.py
```
