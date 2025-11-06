# Development Guide

## Setup

Setup Conda
```shell
conda create -p ./env python=3.11.14
conda activate ./env
```

Install dependencies:

```shell
poetry install
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
