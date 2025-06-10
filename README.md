# Docling LangChain integration

[![PyPI version](https://img.shields.io/pypi/v/langchain-docling)](https://pypi.org/project/langchain-docling/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/langchain-docling)](https://pypi.org/project/langchain-docling/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License MIT](https://img.shields.io/github/license/docling-project/docling-langchain)](https://opensource.org/licenses/MIT)

A [Docling](https://github.com/docling-project/docling) integration for
[LangChain](https://github.com/langchain-ai/langchain/).

## Installation

Simply install `langchain-docling` from your package manager, e.g. pip:
```bash
pip install langchain-docling
```

### Development setup

To develop for Docling Core, you need Python >=3.9 <=3.13 and uv. You can then install from your local clone's root dir:
```bash
uv sync
```

## Usage

### Basic usage

Basic usage of `DoclingLoader` looks as follows:

```python
from langchain_docling import DoclingLoader

FILE_PATH = ["https://arxiv.org/pdf/2408.09869"]  # Docling Technical Report

loader = DoclingLoader(file_path=FILE_PATH)
docs = loader.load()
```

### Advanced usage

When initializing a `DoclingLoader`, you can use the following parameters:

- `file_path`: source as single str (URL or local file) or iterable thereof
- `converter` (optional): any specific Docling converter instance to use
- `convert_kwargs` (optional): any specific kwargs for conversion execution
- `export_type` (optional): export mode to use: `ExportType.DOC_CHUNKS` (default) or
    `ExportType.MARKDOWN`
- `md_export_kwargs` (optional): any specific Markdown export kwargs (for Markdown mode)
- `chunker` (optional): any specific Docling chunker instance to use (for doc-chunk
    mode)
- `meta_extractor` (optional): any specific metadata extractor to use

### Docs and examples

For more details and usage examples, check out
[this page](https://docling-project.github.io/docling/integrations/langchain/).
