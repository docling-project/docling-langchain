# Docling LangChain integration

[![PyPI version](https://img.shields.io/pypi/v/langchain-docling)](https://pypi.org/project/langchain-docling/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/langchain-docling)](https://pypi.org/project/langchain-docling/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License MIT](https://img.shields.io/github/license/DS4SD/docling)](https://opensource.org/licenses/MIT)

A [Docling](https://github.com/DS4SD/docling) integration for
[LangChain](https://github.com/langchain-ai/langchain/).

## Installation

Simply install `langchain-docling` from your package manager, e.g. pip:
```bash
pip install langchain-docling
```

## Usage

Basic usage looks as follows:

```python
from langchain_docling import DoclingLoader

FILE_PATH = ["https://arxiv.org/pdf/2408.09869"]  # Docling Technical Report

loader = DoclingLoader(file_path=FILE_PATH)

docs = loader.load()
```

For end-to-end usage samples check out the [examples](examples/).
