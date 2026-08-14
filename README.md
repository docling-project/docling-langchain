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

For Docling Serve or Managed Docling, install the lightweight service client:
```bash
pip install langchain-docling
```

This installation does not include local AI runtimes such as PyTorch. To convert
documents locally, install the `local` extra instead:

```bash
pip install "langchain-docling[local]"
```

### Development setup

To develop for Docling LangChain, you need Python 3.10 through 3.14 and uv. You can then install from your local clone's
root directory:
```bash
uv sync
```

## Usage

### Basic usage

For local conversion, install the `local` extra and use `DoclingLoader` as follows:

```python
from langchain_docling import DoclingLoader

FILE_PATH = ["https://arxiv.org/pdf/2408.09869"]  # Docling Technical Report

loader = DoclingLoader(file_path=FILE_PATH)
docs = loader.load()
```

### Docling Serve or Managed Docling

Pass Docling's existing `DoclingServiceClient` as the loader's converter to run
conversion remotely. The same client works with
[Docling for IBM watsonx](https://www.ibm.com/products/docling) and a self-hosted
Docling Serve endpoint:

```python
import os

from docling.datamodel.service.options import ConvertDocumentsOptions
from docling.service_client import DoclingServiceClient

from langchain_docling import DoclingLoader

service_url = os.environ["DOCLING_SERVICE_URL"]
api_key = os.environ["DOCLING_API_KEY"]
options = ConvertDocumentsOptions(
    do_ocr=True,
    table_mode="accurate",
)

with DoclingServiceClient(url=service_url, api_key=api_key) as client:
    loader = DoclingLoader(
        file_path=["https://arxiv.org/pdf/2408.09869"],
        converter=client,
        convert_kwargs={"options": options},
    )
    docs = loader.load()
```

The loader does not take ownership of a supplied client. Keep the client open
until `load()` completes or until a `lazy_load()` iterator has been fully
consumed. For a self-hosted endpoint that does not require authentication, omit
`api_key`.

### Advanced usage

When initializing a `DoclingLoader`, you can use the following parameters:

- `file_path`: source as single str (URL or local file) or iterable thereof
- `converter` (optional): a local `DocumentConverter`, remote
    `DoclingServiceClient`, or compatible converter
- `convert_kwargs` (optional): backend-specific kwargs for conversion execution,
    such as `{"options": ConvertDocumentsOptions(...)}` for the service client
- `export_type` (optional): export mode to use: `ExportType.DOC_CHUNKS` (default) or
    `ExportType.MARKDOWN`
- `md_export_kwargs` (optional): any specific Markdown export kwargs (for Markdown mode)
- `chunker` (optional): any specific Docling chunker instance to use (for doc-chunk
    mode)
- `meta_extractor` (optional): any specific metadata extractor to use

### Docs and examples

For more details and usage examples, check out
[this page](https://docling-project.github.io/docling/integrations/langchain/).
