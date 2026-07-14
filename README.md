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

To develop for Docling LangChain, you need Python 3.10 through 3.14 and uv. You can then install from your local clone's
root directory:
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
- `docling_serve_url` (optional): URL of a remote Docling endpoint (`docling-serve`
    or Docling for IBM watsonx). When set (or via the `DOCLING_SERVE_URL` env var),
    conversion runs remotely instead of locally. Cannot be combined with `converter`.
- `api_key` (optional): API key for the remote endpoint; falls back to the
    `DOCLING_API_KEY` env var.
- `docling_serve_options` (optional): options dict for remote conversion, validated
    against Docling's `ConvertDocumentsOptions` (unknown keys are dropped with a
    warning).
- `docling_serve_client_kwargs` (optional): extra keyword arguments for the
    `DoclingServiceClient` constructor, e.g. `{"job_timeout": 600}` (the client
    defaults to a 300s per-file timeout).

### Remote endpoint (optional)

By default `DoclingLoader` converts locally. To offload conversion to a remote
Docling endpoint (self-hosted `docling-serve >= 1.0.0` or Docling for IBM watsonx),
pass a URL (and key):

```python
from langchain_docling import DoclingLoader

loader = DoclingLoader(
    file_path=["https://arxiv.org/pdf/2408.09869"],
    docling_serve_url="https://your-docling-serve.example",
    api_key="...",                       # or set DOCLING_API_KEY
    docling_serve_options={"do_ocr": True},
)
docs = loader.load()
```

`docling_serve_url` / `api_key` also read from the `DOCLING_SERVE_URL` /
`DOCLING_API_KEY` environment variables. The same code targets a self-hosted
`docling-serve` and Docling for IBM watsonx — only the URL and key differ.
Conversion results are returned in-body (avoiding a presigned-artifact round
trip), so the endpoint needs no object-storage (S3) configuration.

For custom/self-signed TLS, the remote path uses `httpx`: set `SSL_CERT_FILE` or
`SSL_CERT_DIR` (a `requests`-style `verify=` is not used).

### Docs and examples

For more details and usage examples, check out
[this page](https://docling-project.github.io/docling/integrations/langchain/).
