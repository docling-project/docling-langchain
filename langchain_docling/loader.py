#
# Copyright IBM Corp. 2025 - 2025
# SPDX-License-Identifier: MIT
#

"""Docling LangChain loader module."""

import os
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Iterable, Iterator, Literal, Optional, Union

from docling.chunking import BaseChunk, BaseChunker, HybridChunker
from docling.datamodel.document import DoclingDocument
from docling.document_converter import DocumentConverter
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from langchain_docling._remote import build_convert_options, convert_source, make_client


class ExportType(str, Enum):
    """Enumeration of available export types."""

    MARKDOWN = "markdown"
    DOC_CHUNKS = "doc_chunks"


class BaseMetaExtractor(ABC):
    """BaseMetaExtractor."""

    @abstractmethod
    def extract_chunk_meta(self, file_path: str, chunk: BaseChunk) -> dict[str, Any]:
        """Extract chunk meta."""
        raise NotImplementedError()

    @abstractmethod
    def extract_dl_doc_meta(
        self, file_path: str, dl_doc: DoclingDocument
    ) -> dict[str, Any]:
        """Extract Docling document meta."""
        raise NotImplementedError()


class MetaExtractor(BaseMetaExtractor):
    """MetaExtractor."""

    def extract_chunk_meta(self, file_path: str, chunk: BaseChunk) -> dict[str, Any]:
        """Extract chunk meta."""
        return {
            "source": file_path,
            "dl_meta": chunk.meta.export_json_dict(),
        }

    def extract_dl_doc_meta(
        self, file_path: str, dl_doc: DoclingDocument
    ) -> dict[str, Any]:
        """Extract Docling document meta."""
        return {"source": file_path}


class DoclingLoader(BaseLoader):
    """Docling Loader."""

    def __init__(
        self,
        file_path: Union[str, Iterable[str]],
        *,
        converter: Optional[DocumentConverter] = None,
        convert_kwargs: Optional[Dict[str, Any]] = None,
        export_type: ExportType = ExportType.DOC_CHUNKS,
        md_export_kwargs: Optional[dict[str, Any]] = None,
        chunker: Optional[BaseChunker] = None,
        meta_extractor: Optional[BaseMetaExtractor] = None,
        docling_serve_url: Optional[str] = None,
        api_key: Optional[str] = None,
        docling_serve_options: Optional[Dict[str, Any]] = None,
        docling_serve_client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize with a file path.

        Args:
            file_path: File source as single str (URL or local file) or Iterable
                thereof.
            converter: Any specific `DocumentConverter` to use. Defaults to `None` (i.e.
                converter defined internally).
            convert_kwargs: Any specific kwargs to pass to conversion invocation.
                Defaults to `None` (i.e. behavior defined internally).
            export_type: The type to export to: either `ExportType.MARKDOWN` (outputs
                Markdown of whole input file) or `ExportType.DOC_CHUNKS` (outputs chunks
                based on chunker).
            md_export_kwargs: Any specific kwargs to pass to Markdown export (in case of
                `ExportType.MARKDOWN`). Defaults to `None` (i.e. behavior defined
                internally).
            chunker: Any specific `BaseChunker` to use (in case of
                `ExportType.DOC_CHUNKS`). Defaults to `None` (i.e. chunker defined
                internally).
            meta_extractor: The extractor instance to use for populating the output
                document metadata; if not set, a system default is used.
            docling_serve_url: URL of a remote Docling endpoint (docling-serve or
                Docling for IBM watsonx). If set (or via the `DOCLING_SERVE_URL`
                env var), conversion is routed remotely instead of using a local
                converter. An empty string is treated as unset. Cannot be
                combined with `converter`.
            api_key: API key for the remote endpoint. Falls back to the
                `DOCLING_API_KEY` env var. Ignored in local mode.
            docling_serve_options: Options dict for remote conversion, validated
                against `ConvertDocumentsOptions` (unknown keys are dropped with a
                warning; invalid values raise on construction). Ignored in local
                mode.
            docling_serve_client_kwargs: Extra keyword arguments forwarded to the
                `DoclingServiceClient` constructor for remote conversion, e.g.
                `{"job_timeout": 600}` (the client defaults to a 300s per-file
                timeout). Ignored in local mode.
        """
        self._file_paths = (
            file_path
            if isinstance(file_path, Iterable) and not isinstance(file_path, str)
            else [file_path]
        )

        docling_serve_url = docling_serve_url or None  # treat "" as unset
        resolved_url = (
            docling_serve_url
            if docling_serve_url is not None
            else os.environ.get("DOCLING_SERVE_URL")
        )
        resolved_key = (
            api_key if api_key is not None else os.environ.get("DOCLING_API_KEY")
        )
        if docling_serve_url is not None and converter is not None:
            raise ValueError(
                "Pass either `converter` (local conversion) or `docling_serve_url` "
                "(remote endpoint), not both."
            )
        if converter is not None:
            self._backend: Literal["local", "remote"] = (
                "local"  # explicit converter wins, even over an env URL
            )
        elif resolved_url:
            self._backend = "remote"
        else:
            self._backend = "local"

        self._docling_serve_url = resolved_url
        self._api_key = resolved_key
        self._docling_serve_client_kwargs = docling_serve_client_kwargs

        if self._backend == "remote":
            if convert_kwargs:
                warnings.warn(
                    "`convert_kwargs` is ignored when using a remote Docling "
                    "endpoint; use `docling_serve_options` instead.",
                    stacklevel=2,
                )
            # Validate options once up front so bad values fail fast rather than
            # on the first (possibly Nth) conversion.
            self._remote_options = build_convert_options(docling_serve_options)
        else:
            ignored = [
                name
                for name, value in (
                    ("api_key", api_key),
                    ("docling_serve_options", docling_serve_options),
                    ("docling_serve_client_kwargs", docling_serve_client_kwargs),
                )
                if value is not None
            ]
            if ignored:
                warnings.warn(
                    f"{ignored} ignored when using the local converter "
                    "(no `docling_serve_url` set).",
                    stacklevel=2,
                )

        self._converter: Optional[DocumentConverter] = (
            (converter or DocumentConverter()) if self._backend == "local" else None
        )
        self._convert_kwargs = convert_kwargs if convert_kwargs is not None else {}
        self._export_type = export_type
        self._md_export_kwargs = (
            md_export_kwargs
            if md_export_kwargs is not None
            else {"image_placeholder": ""}
        )
        if self._export_type == ExportType.DOC_CHUNKS:
            self._chunker: BaseChunker = chunker or HybridChunker()
        self._meta_extractor = meta_extractor or MetaExtractor()

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load documents."""
        if self._backend == "remote":
            assert self._docling_serve_url is not None  # narrows for mypy
            # One client (and connection pool) for all sources in this pass.
            with make_client(
                self._docling_serve_url,
                self._api_key,
                self._docling_serve_client_kwargs,
            ) as client:
                for file_path in self._file_paths:
                    dl_doc = convert_source(client, file_path, self._remote_options)
                    yield from self._emit(file_path, dl_doc)
        else:
            assert self._converter is not None  # narrows for mypy
            for file_path in self._file_paths:
                dl_doc = self._converter.convert(
                    source=file_path,
                    **self._convert_kwargs,
                ).document
                yield from self._emit(file_path, dl_doc)

    def _emit(self, file_path: str, dl_doc: DoclingDocument) -> Iterator[Document]:
        """Yield LangChain documents for one converted Docling document."""
        if self._export_type == ExportType.MARKDOWN:
            yield Document(
                page_content=dl_doc.export_to_markdown(**self._md_export_kwargs),
                metadata=self._meta_extractor.extract_dl_doc_meta(
                    file_path=file_path,
                    dl_doc=dl_doc,
                ),
            )
        elif self._export_type == ExportType.DOC_CHUNKS:
            for chunk in self._chunker.chunk(dl_doc):
                yield Document(
                    page_content=self._chunker.contextualize(chunk=chunk),
                    metadata=self._meta_extractor.extract_chunk_meta(
                        file_path=file_path,
                        chunk=chunk,
                    ),
                )
        else:
            raise ValueError(f"Unexpected export type: {self._export_type}")
