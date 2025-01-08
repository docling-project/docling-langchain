#
# Copyright IBM Corp. 2025 - 2025
# SPDX-License-Identifier: MIT
#

"""Docling LangChain loader module."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Iterable, Iterator, Optional, Union

from docling.chunking import BaseChunk, BaseChunker, HybridChunker
from docling.datamodel.document import DoclingDocument
from docling.document_converter import DocumentConverter
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


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
        """
        self._file_paths = (
            file_path
            if isinstance(file_path, Iterable) and not isinstance(file_path, str)
            else [file_path]
        )

        self._converter: DocumentConverter = converter or DocumentConverter()
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
        for file_path in self._file_paths:
            conv_res = self._converter.convert(
                source=file_path,
                **self._convert_kwargs,
            )
            dl_doc = conv_res.document
            if self._export_type == ExportType.MARKDOWN:
                yield Document(
                    page_content=dl_doc.export_to_markdown(**self._md_export_kwargs),
                    metadata=self._meta_extractor.extract_dl_doc_meta(
                        file_path=file_path,
                        dl_doc=dl_doc,
                    ),
                )
            elif self._export_type == ExportType.DOC_CHUNKS:
                chunk_iter = self._chunker.chunk(dl_doc)
                for chunk in chunk_iter:
                    yield Document(
                        page_content=self._chunker.serialize(chunk=chunk),
                        metadata=self._meta_extractor.extract_chunk_meta(
                            file_path=file_path,
                            chunk=chunk,
                        ),
                    )

            else:
                raise ValueError(f"Unexpected export type: {self._export_type}")
