import json
from unittest.mock import MagicMock

import pytest
from docling.chunking import HierarchicalChunker
from docling.datamodel.document import DoclingDocument as DLDocument

from langchain_docling.loader import DoclingLoader, ExportType

in_json_str = json.dumps(
    {
        "schema_name": "DoclingDocument",
        "version": "1.0.0",
        "name": "sample",
        "origin": {
            "mimetype": "text/html",
            "binary_hash": 42,
            "filename": "sample.html",
        },
        "furniture": {
            "self_ref": "#/furniture",
            "children": [],
            "name": "_root_",
            "label": "unspecified",
        },
        "body": {
            "self_ref": "#/body",
            "children": [{"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}],
            "name": "_root_",
            "label": "unspecified",
        },
        "groups": [],
        "texts": [
            {
                "self_ref": "#/texts/0",
                "parent": {"$ref": "#/body"},
                "children": [],
                "label": "paragraph",
                "prov": [],
                "orig": "Some text",
                "text": "Some text",
            },
            {
                "self_ref": "#/texts/1",
                "parent": {"$ref": "#/body"},
                "children": [],
                "label": "paragraph",
                "prov": [],
                "orig": "Another paragraph",
                "text": "Another paragraph",
            },
        ],
        "pictures": [],
        "tables": [],
        "key_value_items": [],
        "pages": {},
    }
)


out_json_obj = {
    "root": [
        {
            "id": None,
            "metadata": {
                "source": "https://example.com/foo.pdf",
                "dl_meta": {
                    "schema_name": "docling_core.transforms.chunker.DocMeta",
                    "version": "1.0.0",
                    "doc_items": [
                        {
                            "self_ref": "#/texts/0",
                            "parent": {"$ref": "#/body"},
                            "children": [],
                            "label": "paragraph",
                            "prov": [],
                        }
                    ],
                    "origin": {
                        "mimetype": "text/html",
                        "binary_hash": 42,
                        "filename": "sample.html",
                    },
                },
            },
            "page_content": "Some text",
            "type": "Document",
        },
        {
            "id": None,
            "metadata": {
                "source": "https://example.com/foo.pdf",
                "dl_meta": {
                    "schema_name": "docling_core.transforms.chunker.DocMeta",
                    "version": "1.0.0",
                    "doc_items": [
                        {
                            "self_ref": "#/texts/1",
                            "parent": {"$ref": "#/body"},
                            "children": [],
                            "label": "paragraph",
                            "prov": [],
                        }
                    ],
                    "origin": {
                        "mimetype": "text/html",
                        "binary_hash": 42,
                        "filename": "sample.html",
                    },
                },
            },
            "page_content": "Another paragraph",
            "type": "Document",
        },
    ]
}

out_md_obj = {
    "root": [
        {
            "id": None,
            "metadata": {"source": "https://example.com/foo.pdf"},
            "page_content": "Some text\n\nAnother paragraph",
            "type": "Document",
        }
    ]
}


@pytest.mark.requires("docling")
def test_load_as_markdown(monkeypatch: pytest.MonkeyPatch) -> None:

    mock_dl_doc = DLDocument.model_validate_json(in_json_str)
    mock_response = MagicMock()
    mock_response.document = mock_dl_doc

    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter.__init__",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter.convert",
        lambda *args, **kwargs: mock_response,
    )

    loader = DoclingLoader(
        source="https://example.com/foo.pdf",
        export_type=ExportType.MARKDOWN,
    )
    lc_doc_iter = loader.lazy_load()
    act_lc_docs = list(lc_doc_iter)
    assert len(act_lc_docs) == 1

    act_data = {"root": [lc_doc.model_dump() for lc_doc in act_lc_docs]}
    assert act_data == out_md_obj


@pytest.mark.requires("docling")
def test_load_as_doc_chunks(monkeypatch: pytest.MonkeyPatch) -> None:

    mock_dl_doc = DLDocument.model_validate_json(in_json_str)
    mock_response = MagicMock()
    mock_response.document = mock_dl_doc

    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter.__init__",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter.convert",
        lambda *args, **kwargs: mock_response,
    )

    loader = DoclingLoader(
        source="https://example.com/foo.pdf",
        export_type=ExportType.DOC_CHUNKS,
        chunker=HierarchicalChunker(),
    )
    lc_doc_iter = loader.lazy_load()
    act_lc_docs = list(lc_doc_iter)
    assert len(act_lc_docs) == 2

    act_data = {"root": [lc_doc.model_dump() for lc_doc in act_lc_docs]}
    assert act_data == out_json_obj
