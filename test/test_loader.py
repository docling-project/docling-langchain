import json
from unittest.mock import MagicMock

import pytest
from docling.chunking import HierarchicalChunker
from docling.datamodel.document import DoclingDocument

from langchain_docling.loader import DoclingLoader, ExportType

from .test_data_gen_flag import GEN_TEST_DATA


@pytest.mark.requires("docling")
def test_load_as_markdown(monkeypatch: pytest.MonkeyPatch) -> None:

    mock_dl_doc = DoclingDocument.load_from_json("test/data/input/dl_doc_1.json")
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
        file_path="https://example.com/foo.pdf",
        export_type=ExportType.MARKDOWN,
    )
    lc_doc_iter = loader.lazy_load()
    act_lc_docs = list(lc_doc_iter)
    assert len(act_lc_docs) == 1

    act_data = {"root": [lc_doc.model_dump() for lc_doc in act_lc_docs]}
    exp_file = "test/data/output/lc_doc_md_1.json"
    if GEN_TEST_DATA:
        out = json.dumps(act_data, indent=4)
        with open(exp_file, mode="w", encoding="utf-8") as f:
            f.write(f"{out}\n")
    else:
        with open(exp_file, encoding="utf-8") as f:
            exp_data = json.load(f)
        assert act_data == exp_data


@pytest.mark.requires("docling")
def test_load_as_doc_chunks(monkeypatch: pytest.MonkeyPatch) -> None:

    mock_dl_doc = DoclingDocument.load_from_json("test/data/input/dl_doc_1.json")
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
        file_path="https://example.com/foo.pdf",
        export_type=ExportType.DOC_CHUNKS,
        chunker=HierarchicalChunker(),
    )
    lc_doc_iter = loader.lazy_load()
    act_lc_docs = list(lc_doc_iter)
    assert len(act_lc_docs) == 2

    act_data = {"root": [lc_doc.model_dump() for lc_doc in act_lc_docs]}
    exp_file = "test/data/output/lc_doc_chunks_1.json"
    if GEN_TEST_DATA:
        out = json.dumps(act_data, indent=4)
        with open(exp_file, mode="w", encoding="utf-8") as f:
            f.write(f"{out}\n")
    else:
        with open(exp_file, encoding="utf-8") as f:
            exp_data = json.load(f)
        assert act_data == exp_data
