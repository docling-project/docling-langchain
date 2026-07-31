import json
from unittest.mock import MagicMock

import pytest
from docling.chunking import HierarchicalChunker
from docling.datamodel.document import DoclingDocument
from docling.datamodel.service.options import ConvertDocumentsOptions
from docling.service_client import DoclingServiceClient

from langchain_docling.loader import DoclingLoader, ExportType

from .test_data_gen_flag import GEN_TEST_DATA


def test_load_as_markdown() -> None:

    mock_dl_doc = DoclingDocument.load_from_json("test/data/input/dl_doc_1.json")
    mock_response = MagicMock()
    mock_response.document = mock_dl_doc
    converter = MagicMock()
    converter.convert.return_value = mock_response

    loader = DoclingLoader(
        file_path="https://example.com/foo.pdf",
        converter=converter,
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


def test_load_as_doc_chunks() -> None:

    mock_dl_doc = DoclingDocument.load_from_json("test/data/input/dl_doc_1.json")
    mock_response = MagicMock()
    mock_response.document = mock_dl_doc
    converter = MagicMock()
    converter.convert.return_value = mock_response

    loader = DoclingLoader(
        file_path="https://example.com/foo.pdf",
        converter=converter,
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


def test_service_client_load_preserves_chunk_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = "https://example.com/foo.pdf"
    options = ConvertDocumentsOptions(do_ocr=False)
    mock_dl_doc = DoclingDocument.load_from_json("test/data/input/dl_doc_1.json")
    mock_response = MagicMock()
    mock_response.document = mock_dl_doc

    with DoclingServiceClient(url="https://docling.example.com") as client:
        convert = MagicMock(return_value=mock_response)
        monkeypatch.setattr(client, "convert", convert)
        loader = DoclingLoader(
            file_path=file_path,
            converter=client,
            convert_kwargs={"options": options},
            export_type=ExportType.DOC_CHUNKS,
            chunker=HierarchicalChunker(),
        )

        documents = loader.load()

    convert.assert_called_once_with(source=file_path, options=options)
    assert len(documents) == 2
    assert all(document.metadata["source"] == file_path for document in documents)
    assert all("dl_meta" in document.metadata for document in documents)
    assert [document.page_content for document in documents] == [
        chunk.text for chunk in HierarchicalChunker().chunk(mock_dl_doc)
    ]
