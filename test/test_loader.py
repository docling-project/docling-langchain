import json
from unittest.mock import MagicMock

import pytest
from docling.chunking import HierarchicalChunker
from docling.datamodel.document import DoclingDocument

from langchain_docling.loader import DoclingLoader, ExportType

from .test_data_gen_flag import GEN_TEST_DATA


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


def _patch_local_converter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter.__init__",
        lambda *args, **kwargs: None,
    )


def test_remote_backend_selected_by_url() -> None:
    loader = DoclingLoader(
        file_path="x.pdf", docling_serve_url="https://docling.example/api"
    )
    assert loader._backend == "remote"
    assert loader._converter is None  # heavy converter not constructed
    assert loader._docling_serve_url == "https://docling.example/api"


def test_local_backend_is_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_local_converter(monkeypatch)
    loader = DoclingLoader(file_path="x.pdf")
    assert loader._backend == "local"
    assert loader._converter is not None


def test_env_var_selects_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCLING_SERVE_URL", "https://env.example/api")
    monkeypatch.setenv("DOCLING_API_KEY", "envkey")
    loader = DoclingLoader(file_path="x.pdf")
    assert loader._backend == "remote"
    assert loader._docling_serve_url == "https://env.example/api"
    assert loader._api_key == "envkey"


def test_explicit_converter_and_url_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_local_converter(monkeypatch)
    from docling.document_converter import DocumentConverter

    with pytest.raises(ValueError, match="not both"):
        DoclingLoader(
            file_path="x.pdf",
            converter=DocumentConverter(),
            docling_serve_url="https://docling.example/api",
        )


def test_explicit_converter_wins_over_env_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCLING_SERVE_URL", "https://env.example/api")
    _patch_local_converter(monkeypatch)
    from docling.document_converter import DocumentConverter

    loader = DoclingLoader(file_path="x.pdf", converter=DocumentConverter())
    assert loader._backend == "local"


def test_convert_kwargs_in_remote_mode_warns() -> None:
    with pytest.warns(UserWarning, match="convert_kwargs"):
        DoclingLoader(
            file_path="x.pdf",
            docling_serve_url="https://docling.example/api",
            convert_kwargs={"max_num_pages": 1},
        )


def test_docling_serve_options_in_local_mode_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_local_converter(monkeypatch)
    with pytest.warns(UserWarning, match="docling_serve_options"):
        DoclingLoader(file_path="x.pdf", docling_serve_options={"do_ocr": True})


def test_load_remote_as_markdown(install_fake_client) -> None:
    mock_dl_doc = DoclingDocument.load_from_json("test/data/input/dl_doc_1.json")
    install_fake_client(mock_dl_doc)

    loader = DoclingLoader(
        file_path="https://example.com/foo.pdf",
        docling_serve_url="https://docling.example/api",
        export_type=ExportType.MARKDOWN,
    )
    act_lc_docs = list(loader.lazy_load())
    assert len(act_lc_docs) == 1

    # Remote output must equal the SAME golden the local path produces.
    act_data = {"root": [lc_doc.model_dump() for lc_doc in act_lc_docs]}
    with open("test/data/output/lc_doc_md_1.json", encoding="utf-8") as f:
        exp_data = json.load(f)
    assert act_data == exp_data


def test_load_remote_as_doc_chunks(install_fake_client) -> None:
    mock_dl_doc = DoclingDocument.load_from_json("test/data/input/dl_doc_1.json")
    install_fake_client(mock_dl_doc)

    loader = DoclingLoader(
        file_path="https://example.com/foo.pdf",
        docling_serve_url="https://docling.example/api",
        export_type=ExportType.DOC_CHUNKS,
        chunker=HierarchicalChunker(),
    )
    act_lc_docs = list(loader.lazy_load())
    assert len(act_lc_docs) == 2

    act_data = {"root": [lc_doc.model_dump() for lc_doc in act_lc_docs]}
    with open("test/data/output/lc_doc_chunks_1.json", encoding="utf-8") as f:
        exp_data = json.load(f)
    assert act_data == exp_data
