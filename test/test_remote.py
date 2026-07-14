#
# Copyright IBM Corp. 2025 - 2025
# SPDX-License-Identifier: MIT
#

import warnings

import pytest
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import DoclingDocument
from docling.service_client.exceptions import ConversionError, ServiceUnavailableError

from langchain_docling._remote import _filter_known_params, convert_via_endpoint

_INPUT_DOC = "test/data/input/dl_doc_1.json"


def test_filter_known_params_drops_unknown_with_warning() -> None:
    with pytest.warns(UserWarning, match="made_up_field"):
        out = _filter_known_params({"do_ocr": True, "made_up_field": 123})
    assert out == {"do_ocr": True}


def test_filter_known_params_no_warning_when_all_known() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # a UserWarning would raise
        out = _filter_known_params({"do_ocr": True, "image_export_mode": "placeholder"})
    assert out == {"do_ocr": True, "image_export_mode": "placeholder"}


def test_convert_via_endpoint_passes_source_through_and_returns_document(
    install_fake_client,
) -> None:
    from docling.datamodel.service.options import ConvertDocumentsOptions
    from docling.datamodel.service.targets import InBodyTarget

    mock_dl_doc = DoclingDocument.load_from_json(_INPUT_DOC)
    captured: dict = {}
    install_fake_client(mock_dl_doc, captured=captured)

    out = convert_via_endpoint(
        "https://example.com/foo.pdf",
        url="https://docling.example/api",
        api_key="secret",
        options={"do_ocr": True},
    )

    assert out is mock_dl_doc
    # URL string passed through unchanged (NOT wrapped in Path)
    assert captured["source"] == "https://example.com/foo.pdf"
    assert captured["url"] == "https://docling.example/api"
    assert captured["api_key"] == "secret"
    assert isinstance(captured["options"], ConvertDocumentsOptions)
    assert captured["options"].do_ocr is True
    # in-body target is forced so conversion works without server-side artifact
    # storage (the client's default presigned target would need it)
    assert isinstance(captured["target"], InBodyTarget)


def test_convert_via_endpoint_none_api_key_becomes_empty_string(
    install_fake_client,
) -> None:
    mock_dl_doc = DoclingDocument.load_from_json(_INPUT_DOC)
    captured: dict = {}
    install_fake_client(mock_dl_doc, captured=captured)

    convert_via_endpoint("foo.pdf", url="https://docling.example/api", api_key=None)
    assert captured["api_key"] == ""


def test_convert_via_endpoint_raises_on_failed_conversion_status(
    install_fake_client,
) -> None:
    # A non-success ConversionResult must raise (parity with the local path),
    # not silently return an empty document.
    mock_dl_doc = DoclingDocument.load_from_json(_INPUT_DOC)
    install_fake_client(mock_dl_doc, status=ConversionStatus.FAILURE)

    with pytest.raises(ConversionError):
        convert_via_endpoint("foo.pdf", url="https://docling.example/api")


def test_convert_via_endpoint_propagates_typed_client_errors(
    install_fake_client,
) -> None:
    # Typed service-client exceptions propagate unchanged (no wrapping),
    # matching the local converter's error contract.
    install_fake_client(None, submit_error=ServiceUnavailableError("service down"))

    with pytest.raises(ServiceUnavailableError):
        convert_via_endpoint("foo.pdf", url="https://docling.example/api")
