#
# Copyright IBM Corp. 2025 - 2025
# SPDX-License-Identifier: MIT
#

"""Shared pytest fixtures for the docling-langchain test suite."""

from typing import Optional
from unittest.mock import MagicMock

import pytest
from docling.datamodel.base_models import ConversionStatus


@pytest.fixture(autouse=True)
def _clear_remote_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests hermetic.

    An ambient `DOCLING_SERVE_URL` / `DOCLING_API_KEY` in the developer's shell
    would otherwise flip the loader into remote mode and break the local tests.
    """
    monkeypatch.delenv("DOCLING_SERVE_URL", raising=False)
    monkeypatch.delenv("DOCLING_API_KEY", raising=False)


@pytest.fixture
def install_fake_client(monkeypatch: pytest.MonkeyPatch):
    """Return an installer for a fake `DoclingServiceClient`.

    Usage: `install_fake_client(document, captured=..., status=..., submit_error=...)`.
    `captured`, if given, is populated with the client/submit call arguments.
    """

    def _install(
        document,
        *,
        captured: Optional[dict] = None,
        status: ConversionStatus = ConversionStatus.SUCCESS,
        submit_error: Optional[Exception] = None,
    ):
        class _FakeClient:
            def __init__(self, url: str, api_key: str = "", **kwargs) -> None:
                if captured is not None:
                    captured.update(url=url, api_key=api_key, client_kwargs=kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def submit(
                self,
                source,
                options=None,
                output_formats=None,
                headers=None,
                *,
                target=None,
            ):
                if submit_error is not None:
                    raise submit_error
                if captured is not None:
                    captured.update(source=source, options=options, target=target)
                resp = MagicMock()
                resp.document = document
                resp.status = status
                job = MagicMock()
                job.result.return_value = resp
                return job

        monkeypatch.setattr("docling.service_client.DoclingServiceClient", _FakeClient)
        return _FakeClient

    return _install
