#
# Copyright IBM Corp. 2025 - 2025
# SPDX-License-Identifier: MIT
#

"""Remote Docling endpoint conversion via the Docling service client."""

import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

from docling.datamodel.document import ConversionResult, DoclingDocument

if TYPE_CHECKING:
    from docling.datamodel.service.options import ConvertDocumentsOptions
    from docling.service_client import DoclingServiceClient


def _filter_known_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return only options known to ConvertDocumentsOptions, warning on the rest.

    The client-side ``ConvertDocumentsOptions`` model ignores unknown keys
    silently (Pydantic ``extra="ignore"``), so unknown keys are dropped here
    with an explicit warning instead of vanishing without notice.
    """
    from docling.datamodel.service.options import ConvertDocumentsOptions

    known = set(ConvertDocumentsOptions.model_fields.keys())
    out: Dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in params.items():
        if key in known:
            out[key] = value
        else:
            dropped.append(key)
    if dropped:
        warnings.warn(
            f"Ignoring unknown docling_serve_options keys: {sorted(dropped)}. "
            "See docling.datamodel.service.options.ConvertDocumentsOptions for "
            "supported keys.",
            stacklevel=2,
        )
    return out


def build_convert_options(
    options: Optional[Dict[str, Any]],
) -> "ConvertDocumentsOptions":
    """Validate a remote-options dict into a ``ConvertDocumentsOptions``.

    Unknown keys are dropped with a warning; invalid values raise
    ``pydantic.ValidationError``. Call this once up front so bad options fail
    fast rather than on first conversion.
    """
    from docling.datamodel.service.options import ConvertDocumentsOptions

    return ConvertDocumentsOptions.model_validate(_filter_known_params(options or {}))


def make_client(
    url: str,
    api_key: Optional[str] = None,
    client_kwargs: Optional[Dict[str, Any]] = None,
) -> "DoclingServiceClient":
    """Construct a ``DoclingServiceClient`` for the given endpoint.

    ``client_kwargs`` is forwarded to the client constructor (e.g.
    ``job_timeout``, ``http_read_timeout``); by default the client waits up to
    300s per conversion (``job_timeout``).
    """
    try:
        from docling.service_client import DoclingServiceClient
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(
            "Remote Docling conversion requires the Docling service client. "
            "Install or upgrade Docling: pip install -U 'docling>=2.98'."
        ) from exc

    return DoclingServiceClient(url=url, api_key=api_key or "", **(client_kwargs or {}))


def convert_source(
    client: "DoclingServiceClient",
    source: str,
    options: "ConvertDocumentsOptions",
) -> DoclingDocument:
    """Convert one ``source`` via an open client and return its ``DoclingDocument``.

    ``source`` is forwarded unchanged: the client treats an http(s) URL as a
    remote source and anything else as a local path. Conversion uses an in-body
    result target (``submit(target=InBodyTarget())``) so the document is returned
    in the response body. This is deliberate: the default ``convert()`` target is
    a presigned URL, which requires the server to have artifact (object) storage
    configured and to hand back a URL the client then downloads separately — an
    extra round trip with a download-size cap. In-body needs neither.

    A non-success conversion status is raised as ``ConversionError`` to match the
    local converter (whose ``convert()`` raises on failure); other typed
    service-client exceptions propagate to the caller unchanged.
    """
    from docling.datamodel.base_models import ConversionStatus
    from docling.datamodel.service.targets import InBodyTarget
    from docling.service_client.exceptions import ConversionError

    # InBodyTarget guarantees a ConversionResult; submit()'s return type is a
    # union the type checker can't narrow from the target argument.
    result = cast(
        ConversionResult,
        client.submit(source=source, options=options, target=InBodyTarget()).result(),
    )
    if result.status not in (
        ConversionStatus.SUCCESS,
        ConversionStatus.PARTIAL_SUCCESS,
    ):
        raise ConversionError(
            f"Remote Docling conversion failed for {source!r} "
            f"(status={result.status.value})."
        )
    return result.document


def convert_via_endpoint(
    source: str,
    *,
    url: str,
    api_key: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    client_kwargs: Optional[Dict[str, Any]] = None,
) -> DoclingDocument:
    """Convert a single ``source`` via a remote Docling endpoint.

    Convenience wrapper that opens one client for a single conversion. To convert
    multiple sources, reuse one ``make_client(...)`` across ``convert_source(...)``
    calls instead of paying for a client per source.
    """
    with make_client(url, api_key, client_kwargs) as client:
        return convert_source(client, source, build_convert_options(options))
