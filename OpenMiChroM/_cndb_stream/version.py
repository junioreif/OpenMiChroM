"""CNDB file-format metadata and compatibility validation."""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from typing import Any

from .exceptions import (
    CNDBFormatError,
    LegacyCNDBVersionWarning,
    UnsupportedCNDBVersionError,
)

CNDB_FORMAT_NAME = "cndb"
CNDB_FORMAT_VERSION = "1.0.0"
SUPPORTED_CNDB_FORMAT_MAJOR = 1
_KNOWN_FORMAT_NAMES = {"cndb", "openmichrom-cndb", "ndb", "swb"}
_VERSION_RE = re.compile(r"^(?P<major>0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){0,2}$")


def metadata_from_attrs(attrs: Mapping[str, Any]) -> dict[str, str | None]:
    """Extract authoritative format fields from an HDF5 attribute mapping."""

    format_name = _text_value(attrs.get("format"))
    version = _text_value(attrs.get("format_version"))
    if version is None and format_name is not None:
        version = _text_value(attrs.get("version"))
    return {
        "cndb_format": format_name,
        "cndb_format_version": version,
    }


def validate_format_metadata(
    metadata: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, str | None]:
    """Validate CNDB format metadata and return normalized compatibility data.

    Files without version metadata predate OpenMiChroM's explicit version
    marker. They remain readable as legacy files and produce a visible warning.
    Explicit 0.x metadata is treated the same way. Version 1.x is current;
    malformed or future major versions fail before coordinate data is used.
    """

    format_name = _text_value(metadata.get("cndb_format"))
    version = _text_value(metadata.get("cndb_format_version"))

    if format_name is not None and format_name.lower() not in _KNOWN_FORMAT_NAMES:
        raise CNDBFormatError(
            f"Unsupported CNDB format marker {format_name!r} in {source}. "
            f"Known markers are {sorted(_KNOWN_FORMAT_NAMES)}."
        )

    if version is None:
        warnings.warn(
            f"CNDB file {source} has no format-version metadata; treating it as a "
            "legacy pre-1.0 file. Re-save it with a current OpenMiChroM release "
            "to add authoritative metadata.",
            LegacyCNDBVersionWarning,
            stacklevel=2,
        )
        return {
            "cndb_format": format_name or CNDB_FORMAT_NAME,
            "cndb_format_version": None,
            "cndb_format_status": "legacy-missing",
        }

    match = _VERSION_RE.fullmatch(version)
    if match is None:
        raise CNDBFormatError(
            f"Malformed CNDB format version {version!r} in {source}; expected "
            "MAJOR, MAJOR.MINOR, or MAJOR.MINOR.PATCH using non-negative integers."
        )

    major = int(match.group("major"))
    if major == 0:
        warnings.warn(
            f"CNDB file {source} declares legacy format version {version}; "
            f"current writers use {CNDB_FORMAT_VERSION}.",
            LegacyCNDBVersionWarning,
            stacklevel=2,
        )
        status = "legacy-explicit"
    elif major != SUPPORTED_CNDB_FORMAT_MAJOR:
        raise UnsupportedCNDBVersionError(
            f"CNDB file {source} declares unsupported format version {version}; "
            f"this OpenMiChroM release supports major version "
            f"{SUPPORTED_CNDB_FORMAT_MAJOR}."
        )
    else:
        status = "supported"

    return {
        "cndb_format": format_name or CNDB_FORMAT_NAME,
        "cndb_format_version": version,
        "cndb_format_status": status,
    }


def _text_value(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CNDBFormatError("CNDB format metadata is not valid UTF-8.") from exc
    value = str(value).strip()
    return value or None
