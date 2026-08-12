"""HTTP byte-range helpers."""

from __future__ import annotations

import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .exceptions import RangeRequestUnsupportedError, RemoteAccessError


def validate_http_url(url: str) -> str:
    """Return a normalized HTTP(S) URL or raise a useful validation error."""

    if not isinstance(url, str):
        raise TypeError("Remote CNDB URLs must be strings.")
    if any(character.isspace() for character in url):
        raise ValueError(f"Remote CNDB URL contains whitespace: {url!r}.")
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError(
            f"Unsupported remote CNDB URL scheme {parsed.scheme!r}; use http:// or https://."
        )
    if not parsed.netloc or parsed.hostname is None:
        raise ValueError(f"Malformed remote CNDB URL {url!r}: a host is required.")
    return url


def read_range(
    url: str,
    start: int,
    stop_exclusive: int,
    *,
    timeout: float = 30.0,
) -> bytes:
    """Read a byte range from a remote URL using HTTP Range requests.

    The server must return ``206 Partial Content``. A ``200 OK`` response is
    rejected because it usually means the server ignored the range and may be
    sending the full HDF5 file.
    """

    url = validate_http_url(url)
    if start < 0:
        raise ValueError("Range start must be non-negative.")
    if stop_exclusive < start:
        raise ValueError("Range stop must be greater than or equal to start.")
    if stop_exclusive == start:
        return b""

    stop_inclusive = stop_exclusive - 1
    try:
        request = Request(url, headers={"Range": f"bytes={start}-{stop_inclusive}"})
        with urlopen(request, timeout=timeout) as response:
            status = response.getcode()
            if status == 200:
                raise RangeRequestUnsupportedError(
                    "Remote server did not honor the HTTP Range request: it returned 200 OK "
                    "instead of 206 Partial Content. Accepting 200 OK here could accidentally "
                    "download the full HDF5/CNDB file."
                )
            if status != 206:
                raise RangeRequestUnsupportedError(
                    f"Remote server returned status {status} to Range request "
                    f"bytes={start}-{stop_inclusive}."
                )
            final_url = response.geturl()
            validate_http_url(final_url)
            content_range = response.headers.get("Content-Range")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range or "")
            if match is None:
                raise RemoteAccessError(
                    "Remote server returned an invalid Content-Range header for "
                    f"bytes={start}-{stop_inclusive}: {content_range!r}."
                )
            returned_start, returned_stop, total_size = map(int, match.groups())
            expected_stop = min(stop_exclusive, total_size) - 1
            if (
                returned_start != start
                or start >= total_size
                or returned_stop != expected_stop
            ):
                raise RemoteAccessError(
                    "Remote server returned an invalid Content-Range header for "
                    f"bytes={start}-{stop_inclusive}: {content_range!r}."
                )
            expected_length = returned_stop - returned_start + 1
            data = response.read(expected_length + 1)
            if len(data) != expected_length:
                raise RemoteAccessError(
                    f"Remote range bytes={start}-{stop_inclusive} returned {len(data)} "
                    f"bytes; expected exactly {expected_length}."
                )
            return data
    except HTTPError as exc:
        raise RemoteAccessError(
            f"Remote server returned HTTP {exc.code} for Range request "
            f"bytes={start}-{stop_inclusive} at {url}."
        ) from exc
    except URLError as exc:
        raise RemoteAccessError(
            f"Could not access remote CNDB URL {url} for Range request "
            f"bytes={start}-{stop_inclusive}: {exc.reason}"
        ) from exc


class RemoteByteReader:
    """Remote file reader that counts bytes returned from range requests."""

    def __init__(self, url: str, *, timeout: float = 30.0) -> None:
        self.url = validate_http_url(url)
        self.timeout = timeout
        self.bytes_read = 0
        self.file_size: int | None = None

    def read_range(self, start: int, stop_exclusive: int) -> bytes:
        """Read bytes and update the byte counter."""

        if self.file_size is not None and start >= self.file_size:
            return b""
        data = read_range(self.url, start, stop_exclusive, timeout=self.timeout)
        if len(data) < stop_exclusive - start:
            self.file_size = start + len(data)
        self.bytes_read += len(data)
        return data

    def reset_byte_counter(self) -> None:
        """Reset the byte counter."""

        self.bytes_read = 0
