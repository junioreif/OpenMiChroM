"""HTTP byte-range helpers."""

from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .exceptions import RangeRequestUnsupportedError


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
            return response.read()
    except HTTPError as exc:
        raise RangeRequestUnsupportedError(
            f"Remote server returned status {exc.code} to Range request "
            f"bytes={start}-{stop_inclusive}."
        ) from exc
    except URLError as exc:
        raise RangeRequestUnsupportedError(
            f"Could not complete HTTP Range request bytes={start}-{stop_inclusive}: {exc.reason}"
        ) from exc


class RemoteByteReader:
    """Remote file reader that counts bytes returned from range requests."""

    def __init__(self, url: str, *, timeout: float = 30.0) -> None:
        self.url = url
        self.timeout = timeout
        self.bytes_read = 0

    def read_range(self, start: int, stop_exclusive: int) -> bytes:
        """Read bytes and update the byte counter."""

        data = read_range(self.url, start, stop_exclusive, timeout=self.timeout)
        self.bytes_read += len(data)
        return data

    def reset_byte_counter(self) -> None:
        """Reset the byte counter."""

        self.bytes_read = 0
