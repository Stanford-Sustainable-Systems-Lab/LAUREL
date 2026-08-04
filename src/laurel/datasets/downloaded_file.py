"""``DownloadedFileDataset`` fetches a file from a URL and hands downstream nodes its
path rather than its contents.

Most Kedro datasets read bytes into memory. This one deliberately does not: it exists
for large files that are consumed *in place* by software with its own I/O layer, where
Kedro's only useful job is to decide where the file lives and to make sure it got there
intact. In this project that is the OpenStreetMap PBF extract, read by ``osmium`` in
``describe_locations`` and passed to GraphHopper's ``--input`` in ``compute_routes``.

The catalog entry states both endpoints of the transfer: the destination as ``filepath``
and the source as ``save_args["url"]``. A node only *triggers* the download, by returning
a dict of ``save_args`` overrides for that run — usually empty. Configuration and
triggering are therefore separate concerns, and bumping to a new source URL is a catalog
edit rather than a code or parameters change.

Keeping both in the catalog means the destination follows the same
``${runtime_params:data_dir, 'data'}`` convention as every other dataset, instead of
being spelled out in a shell script with an unexpanded ``$SCRATCH`` in it.

Transfers are idempotent: an existing target is left alone unless ``overwrite`` is set,
so re-running the pipeline costs one ``exists()`` call rather than a multi-gigabyte
re-download. Bytes land in a ``.part`` sidecar and are renamed into place only after the
transferred size has been checked against the source, so an interrupted transfer can
never masquerade as a complete file.

``http(s)://`` sources are read through ``fsspec``'s ``HTTPFileSystem``, which requires
``aiohttp`` (a declared dependency of this project, also used by the routing client).
``file://`` sources work with no extra dependency, which is what makes this dataset
testable without network access.
"""

from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

import aiohttp
import fsspec
from fsspec.core import url_to_fs
from kedro.io import AbstractDataset
from kedro.io.core import DatasetError, get_filepath_str, get_protocol_and_path

_BYTES_PER_MB = 1024 * 1024
_SUPPORTED_SOURCE_SCHEMES = ("http", "https", "file")


def _as_gb(n_bytes: int) -> str:
    """Formats a byte count for a log line."""
    return f"{n_bytes / _BYTES_PER_MB / 1024:.1f} GB"


class DownloadedFileDataset(AbstractDataset[dict, str]):
    """``DownloadedFileDataset`` downloads a file from a URL and returns its local path.

    ``save`` takes a dict of per-run ``save_args`` overrides — ``{}`` to use the catalog's
    settings as configured. ``load`` returns the local filesystem path of the downloaded
    file as a string; the file itself is never read into memory.

    Example:
    ::

        >>> dataset = DownloadedFileDataset(
        ...     filepath="data/01_raw/extract.osm.pbf",
        ...     save_args={"url": "https://example.org/extract.osm.pbf"},
        ... )
        >>> dataset.save({})                      # or {"overwrite": True} to re-fetch
        >>> dataset.load()
        'data/01_raw/extract.osm.pbf'
    """

    DEFAULT_SAVE_ARGS: dict[str, Any] = {
        # Source URL. Required in practice; the default exists so that omitting it fails
        # with this class's own message rather than a KeyError.
        "url": None,
        # Leave an existing file alone. Re-downloading tens of gigabytes because a
        # pipeline was re-run is never what anyone wants; set this to force a refresh.
        "overwrite": False,
        # Copy buffer. 16 MB reads amortise request overhead without holding much.
        "chunk_size_mb": 16,
        # Progress-line cadence. Batch jobs log to a file nobody watches live, so this
        # only needs to be often enough to show the transfer is still moving.
        "log_every_mb": 1024,
        # Require the URL's basename to match the destination's. Both are set in this
        # dataset's catalog entry but on separate lines, so this catches a snapshot date
        # bumped in one and not the other -- before the transfer rather than after it.
        "check_name": True,
        # Max seconds to establish the connection to an http(s) source.
        "connect_timeout_secs": 30,
        # Max seconds with no bytes received from an http(s) source before it is
        # considered stalled. There is deliberately no cap on the transfer's total
        # duration: aiohttp's own default (a 300s ClientTimeout.total) bounds the whole
        # request regardless of ongoing progress, which guarantees failure on any file
        # that takes longer than 5 minutes to fetch -- exactly the case this dataset
        # exists for.
        "stall_timeout_secs": 180,
        # Forwarded to fsspec for the source, e.g. request headers. A "client_kwargs" key
        # here overrides the timeout built from the two settings above.
        "storage_options": {},
    }

    def __init__(
        self,
        filepath: str,
        save_args: dict[str, Any] = None,
        fs_args: dict[str, Any] = None,
        metadata: dict[str, Any] = None,
    ):
        """Creates a new instance of ``DownloadedFileDataset``.

        Args:
            filepath: Filepath in POSIX format to the file's destination, optionally
                prefixed with an ``fsspec`` protocol like `s3://`. If no prefix is
                given, the `file` protocol (local filesystem) is used.
            save_args: The source ``url`` and options controlling the transfer. See
                ``DEFAULT_SAVE_ARGS`` for the full set and their defaults.
            fs_args: Extra arguments for the *destination* filesystem. Source-side
                options belong in ``save_args["storage_options"]``.
            metadata: Any arbitrary metadata. This is ignored by Kedro, but may be
                consumed by users or external plugins.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol, **(fs_args or {}))

        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._validate_save_arg_names(save_args, source="the catalog entry")
            self._save_args.update(save_args)

        self.metadata = metadata

    @classmethod
    def _validate_save_arg_names(cls, save_args: dict[str, Any], source: str) -> None:
        """Rejects unrecognised ``save_args`` keys.

        Kedro raises on an unknown *top-level* config key but cannot see inside
        ``save_args``, so without this a misspelled key would be silently ignored and the
        setting it was meant to change would quietly keep its default.
        """
        unknown = sorted(set(save_args) - set(cls.DEFAULT_SAVE_ARGS))
        if unknown:
            raise DatasetError(
                f"Unknown save_args in {source}: {', '.join(unknown)}. "
                f"Valid options are: {', '.join(sorted(cls.DEFAULT_SAVE_ARGS))}."
            )

    def load(self) -> str:
        """Returns the local path of the downloaded file.

        Returns:
            The file's path as a string, ready to hand to software that opens it itself.

        Raises:
            DatasetError: If the file is missing, or if the path is a directory.
        """
        load_path = get_filepath_str(self._filepath, self._protocol)

        if not self._fs.exists(load_path):
            raise DatasetError(
                f"'{load_path}' does not exist. Download it with:\n"
                f"  uv run kedro run --pipeline=download_inputs "
                f"--params=data_dir=<your data directory>\n"
                f"Note that the path above is resolved from 'data_dir', so a run "
                f"pointed at a different directory will look somewhere else."
            )
        if self._fs.isdir(load_path):
            raise DatasetError(
                f"'{load_path}' is a directory, not a file. This is what `wget -P "
                f"<path>` produces: -P names a directory to download *into*, so the "
                f"file usually ends up one level deeper, at "
                f"'{load_path}/{self._filepath.name}'."
            )
        return load_path

    def save(self, overrides: dict) -> None:
        """Downloads this dataset's configured source URL to its filepath.

        Does nothing if the destination already exists and ``overwrite`` is false. The
        transfer goes to a ``.part`` sidecar which is renamed into place only once its
        size has been verified, so an interrupted transfer leaves no usable file behind.

        Args:
            overrides: ``save_args`` to apply for this run only, layered over those from
                the catalog entry. Pass ``{}`` to use the catalog's settings as they are.

        Raises:
            DatasetError: If ``overrides`` is not a dict or names an unknown option, if no
                ``url`` is configured, if the URL's scheme is unsupported, if its basename
                disagrees with the destination's, or if the transfer is incomplete.
        """
        if not isinstance(overrides, dict):
            raise DatasetError(
                f"{type(self).__name__}.save() takes a dict of save_args overrides, not "
                f"{type(overrides).__name__}. A node feeding this dataset should return "
                f"'{{}}' to use the catalog's settings, or e.g. '{{\"url\": ...}}' to "
                f"override one for this run."
            )
        self._validate_save_arg_names(overrides, source="the save() overrides")

        save_path = get_filepath_str(self._filepath, self._protocol)
        save_args = {**deepcopy(self._save_args), **overrides}

        url = save_args["url"]
        if not url:
            raise DatasetError(
                f"No source URL is configured for '{save_path}'. Set save_args.url on "
                f"this dataset's catalog entry, or pass '{{\"url\": ...}}' from the node "
                f"that writes to it."
            )
        self._validate_url(url, check_name=save_args["check_name"])

        if self._fs.exists(save_path) and not save_args["overwrite"]:
            self._logger.info(
                "'%s' already exists (%s); skipping download. To re-fetch it, delete "
                "the file or set save_args.overwrite in the catalog entry.",
                save_path,
                _as_gb(self._fs.size(save_path)),
            )
            return

        parent_dir = self._filepath.parent
        if not self._fs.exists(parent_dir):
            self._fs.makedirs(parent_dir, exist_ok=True)

        part_path = f"{save_path}.part"
        try:
            n_bytes, expected = self._stream(url, part_path, save_args)
        except BaseException:
            self._discard(part_path)
            raise

        if expected is not None and n_bytes != expected:
            self._discard(part_path)
            raise DatasetError(
                f"Incomplete download of '{url}': got {n_bytes} bytes, expected "
                f"{expected}. The partial file has been discarded; re-run to retry."
            )

        self._fs.mv(part_path, save_path)
        self._logger.info("Downloaded '%s' to '%s'.", url, save_path)

    def _validate_url(self, url: str, check_name: bool) -> None:
        """Rejects a URL this dataset cannot or should not fetch, before any I/O."""
        parsed = urlparse(url)
        if parsed.scheme not in _SUPPORTED_SOURCE_SCHEMES:
            raise DatasetError(
                f"Unsupported URL scheme '{parsed.scheme}' in '{url}'. Supported "
                f"schemes are: {', '.join(_SUPPORTED_SOURCE_SCHEMES)}."
            )

        url_name = PurePosixPath(parsed.path).name
        if check_name and url_name != self._filepath.name:
            raise DatasetError(
                f"The URL's filename ('{url_name}') does not match this dataset's "
                f"filepath ('{self._filepath.name}'). Both are set in this dataset's "
                f"catalog entry and are meant to stay in step; update whichever one is "
                f"stale. Set save_args.check_name to false if they are meant to differ."
            )

    def _source_size(self, src_fs: Any, src_path: str) -> int | None:
        """Returns the source's size in bytes, or ``None`` if it does not report one.

        Asked for separately rather than read off the open file: streaming HTTP reads
        (``block_size=0``) never carry a size, so relying on the file object would
        quietly disable the completeness check in ``save``.

        Errors are deliberately not caught here. A source that simply does not advertise
        a length reports ``size: None`` without raising; an exception means the source
        could not be reached at all, and ``fsspec``'s HTTP filesystem reports a refused
        connection or a failed TLS handshake as a ``FileNotFoundError`` on the URL with
        the real cause chained beneath. Swallowing that bought nothing -- the transfer
        below opens the same URL through the same code path and fails on the same error
        two lines later -- while making the log say "does not report a size" about a
        source that was never contacted.
        """
        size = src_fs.info(src_path).get("size")
        if size is None:
            self._logger.warning(
                "'%s' does not report a size; the download cannot be checked for "
                "completeness.",
                src_path,
            )
        return size

    def _source_storage_options(self, url: str, save_args: dict[str, Any]) -> dict[str, Any]:
        """Returns the ``fsspec`` options for opening ``url`` as the transfer source.

        For ``http(s)://`` sources this injects a timeout that bounds only how long a
        connection attempt or an individual read may stall, not how long the whole
        transfer may take -- see ``stall_timeout_secs`` in ``DEFAULT_SAVE_ARGS`` for why
        that distinction matters. An explicit ``client_kwargs`` in ``storage_options``
        (e.g. to add auth headers) is respected as-is and not merged with this timeout.
        """
        storage_options = dict(save_args["storage_options"])
        if urlparse(url).scheme in ("http", "https") and "client_kwargs" not in storage_options:
            storage_options["client_kwargs"] = {
                "timeout": aiohttp.ClientTimeout(
                    total=None,
                    sock_connect=save_args["connect_timeout_secs"],
                    sock_read=save_args["stall_timeout_secs"],
                )
            }
        return storage_options

    def _stream(
        self, url: str, part_path: str, save_args: dict[str, Any]
    ) -> tuple[int, int | None]:
        """Copies ``url`` into ``part_path``, returning bytes written and expected."""
        # int(): a fractional value in the catalog is harmless here but `read` only
        # accepts whole bytes.
        chunk_size = int(save_args["chunk_size_mb"] * _BYTES_PER_MB)
        log_every = int(save_args["log_every_mb"] * _BYTES_PER_MB)
        n_bytes = 0
        next_log = log_every

        src_fs, src_path = url_to_fs(url, **self._source_storage_options(url, save_args))
        expected = self._source_size(src_fs, src_path)
        self._logger.info(
            "Downloading '%s'%s to '%s'.",
            url,
            f" ({_as_gb(expected)})" if expected else "",
            part_path,
        )

        # block_size=0 streams without fsspec's read-ahead cache, which is what a
        # single sequential pass over a large file wants.
        with (
            src_fs.open(src_path, "rb", block_size=0) as src,
            self._fs.open(part_path, "wb") as dst,
        ):
            while chunk := src.read(chunk_size):
                dst.write(chunk)
                n_bytes += len(chunk)
                if n_bytes >= next_log:
                    self._logger.info(
                        "  %s transferred%s.",
                        _as_gb(n_bytes),
                        f" of {_as_gb(expected)}" if expected else "",
                    )
                    next_log += log_every

        return n_bytes, expected

    def _discard(self, part_path: str) -> None:
        """Removes a partial download, without masking the error that caused it."""
        try:
            if self._fs.exists(part_path):
                self._fs.rm(part_path)
        except Exception:  # noqa: BLE001 - cleanup must not replace the real error
            self._logger.warning("Could not remove partial download '%s'.", part_path)

    def _exists(self) -> bool:
        path = get_filepath_str(self._filepath, self._protocol)
        return self._fs.exists(path) and not self._fs.isdir(path)

    def _describe(self) -> dict[str, Any]:
        return {
            "filepath": self._filepath,
            "protocol": self._protocol,
            "save_args": self._save_args,
        }
