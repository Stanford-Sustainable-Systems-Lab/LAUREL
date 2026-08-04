"""Test suite for the DownloadedFileDataset class.

These tests never touch the network. ``DownloadedFileDataset`` accepts ``file://``
sources, so the real streaming code path is exercised against a file in ``tmp_path``;
only the two failure modes that a healthy source cannot produce -- a mid-transfer read
error and a source that overstates its size -- are simulated with ``patch``.
"""

import logging
from pathlib import Path
from unittest.mock import patch

import aiohttp
import pytest
from kedro.io.core import DatasetError

from laurel.datasets.downloaded_file import DownloadedFileDataset

CONTENT = b"pretend this is a very large OpenStreetMap extract" * 40


@pytest.fixture
def source(tmp_path):
    """A local file standing in for a remote download source."""
    path = tmp_path / "source" / "extract.osm.pbf"
    path.parent.mkdir()
    path.write_bytes(CONTENT)
    return path


@pytest.fixture
def dest(tmp_path):
    """The destination path, inside a directory that does not exist yet."""
    return tmp_path / "dest" / "01_raw" / "extract.osm.pbf"


def as_url(path):
    return f"file://{path}"


@pytest.fixture
def dataset(dest, source):
    """A dataset configured the way a catalog entry configures one: both endpoints."""
    return DownloadedFileDataset(
        filepath=str(dest), save_args={"url": as_url(source)}
    )


class TestDownloadedFileDatasetConstructor:
    """Construction, defaults and description."""

    def test_defaults_are_applied(self, dataset):
        assert dataset._save_args["overwrite"] is False
        assert dataset._save_args["check_name"] is True
        assert dataset._save_args["chunk_size_mb"] == 16
        assert dataset._save_args["connect_timeout_secs"] == 30
        assert dataset._save_args["stall_timeout_secs"] == 180

    def test_save_args_override_defaults_without_dropping_them(self, dest):
        dataset = DownloadedFileDataset(
            filepath=str(dest), save_args={"overwrite": True}
        )

        assert dataset._save_args["overwrite"] is True
        assert dataset._save_args["check_name"] is True

    def test_defaults_are_not_mutated_by_an_instance(self, dest):
        DownloadedFileDataset(filepath=str(dest), save_args={"overwrite": True})

        assert DownloadedFileDataset.DEFAULT_SAVE_ARGS["overwrite"] is False

    def test_protocol_is_split_from_the_path(self, dest):
        dataset = DownloadedFileDataset(filepath=str(dest))

        assert dataset._protocol == "file"
        assert str(dataset._filepath) == str(dest)

    def test_describe_reports_string_keys(self, dataset):
        description = dataset._describe()

        assert set(description) == {"filepath", "protocol", "save_args"}
        assert all(isinstance(key, str) for key in description)

    def test_describe_surfaces_the_source_url(self, dataset, source):
        # save_args is in _describe, so the configured source shows up in repr and logs.
        assert dataset._describe()["save_args"]["url"] == as_url(source)

    def test_unknown_save_arg_is_rejected(self, dest, source):
        # save_args is a free-form dict, so a typo here has to be caught by us.
        with pytest.raises(DatasetError) as excinfo:
            DownloadedFileDataset(
                filepath=str(dest), save_args={"urll": as_url(source)}
            )

        assert "urll" in str(excinfo.value)
        assert "Valid options are" in str(excinfo.value)


class TestDownloadedFileDatasetLoad:
    """``load`` returns a path, and explains itself when it cannot."""

    def test_returns_the_local_path(self, dataset, dest):
        dest.parent.mkdir(parents=True)
        dest.write_bytes(CONTENT)

        # A bare path, not a file:// URI -- osmium and GraphHopper need the former.
        assert dataset.load() == str(dest)

    def test_missing_file_names_the_path_and_the_fix(self, dataset, dest):
        with pytest.raises(DatasetError) as excinfo:
            dataset.load()

        message = str(excinfo.value)
        assert str(dest) in message
        assert "download_inputs" in message
        assert "data_dir" in message

    def test_directory_at_the_path_explains_the_wget_p_trap(self, dataset, dest):
        # What `wget -P <path>` leaves behind: a directory where a file was expected.
        dest.mkdir(parents=True)

        with pytest.raises(DatasetError) as excinfo:
            dataset.load()

        assert "is a directory" in str(excinfo.value)
        assert "wget -P" in str(excinfo.value)


class TestDownloadedFileDatasetSave:
    """``save`` transfers the configured file, or refuses for a stated reason."""

    def test_downloads_and_creates_parent_directories(self, dataset, dest):
        dataset.save({})

        assert dest.read_bytes() == CONTENT

    def test_leaves_no_part_file_behind(self, dataset, dest):
        dataset.save({})

        assert not Path(f"{dest}.part").exists()
        assert list(dest.parent.iterdir()) == [dest]

    def test_existing_file_is_left_alone(self, dataset, dest, caplog):
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"already here")

        caplog.set_level(logging.INFO)
        dataset.save({})

        assert dest.read_bytes() == b"already here"
        assert "skipping download" in caplog.text

    def test_overwrite_from_the_catalog_replaces_an_existing_file(self, source, dest):
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"stale")
        dataset = DownloadedFileDataset(
            filepath=str(dest),
            save_args={"url": as_url(source), "overwrite": True},
        )

        dataset.save({})

        assert dest.read_bytes() == CONTENT

    def test_basename_mismatch_is_rejected_before_any_transfer(self, tmp_path, source):
        dest = tmp_path / "dest" / "north-america-260101.osm.pbf"
        dataset = DownloadedFileDataset(
            filepath=str(dest), save_args={"url": as_url(source)}
        )

        with pytest.raises(DatasetError) as excinfo:
            dataset.save({})

        assert "does not match" in str(excinfo.value)
        # Nothing was created -- not even the parent directory.
        assert not dest.parent.exists()

    def test_basename_mismatch_can_be_allowed(self, tmp_path, source):
        dest = tmp_path / "dest" / "renamed.osm.pbf"
        dataset = DownloadedFileDataset(
            filepath=str(dest),
            save_args={"url": as_url(source), "check_name": False},
        )

        dataset.save({})

        assert dest.read_bytes() == CONTENT

    def test_unsupported_scheme_is_rejected(self, dest):
        dataset = DownloadedFileDataset(
            filepath=str(dest), save_args={"url": "ftp://example.org/extract.osm.pbf"}
        )

        with pytest.raises(DatasetError) as excinfo:
            dataset.save({})

        assert "Unsupported URL scheme" in str(excinfo.value)
        assert not dest.parent.exists()

    def test_saving_none_is_rejected_by_kedro(self, dataset):
        with pytest.raises(DatasetError):
            dataset.save(None)

    def test_chunk_size_smaller_than_the_file_still_copies_it_all(self, source, dest):
        dataset = DownloadedFileDataset(
            filepath=str(dest),
            save_args={
                "url": as_url(source),
                "chunk_size_mb": 0.001,
                "log_every_mb": 0.001,
            },
        )

        dataset.save({})

        assert dest.read_bytes() == CONTENT


class TestDownloadedFileDatasetSaveOverrides:
    """``save`` takes per-run ``save_args`` overrides, layered over the catalog's."""

    def test_missing_url_names_the_catalog_key(self, dest):
        dataset = DownloadedFileDataset(filepath=str(dest))

        with pytest.raises(DatasetError) as excinfo:
            dataset.save({})

        assert "save_args.url" in str(excinfo.value)
        assert not dest.parent.exists()

    def test_url_can_be_supplied_by_the_node(self, dest, source):
        dataset = DownloadedFileDataset(filepath=str(dest))

        dataset.save({"url": as_url(source)})

        assert dest.read_bytes() == CONTENT

    def test_override_url_wins_over_the_catalog(self, dataset, tmp_path, dest):
        other = tmp_path / "other" / "extract.osm.pbf"
        other.parent.mkdir()
        other.write_bytes(b"from the override")

        dataset.save({"url": as_url(other)})

        assert dest.read_bytes() == b"from the override"

    def test_override_applies_to_options_other_than_url(self, dataset, dest):
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"stale")

        dataset.save({"overwrite": True})

        assert dest.read_bytes() == CONTENT

    def test_a_bare_url_string_is_rejected_with_the_dict_form(self, dataset, source):
        with pytest.raises(DatasetError) as excinfo:
            dataset.save(as_url(source))

        assert "dict of save_args overrides" in str(excinfo.value)

    def test_unknown_override_is_rejected(self, dataset, dest):
        with pytest.raises(DatasetError) as excinfo:
            dataset.save({"nonsense": 1})

        assert "nonsense" in str(excinfo.value)
        assert not dest.exists()

    def test_overrides_do_not_leak_into_later_saves(self, dataset, dest):
        dataset.save({"overwrite": True})
        dest.write_bytes(b"changed since")

        dataset.save({})

        assert dest.read_bytes() == b"changed since"


class TestDownloadedFileDatasetSourceTimeout:
    """An http(s) source gets a stall timeout, not aiohttp's 5-minute total cap.

    aiohttp.ClientSession defaults to ClientTimeout(total=300), which bounds an entire
    request regardless of ongoing progress -- guaranteed to fire on any transfer slower
    than 5 minutes, which for an 18 GB file at typical Sherlock bandwidth is every time.
    """

    def test_http_source_gets_an_unbounded_total_timeout(self, dataset):
        opts = dataset._source_storage_options(
            "https://example.org/extract.osm.pbf", dataset._save_args
        )

        timeout = opts["client_kwargs"]["timeout"]
        assert timeout.total is None

    def test_http_source_timeout_uses_the_configured_seconds(self, dest, source):
        dataset = DownloadedFileDataset(
            filepath=str(dest),
            save_args={
                "url": as_url(source),
                "connect_timeout_secs": 5,
                "stall_timeout_secs": 45,
            },
        )

        opts = dataset._source_storage_options(
            "https://example.org/extract.osm.pbf", dataset._save_args
        )

        timeout = opts["client_kwargs"]["timeout"]
        assert timeout.sock_connect == 5
        assert timeout.sock_read == 45

    def test_file_source_gets_no_client_kwargs(self, dataset, source):
        # LocalFileSystem has no notion of client_kwargs; injecting one would be a
        # spurious kwarg on a filesystem that doesn't take it.
        opts = dataset._source_storage_options(as_url(source), dataset._save_args)

        assert "client_kwargs" not in opts

    def test_explicit_client_kwargs_are_left_untouched(self, dataset):
        save_args = {
            **dataset._save_args,
            "storage_options": {"client_kwargs": {"headers": {"X-Test": "1"}}},
        }

        opts = dataset._source_storage_options(
            "https://example.org/extract.osm.pbf", save_args
        )

        assert opts["client_kwargs"] == {"headers": {"X-Test": "1"}}

    def test_https_url_forwards_the_timeout_into_url_to_fs(self, dest, source):
        # End-to-end through save(), not just _source_storage_options: proves the
        # timeout is actually plumbed into the real call that opens the source. The
        # https:// URL is never contacted -- url_to_fs is faked to redirect to the
        # local test source -- so this stays network-free.
        from laurel.datasets.downloaded_file import url_to_fs as real_url_to_fs

        captured = {}

        def fake_url_to_fs(url, **kwargs):
            captured.update(kwargs)
            return real_url_to_fs(as_url(source))

        dataset = DownloadedFileDataset(
            filepath=str(dest),
            save_args={"url": "https://example.org/extract.osm.pbf"},
        )

        with patch("laurel.datasets.downloaded_file.url_to_fs", fake_url_to_fs):
            dataset.save({})

        assert dest.read_bytes() == CONTENT
        assert captured["client_kwargs"]["timeout"].total is None
        assert captured["client_kwargs"]["timeout"].sock_read == 180


class TestDownloadedFileDatasetIncompleteTransfer:
    """A partial transfer must never be published as a complete file."""

    def test_short_read_raises_and_publishes_nothing(self, dataset, dest):
        # The source claims more bytes than it will actually yield.
        with patch.object(
            DownloadedFileDataset, "_source_size", return_value=len(CONTENT) + 100
        ):
            with pytest.raises(DatasetError) as excinfo:
                dataset.save({})

        assert "Incomplete download" in str(excinfo.value)
        assert not dest.exists()
        assert not Path(f"{dest}.part").exists()

    def test_unsized_source_is_transferred_without_the_check(self, dataset, dest):
        with patch.object(DownloadedFileDataset, "_source_size", return_value=None):
            dataset.save({})

        assert dest.read_bytes() == CONTENT

    def test_mid_transfer_failure_cleans_up_and_publishes_nothing(self, dataset, dest):
        def fail_after_writing_part(self, url, part_path, save_args):
            Path(part_path).parent.mkdir(parents=True, exist_ok=True)
            Path(part_path).write_bytes(CONTENT[:20])
            raise OSError("connection reset")

        with patch.object(DownloadedFileDataset, "_stream", fail_after_writing_part):
            with pytest.raises(DatasetError):
                dataset.save({})

        assert not dest.exists()
        assert not Path(f"{dest}.part").exists()

    def test_cleanup_failure_does_not_mask_the_real_error(self, dataset, dest, caplog):
        def fail_after_writing_part(self, url, part_path, save_args):
            Path(part_path).parent.mkdir(parents=True, exist_ok=True)
            Path(part_path).write_bytes(CONTENT[:20])
            raise OSError("connection reset")

        with patch.object(DownloadedFileDataset, "_stream", fail_after_writing_part):
            with patch.object(
                dataset._fs, "rm", side_effect=OSError("filesystem is gone")
            ):
                with pytest.raises(DatasetError) as excinfo:
                    dataset.save({})

        assert "connection reset" in str(excinfo.value)
        assert "Could not remove partial download" in caplog.text


class TestDownloadedFileDatasetExists:
    """``exists`` underpins skip-if-present, so a directory must not count."""

    def test_absent(self, dataset):
        assert dataset.exists() is False

    def test_present(self, dataset, dest):
        dest.parent.mkdir(parents=True)
        dest.write_bytes(CONTENT)

        assert dataset.exists() is True

    def test_directory_does_not_count_as_present(self, dataset, dest):
        dest.mkdir(parents=True)

        assert dataset.exists() is False
