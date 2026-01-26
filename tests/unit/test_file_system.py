# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fairseq2.file_system import (
    FileMode,
    FSspecFileSystem,
)


class TestFSspecFileSystem(unittest.TestCase):
    """Tests for FSspecFileSystem wrapper around fsspec."""

    def setUp(self) -> None:
        import fsspec.implementations.local as fs_local  # type: ignore[import-untyped]

        self.fsspec_fs = fs_local.LocalFileSystem()
        self.fs = FSspecFileSystem(self.fsspec_fs, "")
        self.temp_dir = TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_is_local_returns_true_for_local_fsspec(self) -> None:
        self.assertTrue(self.fs.is_local)

    def test_is_file_returns_true_for_file(self) -> None:
        file_path = self.temp_path / "test.txt"
        file_path.write_text("test")

        self.assertTrue(self.fs.is_file(file_path))

    def test_is_file_returns_false_for_directory(self) -> None:
        self.assertFalse(self.fs.is_file(self.temp_path))

    def test_is_file_returns_false_for_nonexistent(self) -> None:
        self.assertFalse(self.fs.is_file(self.temp_path / "nonexistent"))

    def test_is_dir_returns_true_for_directory(self) -> None:
        self.assertTrue(self.fs.is_dir(self.temp_path))

    def test_is_dir_returns_false_for_file(self) -> None:
        file_path = self.temp_path / "test.txt"
        file_path.write_text("test")

        self.assertFalse(self.fs.is_dir(file_path))

    def test_exists_returns_true_for_file(self) -> None:
        file_path = self.temp_path / "test.txt"
        file_path.write_text("test")

        self.assertTrue(self.fs.exists(file_path))

    def test_exists_returns_true_for_directory(self) -> None:
        self.assertTrue(self.fs.exists(self.temp_path))

    def test_exists_returns_false_for_nonexistent(self) -> None:
        self.assertFalse(self.fs.exists(self.temp_path / "nonexistent"))

    def test_open_read_works(self) -> None:
        file_path = self.temp_path / "test.txt"
        file_path.write_bytes(b"test content")

        with self.fs.open(file_path, FileMode.READ) as f:
            content = f.read()

        self.assertEqual(content, b"test content")

    def test_open_write_works(self) -> None:
        file_path = self.temp_path / "test.txt"

        with self.fs.open(file_path, FileMode.WRITE) as f:
            f.write(b"written content")

        self.assertEqual(file_path.read_bytes(), b"written content")

    def test_open_append_works(self) -> None:
        file_path = self.temp_path / "test.txt"
        file_path.write_bytes(b"initial")

        with self.fs.open(file_path, FileMode.APPEND) as f:
            f.write(b" appended")

        self.assertEqual(file_path.read_bytes(), b"initial appended")

    def test_open_text_read_works(self) -> None:
        file_path = self.temp_path / "test.txt"
        file_path.write_text("test content", encoding="utf-8")

        with self.fs.open_text(file_path, FileMode.READ) as f:
            content = f.read()

        self.assertEqual(content, "test content")

    def test_open_text_write_works(self) -> None:
        file_path = self.temp_path / "test.txt"

        with self.fs.open_text(file_path, FileMode.WRITE) as f:
            f.write("written content")

        self.assertEqual(file_path.read_text(encoding="utf-8"), "written content")

    def test_open_text_append_works(self) -> None:
        file_path = self.temp_path / "test.txt"
        file_path.write_text("initial", encoding="utf-8")

        with self.fs.open_text(file_path, FileMode.APPEND) as f:
            f.write(" appended")

        self.assertEqual(file_path.read_text(encoding="utf-8"), "initial appended")

    def test_move_works(self) -> None:
        old_path = self.temp_path / "old.txt"
        new_path = self.temp_path / "new.txt"
        old_path.write_text("content")

        self.fs.move(old_path, new_path)

        self.assertFalse(old_path.exists())
        self.assertTrue(new_path.exists())
        self.assertEqual(new_path.read_text(), "content")

    def test_remove_works(self) -> None:
        file_path = self.temp_path / "test.txt"
        file_path.write_text("test")

        self.fs.remove(file_path)

        self.assertFalse(file_path.exists())

    def test_make_directory_works(self) -> None:
        dir_path = self.temp_path / "new_dir"

        self.fs.make_directory(dir_path)

        self.assertTrue(dir_path.exists())
        self.assertTrue(dir_path.is_dir())

    def test_make_directory_nested_works(self) -> None:
        dir_path = self.temp_path / "subdir" / "nested"

        self.fs.make_directory(dir_path)

        self.assertTrue(dir_path.exists())
        self.assertTrue(dir_path.is_dir())

    def test_make_directory_exists_ok(self) -> None:
        dir_path = self.temp_path / "existing"
        dir_path.mkdir()

        self.fs.make_directory(dir_path)

        self.assertTrue(dir_path.exists())

    def test_copy_directory_works(self) -> None:
        source = self.temp_path / "source"
        source.mkdir()
        (source / "file.txt").write_text("content")

        target = self.temp_path / "target"

        self.fs.copy_directory(source, target)

        self.assertTrue(target.exists())
        self.assertTrue((target / "file.txt").exists())
        self.assertEqual((target / "file.txt").read_text(), "content")

    def test_glob_works(self) -> None:
        (self.temp_path / "file1.txt").write_text("1")
        (self.temp_path / "file2.txt").write_text("2")
        (self.temp_path / "file.py").write_text("3")

        result = list(self.fs.glob(self.temp_path, "*.txt"))

        self.assertEqual(len(result), 2)
        names = [Path(p).name for p in result]
        self.assertIn("file1.txt", names)
        self.assertIn("file2.txt", names)

    def test_walk_directory_works(self) -> None:
        subdir = self.temp_path / "subdir"
        subdir.mkdir()
        (self.temp_path / "root.txt").write_text("root")
        (subdir / "sub.txt").write_text("sub")

        result = list(self.fs.walk_directory(self.temp_path, on_error=None))

        self.assertGreaterEqual(len(result), 1)

    def test_tmp_directory_works(self) -> None:
        with self.fs.tmp_directory(self.temp_path) as tmp_path:
            self.assertTrue(tmp_path.exists())
            self.assertTrue(tmp_path.is_dir())
            self.assertEqual(tmp_path.parent, self.temp_path)

        self.assertFalse(tmp_path.exists())

    def test_resolve_works(self) -> None:
        relative_path = Path(".")
        resolved = self.fs.resolve(relative_path)

        self.assertTrue(resolved.is_absolute())


class TestFSspecFileSystemWithPrefix(unittest.TestCase):
    """Tests for FSspecFileSystem with URI prefix handling."""

    def setUp(self) -> None:
        import fsspec.implementations.local as fs_local  # type: ignore[import-untyped]

        self.fsspec_fs = fs_local.LocalFileSystem()
        self.prefix = "file://"
        self.fs = FSspecFileSystem(self.fsspec_fs, self.prefix)
        self.temp_dir = TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_get_short_uri_removes_prefix(self) -> None:
        path = f"{self.prefix}/some/path"
        result = self.fs.get_short_uri(path)

        self.assertEqual(result, "some/path")

    def test_get_short_uri_with_sequence(self) -> None:
        paths = [
            f"{self.prefix}/path1",
            f"{self.prefix}/path2",
        ]
        result = self.fs.get_short_uri(paths)

        self.assertEqual(result, ["path1", "path2"])

    def test_get_short_uri_raises_for_invalid_prefix(self) -> None:
        path = Path("/some/path")  # No prefix

        with self.assertRaises(ValueError) as ctx:
            self.fs.get_short_uri(path)

        self.assertIn("does not start with prefix", str(ctx.exception))

    def test_get_long_uri_adds_prefix(self) -> None:
        path = "/some/path"
        result = self.fs.get_long_uri(path)

        self.assertEqual(result, f"{self.prefix}/some/path")

    def test_get_long_uri_with_sequence(self) -> None:
        paths = ["/path1", "/path2"]
        result = self.fs.get_long_uri(paths)

        self.assertEqual(result, [f"{self.prefix}/path1", f"{self.prefix}/path2"])

    def test_get_long_uri_raises_for_already_prefixed(self) -> None:
        path = f"{self.prefix}/some/path"

        with self.assertRaises(ValueError) as ctx:
            self.fs.get_long_uri(path)

        self.assertIn("already starts with prefix", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


class TestNativeLocalFileSystemEquivalence(unittest.TestCase):
    """Tests that NativeLocalFileSystem and FSspecFileSystem (local) behave equivalently."""

    def setUp(self) -> None:
        import fsspec.implementations.local as fs_local  # type: ignore[import-untyped]

        from fairseq2.file_system import NativeLocalFileSystem

        self.fsspec_fs = FSspecFileSystem(fs_local.LocalFileSystem(), "")
        self.native_fs = NativeLocalFileSystem()
        self.temp_dir = TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_is_local_equivalent(self) -> None:
        self.assertEqual(self.fsspec_fs.is_local, self.native_fs.is_local)

    def test_is_file_equivalent(self) -> None:
        file_path = self.temp_path / "test.txt"
        file_path.write_text("test")

        self.assertEqual(
            self.fsspec_fs.is_file(file_path), self.native_fs.is_file(file_path)
        )
        self.assertEqual(
            self.fsspec_fs.is_file(self.temp_path), self.native_fs.is_file(self.temp_path)
        )
        self.assertEqual(
            self.fsspec_fs.is_file(self.temp_path / "nonexistent"),
            self.native_fs.is_file(self.temp_path / "nonexistent"),
        )

    def test_is_dir_equivalent(self) -> None:
        file_path = self.temp_path / "test.txt"
        file_path.write_text("test")

        self.assertEqual(
            self.fsspec_fs.is_dir(self.temp_path), self.native_fs.is_dir(self.temp_path)
        )
        self.assertEqual(
            self.fsspec_fs.is_dir(file_path), self.native_fs.is_dir(file_path)
        )

    def test_exists_equivalent(self) -> None:
        file_path = self.temp_path / "test.txt"
        file_path.write_text("test")

        self.assertEqual(
            self.fsspec_fs.exists(file_path), self.native_fs.exists(file_path)
        )
        self.assertEqual(
            self.fsspec_fs.exists(self.temp_path), self.native_fs.exists(self.temp_path)
        )
        self.assertEqual(
            self.fsspec_fs.exists(self.temp_path / "nonexistent"),
            self.native_fs.exists(self.temp_path / "nonexistent"),
        )

    def test_open_read_equivalent(self) -> None:
        file_path = self.temp_path / "test.txt"
        file_path.write_bytes(b"test content")

        with self.fsspec_fs.open(file_path, FileMode.READ) as f1:
            content1 = f1.read()
        with self.native_fs.open(file_path, FileMode.READ) as f2:
            content2 = f2.read()

        self.assertEqual(content1, content2)

    def test_open_write_equivalent(self) -> None:
        file1 = self.temp_path / "test1.txt"
        file2 = self.temp_path / "test2.txt"

        with self.fsspec_fs.open(file1, FileMode.WRITE) as f:
            f.write(b"written")
        with self.native_fs.open(file2, FileMode.WRITE) as f:
            f.write(b"written")

        self.assertEqual(file1.read_bytes(), file2.read_bytes())

    def test_cat_equivalent(self) -> None:
        file_path = self.temp_path / "test.txt"
        file_path.write_bytes(b"test content")

        self.assertEqual(
            self.fsspec_fs.cat(file_path), self.native_fs.cat(file_path)
        )

    def test_open_text_read_equivalent(self) -> None:
        file_path = self.temp_path / "test.txt"
        file_path.write_text("test content", encoding="utf-8")

        with self.fsspec_fs.open_text(file_path, FileMode.READ) as f1:
            content1 = f1.read()
        with self.native_fs.open_text(file_path, FileMode.READ) as f2:
            content2 = f2.read()

        self.assertEqual(content1, content2)

    def test_glob_equivalent(self) -> None:
        (self.temp_path / "file1.txt").write_text("1")
        (self.temp_path / "file2.txt").write_text("2")
        (self.temp_path / "file.py").write_text("3")

        result1 = sorted([Path(p).name for p in self.fsspec_fs.glob(self.temp_path, "*.txt")])
        result2 = sorted([Path(p).name for p in self.native_fs.glob(self.temp_path, "*.txt")])

        self.assertEqual(result1, result2)

    def test_resolve_equivalent(self) -> None:
        relative_path = Path(".")

        self.assertEqual(
            self.fsspec_fs.resolve(relative_path), self.native_fs.resolve(relative_path)
        )
