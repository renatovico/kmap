"""CLI integration tests — all operations go through ``kllm`` entry-point."""

import subprocess
import sys

import pytest

from kllm.cli import _parse_token_count


def _run_cli(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run the kllm CLI via subprocess and return the result."""
    return subprocess.run(
        [sys.executable, "-m", "kllm.cli", *args],
        capture_output=True,
        text=True,
        check=check,
    )


class TestCLIHelp:
    def test_help_flag(self):
        r = _run_cli("--help", check=False)
        assert r.returncode == 0
        assert "kllm" in r.stdout

    def test_no_command_fails(self):
        r = _run_cli(check=False)
        assert r.returncode != 0

    def test_create_help(self):
        r = _run_cli("create", "--help", check=False)
        assert r.returncode == 0
        assert "--model" in r.stdout

    def test_infer_help(self):
        r = _run_cli("infer", "--help", check=False)
        assert r.returncode == 0
        assert "chip_path" in r.stdout
        assert "--max-tokens" in r.stdout

    def test_export_hdl_help(self):
        r = _run_cli("export-hdl", "--help", check=False)
        assert r.returncode == 0
        assert "--format" in r.stdout


class TestParseTokenCount:
    def test_plain_number(self):
        assert _parse_token_count("50") == 50

    def test_k_suffix_lower(self):
        assert _parse_token_count("128k") == 131072

    def test_k_suffix_upper(self):
        assert _parse_token_count("128K") == 131072

    def test_m_suffix_lower(self):
        assert _parse_token_count("1m") == 1048576

    def test_m_suffix_upper(self):
        assert _parse_token_count("2M") == 2 * 1048576

    def test_whitespace_ignored(self):
        assert _parse_token_count("  64k  ") == 65536

    def test_invalid_raises(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_token_count("abc")

    def test_empty_raises(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_token_count("")


class TestCLIParser:
    def test_create_requires_model(self):
        r = _run_cli("create", "/tmp/test_chip", check=False)
        assert r.returncode != 0

    def test_infer_nonexistent_chip_fails(self):
        r = _run_cli("infer", "/nonexistent_chip_path",
                      "--max-tokens", "1", "hello", check=False)
        assert r.returncode != 0

    def test_export_hdl_nonexistent_chip_fails(self):
        r = _run_cli("export-hdl", "/nonexistent_chip_path", check=False)
        assert r.returncode != 0
