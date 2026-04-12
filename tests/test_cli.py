"""CLI integration tests — all operations go through ``kllm`` entry-point."""

import subprocess
import sys


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
        assert "--mode" in r.stdout

    def test_missing_mode_fails(self):
        r = _run_cli(check=False)
        assert r.returncode != 0


class TestCLIParser:
    def test_engine_flag_accepted(self):
        r = _run_cli("--mode", "inference", "--engine", "bitlogic", "--text", "hi",
                      "--save-dir", "/nonexistent", check=False)
        assert "unrecognized arguments" not in r.stderr

    def test_engine_standard_accepted(self):
        r = _run_cli("--mode", "inference", "--engine", "standard", "--text", "hi",
                      "--save-dir", "/nonexistent", check=False)
        assert "unrecognized arguments" not in r.stderr

    def test_max_tokens_flag(self):
        r = _run_cli("--mode", "generate", "--max-tokens", "10", "--text", "hi",
                      "--save-dir", "/nonexistent", check=False)
        assert "unrecognized arguments" not in r.stderr

    def test_solver_timeout_flag(self):
        r = _run_cli("--mode", "compile", "--solver-timeout", "500",
                      "--save-dir", "/nonexistent", check=False)
        assert "unrecognized arguments" not in r.stderr

    def test_max_layers_flag(self):
        r = _run_cli("--mode", "compare", "--max-layers", "3", "--text", "hi",
                      "--save-dir", "/nonexistent", check=False)
        assert "unrecognized arguments" not in r.stderr

    def test_invalid_mode_fails(self):
        r = _run_cli("--mode", "invalid", check=False)
        assert r.returncode != 0
        assert "invalid choice" in r.stderr
