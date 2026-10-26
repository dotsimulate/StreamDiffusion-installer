import subprocess
import sys

import pytest

from sd_installer.installer import CUDA_LINK_WHEELS
from sd_installer.verifier import VERIFICATION_CHECKS, cuda_link_check_code, expected_cuda_link_version

ON_CP311_WINDOWS = sys.platform == "win32" and sys.version_info[:2] == (3, 11)


def test_expected_version_follows_the_wheel_url():
    assert expected_cuda_link_version() == "1.14.1"
    assert f"/v{expected_cuda_link_version()}/" in CUDA_LINK_WHEELS[(3, 11)]


def test_verifier_uses_the_exact_version_check():
    name, code, description = next(c for c in VERIFICATION_CHECKS if c[0] == "cuda-link")
    assert code == cuda_link_check_code("1.14.1")
    assert "1.14.1" in description


def run_check(tmp_path, version):
    pkg = tmp_path / "cuda_link"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    return subprocess.run([sys.executable, "-c", cuda_link_check_code("1.14.1")], cwd=tmp_path,
                          capture_output=True, text=True, timeout=60)


@pytest.mark.skipif(not ON_CP311_WINDOWS, reason="the check only runs on cp311 Windows")
def test_matching_version_passes(tmp_path):
    result = run_check(tmp_path, "1.14.1")
    assert result.returncode == 0 and result.stdout.strip() == "1.14.1"


@pytest.mark.skipif(not ON_CP311_WINDOWS, reason="the check only runs on cp311 Windows")
def test_left_over_older_version_fails(tmp_path):
    result = run_check(tmp_path, "1.12.2")
    assert result.returncode != 0
    assert "cuda-link 1.12.2 installed, 1.14.1 required" in result.stderr


@pytest.mark.skipif(ON_CP311_WINDOWS, reason="other platforms skip")
def test_other_platforms_skip(tmp_path):
    result = run_check(tmp_path, "0.0.0")
    assert result.returncode == 0 and result.stdout.startswith("SKIP")
