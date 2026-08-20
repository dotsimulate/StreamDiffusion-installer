"""
TensorRT installation for StreamDiffusionTD

Standalone module that doesn't rely on streamdiffusion package imports.
"""

import importlib
import platform
import subprocess
import sys
import time
from typing import Optional


def run_pip(command: str, retries: int = 2):
    """Run pip command with proper error handling; retry a couple times for flaky indexes."""
    args = [sys.executable, "-m", "pip"] + command.split()
    for attempt in range(retries + 1):
        try:
            return subprocess.check_call(args)
        except subprocess.CalledProcessError:
            if attempt >= retries:
                raise
            print(f"  pip failed (attempt {attempt + 1}/{retries + 1}), retrying: {command}")
            time.sleep(3)


def is_installed(package_name: str) -> bool:
    """Check if a package is installed"""
    try:
        __import__(package_name.replace("-", "_"))
        return True
    except ImportError:
        return False


def version(package_name: str) -> Optional[str]:
    """Get version of installed package"""
    try:
        import importlib.metadata

        return importlib.metadata.version(package_name)
    except Exception:
        return None


def _import_ok(module: str) -> bool:
    """True if `module` imports in a fresh interpreter. A subprocess avoids this
    process's stale import caches after a pip install ran in another subprocess.
    The module name is passed as an argv value (not interpolated into the -c
    string) so it can't be abused to inject arbitrary code."""
    return (
        subprocess.run(
            [sys.executable, "-c", "import importlib, sys; importlib.import_module(sys.argv[1])", module],
            capture_output=True,
        ).returncode
        == 0
    )


def ensure_wrapper(module: str, spec: str, index_url: str):
    """Repair the empty-wrapper state: pip reports the dist already satisfied
    (dist-info present) but the top-level package dir is missing, so `import
    <module>` fails. --force-reinstall rebuilds the wrapper from sdist; --no-deps
    leaves the intact bindings/libs wheels untouched. Best-effort: if the repair
    pip call itself fails (e.g. retries exhausted on a flaky index), warn and
    return rather than raising, so install() still reaches the remaining steps
    and verify() instead of aborting with an unhandled traceback.
    """
    if _import_ok(module):
        return
    print(f"'{module}' import failed after install; repairing wrapper package...")
    try:
        run_pip(f"install --force-reinstall --no-deps --no-cache-dir --extra-index-url {index_url} {spec}")
    except subprocess.CalledProcessError:
        print(f"WARNING: repair install for '{module}' failed; continuing without it.")
        return
    if not _import_ok(module):
        print(f"WARNING: '{module}' still not importable after repair.")


def verify(cu: Optional[str] = None) -> bool:
    """
    Verify the TensorRT install by importing/checking every package installed
    by install(). Unlike the main sd_installer verifier ([8/8] VERIFICATION_CHECKS
    in verifier.py), this covers the TensorRT step specifically, since TensorRT is
    a separate, optional UI-button install and previously had no verification at all.

    Args:
        cu: CUDA version string used for the install (e.g. "12.8"). Auto-detected if None.

    Returns:
        True if all applicable checks passed, False otherwise.
    """
    if cu is None:
        cu = get_cuda_version_from_torch()
    cuda_major = cu.split(".")[0] if cu else "12"

    print()
    print("TensorRT Verification")
    print("=" * 40)

    checks = []  # (name, passed, detail)

    def try_import(mod_name: str, label: Optional[str] = None):
        label = label or mod_name
        try:
            mod = __import__(mod_name)
            checks.append((label, True, getattr(mod, "__version__", "OK")))
        except Exception as e:
            checks.append((label, False, str(e)))

    def check_dist(dist_name: str):
        v = version(dist_name)
        checks.append((dist_name, v is not None, v or "not installed"))

    # Functional checks - these actually load the native libs.
    try_import("tensorrt")
    try_import("polygraphy")
    try_import("onnx_graphsurgeon", "onnx-graphsurgeon")

    # Distribution-presence checks - avoid heavy/CUDA-init imports for these.
    if cuda_major == "12":
        check_dist("nvidia-cudnn-cu12")
        check_dist("nvidia-modelopt")
        check_dist("cupy-cuda12x")
    elif cuda_major == "11":
        check_dist("nvidia-cudnn-cu11")

    if platform.system() == "Windows":
        check_dist("pywin32")
        try_import("triton")

    passed = sum(1 for _, ok, _ in checks if ok)
    failed = len(checks) - passed

    for name, ok, detail in checks:
        print(f"{'  OK' if ok else 'FAIL'}: {name}: {detail}")

    print()
    print(f"Results: {passed} passed, {failed} failed")

    return failed == 0


def get_cuda_version_from_torch() -> Optional[str]:
    """Get CUDA version from installed PyTorch"""
    try:
        import torch
    except ImportError:
        return None

    cuda_version = torch.version.cuda
    if cuda_version:
        # Return full version like "12.8" for better detection
        major_minor = ".".join(cuda_version.split(".")[:2])
        return major_minor
    return None


def install(cu: Optional[str] = None):
    """
    Install TensorRT and related packages.

    Args:
        cu: CUDA version string like "12.8" or "11.8". Auto-detected if None.
    """
    if cu is None:
        cu = get_cuda_version_from_torch()

    if cu is None:
        print("Could not detect CUDA version. Please specify manually.")
        return False

    print(f"Detected CUDA version: {cu}")
    print("Installing TensorRT requirements...")

    # Determine CUDA major version for package selection
    cuda_major = cu.split(".")[0] if cu else "12"
    cuda_version_float = float(cu) if cu else 12.0

    # Uninstall old TensorRT versions
    if is_installed("tensorrt"):
        current_version_str = version("tensorrt")
        if current_version_str:
            try:
                from packaging.version import Version

                current_version = Version(current_version_str)
                if current_version < Version("10.8.0"):
                    print("Uninstalling old TensorRT version...")
                    run_pip("uninstall -y tensorrt")
            except Exception:
                # If packaging is not available, check version string directly
                if current_version_str.startswith("9."):
                    print("Uninstalling old TensorRT version...")
                    run_pip("uninstall -y tensorrt")

    # For CUDA 12.8+ (RTX 5090/Blackwell support), use TensorRT 10.16+
    # 10.16.1.11 is the first Blackwell-Windows-production release and fixes
    # the 78% FP8 perf regression that shipped in 10.12–10.13 on SM_120.
    if cuda_version_float >= 12.8:
        print("Installing TensorRT 10.16+ for CUDA 12.8+ (Blackwell GPU support)...")

        # Install cuDNN 9 for CUDA 12
        cudnn_name = "nvidia-cudnn-cu12==9.7.1.26"
        print(f"Installing cuDNN: {cudnn_name}")
        run_pip(f"install {cudnn_name} --no-cache-dir")

        # tensorrt_cu12 is the CUDA 12 wrapper that owns tensorrt/__init__.py
        # and depends on tensorrt_cu12_libs + tensorrt_cu12_bindings.
        # All three are normal wheels with Requires-Dist (no pip-inside-pip).
        trt_version = "10.16.1.11"
        print(f"Installing TensorRT {trt_version} for CUDA {cu}...")
        run_pip(f"install --extra-index-url https://pypi.nvidia.com tensorrt_cu12=={trt_version} --no-cache-dir")
        ensure_wrapper("tensorrt", f"tensorrt_cu12=={trt_version}", "https://pypi.nvidia.com")

    elif cuda_major == "12":
        print("Installing TensorRT for CUDA 12.x...")

        # Install cuDNN for CUDA 12
        cudnn_name = "nvidia-cudnn-cu12==9.7.1.26"
        print(f"Installing cuDNN: {cudnn_name}")
        run_pip(f"install {cudnn_name} --no-cache-dir")

        # tensorrt_cu12 is the CUDA 12 wrapper that owns tensorrt/__init__.py
        # and depends on tensorrt_cu12_libs + tensorrt_cu12_bindings.
        # All three are normal wheels with Requires-Dist (no pip-inside-pip).
        trt_version = "10.16.1.11"
        print(f"Installing TensorRT {trt_version} for CUDA {cu}...")
        run_pip(f"install --extra-index-url https://pypi.nvidia.com tensorrt_cu12=={trt_version} --no-cache-dir")
        ensure_wrapper("tensorrt", f"tensorrt_cu12=={trt_version}", "https://pypi.nvidia.com")

    elif cuda_major == "11":
        print("Installing TensorRT for CUDA 11.x...")

        # Install cuDNN for CUDA 11
        cudnn_name = "nvidia-cudnn-cu11==8.9.7.29"
        print(f"Installing cuDNN: {cudnn_name}")
        run_pip(f"install {cudnn_name} --no-cache-dir")

        # Install TensorRT for CUDA 11
        tensorrt_version = "tensorrt==9.0.1.post11.dev4"
        print(f"Installing TensorRT for CUDA {cu}: {tensorrt_version}")
        run_pip(f"install --extra-index-url https://pypi.nvidia.com {tensorrt_version} --no-cache-dir")
        ensure_wrapper("tensorrt", tensorrt_version, "https://pypi.nvidia.com")
    else:
        print(f"Unsupported CUDA version: {cu}")
        print("Supported versions: CUDA 11.x, 12.x, 12.8+")
        return False

    # Install additional TensorRT tools
    if not is_installed("polygraphy"):
        print("Installing polygraphy...")
        run_pip("install polygraphy==0.49.26 --extra-index-url https://pypi.nvidia.com --no-cache-dir")
    if not is_installed("onnx_graphsurgeon"):
        print("Installing onnx-graphsurgeon...")
        run_pip("install onnx-graphsurgeon==0.6.1 --extra-index-url https://pypi.nvidia.com --no-cache-dir")

    # FP8 quantization dependencies (CUDA 12 only).
    # Previously missing — caused ImportError in fp8_quantize.py when users enabled FP8.
    # modelopt is pinned to 0.43.0 (the proven pin in tests/quality/manifest.json); an unbounded
    # ">=0.19.0" spec floats to 0.45.0, whose [onnx] extra force-upgrades onnx to 1.21.0, which
    # breaks FP8 quant (external-data loading -> negative QDQ scale). setup.py pins onnx==1.19.1.
    if cuda_major == "12":
        print("Installing FP8 quantization dependencies (modelopt, cupy)...")
        # nvidia-modelopt[onnx]==0.43.0 hard-pins onnxruntime-gpu==1.22.0 on Windows (its METADATA:
        # `Requires-Dist: onnxruntime-gpu==1.22.0; platform_system == "Windows" and extra == "onnx"`),
        # so requesting the [onnx] extra force-downgrades our setup.py-authoritative
        # onnxruntime-gpu==1.24.4 to 1.22.0 (~215 MB) only to re-install 1.24.4 (~207 MB) seconds
        # later. Install modelopt WITHOUT the extra and enumerate the extra's deps explicitly,
        # substituting our own onnx/onnxruntime-gpu pins — modelopt core has no onnx requirement, so
        # 1.22.0 never enters the resolve. modelopt is version-pinned, so this dep list is
        # deterministic; re-check it against nvidia_modelopt-<ver>.dist-info/METADATA (the
        # `extra == "onnx"` Requires-Dist lines) if the modelopt pin is ever bumped. onnx-graphsurgeon
        # and polygraphy (also [onnx] deps) are installed above from the NVIDIA index — already
        # satisfied, intentionally not re-listed.
        run_pip(
            "install nvidia-modelopt==0.43.0 "
            "cppimport lief ml_dtypes onnxconverter-common~=1.16.0 onnxscript onnxslim>=0.1.76 "
            "onnx==1.19.1 onnxruntime-gpu==1.24.4 "
            "cupy-cuda12x==13.6.0 numpy==1.26.4 --no-cache-dir"
        )

    if platform.system() == "Windows" and not is_installed("pywin32"):
        print("Installing pywin32...")
        run_pip("install pywin32==311 --no-cache-dir")
    if platform.system() == "Windows" and not is_installed("triton"):
        print("Installing triton-windows...")
        run_pip("install triton-windows==3.4.0.post21 --no-cache-dir")

    # verify() runs in-process; drop any modules install() may have already imported
    # (via is_installed()) and invalidate the finder caches, so verify() picks up a
    # module ensure_wrapper() just rebuilt on disk rather than a stale sys.modules entry.
    for _m in ("tensorrt", "polygraphy", "onnx_graphsurgeon"):
        sys.modules.pop(_m, None)
    importlib.invalidate_caches()
    ok = verify(cu)
    if ok:
        print("TensorRT installation completed successfully!")
    else:
        print("TensorRT installation completed, but verification found issues (see above).")
    return ok


if __name__ == "__main__":
    install()
