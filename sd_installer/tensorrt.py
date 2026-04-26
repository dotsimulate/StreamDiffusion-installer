"""
TensorRT installation for StreamDiffusionTD

Standalone module that doesn't rely on streamdiffusion package imports.
"""

import platform
import subprocess
import sys
from typing import Optional


def run_pip(command: str):
    """Run pip command with proper error handling"""
    return subprocess.check_call([sys.executable, "-m", "pip"] + command.split())


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
    else:
        print(f"Unsupported CUDA version: {cu}")
        print("Supported versions: CUDA 11.x, 12.x, 12.8+")
        return False

    # Install additional TensorRT tools
    if not is_installed("polygraphy"):
        print("Installing polygraphy...")
        run_pip("install polygraphy==0.49.26 --extra-index-url https://pypi.ngc.nvidia.com --no-cache-dir")
    if not is_installed("onnx_graphsurgeon"):
        print("Installing onnx-graphsurgeon...")
        run_pip("install onnx-graphsurgeon==0.6.1 --extra-index-url https://pypi.ngc.nvidia.com --no-cache-dir")

    # FP8 quantization dependencies (CUDA 12 only).
    # Previously missing — caused ImportError in fp8_quantize.py when users enabled FP8.
    # Aligns with FLUX pyproject.toml (nvidia-modelopt >= 0.19.0).
    if cuda_major == "12":
        print("Installing FP8 quantization dependencies (modelopt, cupy)...")
        run_pip("install nvidia-modelopt[onnx]>=0.19.0 cupy-cuda12x==13.6.0 numpy==1.26.4 --no-cache-dir")
        # modelopt's resolver downgrades onnxruntime-gpu to 1.22.0; re-assert 1.24.4.
        # --no-deps avoids triggering a conflicting re-solve.
        run_pip("install onnxruntime-gpu==1.24.4 --no-deps --no-cache-dir")

    if platform.system() == "Windows" and not is_installed("pywin32"):
        print("Installing pywin32...")
        run_pip("install pywin32==306 --no-cache-dir")
    if platform.system() == "Windows" and not is_installed("triton"):
        print("Installing triton-windows...")
        run_pip("install triton-windows==3.4.0.post21 --no-cache-dir")

    print("TensorRT installation completed successfully!")
    return True


if __name__ == "__main__":
    install()
