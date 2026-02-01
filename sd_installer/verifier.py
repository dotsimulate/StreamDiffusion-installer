"""
StreamDiffusionTD Installation Verifier

Runs import tests to verify installation is working correctly.
"""

import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class VerificationResult:
    """Result of a single verification check."""
    name: str
    passed: bool
    message: str
    error: Optional[str] = None


# Verification checks - each is a (name, python_code, success_message) tuple
VERIFICATION_CHECKS = [
    (
        "torch CUDA",
        "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'{torch.__version__}+cu{torch.version.cuda}')",
        "PyTorch with CUDA"
    ),
    (
        "StreamDiffusion",
        "from streamdiffusion.config import load_config; print('OK')",
        "StreamDiffusion core"
    ),
    (
        "timm RotaryEmbedding",
        "from timm.layers import RotaryEmbedding; print('OK')",
        "timm (>=1.0.24 required)"
    ),
    (
        "mediapipe",
        "import mediapipe as mp; mp.solutions.drawing_utils; print('OK')",
        "mediapipe solutions"
    ),
    (
        "transformers MT5",
        "from transformers import MT5Tokenizer; print('OK')",
        "transformers (MT5Tokenizer)"
    ),
    (
        "huggingface_hub",
        "from huggingface_hub import hf_hub_download; print('OK')",
        "huggingface_hub"
    ),
    (
        "numpy version",
        "import numpy; v = numpy.__version__; assert v.startswith('1.'), f'numpy 2.x detected: {v}'; print(v)",
        "numpy (<2.0.0 required)"
    ),
    (
        "diffusers fork",
        "import inspect; from diffusers.models.attention_processor import Attention; assert 'kvo_cache' in inspect.signature(Attention.forward).parameters, 'Missing kvo_cache'; print('OK')",
        "diffusers (varshith15 fork with kvo_cache)"
    ),
    (
        "accelerate",
        "from accelerate import Accelerator; print('OK')",
        "accelerate"
    ),
    (
        "controlnet_aux",
        "from controlnet_aux import OpenposeDetector; print('OK')",
        "controlnet_aux"
    ),
    (
        "peft (USE_PEFT_BACKEND)",
        "from diffusers.utils import USE_PEFT_BACKEND; assert USE_PEFT_BACKEND, 'peft not detected'; print('OK')",
        "peft (required for Cached Attention/StreamV2V)"
    ),
]


class Verifier:
    """Verifies StreamDiffusionTD installation by running import tests."""

    def __init__(self, python_exe: str):
        """
        Initialize verifier.

        Args:
            python_exe: Path to Python executable to test.
        """
        self.python_exe = python_exe

    def check(self, name: str, code: str, description: str) -> VerificationResult:
        """
        Run a single verification check.

        Args:
            name: Short name for the check
            code: Python code to execute
            description: Human-readable description

        Returns:
            VerificationResult with pass/fail status
        """
        try:
            result = subprocess.run(
                [self.python_exe, "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                output = result.stdout.strip()
                return VerificationResult(
                    name=name,
                    passed=True,
                    message=f"{description}: {output}",
                )
            else:
                return VerificationResult(
                    name=name,
                    passed=False,
                    message=f"{description}: FAILED",
                    error=result.stderr.strip(),
                )

        except subprocess.TimeoutExpired:
            return VerificationResult(
                name=name,
                passed=False,
                message=f"{description}: TIMEOUT",
                error="Check timed out after 30 seconds",
            )
        except Exception as e:
            return VerificationResult(
                name=name,
                passed=False,
                message=f"{description}: ERROR",
                error=str(e),
            )

    def run_all(self, verbose: bool = True) -> bool:
        """
        Run all verification checks.

        Args:
            verbose: If True, print results as they complete.

        Returns:
            True if all checks passed, False otherwise.
        """
        results = []
        passed = 0
        failed = 0

        for name, code, description in VERIFICATION_CHECKS:
            result = self.check(name, code, description)
            results.append(result)

            if result.passed:
                passed += 1
                if verbose:
                    print(f"  OK: {result.message}")
            else:
                failed += 1
                if verbose:
                    print(f"FAIL: {result.message}")
                    if result.error:
                        # Print first line of error
                        error_line = result.error.split('\n')[-1]
                        print(f"      {error_line}")

        if verbose:
            print()
            print(f"Results: {passed} passed, {failed} failed")

        return failed == 0

    def diagnose(self) -> dict:
        """
        Run diagnostics and return detailed information.

        Returns:
            Dictionary with diagnostic information.
        """
        info = {
            "python_exe": self.python_exe,
            "checks": [],
            "versions": {},
        }

        # Run all checks and collect detailed info
        for name, code, description in VERIFICATION_CHECKS:
            result = self.check(name, code, description)
            info["checks"].append({
                "name": name,
                "passed": result.passed,
                "message": result.message,
                "error": result.error,
            })

        # Get version information for key packages
        version_checks = [
            ("torch", "import torch; print(torch.__version__)"),
            ("numpy", "import numpy; print(numpy.__version__)"),
            ("transformers", "import transformers; print(transformers.__version__)"),
            ("diffusers", "import diffusers; print(diffusers.__version__)"),
            ("accelerate", "import accelerate; print(accelerate.__version__)"),
            ("huggingface_hub", "import huggingface_hub; print(huggingface_hub.__version__)"),
            ("mediapipe", "import mediapipe; print(mediapipe.__version__)"),
            ("timm", "import timm; print(timm.__version__)"),
            ("xformers", "import xformers; print(xformers.__version__)"),
            ("onnx", "import onnx; print(onnx.__version__)"),
            ("onnxruntime", "import onnxruntime; print(onnxruntime.__version__)"),
            ("peft", "import peft; print(peft.__version__)"),
        ]

        for pkg, code in version_checks:
            try:
                result = subprocess.run(
                    [self.python_exe, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    info["versions"][pkg] = result.stdout.strip()
                else:
                    info["versions"][pkg] = "NOT INSTALLED"
            except Exception:
                info["versions"][pkg] = "ERROR"

        return info


# Known errors and their fixes - used by repair module
KNOWN_ERRORS = {
    "cannot import name 'RotaryEmbedding'": {
        "cause": "timm version too old (1.0.23)",
        "fix": "pip install timm>=1.0.24",
    },
    "cannot import name 'cached_download'": {
        "cause": "huggingface_hub version incompatible",
        "fix": "pip install huggingface_hub==0.35.0",
    },
    "cannot import name 'MT5Tokenizer'": {
        "cause": "transformers version too new (>=5.0.0)",
        "fix": "pip install transformers==4.56.0",
    },
    "module 'mediapipe' has no attribute 'solutions'": {
        "cause": "mediapipe dependency conflict",
        "fix": "pip install --no-deps mediapipe==0.10.21",
    },
    "No module named 'torch.distributed.device_mesh'": {
        "cause": "accelerate version incompatible",
        "fix": "pip install accelerate==1.10.0",
    },
    "'onnx.helper' has no attribute 'float32_to_bfloat16'": {
        "cause": "onnx version too new",
        "fix": "pip install onnx==1.18.0",
    },
    "Missing kvo_cache": {
        "cause": "Wrong diffusers installed (vanilla instead of varshith15 fork)",
        "fix": "pip install --force-reinstall --no-deps 'diffusers @ git+https://github.com/varshith15/diffusers.git@3e3b72f557e91546894340edabc845e894f00922'",
    },
    "unexpected keyword argument 'kvo_cache'": {
        "cause": "Wrong diffusers installed (vanilla instead of varshith15 fork)",
        "fix": "pip install --force-reinstall --no-deps 'diffusers @ git+https://github.com/varshith15/diffusers.git@3e3b72f557e91546894340edabc845e894f00922'",
    },
    "Linear.forward() takes 2 positional arguments but 3 were given": {
        "cause": "peft not installed - Cached Attention (StreamV2V) requires peft for USE_PEFT_BACKEND=True",
        "fix": "pip install peft==0.17.1",
    },
}


def match_known_error(error_text: str) -> Optional[dict]:
    """
    Match error text against known errors.

    Args:
        error_text: Error message to match

    Returns:
        Dict with cause and fix if matched, None otherwise
    """
    for pattern, info in KNOWN_ERRORS.items():
        if pattern in error_text:
            return info
    return None
