"""
StreamDiffusionTD Installer

Correct installation sequence that lets setup.py handle dependency versions.

Philosophy:
1. PyTorch FIRST - Everything depends on it, pin CUDA version
2. numpy LOCKED - Before and after other installs (numpy 2.x breaks everything)
3. Let setup.py handle most deps - Single source of truth
4. --no-deps for conflict-prone packages - mediapipe, controlnet_aux, opencv
5. Verify imports - Catch failures immediately
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Callable

# Version pins - packages NOT in setup.py that must be manually pinned
MANUAL_PINS = {
    "numpy": "1.26.4",
    "timm": ">=1.0.24",
    "opencv-python": "4.8.1.78",
}

# PyTorch configurations by CUDA version
PYTORCH_CONFIGS = {
    "cu118": {
        "torch": "2.4.0",
        "torchvision": "0.19.0",
        "torchaudio": None,
        "index_url": "https://download.pytorch.org/whl/cu118",
        "cuda_python": "11.8.7",
        "xformers": "0.0.30",
    },
    "cu121": {
        "torch": "2.4.0",
        "torchvision": "0.19.0",
        "torchaudio": None,
        "index_url": "https://download.pytorch.org/whl/cu121",
        "cuda_python": "12.9.0",
        "xformers": "0.0.30",
    },
    "cu124": {
        "torch": "2.4.0",
        "torchvision": "0.19.0",
        "torchaudio": None,
        "index_url": "https://download.pytorch.org/whl/cu121",  # cu124 uses cu121 index
        "cuda_python": "12.9.0",
        "xformers": None,  # Skip - causes conflicts
    },
    "cu128": {
        "torch": "2.7.0",
        "torchvision": "0.22.0",
        "torchaudio": "2.7.0",
        "index_url": "https://download.pytorch.org/whl/cu128",
        "cuda_python": "12.9.0",
        "xformers": None,  # Not needed - PyTorch 2.7+ has native SDPA
    },
}


class Installer:
    """Handles StreamDiffusionTD installation with correct dependency ordering."""

    def __init__(
        self,
        base_folder: str,
        cuda_version: str = "cu128",
        no_cache: bool = False,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ):
        """
        Initialize installer.

        Args:
            base_folder: Path to StreamDiffusionTD folder
            cuda_version: CUDA version (cu118, cu121, cu124, cu128)
            no_cache: If True, use --no-cache-dir for pip
            progress_callback: Optional callback(message, step, total_steps)
        """
        self.base_folder = Path(base_folder).resolve()
        self.cuda_version = cuda_version
        self.no_cache = no_cache
        self.progress_callback = progress_callback

        self.venv_path = self.base_folder / "venv"
        # setup.py is directly in base_folder (base_folder IS the StreamDiffusion repo root)
        self.streamdiffusion_path = self.base_folder

        # Validate CUDA version
        if cuda_version not in PYTORCH_CONFIGS:
            raise ValueError(
                f"Unsupported CUDA version: {cuda_version}. "
                f"Supported: {list(PYTORCH_CONFIGS.keys())}"
            )

        self.pytorch_config = PYTORCH_CONFIGS[cuda_version]

    @property
    def python_exe(self) -> Path:
        """Path to Python executable in venv."""
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "python.exe"
        return self.venv_path / "bin" / "python"

    @property
    def pip_args(self) -> list:
        """Base pip arguments."""
        args = [str(self.python_exe), "-m", "pip", "install"]
        if self.no_cache:
            args.append("--no-cache-dir")
        return args

    def _report_progress(self, message: str, step: int, total: int):
        """Report progress to callback if set."""
        print(f"[{step}/{total}] {message}")
        if self.progress_callback:
            self.progress_callback(message, step, total)

    def _run_pip(self, args: list, check: bool = True, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Run pip with given arguments."""
        cmd = self.pip_args + args
        work_dir = cwd or self.base_folder
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(work_dir),
        )
        if check and result.returncode != 0:
            print(f"  STDERR: {result.stderr}")
            raise RuntimeError(f"pip failed: {result.stderr}")
        return result

    def _run_python(self, code: str) -> subprocess.CompletedProcess:
        """Run Python code in venv."""
        return subprocess.run(
            [str(self.python_exe), "-c", code],
            capture_output=True,
            text=True,
        )

    def create_venv(self, python_exe: Optional[str] = None) -> bool:
        """
        Create virtual environment.

        Args:
            python_exe: Python executable to use. If None, uses sys.executable.

        Returns:
            True if created, False if already exists.
        """
        if self.venv_path.exists():
            print(f"Virtual environment already exists at: {self.venv_path}")
            return False

        python = python_exe or sys.executable
        print(f"Creating virtual environment with: {python}")
        subprocess.run(
            [python, "-m", "venv", str(self.venv_path)],
            check=True,
        )
        return True

    def phase1_foundation(self):
        """Phase 1: Install pip, setuptools, wheel, and lock numpy."""
        self._report_progress("Installing pip, setuptools, wheel...", 1, 8)
        self._run_pip(["--upgrade", "pip", "setuptools", "wheel"])

        self._report_progress(f"Locking numpy=={MANUAL_PINS['numpy']} (prevents 2.x conflicts)...", 1, 8)
        self._run_pip([f"numpy=={MANUAL_PINS['numpy']}", "--force-reinstall"])

    def phase2_pytorch(self):
        """Phase 2: Install PyTorch with correct CUDA version."""
        config = self.pytorch_config
        self._report_progress(f"Installing PyTorch {config['torch']} with CUDA {self.cuda_version}...", 2, 8)

        # Build torch install command
        packages = [f"torch=={config['torch']}", f"torchvision=={config['torchvision']}"]
        if config["torchaudio"]:
            packages.append(f"torchaudio=={config['torchaudio']}")

        self._run_pip(packages + ["--index-url", config["index_url"]])

        # Install cuda-python
        self._report_progress(f"Installing cuda-python=={config['cuda_python']}...", 2, 8)
        self._run_pip([f"cuda-python=={config['cuda_python']}"])

        # Verify PyTorch CUDA
        result = self._run_python(
            "import torch; "
            "assert torch.cuda.is_available(), 'CUDA not available!'; "
            "print(f'PyTorch {torch.__version__} CUDA {torch.version.cuda}')"
        )
        if result.returncode != 0:
            raise RuntimeError(f"PyTorch CUDA verification failed: {result.stderr}")
        print(f"  Verified: {result.stdout.strip()}")

    def phase3_xformers(self):
        """Phase 3: Install xformers if needed for this CUDA version."""
        config = self.pytorch_config
        if config["xformers"]:
            self._report_progress(f"Installing xformers=={config['xformers']}...", 3, 8)
            self._run_pip([f"xformers=={config['xformers']}"])
        else:
            self._report_progress("Skipping xformers (not needed for this CUDA version)...", 3, 8)

    def phase4_streamdiffusion(self):
        """Phase 4: Install StreamDiffusion - let setup.py handle versions."""
        self._report_progress("Installing StreamDiffusion (daydream fork)...", 4, 8)

        # Install from StreamDiffusion directory where setup.py lives
        # The -e flag makes it editable, setup.py handles all pinned versions
        self._run_pip(["-e", ".[tensorrt,controlnet,ipadapter]"], check=True, cwd=self.streamdiffusion_path)

    def phase5_missing_pins(self):
        """Phase 5: Install packages not pinned in setup.py."""
        self._report_progress("Installing packages not in setup.py (timm)...", 5, 8)
        self._run_pip([f"timm{MANUAL_PINS['timm']}"])

    def phase6_conflict_prone(self):
        """Phase 6: Fix conflict-prone packages with --no-deps."""
        self._report_progress("Fixing conflict-prone packages...", 6, 8)

        # Remove conflicting opencv variants
        subprocess.run(
            [str(self.python_exe), "-m", "pip", "uninstall", "-y",
             "opencv-python-headless", "opencv-contrib-python"],
            capture_output=True,
        )

        # Install correct opencv
        self._run_pip(["--no-deps", f"opencv-python=={MANUAL_PINS['opencv-python']}"])

    def phase7_numpy_lock(self):
        """Phase 7: Final numpy lock (other packages may have upgraded it)."""
        self._report_progress(f"Final numpy lock (numpy=={MANUAL_PINS['numpy']})...", 7, 8)
        self._run_pip([f"numpy=={MANUAL_PINS['numpy']}", "--force-reinstall"])

    def phase8_verify(self) -> bool:
        """Phase 8: Verify installation with import tests."""
        from .verifier import Verifier

        self._report_progress("Verifying installation...", 8, 8)
        verifier = Verifier(str(self.python_exe))
        return verifier.run_all()

    def install(self, python_exe: Optional[str] = None) -> bool:
        """
        Run full installation.

        Args:
            python_exe: Python executable for creating venv. If None, uses sys.executable.

        Returns:
            True if installation and verification succeeded.
        """
        print("=" * 50)
        print(" StreamDiffusionTD v0.3.1 Installation")
        print(" Daydream Fork with StreamV2V")
        print("=" * 50)
        print()
        print(f"Base folder: {self.base_folder}")
        print(f"CUDA version: {self.cuda_version}")
        print()

        # Create venv if needed
        self.create_venv(python_exe)

        # Run installation phases
        self.phase1_foundation()
        self.phase2_pytorch()
        self.phase3_xformers()
        self.phase4_streamdiffusion()
        self.phase5_missing_pins()
        self.phase6_conflict_prone()
        self.phase7_numpy_lock()
        success = self.phase8_verify()

        print()
        print("=" * 50)
        if success:
            print(" Installation Complete - All checks passed!")
        else:
            print(" Installation Complete - Some checks failed!")
            print(" Run 'python -m sd_installer diagnose' for details")
        print("=" * 50)

        return success

    def generate_batch_file(self, output_path: Optional[str] = None) -> str:
        """
        Generate a standalone batch file for installation.

        This is for users who prefer running a .bat file directly.

        Args:
            output_path: Where to write the batch file. Default: base_folder/Install_StreamDiffusion.bat

        Returns:
            Path to the generated batch file.
        """
        if output_path is None:
            output_path = self.base_folder / "Install_StreamDiffusion.bat"

        config = self.pytorch_config
        no_cache = "--no-cache-dir" if self.no_cache else ""

        # Build torch packages string
        torch_packages = f"torch=={config['torch']} torchvision=={config['torchvision']}"
        if config["torchaudio"]:
            torch_packages += f" torchaudio=={config['torchaudio']}"

        # xformers line
        if config["xformers"]:
            xformers_line = f'python -m pip install {no_cache} xformers=={config["xformers"]}'
        else:
            xformers_line = "echo Skipping xformers (not needed for this CUDA version)"

        content = f'''@echo off
echo ========================================
echo  StreamDiffusionTD v0.3.1 Installation
echo  Daydream Fork with StreamV2V
echo ========================================

cd /d "{self.base_folder}"

rem === PHASE 1: VENV SETUP ===
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call "venv\\Scripts\\activate.bat"

rem === PHASE 2: FOUNDATION (order matters!) ===
echo [1/8] Installing pip, setuptools, wheel...
python -m pip install {no_cache} --upgrade pip setuptools wheel

echo [1/8] Locking numpy FIRST (prevents 2.x conflicts)...
python -m pip install {no_cache} "numpy=={MANUAL_PINS['numpy']}" --force-reinstall

rem === PHASE 3: PYTORCH (must be before StreamDiffusion) ===
echo [2/8] Installing PyTorch with CUDA {self.cuda_version}...
python -m pip install {no_cache} {torch_packages} --index-url {config['index_url']}
python -m pip install {no_cache} cuda-python=={config['cuda_python']}

rem Verify PyTorch CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA FAILED'; print(f'PyTorch {{torch.__version__}} CUDA {{torch.version.cuda}}')"

rem === PHASE 4: XFORMERS ===
echo [3/8] Installing xformers...
{xformers_line}

rem === PHASE 5: STREAMDIFFUSION (setup.py handles most deps) ===
echo [4/8] Installing StreamDiffusion (daydream fork)...
rem setup.py is in base folder (base folder IS StreamDiffusion)
python -m pip install {no_cache} -e ".[tensorrt,controlnet,ipadapter]"

rem === PHASE 6: MISSING PINS (not in setup.py) ===
echo [5/8] Installing packages not pinned in setup.py...
python -m pip install {no_cache} "timm{MANUAL_PINS['timm']}"

rem === PHASE 7: CONFLICT-PRONE PACKAGES (--no-deps) ===
echo [6/8] Fixing conflict-prone packages...
pip uninstall -y opencv-python-headless opencv-contrib-python 2>nul
python -m pip install {no_cache} --no-deps "opencv-python=={MANUAL_PINS['opencv-python']}"

rem === PHASE 8: FINAL NUMPY LOCK ===
echo [7/8] Final numpy lock (other packages may have upgraded it)...
python -m pip install {no_cache} "numpy=={MANUAL_PINS['numpy']}" --force-reinstall

rem === PHASE 9: VERIFICATION ===
echo [8/8] Verifying installation...
python -c "import torch; assert torch.cuda.is_available(); print('torch CUDA: OK')"
python -c "from streamdiffusion.config import load_config; print('StreamDiffusion: OK')"
python -c "from timm.layers import RotaryEmbedding; print('timm RotaryEmbedding: OK')"
python -c "import mediapipe as mp; mp.solutions.drawing_utils; print('mediapipe: OK')"
python -c "from transformers import MT5Tokenizer; print('transformers MT5: OK')"
python -c "from huggingface_hub import hf_hub_download; print('huggingface_hub: OK')"
python -c "import numpy; assert numpy.__version__.startswith('1.'); print('numpy %%s: OK' %% numpy.__version__)"
python -c "from diffusers.models.attention_processor import Attention; print('diffusers (varshith15 fork): OK')"

echo ========================================
echo Installation Complete
echo ========================================
pause
'''

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Generated batch file: {output_path}")
        return str(output_path)
