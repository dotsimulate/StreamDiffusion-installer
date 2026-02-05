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
    "python-osc": "",  # Required for TouchDesigner OSC communication
    "peft": "0.17.1",  # Required for Cached Attention (StreamV2V) - enables USE_PEFT_BACKEND
}

# Pre-built insightface wheels for Windows (PyPI has no Windows wheels, requires C++ build tools)
# Source: https://github.com/Gourieff/Assets
INSIGHTFACE_WHEELS = {
    (3, 10): "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl",
    (3, 11): "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp311-cp311-win_amd64.whl",
    (3, 12): "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp312-cp312-win_amd64.whl",
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

    def phase3b_insightface(self):
        """Phase 3b: Pre-install insightface from pre-built wheel (Windows only).

        PyPI has no Windows wheels for insightface - only source distribution that
        requires Visual C++ Build Tools. Pre-installing from Gourieff's pre-built
        wheels prevents build failures for users without build tools.
        """
        if sys.platform != "win32":
            return  # Only needed on Windows

        # Detect Python version in venv
        result = self._run_python("import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if result.returncode != 0:
            print("  WARNING: Could not detect venv Python version, skipping insightface pre-install")
            return

        version_str = result.stdout.strip()
        try:
            major, minor = map(int, version_str.split('.'))
            py_version = (major, minor)
        except ValueError:
            print(f"  WARNING: Could not parse Python version '{version_str}', skipping insightface pre-install")
            return

        wheel_url = INSIGHTFACE_WHEELS.get(py_version)
        if wheel_url:
            self._report_progress(f"Installing insightface from pre-built wheel (Python {version_str})...", 3, 8)
            self._run_pip([wheel_url], check=False)  # Don't fail if wheel install fails
        else:
            print(f"  WARNING: No pre-built insightface wheel for Python {version_str}")
            print("  insightface will be built from source (requires Visual C++ Build Tools)")

    def phase4_streamdiffusion(self):
        """Phase 4: Install StreamDiffusion - let setup.py handle versions."""
        self._report_progress("Installing StreamDiffusion (daydream fork)...", 4, 8)

        # Install from StreamDiffusion directory where setup.py lives
        # The -e flag makes it editable, setup.py handles all pinned versions
        self._run_pip(["-e", ".[tensorrt,controlnet,ipadapter]"], check=True, cwd=self.streamdiffusion_path)

    def phase5_missing_pins(self):
        """Phase 5: Install packages not pinned in setup.py and fix diffusers."""
        self._report_progress("Installing packages not in setup.py (timm, python-osc, peft)...", 5, 8)
        self._run_pip([f"timm{MANUAL_PINS['timm']}"])
        self._run_pip(["python-osc"])  # Required for TouchDesigner OSC communication
        self._run_pip([f"peft=={MANUAL_PINS['peft']}"])  # Required for Cached Attention (StreamV2V)

        # Force reinstall varshith15 diffusers (other deps may have overwritten it)
        self._report_progress("Ensuring varshith15 diffusers fork with kvo_cache support...", 5, 8)
        self._run_pip([
            "--force-reinstall", "--no-deps",
            "diffusers @ git+https://github.com/varshith15/diffusers.git@3e3b72f557e91546894340edabc845e894f00922"
        ])

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
        self.phase3b_insightface()  # Pre-install insightface from wheel (Windows)
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

    def generate_batch_file(self, output_path: Optional[str] = None, python_exe: Optional[str] = None) -> str:
        """
        Generate a standalone batch file for installation.

        Just calls the CLI install command - no duplicated logic.

        Args:
            output_path: Where to write the batch file. Default: base_folder/Install_StreamDiffusion.bat
            python_exe: Python executable path. Default: "py -3.11" on Windows.

        Returns:
            Path to the generated batch file.
        """
        if output_path is None:
            output_path = self.base_folder / "Install_StreamDiffusion.bat"

        # Use provided python path, or default to py -3.11 launcher
        if python_exe:
            python_cmd = f'"{python_exe}"'
        else:
            python_cmd = "py -3.11"

        no_cache_flag = "--no-cache" if self.no_cache else ""

        content = f'''@echo off
echo ========================================
echo  StreamDiffusionTD v0.3.1 Installation
echo  Daydream Fork with StreamV2V
echo ========================================

cd /d "{self.base_folder}"
cd StreamDiffusion-installer

{python_cmd} -m sd_installer --base-folder "{self.base_folder}" install --cuda {self.cuda_version} {no_cache_flag}

pause
'''

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Generated batch file: {output_path}")
        return str(output_path)
