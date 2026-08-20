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

import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

# Version pins - packages NOT in setup.py that must be manually pinned
MANUAL_PINS = {
    "numpy": "1.26.4",
    "timm": ">=1.0.24",
    "opencv-python": "4.8.1.78",
    "python-osc": "",  # Required for TouchDesigner OSC communication
    "peft": "0.17.1",  # Required for Cached Attention (StreamV2V) - enables USE_PEFT_BACKEND
    "protobuf": "4.25.8",  # Required by mediapipe, onnx/TensorRT - protobuf 6.x breaks serialization, setup.py requires >=4.25.8
    # Security floor pins (transitive deps — pip resolves these on fresh install, but floor ensures upgrade on update)
    "idna": ">=3.16",  # CVE-2026-45409: punycode resource exhaustion
    "Mako": ">=1.3.12",  # CVE-2026-44307: Windows backslash path traversal
    "urllib3": ">=2.7.0",  # CVE-2026-44432/44431: response over-decompression; cross-origin redirect
    # Most Hub repos (e.g. stabilityai/sd-turbo) are Xet-backed. Without hf_xet, huggingface_hub
    # falls back to plain HTTP through cas-bridge.xethub.hf.co, which is markedly slower and far
    # more prone to read-timeout/retry storms on unstable links.
    "hf_xet": "",
}

# Pre-built insightface wheels for Windows (PyPI has no Windows wheels, requires C++ build tools)
# Source: https://github.com/Gourieff/Assets
INSIGHTFACE_WHEELS = {
    (3, 10): "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl",
    (3, 11): "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp311-cp311-win_amd64.whl",
    (3, 12): "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp312-cp312-win_amd64.whl",
}

# stringzilla 5.1.2 (latest; pulled in transitively via insightface -> albumentations ->
# albucore's unbounded `stringzilla>=3.10.4`) ships no cp311 win_amd64 wheel, so pip falls back
# to building it from source -- which reliably fatal-errors (C1001 internal compiler error in
# stringzilla/cipher/icelake.h under /LTCG, linker exit 0xC0000005) on MSVC toolsets <= 14.38
# (VS 2022 17.8 and earlier). simsimd is albucore's other SIMD dependency and carries the same
# risk profile. Passed via --only-binary in pip_args below so it applies to every install in
# this session, including the transitive pull during phase3b_insightface. This is self-healing:
# pip falls back to the newest wheel-bearing version (5.1.1 today), and the moment upstream
# publishes a 5.1.2+ Windows wheel, pip picks it up automatically -- no manual bump needed.
NO_SOURCE_BUILD = ["stringzilla", "simsimd"]

# Pre-built cuda-link wheel (CUDA-IPC zero-copy transport). setup.py's cuda-link pin lives only in
# the optional cuda_ipc extra as a git ref (compiled cp311 extension) — installing that extra would
# force an MSVC/nvcc source build. Install the prebuilt wheel directly instead, --no-deps, so this
# extra is never triggered. Only a cp311 wheel is published.
CUDA_LINK_WHEELS = {
    (
        3,
        11,
    ): "https://github.com/forkni/cuda-link/releases/download/v1.12.1/cuda_link-1.12.1-cp311-cp311-win_amd64.whl",
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
        "torch": "2.8.0",
        "torchvision": "0.23.0",
        "torchaudio": None,
        "index_url": "https://download.pytorch.org/whl/cu128",
        "cuda_python": "12.9.0",
        "xformers": None,  # Not needed - PyTorch 2.7+ has native SDPA
        # Two transitive nvidia-*-cu12 deps otherwise float unconstrained: cuda-bindings (pulled
        # in by cuda-python) requires nvidia-cuda-nvrtc-cu12 with no upper bound, and
        # nvidia-cudnn-cu12 (pulled in later via onnxruntime-gpu[cudnn]) requires
        # nvidia-cublas-cu12 with no bound at all -- so pip silently resolves both to whatever is
        # newest (12.9.x as of this writing), drifting the install off the CUDA 12.8 toolchain
        # this config targets. Pinned here for install reproducibility only: on Windows these
        # wheels' DLLs are never actually loaded at runtime (torch's own bundled CUDA 12.8/cuDNN
        # 9.x in torch/lib is what's on the DLL search path -- see cuda_dll_path.py), so this
        # does not change GPU behavior, only what gets downloaded.
        "cuda_pins": {
            "nvidia-cublas-cu12": "12.8.5.5",
            "nvidia-cuda-nvrtc-cu12": "12.8.93",
        },
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
            raise ValueError(f"Unsupported CUDA version: {cuda_version}. Supported: {list(PYTORCH_CONFIGS.keys())}")

        self.pytorch_config = PYTORCH_CONFIGS[cuda_version]

        # Populated during install() for the on-failure diagnostic report (see report.py).
        self.current_phase: Optional[str] = None
        self._last_pip_stderr: Optional[str] = None

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
        # See NO_SOURCE_BUILD above: never let these build from source on Windows.
        args.append(f"--only-binary={','.join(NO_SOURCE_BUILD)}")
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
            self._last_pip_stderr = result.stderr
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

        # Pin unconstrained transitive CUDA deps (see "cuda_pins" comment on PYTORCH_CONFIGS
        # above) before anything downstream can float them to a newer, unpinned release.
        cuda_pins = config.get("cuda_pins")
        if cuda_pins:
            self._report_progress("Pinning CUDA 12.8 transitive deps for reproducibility...", 2, 8)
            self._run_pip([f"{pkg}=={version}" for pkg, version in cuda_pins.items()])

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
            major, minor = map(int, version_str.split("."))
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

    def phase4b_cuda_link(self):
        """Phase 4b: Install cuda-link from pre-built wheel (CUDA-IPC zero-copy transport).

        Not covered by any setup.py extra actually installed above (cuda_ipc is intentionally
        skipped to avoid a source build) — install the compiled wheel directly. Non-fatal: if no
        wheel exists for this venv's Python, CUDA-IPC falls back to the mirror-DAT transport.
        """
        result = self._run_python("import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if result.returncode != 0:
            print("  WARNING: Could not detect venv Python version, skipping cuda-link pre-install")
            return

        version_str = result.stdout.strip()
        try:
            major, minor = map(int, version_str.split("."))
            py_version = (major, minor)
        except ValueError:
            print(f"  WARNING: Could not parse Python version '{version_str}', skipping cuda-link pre-install")
            return

        wheel_url = CUDA_LINK_WHEELS.get(py_version)
        if wheel_url:
            self._report_progress(f"Installing cuda-link 1.12.1 from pre-built wheel (Python {version_str})...", 4, 8)
            self._run_pip(["--no-deps", wheel_url], check=False)
        else:
            print(f"  WARNING: No pre-built cuda-link wheel for Python {version_str}")
            print("  CUDA-IPC zero-copy export will fall back to the mirror-DAT transport")

    def phase4c_cuda_link_env(self):
        """Phase 4c: Persist CUDALINK_LIB_PATH, CUDALINK_DOORBELL, and SDTD_BASE_FOLDER_PATH
        (Windows only).

        CUDALINK_LIB_PATH -> this venv's site-packages:
        TouchDesigner's CUDALinkBootstrap.py reads CUDALINK_LIB_PATH at Text DAT import time to
        enable "library mode" (sys.path injection of the installed cuda_link package, aliasing the
        14 mirror DAT names). Persisting it here via `setx` means every TD process launched after
        this install inherits it automatically -- no manual env-var step.

        CUDALINK_DOORBELL=1:
        Enables the Win32 named-event doorbell so the cuda-link native wait backend reaches its
        low-latency target instead of silently falling back to poll-sleep. The SD<->TD topology is
        bidirectional -- TD's Sender and SD's Exporter are each a producer on their own IPC leg --
        and the doorbell event is only created by a producer whose CUDALINK_DOORBELL=1. TD's Sender
        runs inside TD's own bundled-Python *process*, which reads its environment from user/system
        scope only; a runtime `os.environ.setdefault` (as used for
        CUDALINK_TORCH_GPU_WAIT_ADAPTIVE in td_manager.py) cannot reach a separate process, so this
        must be persisted here instead. CUDALINK_WAIT_BACKEND is deliberately left unset -- its
        default "auto" already selects the native path.

        SDTD_BASE_FOLDER_PATH -> this install's base_folder (StreamDiffusion repo root):
        Same cross-process problem as CUDALINK_DOORBELL -- TD's Python process needs a reliable
        anchor to the repo root for the inference-side error-report dump
        (streamdiffusion.utils.diagnostics.write_error_report), and a runtime os.environ.setdefault
        in td_manager.py can't reach that separate process. Persisting it here means every
        TD-launched Python inherits it without a manual env-var step.

        setx writes to HKCU\\Environment (user scope) and only affects processes started
        *after* it runs, so TD must be (re)started after installation to pick it up. This
        intentionally overwrites any prior manual value (e.g. an older cuda_link_lib\\ target).
        Non-fatal: if setx fails or this isn't Windows, TD simply falls back to the mirror-DAT
        classic mode (for CUDALINK_LIB_PATH), the poll-sleep wait backend (for CUDALINK_DOORBELL),
        or the diagnostics module's own __file__-relative fallback (for SDTD_BASE_FOLDER_PATH).
        """
        if sys.platform != "win32":
            return  # setx is a Windows-only mechanism; non-Windows TD launches are unaffected

        result = self._run_python("import sysconfig; print(sysconfig.get_paths()['purelib'])")
        if result.returncode != 0 or not result.stdout.strip():
            print("  WARNING: Could not resolve venv site-packages path, skipping CUDALINK_LIB_PATH setup")
        else:
            site_packages = result.stdout.strip()
            self._report_progress(f"Persisting CUDALINK_LIB_PATH -> {site_packages}", 4, 8)
            try:
                setx_result = subprocess.run(
                    ["setx", "CUDALINK_LIB_PATH", site_packages],
                    capture_output=True,
                    text=True,
                )
            except OSError as setx_exc:
                print(f"  WARNING: setx failed to persist CUDALINK_LIB_PATH: {setx_exc}")
            else:
                if setx_result.returncode != 0:
                    print(f"  WARNING: setx failed to persist CUDALINK_LIB_PATH: {setx_result.stderr.strip()}")
                else:
                    print("  CUDALINK_LIB_PATH persisted for this user account.")
                    print("  Restart TouchDesigner (and any open shells) to pick up the new environment variable.")

        # CUDALINK_DOORBELL=1 enables the Win32 named-event doorbell so the cuda-link native wait
        # backend reaches its low-latency target. Must be set on the *producer* side, and SD's TD
        # topology is bidirectional (TD Sender + SD Exporter are both producers). TD's Sender runs
        # in TD's own bundled-Python *process*, which reads env from user/system scope only -- a
        # runtime os.environ.setdefault in td_manager.py can't reach it, so it must be persisted
        # here. Independent of the site-packages resolution above, so it runs even if that warned.
        try:
            db_result = subprocess.run(["setx", "CUDALINK_DOORBELL", "1"], capture_output=True, text=True)
        except OSError as setx_exc:
            print(f"  WARNING: setx failed to persist CUDALINK_DOORBELL: {setx_exc}")
        else:
            if db_result.returncode != 0:
                print(f"  WARNING: setx failed to persist CUDALINK_DOORBELL: {db_result.stderr.strip()}")
            else:
                print("  CUDALINK_DOORBELL=1 persisted (enables doorbell/native-wait IPC fast path).")

        # SDTD_BASE_FOLDER_PATH -> repo root, so TD's Python process can locate error_reports/
        # without a manual env-var step. Independent of the blocks above, so it runs even if
        # either warned.
        base_result = subprocess.run(
            ["setx", "SDTD_BASE_FOLDER_PATH", str(self.base_folder)], capture_output=True, text=True
        )
        if base_result.returncode != 0:
            print(f"  WARNING: setx failed to persist SDTD_BASE_FOLDER_PATH: {base_result.stderr.strip()}")
        else:
            print(f"  SDTD_BASE_FOLDER_PATH={self.base_folder} persisted (anchors error-report dumps).")

    def phase5_missing_pins(self):
        """Phase 5: Install packages not pinned in setup.py and fix diffusers."""
        self._report_progress("Installing packages not in setup.py (timm, python-osc, peft, hf_xet)...", 5, 8)
        self._run_pip([f"timm{MANUAL_PINS['timm']}"])
        self._run_pip(["python-osc"])  # Required for TouchDesigner OSC communication
        self._run_pip([f"peft=={MANUAL_PINS['peft']}"])  # Required for Cached Attention (StreamV2V)
        self._run_pip(["hf_xet"])  # Native Xet transport for HuggingFace Hub downloads

        # Force reinstall varshith15 diffusers (other deps may have overwritten it)
        self._report_progress("Ensuring varshith15 diffusers fork with kvo_cache support...", 5, 8)
        self._run_pip(
            [
                "--force-reinstall",
                "--no-deps",
                "diffusers @ git+https://github.com/varshith15/diffusers.git@3e3b72f557e91546894340edabc845e894f00922",
            ]
        )

    def phase6_conflict_prone(self):
        """Phase 6: Fix conflict-prone packages with --no-deps."""
        self._report_progress("Fixing conflict-prone packages...", 6, 8)

        # Remove conflicting opencv variants
        subprocess.run(
            [str(self.python_exe), "-m", "pip", "uninstall", "-y", "opencv-python-headless", "opencv-contrib-python"],
            capture_output=True,
        )

        # Install correct opencv
        self._run_pip(["--no-deps", f"opencv-python=={MANUAL_PINS['opencv-python']}"])

    def phase7_numpy_lock(self):
        """Phase 7: Final numpy/protobuf lock + security floor pins."""
        self._report_progress(f"Final numpy lock (numpy=={MANUAL_PINS['numpy']})...", 7, 8)
        self._run_pip([f"numpy=={MANUAL_PINS['numpy']}", "--force-reinstall"])

        self._report_progress(f"Final protobuf lock (protobuf=={MANUAL_PINS['protobuf']})...", 7, 8)
        self._run_pip([f"protobuf=={MANUAL_PINS['protobuf']}", "--force-reinstall"])

        self._report_progress("Applying security floor pins (idna, Mako, urllib3)...", 7, 8)
        self._run_pip(
            [
                f"idna{MANUAL_PINS['idna']}",
                f"Mako{MANUAL_PINS['Mako']}",
                f"urllib3{MANUAL_PINS['urllib3']}",
            ]
        )

    def phase8_verify(self) -> bool:
        """Phase 8: Verify installation with import tests."""
        from .verifier import Verifier

        self._report_progress("Verifying installation...", 8, 8)
        verifier = Verifier(str(self.python_exe))
        return verifier.run_all()

    def _preflight_check_ngc_index(self) -> None:
        """Warn if pip config still references the decommissioned NVIDIA NGC PyPI mirror.

        pypi.ngc.nvidia.com was retired and is now NXDOMAIN; the live replacement is
        pypi.nvidia.com. The dead index is typically injected into a user/global pip.ini
        by the deprecated nvidia-pyindex shim, and pip retries it (5x per package, per
        index) on every single install below -- a wall of NameResolutionError warnings
        that can bury real errors, plus full DNS-timeout delays on networks that
        blackhole rather than NXDOMAIN. We cannot edit another user's pip.ini from here,
        so surface one clear, actionable warning instead of letting it happen silently.
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "config", "debug"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if "pypi.ngc.nvidia.com" in result.stdout:
                print()
                print("  WARNING: pip config references the decommissioned NVIDIA NGC PyPI")
                print("  mirror (pypi.ngc.nvidia.com), which no longer resolves. Every pip")
                print("  install below will retry against it before falling back to PyPI --")
                print("  expect noisy NameResolutionError warnings (and possible delays).")
                print("  Fix: remove the 'extra-index-url' / 'trusted-host' lines referencing")
                print("  pypi.ngc.nvidia.com from pip.ini, typically found at:")
                print("    %APPDATA%\\pip\\pip.ini")
                print("    %USERPROFILE%\\pip\\pip.ini")
                print("    %PROGRAMDATA%\\pip\\pip.ini")
                print("  (auto-generated by the deprecated nvidia-pyindex package)")
                print()
        except Exception:
            # Diagnostic-only -- must never break the actual install.
            pass

    def _write_install_error_report(self, exc: BaseException) -> None:
        """Best-effort diagnostic dump on install failure. Never raises -- a bug here must
        not mask the real installation error, which the caller re-raises regardless."""
        try:
            from .report import write_error_report

            report_path = write_error_report(
                self.base_folder / "error_reports",
                {
                    "stage": "installation",
                    "exc": exc,
                    "phase": self.current_phase,
                    "python_exe": str(self.python_exe),
                    "base_folder": str(self.base_folder),
                    "cuda_version": self.cuda_version,
                    "pytorch_config": self.pytorch_config,
                    "venv_path": str(self.venv_path),
                    "pip_stderr": self._last_pip_stderr,
                },
            )
            if report_path:
                print(f"\n  Error report written to: {report_path}")
        except Exception as report_exc:
            print(f"  WARNING: Failed to generate error report: {report_exc}")

    def install(self, python_exe: Optional[str] = None) -> bool:
        """
        Run full installation.

        Args:
            python_exe: Python executable for creating venv. If None, uses sys.executable.

        Returns:
            True if installation and verification succeeded.
        """
        print("=" * 50)
        print(" StreamDiffusionTD v0.4.0 Installation")
        print(" Daydream Fork with StreamV2V")
        print("=" * 50)
        print()
        print(f"Base folder: {self.base_folder}")
        print(f"CUDA version: {self.cuda_version}")
        print()

        self._preflight_check_ngc_index()

        # Create venv if needed
        self.create_venv(python_exe)

        # Run installation phases. current_phase is tracked so a failure report (see
        # _write_install_error_report) can name the phase that was running when it broke.
        phases = [
            ("phase1_foundation", self.phase1_foundation),
            ("phase2_pytorch", self.phase2_pytorch),
            ("phase3_xformers", self.phase3_xformers),
            ("phase3b_insightface", self.phase3b_insightface),  # insightface from wheel (Windows)
            ("phase4_streamdiffusion", self.phase4_streamdiffusion),
            ("phase4b_cuda_link", self.phase4b_cuda_link),  # cuda-link from wheel (CUDA-IPC transport)
            ("phase4c_cuda_link_env", self.phase4c_cuda_link_env),  # CUDALINK_LIB_PATH -> venv (TD library mode)
            ("phase5_missing_pins", self.phase5_missing_pins),
            ("phase6_conflict_prone", self.phase6_conflict_prone),
            ("phase7_numpy_lock", self.phase7_numpy_lock),
        ]

        try:
            for name, phase_fn in phases:
                self.current_phase = name
                phase_fn()
            self.current_phase = "phase8_verify"
            success = self.phase8_verify()
        except Exception as exc:
            self._write_install_error_report(exc)
            raise

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
echo  StreamDiffusionTD v0.4.0 Installation
echo  Daydream Fork with StreamV2V
echo ========================================

cd /d "{self.base_folder}"
cd StreamDiffusion-installer

{python_cmd} -m sd_installer --base-folder "{self.base_folder}" install --cuda {self.cuda_version} {no_cache_flag}

pause
'''

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Generated batch file: {output_path}")
        return str(output_path)
