"""
StreamDiffusionTD Installer CLI

Command-line interface for the installer module.

Usage:
    python -m sd_installer check              # Check system readiness
    python -m sd_installer install            # Fresh install (auto-detects CUDA)
    python -m sd_installer install --cuda cu128  # Install with specific CUDA
    python -m sd_installer verify             # Verify existing installation
    python -m sd_installer diagnose           # Detailed diagnostics
    python -m sd_installer repair             # Auto-fix known issues
    python -m sd_installer generate-bat       # Generate standalone batch file
    python -m sd_installer install-tensorrt   # Install TensorRT packages
"""

import argparse
import os
import sys
from pathlib import Path


def find_base_folder() -> Path:
    """
    Find the StreamDiffusion base folder (where setup.py lives).

    Runtime structure:
        C:\StreamDiffusion\              <- base_folder (has setup.py)
        ├── src/
        ├── setup.py                     <- we look for this
        ├── venv/
        ├── streamdiffusionTD/
        └── StreamDiffusion-installer/   <- installer cloned here
            └── sd_installer/            <- this package
    """
    # Try current directory (user ran from StreamDiffusion root)
    cwd = Path.cwd()
    if (cwd / "setup.py").exists() and (cwd / "src").exists():
        return cwd

    # Try parent of current directory (user is in StreamDiffusion-installer/)
    parent = cwd.parent
    if (parent / "setup.py").exists() and (parent / "src").exists():
        return parent

    # Try to find from this file's location
    # __file__ = .../StreamDiffusion-installer/sd_installer/cli.py
    # We want: .../StreamDiffusion/
    this_file = Path(__file__).resolve()
    sd_installer_pkg = this_file.parent          # sd_installer/
    installer_repo = sd_installer_pkg.parent     # StreamDiffusion-installer/
    base = installer_repo.parent                 # StreamDiffusion/
    if (base / "setup.py").exists():
        return base

    raise RuntimeError(
        "Could not find StreamDiffusion base folder (where setup.py is).\n"
        "Run this from the StreamDiffusion directory or pass --base-folder explicitly."
    )


def detect_cuda_version() -> str:
    """Attempt to detect CUDA version from system."""
    # Default to cu128 (recommended)
    return "cu128"


def cmd_check(args):
    """Check system readiness for installation."""
    print("System Check")
    print("=" * 40)

    # Check Python version
    py_version = sys.version_info
    print(f"Python: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version.major == 3 and py_version.minor in (10, 11):
        print("  OK: Python version supported")
    else:
        print("  WARNING: Python 3.10 or 3.11 recommended")

    # Check if venv exists
    try:
        base = Path(args.base_folder) if args.base_folder else find_base_folder()
        venv_path = base / "venv"
        if venv_path.exists():
            print(f"Venv: Found at {venv_path}")
        else:
            print(f"Venv: Not found (will be created during install)")

        # Check StreamDiffusion setup.py (base folder IS StreamDiffusion)
        setup_py = base / "setup.py"
        if setup_py.exists():
            print(f"StreamDiffusion: Found at {base}")
        else:
            print(f"StreamDiffusion: setup.py NOT FOUND at {base}")
            return 1

    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    print()
    print("System check complete. Ready for installation.")
    return 0


def cmd_install(args):
    """Run fresh installation."""
    from .installer import Installer

    try:
        base = Path(args.base_folder) if args.base_folder else find_base_folder()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    cuda = args.cuda or detect_cuda_version()
    print(f"Installing with CUDA {cuda}...")

    installer = Installer(
        base_folder=str(base),
        cuda_version=cuda,
        no_cache=args.no_cache,
    )

    success = installer.install(python_exe=args.python)
    return 0 if success else 1


def cmd_verify(args):
    """Verify existing installation."""
    from .verifier import Verifier

    try:
        base = Path(args.base_folder) if args.base_folder else find_base_folder()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    # Find Python executable in venv
    venv_path = base / "venv"
    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"

    if not python_exe.exists():
        print(f"ERROR: Virtual environment not found at {venv_path}")
        print("Run 'python -m sd_installer install' first.")
        return 1

    print("Verifying StreamDiffusionTD Installation")
    print("=" * 40)

    verifier = Verifier(str(python_exe))
    success = verifier.run_all(verbose=True)

    return 0 if success else 1


def cmd_diagnose(args):
    """Run detailed diagnostics."""
    from .verifier import Verifier

    try:
        base = Path(args.base_folder) if args.base_folder else find_base_folder()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    # Find Python executable in venv
    venv_path = base / "venv"
    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"

    if not python_exe.exists():
        print(f"ERROR: Virtual environment not found at {venv_path}")
        return 1

    print("StreamDiffusionTD Diagnostics")
    print("=" * 40)

    verifier = Verifier(str(python_exe))
    info = verifier.diagnose()

    print("\nPackage Versions:")
    print("-" * 40)
    for pkg, version in info["versions"].items():
        print(f"  {pkg}: {version}")

    print("\nVerification Checks:")
    print("-" * 40)
    for check in info["checks"]:
        status = "OK" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['name']}")
        if check["error"]:
            # Print just the last line of the error
            error_line = check["error"].split('\n')[-1][:60]
            print(f"         {error_line}")

    return 0


def cmd_repair(args):
    """Auto-fix known issues."""
    from .verifier import Verifier, KNOWN_ERRORS

    try:
        base = Path(args.base_folder) if args.base_folder else find_base_folder()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    # Find Python executable in venv
    venv_path = base / "venv"
    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"

    if not python_exe.exists():
        print(f"ERROR: Virtual environment not found at {venv_path}")
        return 1

    print("StreamDiffusionTD Auto-Repair")
    print("=" * 40)

    # First, diagnose
    verifier = Verifier(str(python_exe))
    info = verifier.diagnose()

    # Find failed checks and match to known fixes
    fixes_needed = []
    for check in info["checks"]:
        if not check["passed"] and check["error"]:
            from .verifier import match_known_error
            fix = match_known_error(check["error"])
            if fix:
                fixes_needed.append((check["name"], fix))

    if not fixes_needed:
        # Check for common issues even if no direct match
        # numpy 2.x
        numpy_ver = info["versions"].get("numpy", "")
        if numpy_ver.startswith("2."):
            fixes_needed.append(("numpy version", {
                "cause": f"numpy {numpy_ver} detected (2.x breaks things)",
                "fix": "pip install numpy==1.26.4 --force-reinstall"
            }))

    if not fixes_needed:
        print("No known issues detected that can be auto-fixed.")
        print("Run 'python -m sd_installer diagnose' for detailed information.")
        return 0

    print(f"Found {len(fixes_needed)} issue(s) to fix:\n")
    for name, fix in fixes_needed:
        print(f"  {name}:")
        print(f"    Cause: {fix['cause']}")
        print(f"    Fix: {fix['fix']}")
        print()

    if not args.yes:
        response = input("Apply fixes? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0

    # Apply fixes
    import subprocess
    for name, fix in fixes_needed:
        print(f"Applying fix for {name}...")
        cmd = [str(python_exe), "-m", "pip"] + fix["fix"].replace("pip ", "").split()
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  OK")
        else:
            print(f"  FAILED: {result.stderr}")

    # Re-verify
    print("\nRe-verifying...")
    success = verifier.run_all(verbose=True)

    return 0 if success else 1


def cmd_generate_bat(args):
    """Generate standalone batch file."""
    from .installer import Installer

    try:
        base = Path(args.base_folder) if args.base_folder else find_base_folder()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    cuda = args.cuda or detect_cuda_version()

    installer = Installer(
        base_folder=str(base),
        cuda_version=cuda,
        no_cache=args.no_cache,
    )

    output = args.output or str(base / "Install_StreamDiffusion.bat")
    installer.generate_batch_file(output, python_exe=args.python)

    print(f"\nGenerated: {output}")
    print("Run this batch file to install StreamDiffusionTD.")
    return 0


def cmd_install_tensorrt(args):
    """Install TensorRT packages."""
    from .tensorrt import install, get_cuda_version_from_torch

    print("StreamDiffusionTD TensorRT Installation")
    print("=" * 40)

    # Get CUDA version
    cuda = args.cuda
    if not cuda:
        cuda = get_cuda_version_from_torch()
        if not cuda:
            print("ERROR: Could not detect CUDA version.")
            print("Make sure PyTorch is installed, or specify --cuda manually.")
            return 1

    print(f"CUDA version: {cuda}")
    print()

    success = install(cu=cuda)
    return 0 if success else 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="sd_installer",
        description="StreamDiffusionTD Installer CLI",
    )
    parser.add_argument(
        "--base-folder",
        help="Path to StreamDiffusionTD folder (auto-detected if not specified)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # check command
    check_parser = subparsers.add_parser("check", help="Check system readiness")
    check_parser.set_defaults(func=cmd_check)

    # install command
    install_parser = subparsers.add_parser("install", help="Run fresh installation")
    install_parser.add_argument(
        "--cuda",
        choices=["cu118", "cu121", "cu124", "cu128"],
        help="CUDA version (default: cu128)",
    )
    install_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Use --no-cache-dir for pip",
    )
    install_parser.add_argument(
        "--python",
        help="Python executable to use for creating venv",
    )
    install_parser.set_defaults(func=cmd_install)

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify existing installation")
    verify_parser.set_defaults(func=cmd_verify)

    # diagnose command
    diagnose_parser = subparsers.add_parser("diagnose", help="Run detailed diagnostics")
    diagnose_parser.set_defaults(func=cmd_diagnose)

    # repair command
    repair_parser = subparsers.add_parser("repair", help="Auto-fix known issues")
    repair_parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Apply fixes without prompting",
    )
    repair_parser.set_defaults(func=cmd_repair)

    # generate-bat command
    bat_parser = subparsers.add_parser("generate-bat", help="Generate standalone batch file")
    bat_parser.add_argument(
        "--cuda",
        choices=["cu118", "cu121", "cu124", "cu128"],
        help="CUDA version (default: cu128)",
    )
    bat_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Use --no-cache-dir for pip",
    )
    bat_parser.add_argument(
        "--output",
        help="Output path for batch file",
    )
    bat_parser.add_argument(
        "--python",
        help="Python executable path to use for venv creation (embeds in batch file)",
    )
    bat_parser.set_defaults(func=cmd_generate_bat)

    # install-tensorrt command
    trt_parser = subparsers.add_parser("install-tensorrt", help="Install TensorRT packages")
    trt_parser.add_argument(
        "--cuda",
        help="CUDA version (e.g., 12.1, 12.8). Auto-detected from PyTorch if not specified.",
    )
    trt_parser.set_defaults(func=cmd_install_tensorrt)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
