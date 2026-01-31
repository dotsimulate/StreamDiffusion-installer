# StreamDiffusion Installer

CLI tool for installing and repairing StreamDiffusion environments used by StreamDiffusionTD.

## Installation

Clone this repo into your StreamDiffusion folder:

```bash
cd /path/to/StreamDiffusion
git clone https://github.com/dotsimulate/StreamDiffusion-installer.git
```

## Usage

Run from the StreamDiffusion folder or from inside StreamDiffusion-installer:

```bash
# Check system readiness
python -m sd_installer check

# Fresh install with CUDA 12.8 (recommended)
python -m sd_installer install --cuda cu128

# Install with specific CUDA version
python -m sd_installer install --cuda cu121

# Verify existing installation
python -m sd_installer verify

# Detailed diagnostics
python -m sd_installer diagnose

# Auto-fix known issues
python -m sd_installer repair

# Generate standalone batch file
python -m sd_installer generate-bat --cuda cu128
```

## Supported CUDA Versions

| CUDA | PyTorch | GPU Support |
|------|---------|-------------|
| cu118 | 2.4.0 | RTX 20/30 series |
| cu121 | 2.4.0 | RTX 30/40 series |
| cu124 | 2.4.0 | RTX 30/40 series |
| cu128 | 2.7.0 | RTX 30/40/50 series (recommended) |

## What This Fixes

The installer uses the correct dependency sequence:

1. **numpy locked first** - Prevents numpy 2.x conflicts
2. **PyTorch with correct CUDA** - Before any other packages
3. **setup.py handles versions** - Single source of truth for pins
4. **Post-install fixes** - timm, opencv with --no-deps
5. **Verification** - Import tests catch failures immediately

## Known Issues Fixed

| Error | Cause | Fix |
|-------|-------|-----|
| `cannot import name 'RotaryEmbedding'` | timm too old | timm>=1.0.24 |
| `cannot import name 'cached_download'` | huggingface_hub version | huggingface_hub==0.35.0 |
| `cannot import name 'MT5Tokenizer'` | transformers too new | transformers==4.56.0 |
| `module 'mediapipe' has no attribute 'solutions'` | mediapipe conflict | --no-deps mediapipe==0.10.21 |
| `numpy 2.x` errors | numpy upgraded by dependencies | numpy==1.26.4 locked |

## Folder Structure

At runtime, the expected structure is:

```
StreamDiffusion/                  <- base folder (par.Basefolder in TD)
├── src/
├── setup.py
├── venv/                         <- created by installer
├── streamdiffusionTD/            <- TD extension files
└── StreamDiffusion-installer/    <- this repo
    └── sd_installer/
```