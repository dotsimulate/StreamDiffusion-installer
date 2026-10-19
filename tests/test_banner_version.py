from sd_installer.installer import Installer, banner_version, tox_version


def test_fallback_when_unset(monkeypatch):
    monkeypatch.delenv("SDTD_TOX_VERSION", raising=False)
    assert banner_version() == "v0.4.x"


def test_fallback_when_empty(monkeypatch):
    monkeypatch.setenv("SDTD_TOX_VERSION", "  ")
    assert banner_version() == "v0.4.x"


def test_operator_version(monkeypatch):
    monkeypatch.setenv("SDTD_TOX_VERSION", "0.4.2cloud15")
    assert banner_version() == "v0.4.2cloud15"


def test_existing_v_prefix_kept(monkeypatch):
    monkeypatch.setenv("SDTD_TOX_VERSION", "v0.4.2")
    assert banner_version() == "v0.4.2"


def test_bat_unsafe_characters_dropped(monkeypatch):
    monkeypatch.setenv("SDTD_TOX_VERSION", '0.4.2 & del x | y > z %PATH% "^')
    assert tox_version() == "0.4.2delxyzPATH"


def test_bat_carries_version(monkeypatch, tmp_path):
    monkeypatch.setenv("SDTD_TOX_VERSION", "0.4.2cloud15")
    out = tmp_path / "install.bat"
    Installer(str(tmp_path)).generate_batch_file(str(out))
    text = out.read_text(encoding="utf-8")
    assert 'set "SDTD_TOX_VERSION=0.4.2cloud15"' in text
    assert "echo  StreamDiffusionTD v0.4.2cloud15 Installation" in text
    assert text.index("set ") < text.index("-m sd_installer")


def test_bat_without_version(monkeypatch, tmp_path):
    monkeypatch.delenv("SDTD_TOX_VERSION", raising=False)
    out = tmp_path / "install.bat"
    Installer(str(tmp_path)).generate_batch_file(str(out))
    text = out.read_text(encoding="utf-8")
    assert "SDTD_TOX_VERSION" not in text
    assert "echo  StreamDiffusionTD v0.4.x Installation" in text
