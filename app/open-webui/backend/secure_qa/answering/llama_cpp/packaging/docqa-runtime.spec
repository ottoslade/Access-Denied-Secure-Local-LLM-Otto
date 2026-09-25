# PyInstaller spec - builds dist/DocQA-Runtime/docqa-runtime.exe (one-folder: faster start than one-file,
# and antivirus tools are less suspicious of it). Build with scripts/windows/build_exe.ps1.
# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None
root = os.path.abspath(os.path.join(SPECPATH, ".."))

a = Analysis(
    [os.path.join(root, "packaging", "entry.py")],
    pathex=[os.path.join(root, "src")],
    binaries=[],
    datas=[],
    hiddenimports=["docqa_runtime.stub_server", "docqa_runtime.stubmodels", "docqa_runtime.bench"],
    excludes=["tkinter", "unittest", "pydoc", "numpy"],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="docqa-runtime",
    console=True,              # it is a CLI; the desktop app will launch it with CREATE_NO_WINDOW
    upx=False,                 # UPX-packed binaries trigger antivirus false positives
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=False, name="DocQA-Runtime")
