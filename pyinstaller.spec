# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for DSR_Extract
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all submodules for our utils package
hidden_imports = [
    'utils.pattern_matcher',
    'utils.fir_parser',
    'utils.document_processor',
    'pickle',
    'traceback',
    'concurrent.futures',
    'glob',
    'secrets',
    'tempfile',
    'shutil',
    'json',
    'csv',
    'io',
    'threading',
    'time',
    're',
    'os',
    'sys',
    'logging',
    'waitress',
    'waitress.server',
    'waitress.utilities',
]

# Collect data files (templates, static files, etc.)
datas = [
    ('templates/', 'templates/'),
    ('static/', 'static/'),
    ('utils/', 'utils/'),
]

# WSGI application entry point
wsgi_app = 'app:app'  # This tells PyInstaller to use app as WSGI application

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'unittest',
        'pydoc',
        'pdb',
        'test',
        'tests',
        'setuptools',
        'pip',
        'wheel',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DSR_Extract',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='KASHVA.ico' if os.path.exists('KASHVA.ico') else None,
)

# For macOS app bundle (if needed)
# app = BUNDLE(
#     exe,
#     name='DSR_Extract.app',
#     icon='KASHVA.ico',
#     bundle_identifier='com.yourcompany.dsr_extract',
#     version='1.0.0',
#     info_plist={
#         'NSPrincipalClass': 'NSApplication',
#         'NSAppleScriptEnabled': False,
#         'CFBundleDocumentTypes': [],
#         'CFBundleURLTypes': [],
#     },
# )