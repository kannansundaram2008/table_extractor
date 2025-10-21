# cx_Freeze setup for DSR_Extract
import sys
from cx_Freeze import setup, Executable

# Configuration for cx_Freeze
build_exe_options = {
    "packages": [
        "flask",
        "docx",
        "openpyxl",
        "reportlab",
        "dateutil",
        "pypandoc",
        "utils",
        "os",
        "sys",
        "logging",
        "secrets",
        "tempfile",
        "shutil",
        "json",
        "csv",
        "io",
        "threading",
        "time",
        "re",
    ],
    "includes": [
        "pickle",
        "traceback",
        "concurrent.futures",
        "glob",
    ],
    "include_files": [
        ("templates/", "templates/"),
        ("static/", "static/"),
        ("utils/", "utils/"),
    ],
    "excludes": [
        "tkinter",
        "unittest",
        "pydoc",
        "pdb",
        "test",
        "tests",
        "setuptools",
        "pip",
        "wheel",
    ],
    "include_msvcr": True,
    "optimize": 2,
}

# Create executable
executables = [
    Executable(
        "app.py",
        base=None,  # Console application
        target_name="DSR_Extract.exe" if sys.platform == "win32" else "DSR_Extract",
        icon="KASHVA.ico" if os.path.exists("KASHVA.ico") else None,
    )
]

setup(
    name="DSR_Extract",
    version="1.0.0",
    description="FIR Document Processing Application",
    author="Your Name",
    options={"build_exe": build_exe_options},
    executables=executables
)