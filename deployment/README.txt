DSR Extract - FIR Document Processing Application
===============================================

Version: 1.0.0
Build Date: 2025-10-14
Platform: Windows

DESCRIPTION
-----------
DSR Extract is a sophisticated Flask web application for parsing and extracting structured data from FIR (First Information Report) documents. The application intelligently identifies FIR table rows and extracts comprehensive structured information including complainant details, victim information, accused details, property information, and legal case data.

FEATURES
--------
- Intelligent Pattern Recognition for FIR table rows
- Structured Data Extraction (Police Station, CR Number, Section of Law, etc.)
- Multi-format Support (.doc and .docx files)
- Background Processing with real-time progress tracking
- Multiple Export Formats (CSV, Excel, PDF)
- Production-ready WSGI deployment with Waitress server
- Modern web interface with responsive design

INSTALLATION & USAGE
--------------------
1. Extract all files to your desired location
2. Double-click DSR_Extract.exe to start the application
3. Open your web browser and navigate to http://127.0.0.1:5000/
4. Upload FIR documents (.doc or .docx files)
5. View extracted data and export results

SYSTEM REQUIREMENTS
------------------
- Windows 7 or later
- 100MB free disk space
- 512MB RAM
- Internet connection (for initial setup)

OPTIONAL DEPENDENCIES
--------------------
For .doc file support, install Pandoc:

📥 DOWNLOAD PANDOC
-----------------
1. Go to: https://pandoc.org/installing.html
2. Download the latest version for Windows
3. Run the installer and follow the setup wizard
4. Pandoc will be installed to your system PATH

🔧 ALTERNATIVE: CHOCOLATEY INSTALLATION
--------------------------------------
Open Command Prompt as Administrator and run:
choco install pandoc

🔧 ALTERNATIVE: WINGET INSTALLATION
----------------------------------
Open Command Prompt as Administrator and run:
winget install JohnMacFarlane.Pandoc

✅ VERIFY PANDOC INSTALLATION
----------------------------
Open Command Prompt and run:
pandoc --version

You should see version information if Pandoc is installed correctly.

SUPPORTED FILE FORMATS
---------------------
- Microsoft Word .docx files (fully supported)
- Microsoft Word .doc files (requires Pandoc installation)

EXPORT OPTIONS
-------------
- CSV: Raw table data export
- Excel: Formatted spreadsheet with multiple sheets
- PDF: Professional report format

TECHNICAL DETAILS
----------------
- Built with Python 3.13
- Flask web framework
- Waitress WSGI server for production deployment
- PyInstaller packaging
- Thread-safe processing with background jobs

SUPPORT
-------
For technical support or questions, please refer to the source code documentation or contact the development team.

LICENSE
-------
This application is distributed under the MIT License. See LICENSE file for details.

BUILD INFORMATION
----------------
- PyInstaller Version: 6.16.0
- Python Version: 3.13.7
- Platform: Windows 11
- Architecture: 64-bit

TROUBLESHOOTING
---------------
1. If the application doesn't start, ensure all files are in the same directory
2. For .doc file support, install Pandoc (see OPTIONAL DEPENDENCIES section above)
3. Check Windows Event Viewer for detailed error messages
4. Ensure no other application is using port 5000

🔍 .DOC FILE TROUBLESHOOTING
---------------------------
❌ "Error: Pandoc not found" when uploading .doc files
   ✅ Solution: Install Pandoc (see OPTIONAL DEPENDENCIES section)

❌ .doc files are skipped during processing
   ✅ Solution: Verify Pandoc installation with: pandoc --version

❌ "Failed to convert .doc to .docx" error
   ✅ Solution: Ensure .doc file is not corrupted and Pandoc is properly installed

❌ Processing hangs on .doc files
   ✅ Solution: Check if .doc file is password-protected or contains complex formatting

📝 NOTE: .DOCX FILES WORK WITHOUT PANDOC
   - .docx files are fully supported without additional software
   - Only .doc files require Pandoc installation

CHANGELOG
---------
v1.0.0 (2025-10-14)
- Initial release
- Full FIR parsing functionality
- WSGI production deployment
- Multi-format export support