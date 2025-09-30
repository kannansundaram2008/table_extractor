# DSR_Extract

A Flask web application for extracting specific data from tables in Word documents based on pattern matching.

## Features

- Upload multiple .doc and .docx files
- Background processing with progress tracking
- Pattern-based extraction of table rows (e.g., legal case data with dates, keywords like IPC/BNS, and time/direction info)
- Export results to CSV, Excel, or PDF
- Web-based UI for easy interaction

## Installation

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python app.py`

## Usage

1. Open the web app in your browser (usually http://127.0.0.1:5000/)
2. Upload Word documents containing tables.
3. Wait for processing to complete.
4. View results and export in desired format.

## Project Structure

- `app.py`: Main Flask application
- `utils/`: Utility modules for document processing and pattern matching
- `templates/`: HTML templates for the web UI
- `static/`: CSS and other static files
- `tests/`: Unit tests

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
