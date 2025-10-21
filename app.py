import logging
import pickle
import traceback
from flask import Flask, request, render_template, session, send_file, jsonify
import os
import io
import csv
import openpyxl
import threading
import time
import tempfile
import shutil
import json
import re
import secrets
from utils.document_processor import process_folder
from utils.fir_parser import parse_fir_row
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Configure logging
def setup_logging():
    """Setup logging configuration."""
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def cleanup_temp_dir(temp_dir):
    """Safely cleanup temporary directory and its contents."""
    if not temp_dir or not os.path.exists(temp_dir):
        return

    try:
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except OSError as e:
        logger.error(f"Failed to cleanup temporary directory {temp_dir}: {str(e)}")

def cleanup_expired_sessions():
    """Clean up old session data and temporary files."""
    # This would be called periodically or on shutdown
    # For now, we'll clean up when sessions are accessed
    pass

def strip_html_tags(text):
    """Remove HTML tags from a string."""
    return re.sub(r'<[^>]+>', '', text)

def get_secret_key():
    """Get or generate a secure secret key."""
    secret_key = os.environ.get('FLASK_SECRET_KEY')
    if secret_key:
        return secret_key

    # Generate a secure random key if not provided
    return secrets.token_hex(32)

def secure_filename(filename):
    """Sanitize filename to prevent path traversal and other security issues."""
    import string

    # Remove path separators and dangerous characters
    filename = os.path.basename(filename)  # Remove any path components

    # Keep only safe characters
    safe_chars = string.ascii_letters + string.digits + '.-_'
    filename = ''.join(c for c in filename if c in safe_chars)

    # Ensure filename is not empty and not too long
    if not filename:
        filename = 'unnamed_file'
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext

    return filename

def validate_file_upload(file):
    """Validate uploaded file for security and size constraints."""
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit

    if not file or file.filename == '':
        return False, "No file selected"

    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size > MAX_FILE_SIZE:
        return False, f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"

    if file_size == 0:
        return False, "Empty file not allowed"

    # Check file extension
    allowed_extensions = {'.doc', '.docx'}
    file_ext = os.path.splitext(file.filename.lower())[1]

    if file_ext not in allowed_extensions:
        return False, f"File type not allowed. Only {', '.join(allowed_extensions)} files are permitted"

    return True, "File is valid"

app = Flask(__name__,
           template_folder='templates',
           static_folder='static')
app.secret_key = get_secret_key()

# Security configurations
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max request size

# Thread-safe global state management
class JobManager:
    def __init__(self):
        self.lock = threading.Lock()
        self._progress = {
            'total_files': 0,
            'processed_files': 0,
            'status': 'Idle',
            'start_time': None,
            'elapsed_time': 0
        }
        self._job_queue = []
        self._current_job = None

    @property
    def progress(self):
        with self.lock:
            if self._progress['start_time']:
                self._progress['elapsed_time'] = int(time.time() - self._progress['start_time'])
            return self._progress.copy()

    @progress.setter
    def progress(self, value):
        with self.lock:
            self._progress.update(value)

    @property
    def job_queue(self):
        with self.lock:
            return self._job_queue.copy()

    def add_job(self, job):
        with self.lock:
            self._job_queue.append(job)

    def get_next_job(self):
        with self.lock:
            if self._job_queue and self._current_job is None:
                return self._job_queue.pop(0)
            return None

    @property
    def current_job(self):
        with self.lock:
            return self._current_job

    @current_job.setter
    def current_job(self, value):
        with self.lock:
            self._current_job = value

# Global job manager instance
job_manager = JobManager()

def process_job(job):
    temp_dir = job.get('temp_dir')
    logger.info(f"Starting job processing for directory: {temp_dir}")

    try:
        job_manager.current_job = job
        job_manager.progress = {
            'status': 'Processing',
            'total_files': len(job['files']),
            'processed_files': 0,
            'start_time': time.time(),
            'elapsed_time': 0
        }

        logger.info(f"Processing {len(job['files'])} files in {temp_dir}")
        result = process_folder(job['temp_dir'])

        job_manager.progress = {
            'status': 'Completed',
            'result_path': os.path.join(job['temp_dir'], 'result.pkl')
        }

        # Save result
        result_path = os.path.join(job['temp_dir'], 'result.pkl')
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)

        logger.info(f"Job completed successfully. Processed {result.get('processed_files', 0)} files")

    except Exception as e:
        logger.error(f"Job failed: {str(e)}", extra={'traceback': traceback.format_exc()})
        job_manager.progress = {
            'status': 'Error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    finally:
        job_manager.current_job = None
        process_next_job()

def process_next_job():
    next_job = job_manager.get_next_job()
    if next_job:
        thread = threading.Thread(target=process_job, args=(next_job,))
        thread.daemon = True  # Ensure thread doesn't prevent shutdown
        thread.start()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return render_template('index.html', error="No files selected")

        temp_dir = tempfile.mkdtemp()
        uploaded_files = []
        errors = []

        for f in files:
            if f.filename == '':
                continue

            # Validate file
            is_valid, error_msg = validate_file_upload(f)
            if not is_valid:
                errors.append(f"File '{f.filename}': {error_msg}")
                continue

            try:
                # Secure the filename
                safe_name = secure_filename(f.filename)
                file_path = os.path.join(temp_dir, safe_name)

                # Ensure parent directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Save file
                f.save(file_path)
                uploaded_files.append(f)

            except Exception as e:
                errors.append(f"Error saving file '{f.filename}': {str(e)}")

        if not uploaded_files:
            # Clean up temp directory if no files were uploaded successfully
            try:
                shutil.rmtree(temp_dir)
            except OSError:
                pass
            return render_template('index.html', error="No valid files could be uploaded. Errors: " + "; ".join(errors))

        if errors:
            # Log warnings for files that failed
            logger.warning(f"Some files failed to upload: {errors}")

        job = {'temp_dir': temp_dir, 'files': uploaded_files}
        job_manager.add_job(job)

        # Save job info in session for tracking
        session['temp_dir'] = temp_dir
        session['uploaded_count'] = len(uploaded_files)
        session.modified = True

        process_next_job()
        return render_template('processing.html')

    return render_template('index.html')

@app.route('/progress')
def get_progress():
    return jsonify(job_manager.progress)

@app.route('/results')
def results():
    temp_dir = session.get('temp_dir')
    if not temp_dir:
        return "No results available. Please process files first."
    result_path = os.path.join(temp_dir, 'result.pkl')
    if not os.path.exists(result_path):
        return "No results available. Please process files first."
    with open(result_path, 'rb') as f:
        result = pickle.load(f)
    matched_rows_json = json.dumps(result.get('matched_rows', []))
    return render_template('results.html', result=result, matched_rows_json=matched_rows_json)

@app.route('/further_extraction')
def further_extraction():
    temp_dir = session.get('temp_dir')
    if not temp_dir:
        return "No results available. Please process files first."
    result_path = os.path.join(temp_dir, 'result.pkl')
    if not os.path.exists(result_path):
        return "No results available. Please process files first."
    with open(result_path, 'rb') as f:
        result = pickle.load(f)
    parsed_rows = []
    for r in result.get('matched_rows', []):
        parsed_row = parse_fir_row(r.get('row', []))
        parsed_rows.append({
            'source_file': r.get('source_file', ''),
            'table_index': r.get('table_index', ''),
            'parsed': parsed_row
        })
    return render_template('further_extraction.html', parsed_rows=parsed_rows)

@app.route('/export_csv')
def export_csv():
    temp_dir = session.get('temp_dir')
    if not temp_dir:
        return "No data to export"
    result_path = os.path.join(temp_dir, 'result.pkl')
    if not os.path.exists(result_path):
        return "No data to export"
    try:
        with open(result_path, 'rb') as f:
            result = pickle.load(f)
    except Exception as e:
        return f"Error loading result data: {str(e)}"
    if not result or not result.get('matched_rows'):
        return "No data to export"
    output = io.StringIO()
    writer = csv.writer(output)
    max_cells = max(len(r['row']) for r in result.get('matched_rows', [])) if result.get('matched_rows') else 0
    headers = ['Source File', 'Table Index'] + [f'Cell {i+1}' for i in range(max_cells)]
    writer.writerow(headers)
    for r in result.get('matched_rows', []):
        writer.writerow([strip_html_tags(r.get('source_file', '')) , strip_html_tags(str(r.get('table_index', '')))] + [strip_html_tags(str(cell)) for cell in r.get('row', [])])
    output.seek(0)
    response = send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='results.csv')
    response.headers["Content-Type"] = "text/csv"
    return response

@app.route('/export_excel')
def export_excel():
    temp_dir = session.get('temp_dir')
    if not temp_dir:
        return "No data to export"
    result_path = os.path.join(temp_dir, 'result.pkl')
    if not os.path.exists(result_path):
        return "No data to export"
    with open(result_path, 'rb') as f:
        result = pickle.load(f)
    if not result or not result.get('matched_rows'):
        return "No data to export"
    wb = openpyxl.Workbook()
    ws = wb.active
    max_cells = max(len(r['row']) for r in result.get('matched_rows', [])) if result.get('matched_rows') else 0
    headers = ['Source File', 'Table Index'] + [f'Cell {i+1}' for i in range(max_cells)]
    ws.append(headers)
    for r in result.get('matched_rows', []):
        ws.append([strip_html_tags(r.get('source_file', '')) , strip_html_tags(str(r.get('table_index', '')))] + [strip_html_tags(str(cell)) for cell in r.get('row', [])])
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='results.xlsx')

@app.route('/export_pdf')
def export_pdf():
    try:
        temp_dir = session.get('temp_dir')
        if not temp_dir:
            return "No data to export"
        result_path = os.path.join(temp_dir, 'result.pkl')
        if not os.path.exists(result_path):
            return "No data to export"
        with open(result_path, 'rb') as f:
            result = pickle.load(f)
        if not result or not result.get('matched_rows'):
            return "No data to export"
        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title = Paragraph("Extraction Results", styles['Title'])
        story.append(title)

        # Summary
        summary = Paragraph(f"Processed {result.get('processed_files', 0)} document(s). Matched {len(result.get('matched_rows', []))} row(s).", styles['Normal'])
        story.append(summary)

        # Table data
        max_cells = max(len(r['row']) for r in result.get('matched_rows', [])) if result.get('matched_rows') else 0
        headers = ['Source File', 'Table Index'] + [f'Cell {i+1}' for i in range(max_cells)]
        data = [headers]
        for r in result.get('matched_rows', []):
            row = [r.get('source_file', ''), str(r.get('table_index', ''))] + [cell or '' for cell in r.get('row', [])]
            data.append(row)

        # Create table
        # Wrap cell content in Paragraphs to enable wrapping and dynamic height
        styles = getSampleStyleSheet()
        wrapped_data = []
        for row in data:
            wrapped_row = []
            for cell in row:
                wrapped_row.append(Paragraph(strip_html_tags(str(cell)), styles['BodyText']))
            wrapped_data.append(wrapped_row)

        table = Table(wrapped_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(table)

        doc.build(story)
        output.seek(0)
        return send_file(output, mimetype='application/pdf', as_attachment=True, download_name='results.pdf')
    except Exception as e:
        import traceback
        error_msg = f"Error generating PDF: {str(e)}"
        logger.error(error_msg, extra={'traceback': traceback.format_exc()})
        return f"Internal Server Error: {str(e)}", 500

@app.route('/export_further_csv')
def export_further_csv():
    temp_dir = session.get('temp_dir')
    if not temp_dir:
        return "No data to export"
    result_path = os.path.join(temp_dir, 'result.pkl')
    if not os.path.exists(result_path):
        return "No data to export"
    with open(result_path, 'rb') as f:
        result = pickle.load(f)
    if not result or not result.get('matched_rows'):
        return "No data to export"
    parsed_rows = []
    for r in result.get('matched_rows', []):
        parsed_row = parse_fir_row(r.get('row', []))
        parsed_rows.append({
            'source_file': r.get('source_file', ''),
            'table_index': r.get('table_index', ''),
            'parsed': parsed_row
        })
    output = io.StringIO()
    writer = csv.writer(output)
    headers = ['Source File', 'Table Index', 'Police Station', 'CR No', 'Section of Law', 'Date of Occurrence', 'Date of Report', 'Place of Occurrence', 'Complainant Name', 'Complainant Address', 'Complainant Sex', 'Complainant Age', 'Victims', 'Accused', 'Property Lost', 'Property Recovered', 'Property Seized', 'Gist']
    writer.writerow(headers)
    for pr in parsed_rows:
        victims = '; '.join([f"{v['name']}, {v['age']}, {v['address']}, {v['sex']}" for v in pr['parsed'].get('victims', [])])
        accused = '; '.join([f"{a['name']}, {a['age']}, {a['address']}, {a['sex']}" for a in pr['parsed'].get('accused', [])])
        prop_lost = '; '.join([f"{p['item']}: {p['quantity']}" for p in pr['parsed'].get('property_lost', [])])
        prop_recovered = '; '.join([f"{p['item']}: {p['quantity']}" for p in pr['parsed'].get('property_recovered', [])])
        prop_seized = '; '.join([f"{p['item']}: {p['quantity']}" for p in pr['parsed'].get('property_seized', [])])
        row = [
            pr['source_file'],
            pr['table_index'],
            pr['parsed'].get('police_station', ''),
            pr['parsed'].get('cr_no', ''),
            pr['parsed'].get('section_of_law', ''),
            pr['parsed'].get('date_of_occurrence', ''),
            pr['parsed'].get('date_of_report', ''),
            pr['parsed'].get('place_of_occurrence', ''),
            pr['parsed']['complainant'].get('name', ''),
            pr['parsed']['complainant'].get('address', ''),
            pr['parsed']['complainant'].get('sex', ''),
            pr['parsed']['complainant'].get('age', ''),
            victims,
            accused,
            prop_lost,
            prop_recovered,
            prop_seized,
            pr['parsed'].get('gist', '')
        ]
        row = [strip_html_tags(str(item)) for item in row]
        writer.writerow(row)
    output.seek(0)
    response = send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='further_results.csv')
    response.headers["Content-Type"] = "text/csv"
    return response

@app.route('/export_further_excel')
def export_further_excel():
    temp_dir = session.get('temp_dir')
    if not temp_dir:
        return "No data to export"
    result_path = os.path.join(temp_dir, 'result.pkl')
    if not os.path.exists(result_path):
        return "No data to export"
    with open(result_path, 'rb') as f:
        result = pickle.load(f)
    if not result or not result.get('matched_rows'):
        return "No data to export"
    parsed_rows = []
    for r in result.get('matched_rows', []):
        parsed_row = parse_fir_row(r.get('row', []))
        parsed_rows.append({
            'source_file': r.get('source_file', ''),
            'table_index': r.get('table_index', ''),
            'parsed': parsed_row
        })
    wb = openpyxl.Workbook()
    ws = wb.active
    headers = ['Source File', 'Table Index', 'Police Station', 'CR No', 'Section of Law', 'Date of Occurrence', 'Date of Report', 'Place of Occurrence', 'Complainant Name', 'Complainant Address', 'Complainant Sex', 'Complainant Age', 'Victims', 'Accused', 'Property Lost', 'Property Recovered', 'Property Seized', 'Gist']
    ws.append(headers)
    for pr in parsed_rows:
        victims = '; '.join([f"{v['name']}, {v['age']}, {v['address']}, {v['sex']}" for v in pr['parsed'].get('victims', [])])
        accused = '; '.join([f"{a['name']}, {a['age']}, {a['address']}, {a['sex']}" for a in pr['parsed'].get('accused', [])])
        prop_lost = '; '.join([f"{p['item']}: {p['quantity']}" for p in pr['parsed'].get('property_lost', [])])
        prop_recovered = '; '.join([f"{p['item']}: {p['quantity']}" for p in pr['parsed'].get('property_recovered', [])])
        prop_seized = '; '.join([f"{p['item']}: {p['quantity']}" for p in pr['parsed'].get('property_seized', [])])
        row = [
            pr['source_file'],
            pr['table_index'],
            pr['parsed'].get('police_station', ''),
            pr['parsed'].get('cr_no', ''),
            pr['parsed'].get('section_of_law', ''),
            pr['parsed'].get('date_of_occurrence', ''),
            pr['parsed'].get('date_of_report', ''),
            pr['parsed'].get('place_of_occurrence', ''),
            pr['parsed']['complainant'].get('name', ''),
            pr['parsed']['complainant'].get('address', ''),
            pr['parsed']['complainant'].get('sex', ''),
            pr['parsed']['complainant'].get('age', ''),
            victims,
            accused,
            prop_lost,
            prop_recovered,
            prop_seized,
            pr['parsed'].get('gist', '')
        ]
        row = [strip_html_tags(str(item)) for item in row]
        ws.append(row)
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='further_results.xlsx')

@app.route('/export_further_pdf')
def export_further_pdf():
    try:
        temp_dir = session.get('temp_dir')
        if not temp_dir:
            return "No data to export"
        result_path = os.path.join(temp_dir, 'result.pkl')
        if not os.path.exists(result_path):
            return "No data to export"
        with open(result_path, 'rb') as f:
            result = pickle.load(f)
        if not result or not result.get('matched_rows'):
            return "No data to export"
        parsed_rows = []
        for r in result.get('matched_rows', []):
            parsed_row = parse_fir_row(r.get('row', []))
            parsed_rows.append({
                'source_file': r.get('source_file', ''),
                'table_index': r.get('table_index', ''),
                'parsed': parsed_row
            })
        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title = Paragraph("Further Extraction Results", styles['Title'])
        story.append(title)

        # Summary
        summary = Paragraph(f"Processed {result.get('processed_files', 0)} document(s). Matched {len(result.get('matched_rows', []))} row(s).", styles['Normal'])
        story.append(summary)

        # Table data
        headers = ['Source File', 'Table Index', 'Police Station', 'CR No', 'Section of Law', 'Date of Occurrence', 'Date of Report', 'Place of Occurrence', 'Complainant Name', 'Complainant Address', 'Complainant Sex', 'Complainant Age', 'Victims', 'Accused', 'Property Lost', 'Property Recovered', 'Property Seized', 'Gist']
        data = [headers]
        for pr in parsed_rows:
            victims = '; '.join([f"{v['name']}, {v['age']}, {v['address']}, {v['sex']}" for v in pr['parsed'].get('victims', [])])
            accused = '; '.join([f"{a['name']}, {a['age']}, {a['address']}, {a['sex']}" for a in pr['parsed'].get('accused', [])])
            prop_lost = '; '.join([f"{p['item']}: {p['quantity']}" for p in pr['parsed'].get('property_lost', [])])
            prop_recovered = '; '.join([f"{p['item']}: {p['quantity']}" for p in pr['parsed'].get('property_recovered', [])])
            prop_seized = '; '.join([f"{p['item']}: {p['quantity']}" for p in pr['parsed'].get('property_seized', [])])
            row = [
                pr['source_file'],
                pr['table_index'],
                pr['parsed'].get('police_station', ''),
                pr['parsed'].get('cr_no', ''),
                pr['parsed'].get('section_of_law', ''),
                pr['parsed'].get('date_of_occurrence', ''),
                pr['parsed'].get('date_of_report', ''),
                pr['parsed'].get('place_of_occurrence', ''),
                pr['parsed']['complainant'].get('name', ''),
                pr['parsed']['complainant'].get('address', ''),
                pr['parsed']['complainant'].get('sex', ''),
                pr['parsed']['complainant'].get('age', ''),
                victims,
                accused,
                prop_lost,
                prop_recovered,
                prop_seized,
                pr['parsed'].get('gist', '')
            ]
            data.append(row)

        # Create table
        # Wrap cell content in Paragraphs to enable wrapping and dynamic height
        wrapped_data = []
        for row in data:
            wrapped_row = []
            for cell in row:
                wrapped_row.append(Paragraph(strip_html_tags(str(cell)), styles['BodyText']))
            wrapped_data.append(wrapped_row)

        table = Table(wrapped_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(table)

        doc.build(story)
        output.seek(0)
        return send_file(output, mimetype='application/pdf', as_attachment=True, download_name='further_results.pdf')
    except Exception as e:
        import traceback
        error_msg = f"Error generating PDF: {str(e)}"
        logger.error(error_msg, extra={'traceback': traceback.format_exc()})
        return f"Internal Server Error: {str(e)}", 500

def main():
    """Main application entry point with WSGI support."""
    import os

    # Check if running as PyInstaller bundle (frozen)
    is_frozen = getattr(sys, 'frozen', False)

    # Check if running in production mode (packaged or env var set)
    production_mode = (os.environ.get('PRODUCTION_MODE', 'false').lower() == 'true' or is_frozen)

    if production_mode:
        # Use Waitress WSGI server for production
        from waitress import serve
        port = int(os.environ.get('PORT', 5000))
        host = os.environ.get('HOST', '127.0.0.1')

        print(f"Starting DSR_Extract in production mode on {host}:{port}")
        print("Using Waitress WSGI server")
        serve(app, host=host, port=port)
    else:
        # Use Flask development server
        port = int(os.environ.get('PORT', 5000))
        host = os.environ.get('HOST', '127.0.0.1')

        print(f"Starting DSR_Extract in development mode on {host}:{port}")
        app.run(debug=True, host=host, port=port)

if __name__ == '__main__':
    main()
