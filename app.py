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
from utils.document_processor import process_folder
from utils.fir_parser import parse_fir_row
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

app = Flask(__name__)
app.secret_key = 'dsr_extract_secret_key'

progress = {
    'total_files': 0,
    'processed_files': 0,
    'status': 'Idle',
    'start_time': None,
    'elapsed_time': 0
}

job_queue = []
current_job = None
lock = threading.Lock()

def process_job(job):
    global progress, current_job
    import traceback
    try:
        with lock:
            current_job = job
            progress['status'] = 'Processing'
            progress['total_files'] = len(job['files'])
            progress['processed_files'] = 0
            progress['start_time'] = time.time()
            progress['elapsed_time'] = 0
        result = process_folder(job['temp_dir'])
        with lock:
            progress['status'] = 'Completed'
            result_path = os.path.join(job['temp_dir'], 'result.pkl')
            import pickle
            with open(result_path, 'wb') as f:
                pickle.dump(result, f)
            progress['result_path'] = result_path
    except Exception as e:
        with lock:
            progress['status'] = 'Error'
            progress['error'] = str(e)
            progress['traceback'] = traceback.format_exc()
    finally:
        with lock:
            current_job = None
        process_next_job()

def process_next_job():
    global job_queue
    with lock:
        if job_queue and current_job is None:
            next_job = job_queue.pop(0)
            thread = threading.Thread(target=process_job, args=(next_job,))
            thread.start()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('files')
        if files:
            temp_dir = tempfile.mkdtemp()
            for f in files:
                if f.filename.lower().endswith(('.doc', '.docx')) and not f.filename.startswith('$'):
                    safe_name = f.filename.replace(' ', '_')
                    file_path = os.path.join(temp_dir, safe_name)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    f.save(file_path)
            job = {'temp_dir': temp_dir, 'files': files}
            with lock:
                job_queue.append(job)
            session['temp_dir'] = temp_dir
            process_next_job()
            return render_template('processing.html')
    return render_template('index.html')

@app.route('/progress')
def get_progress():
    with lock:
        if progress['start_time']:
            progress['elapsed_time'] = int(time.time() - progress['start_time'])
        return jsonify(progress)

@app.route('/results')
def results():
    temp_dir = session.get('temp_dir')
    if not temp_dir:
        return "No results available. Please process files first."
    result_path = os.path.join(temp_dir, 'result.pkl')
    if not os.path.exists(result_path):
        return "No results available. Please process files first."
    import pickle
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
    import pickle
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
    import pickle
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
        writer.writerow([r.get('source_file', '') , r.get('table_index', '')] + r.get('row', []))
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
    import pickle
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
        ws.append([r.get('source_file', '') , r.get('table_index', '')] + r.get('row', []))
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
        import pickle
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
                wrapped_row.append(Paragraph(str(cell), styles['BodyText']))
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
        error_msg = f"Error generating PDF: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return f"Internal Server Error: {str(e)}", 500

@app.route('/export_further_csv')
def export_further_csv():
    temp_dir = session.get('temp_dir')
    if not temp_dir:
        return "No data to export"
    result_path = os.path.join(temp_dir, 'result.pkl')
    if not os.path.exists(result_path):
        return "No data to export"
    import pickle
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
    import pickle
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
        import pickle
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
                wrapped_row.append(Paragraph(str(cell), styles['BodyText']))
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
        error_msg = f"Error generating PDF: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return f"Internal Server Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
