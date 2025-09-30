from flask import Flask, request, render_template, session, send_file, jsonify
import os
import io
import csv
import openpyxl
import threading
import time
import tempfile
import shutil
from utils.document_processor import process_folder
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

app = Flask(__name__)
app.secret_key = 'dsr_extract_secret_key'

progress = {
    'total_files': 0,
    'processed_files': 0,
    'status': 'Idle'
}

def background_process(folder_path):
    global progress
    progress['status'] = 'Processing'
    result = process_folder(folder_path)
    progress['status'] = 'Completed'
    # Save result to a file in temp folder to persist across requests
    result_path = os.path.join(folder_path, 'result.pkl')
    import pickle
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)
    progress['result_path'] = result_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('files')
        if files:
            # Clean up old temp_dir if exists
            old_temp_dir = session.get('temp_dir')
            if old_temp_dir and os.path.exists(old_temp_dir):
                shutil.rmtree(old_temp_dir, ignore_errors=True)
            temp_dir = tempfile.mkdtemp()
            for f in files:
                if f.filename.lower().endswith(('.doc', '.docx')) and not f.filename.startswith('$'):
                    safe_name = f.filename.replace(' ', '_')
                    file_path = os.path.join(temp_dir, safe_name)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    f.save(file_path)
            # Start background processing thread
            thread = threading.Thread(target=background_process, args=(temp_dir,))
            thread.start()
            session['temp_dir'] = temp_dir
            return render_template('processing.html')
    return render_template('index.html')

@app.route('/progress')
def get_progress():
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
    return render_template('results.html', result=result)

@app.route('/export_csv')
def export_csv():
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
    output = io.StringIO()
    writer = csv.writer(output)
    max_cells = max(len(r['row']) for r in result.get('matched_rows', [])) if result.get('matched_rows') else 0
    headers = ['Source File', 'Table Index'] + [f'Cell {i+1}' for i in range(max_cells)]
    writer.writerow(headers)
    for r in result.get('matched_rows', []):
        writer.writerow([r.get('source_file', '') , r.get('table_index', '')] + r.get('row', []))
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='results.csv')

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
    p = canvas.Canvas(output, pagesize=letter)
    width, height = letter
    y = height - 40
    p.setFont("Helvetica-Bold", 14)
    p.drawString(40, y, "Extraction Results")
    y -= 30
    p.setFont("Helvetica", 10)
    p.drawString(40, y, f"Processed {result.get('processed_files', 0)} document(s). Matched {len(result.get('matched_rows', []))} row(s).")
    y -= 20
    max_cells = max(len(r['row']) for r in result.get('matched_rows', [])) if result.get('matched_rows') else 0
    headers = ['Source File', 'Table Index'] + [f'Cell {i+1}' for i in range(max_cells)]
    p.setFont("Helvetica-Bold", 8)
    x = 40
    for header in headers:
        p.drawString(x, y, header)
        x += 100
    y -= 15
    p.setFont("Helvetica", 8)
    for r in result.get('matched_rows', []):
        x = 40
        if y < 40:
            p.showPage()
            y = height - 40
        p.drawString(x, y, r.get('source_file', '') or '')
        x += 100
        p.drawString(x, y, str(r.get('table_index', '')) or '')
        x += 100
        for cell in r.get('row', []):
            p.drawString(x, y, (cell or '')[:15])
            x += 100
        y -= 15
    p.save()
    output.seek(0)
    return send_file(output, mimetype='application/pdf', as_attachment=True, download_name='results.pdf')

if __name__ == '__main__':
    app.run(debug=True)
