import os
import glob
import pypandoc
import concurrent.futures
from docx import Document
from utils.pattern_matcher import match_row

def discard_adjacent_duplicates(row):
    """
    Discard the first value if adjacent cells have the same value.
    """
    if not row:
        return row
    result = [row[0]]
    for cell in row[1:]:
        if cell != result[-1]:
            result.append(cell)
    return result

def convert_doc_to_docx(doc_path):
    """Convert .doc file to .docx with proper error handling."""
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Document file not found: {doc_path}")

    output_path = doc_path.replace('.doc', '.docx')
    try:
        pypandoc.convert_file(doc_path, 'docx', outputfile=output_path)
        return output_path
    except Exception as e:
        # Clean up partial output file if it exists
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass  # Ignore cleanup errors
        raise RuntimeError(f"Failed to convert .doc to .docx: {str(e)}")

def extract_tables_from_docx(docx_path):
    doc = Document(docx_path)
    tables = []
    for table in doc.tables:
        table_data = []
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            table_data.append(row_data)
        tables.append(table_data)
    return tables

def process_single_file(file_path, folder_path):
    matched_rows = []
    processed_files = 0
    errors = []
    file = os.path.relpath(file_path, folder_path)
    try:
        if file.lower().endswith('.doc'):
            docx_path = convert_doc_to_docx(file_path)
        else:
            docx_path = file_path
        tables = extract_tables_from_docx(docx_path)
        for table_index, table in enumerate(tables):
            row_index = 0
            while row_index < len(table):
                row = discard_adjacent_duplicates(table[row_index])
                if match_row(row):
                    matched_row = {'row': row[:], 'source_file': file, 'table_index': table_index}
                    # Check next row for single merged cell to append
                    if row_index + 1 < len(table):
                        next_row = discard_adjacent_duplicates(table[row_index + 1])
                        if len(next_row) == 1:
                            matched_row['row'].append(next_row[0])
                            row_index += 1  # Skip next row as it is appended
                    matched_rows.append(matched_row)
                row_index += 1
        processed_files += 1
        if file.lower().endswith('.doc') and os.path.exists(docx_path):
            try:
                os.remove(docx_path)  # clean up temporary docx
            except OSError as cleanup_error:
                errors.append(f"Warning: Failed to cleanup temporary file {docx_path}: {str(cleanup_error)}")
    except Exception as e:
        errors.append(f"Error processing {file}: {str(e)}")
        # Cleanup temporary file even on error
        if file.lower().endswith('.doc') and os.path.exists(docx_path):
            try:
                os.remove(docx_path)
            except OSError:
                pass  # Ignore cleanup errors
    return {'matched_rows': matched_rows, 'processed_files': processed_files, 'errors': errors}

def process_folder(folder_path):
    matched_rows = []
    processed_files = 0
    errors = []
    files = glob.glob(os.path.join(folder_path, '**', '*.doc'), recursive=True) + glob.glob(os.path.join(folder_path, '**', '*.docx'), recursive=True)
    files = [f for f in files if not os.path.basename(f).startswith('$')]
    # Magical parallel processing for faster extraction
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_file, file_path, folder_path) for file_path in files]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            matched_rows.extend(result['matched_rows'])
            processed_files += result['processed_files']
            errors.extend(result['errors'])
    return {'matched_rows': matched_rows, 'processed_files': processed_files, 'errors': errors}
