from docx import Document

doc = Document()
table = doc.add_table(rows=3, cols=4)
table.cell(0, 0).text = 'IPC 1/2023'
table.cell(0, 1).text = 'IPC 1/2023'  # adjacent same
table.cell(0, 2).text = 'BNS'
table.cell(0, 3).text = 'Act'

table.cell(1, 0).text = 'IPC 2/2023'
table.cell(1, 1).text = 'East'
table.cell(1, 2).text = '10 hrs'
table.cell(1, 3).text = 'Direction'

table.cell(2, 0).text = 'IPC 3/2023'
table.cell(2, 1).text = 'IPC 3/2023'  # adjacent same
table.cell(2, 2).text = 'IPC 3/2023'  # another same
table.cell(2, 3).text = 'U/S'

doc.save('tests/test_data/sample.docx')
