from docx import Document

doc = Document()
table = doc.add_table(rows=2, cols=9)
# Headers (optional, but for clarity)
table.cell(0, 0).text = 'Police Station, CR No, Section'
table.cell(0, 1).text = 'Dates, Times, Place'
table.cell(0, 2).text = 'Complainant'
table.cell(0, 3).text = 'Victims'
table.cell(0, 4).text = 'Property'
table.cell(0, 5).text = 'Accused'
table.cell(0, 6).text = 'Gist'

# Sample row 1
table.cell(1, 0).text = 'Sample Police Station PS 123/2023 U/S 420 IPC'
table.cell(1, 1).text = '15/06/2023 10:30hrs to 16/06/2023 11:00hrs Sample Place East'
table.cell(1, 2).text = 'John Doe (25) S/o Father Address: 123 Street'
table.cell(1, 3).text = '1. Jane Doe (22) D/o Mother Address: 456 Avenue\n2. Bob Smith (30) S/o Parent Address: 789 Road'
table.cell(1, 4).text = 'Property Lost: Mobile Phone 1 no, Cash 5000 Rs\nProperty Recovered: None\nProperty Seized: Knife 1 no'
table.cell(1, 5).text = '1. Villain One (28) S/o Bad Address: Evil Street\n2. Villain Two (35) S/o Worse Address: Dark Alley'
table.cell(1, 6).text = 'Theft case details'

doc.save('tests/test_data/sample.docx')
