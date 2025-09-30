✅ Core Functional Requirements
- 📁 Folder Selection via UI
- Allow the user to select a root folder from the Flask interface.
- 🔍 Recursive File Processing
- Traverse all subdirectories within the selected folder.
- Process every .doc and .docx file except those whose filenames start with $ (to exclude temporary or auto-saved files).
- 📄 Table Extraction Logic
For each document:
- Extract all tables.
- Identify rows where:
- Primary Cell Pattern:
- Contains a string matching \d{1,4}/\d{1,4}
- AND includes one or more of the following keywords:
- "IPC", "BNS", "Act"
- Any variation of "u/s" (case-insensitive, with or without spaces)
- Adjacent Cell Pattern:
- Contains either:
- A time reference like "hrs" (case-insensitive), or
- A date in any format
- AND includes a direction keyword such as "east", "west", "north", "south", "NE", "SE", etc.
- 🧩 Merged Row Handling
- If the next row consists of a single merged cell (i.e., one cell spanning multiple columns), treat it as a continuation of the previous row.
- Append its content as the last cell of the previously matched row.
- Ensure this logic works across different table structures and merged cell formats (e.g., gridSpan, colSpan, or merged XML tags in python-docx).
- 📊 Result Aggregation
- Combine all matched rows from all documents into a unified result table.
- Display the final results on a dedicated results page in the Flask UI.

💡 Optional Enhancements
- 📤 Export Options: Allow export of results to CSV or Excel.
- 📈 Summary Stats: Show number of files processed, number of matches found.
- 🚫 Error Handling: Log and skip unreadable or malformed documents.
- 🔗 Source Linking: Include filename or path reference for each matched row.

Would you like me to help scaffold the folder traversal and file filtering logic next? Or dive into the merged-cell detection strategy using python-docx?
