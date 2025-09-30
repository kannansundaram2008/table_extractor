# TODO List for DSR Extract Enhancements

## 1. Modify Primary Patterns ✅
- Update PRIMARY_REGEX in utils/pattern_matcher.py to include space between / (e.g., r'\d{1,4}\s*/\s*\d{1,4}')

## 2. Enhance File Upload ✅
- Modify templates/index.html to support drag-and-drop file upload
- Update app.py to handle single/multiple file uploads without requiring webkitdirectory
- Allow both folder and individual file selection

## 3. Advanced Table Processing ✅
- Improve handling of merged cells in document_processor.py
- Add detection for table headers or more complex table structures
- Enhance discard_adjacent_duplicates and row matching logic

## 4. Real-Time Progress with Time Elapsed ✅
- Set total_files in progress dict in app.py
- Add start_time to progress tracking
- Update processing.html to display elapsed time
- Ensure progress polling includes time information

## 5. Result Visualization ✅
- Add charts to templates/results.html using Chart.js
- Visualize matched rows per file or processing statistics
- Include summary charts in the modal

## 6. Queue Processing ✅
- Implement a job queue in app.py for multiple processing requests
- Process jobs sequentially in background
- Update UI to show queue status and allow multiple uploads

## 7. Testing and Validation ✅
- Run existing tests to ensure no regressions
- Test new features manually
- Update tests if necessary
