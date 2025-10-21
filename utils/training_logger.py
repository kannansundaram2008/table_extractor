"""
Training Logger for ML Model Training Data Collection

This module provides comprehensive logging capabilities for FIR document
processing, including extraction results, performance metrics, and
integration with the training data collection system.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import threading
from dataclasses import dataclass, asdict
import queue
import traceback

# Import data collector
from utils.data_collector import get_data_collector, TrainingExample

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ExtractionLogEntry:
    """Represents a single extraction log entry."""
    timestamp: str
    session_id: str
    source_file: str
    extraction_type: str  # 'fir_row', 'full_document', etc.
    input_text: str
    extracted_data: Dict[str, Any]
    confidence_scores: Dict[str, float]
    processing_time: float
    success: bool
    error_message: str = ""
    model_version: str = "1.0"
    processing_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.processing_metadata is None:
            self.processing_metadata = {}

@dataclass
class PerformanceMetrics:
    """Performance metrics for extraction operations."""
    total_extractions: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    average_processing_time: float = 0.0
    total_processing_time: float = 0.0
    extraction_types: Dict[str, int] = None
    error_types: Dict[str, int] = None

    def __post_init__(self):
        if self.extraction_types is None:
            self.extraction_types = {}
        if self.error_types is None:
            self.error_types = {}

class TrainingLogger:
    """
    Comprehensive logging system for training data collection.
    """

    def __init__(
        self,
        log_directory: str = "models/training_data/logs",
        enable_data_collection: bool = True,
        buffer_size: int = 1000,
        auto_flush_interval: int = 300  # 5 minutes
    ):
        """
        Initialize the training logger.

        Args:
            log_directory: Directory for storing log files
            enable_data_collection: Whether to collect data for training
            buffer_size: Number of log entries to buffer before writing
            auto_flush_interval: Auto-flush interval in seconds
        """
        self.log_directory = Path(log_directory)
        self.enable_data_collection = enable_data_collection
        self.buffer_size = buffer_size
        self.auto_flush_interval = auto_flush_interval

        # Create log directory
        self.log_directory.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.log_buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.performance_metrics = PerformanceMetrics()
        self.session_id = self._generate_session_id()

        # Data collector integration
        self.data_collector = get_data_collector() if enable_data_collection else None

        # File paths
        self.extraction_log_file = self.log_directory / "extraction_logs.jsonl"
        self.performance_log_file = self.log_directory / "performance_metrics.json"
        self.error_log_file = self.log_directory / "errors.jsonl"

        # Start background threads
        self._start_background_threads()

        logger.info(f"TrainingLogger initialized with session {self.session_id}")

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"session_{int(time.time())}_{id(self)}"

    def _start_background_threads(self) -> None:
        """Start background threads for log processing."""
        # Log writer thread
        self.log_writer_thread = threading.Thread(
            target=self._log_writer_worker,
            daemon=True
        )
        self.log_writer_thread.start()

        # Auto-flush timer thread
        self.flush_timer_thread = threading.Thread(
            target=self._flush_timer_worker,
            daemon=True
        )
        self.flush_timer_thread.start()

    def log_extraction(
        self,
        source_file: str,
        extraction_type: str,
        input_text: str,
        extracted_data: Dict[str, Any],
        confidence_scores: Dict[str, float],
        processing_time: float,
        success: bool = True,
        error_message: str = "",
        model_version: str = "1.0",
        processing_metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Log an extraction operation.

        Args:
            source_file: Path to the source document
            extraction_type: Type of extraction performed
            input_text: Input text that was processed
            extracted_data: Extracted data results
            confidence_scores: Confidence scores for extracted fields
            processing_time: Time taken for processing
            success: Whether extraction was successful
            error_message: Error message if extraction failed
            model_version: Version of the model used
            processing_metadata: Additional processing information

        Returns:
            Training example ID if data collection is enabled, None otherwise
        """
        # Create log entry
        log_entry = ExtractionLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=self.session_id,
            source_file=source_file,
            extraction_type=extraction_type,
            input_text=input_text,
            extracted_data=extracted_data,
            confidence_scores=confidence_scores,
            processing_time=processing_time,
            success=success,
            error_message=error_message,
            model_version=model_version,
            processing_metadata=processing_metadata or {}
        )

        # Add to buffer
        try:
            self.log_buffer.put(log_entry, block=False)
        except queue.Full:
            logger.warning("Log buffer full, dropping oldest entry")
            try:
                self.log_buffer.get_nowait()  # Remove oldest
                self.log_buffer.put(log_entry, block=False)
            except queue.Empty:
                pass

        # Update performance metrics
        self._update_performance_metrics(log_entry)

        # Collect training data if enabled
        training_example_id = None
        if self.enable_data_collection and self.data_collector and success:
            try:
                # For training data, we need expected output (ground truth)
                # This would typically come from manual annotation or known correct data
                training_example_id = self._collect_training_data(log_entry)
            except Exception as e:
                logger.warning(f"Failed to collect training data: {e}")

        return training_example_id

    def _update_performance_metrics(self, log_entry: ExtractionLogEntry) -> None:
        """Update performance metrics based on log entry."""
        self.performance_metrics.total_extractions += 1
        self.performance_metrics.total_processing_time += log_entry.processing_time

        if log_entry.success:
            self.performance_metrics.successful_extractions += 1
        else:
            self.performance_metrics.failed_extractions += 1

            # Track error types
            if log_entry.error_message:
                error_type = self._categorize_error(log_entry.error_message)
                self.performance_metrics.error_types[error_type] = \
                    self.performance_metrics.error_types.get(error_type, 0) + 1

        # Track extraction types
        self.performance_metrics.extraction_types[log_entry.extraction_type] = \
            self.performance_metrics.extraction_types.get(log_entry.extraction_type, 0) + 1

        # Update average processing time
        if self.performance_metrics.total_extractions > 0:
            self.performance_metrics.average_processing_time = \
                self.performance_metrics.total_processing_time / self.performance_metrics.total_extractions

    def _categorize_error(self, error_message: str) -> str:
        """Categorize error messages for better tracking."""
        error_lower = error_message.lower()

        if "timeout" in error_lower:
            return "timeout"
        elif "parsing" in error_lower:
            return "parsing_error"
        elif "encoding" in error_lower or "decode" in error_lower:
            return "encoding_error"
        elif "memory" in error_lower:
            return "memory_error"
        elif "file" in error_lower or "path" in error_lower:
            return "file_error"
        else:
            return "other_error"

    def _collect_training_data(self, log_entry: ExtractionLogEntry) -> Optional[str]:
        """
        Collect training data from log entry.

        Note: This requires expected output (ground truth) to be available.
        In practice, this would be provided separately or through annotation.
        """
        # For now, we'll create a placeholder for expected output
        # In a real implementation, this would come from:
        # 1. Manual annotation
        # 2. Known correct data
        # 3. Synthetic data generation

        # Skip if no expected output available
        # This is a placeholder - in practice, expected_output would be provided
        expected_output = {}  # Would need to be populated with ground truth

        if not expected_output:
            return None

        return self.data_collector.collect_extraction_result(
            source_file=log_entry.source_file,
            extraction_input=log_entry.input_text,
            expected_output=expected_output,
            actual_output=log_entry.extracted_data,
            confidence_scores=log_entry.confidence_scores,
            processing_metadata={
                **log_entry.processing_metadata,
                "extraction_type": log_entry.extraction_type,
                "session_id": log_entry.session_id
            },
            model_version=log_entry.model_version
        )

    def _log_writer_worker(self) -> None:
        """Background worker for writing logs to files."""
        while True:
            try:
                # Get log entry from buffer
                log_entry = self.log_buffer.get(timeout=1.0)

                # Write to extraction log file
                self._write_extraction_log(log_entry)

                # Write errors to separate file
                if not log_entry.success:
                    self._write_error_log(log_entry)

                self.log_buffer.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in log writer worker: {e}")

    def _write_extraction_log(self, log_entry: ExtractionLogEntry) -> None:
        """Write extraction log entry to file."""
        try:
            with open(self.extraction_log_file, 'a', encoding='utf-8') as f:
                json.dump(asdict(log_entry), f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            logger.error(f"Error writing extraction log: {e}")

    def _write_error_log(self, log_entry: ExtractionLogEntry) -> None:
        """Write error log entry to separate file."""
        try:
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                error_entry = {
                    "timestamp": log_entry.timestamp,
                    "session_id": log_entry.session_id,
                    "source_file": log_entry.source_file,
                    "extraction_type": log_entry.extraction_type,
                    "error_message": log_entry.error_message,
                    "processing_time": log_entry.processing_time,
                    "input_text_length": len(log_entry.input_text)
                }
                json.dump(error_entry, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            logger.error(f"Error writing error log: {e}")

    def _flush_timer_worker(self) -> None:
        """Background worker for periodic flushing."""
        last_flush = time.time()

        while True:
            try:
                time.sleep(60)  # Check every minute

                if time.time() - last_flush >= self.auto_flush_interval:
                    self.flush()
                    last_flush = time.time()

            except Exception as e:
                logger.error(f"Error in flush timer worker: {e}")

    def flush(self) -> None:
        """Flush all pending log entries and save performance metrics."""
        try:
            # Process remaining items in buffer
            while not self.log_buffer.empty():
                try:
                    log_entry = self.log_buffer.get_nowait()
                    self._write_extraction_log(log_entry)
                    if not log_entry.success:
                        self._write_error_log(log_entry)
                    self.log_buffer.task_done()
                except queue.Empty:
                    break

            # Save performance metrics
            self._save_performance_metrics()

            logger.info("Training logger flushed successfully")

        except Exception as e:
            logger.error(f"Error during flush: {e}")

    def _save_performance_metrics(self) -> None:
        """Save current performance metrics to file."""
        try:
            with open(self.performance_log_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "session_id": self.session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metrics": asdict(self.performance_metrics)
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        return {
            "session_id": self.session_id,
            "total_extractions": self.performance_metrics.total_extractions,
            "successful_extractions": self.performance_metrics.successful_extractions,
            "failed_extractions": self.performance_metrics.failed_extractions,
            "success_rate": (
                self.performance_metrics.successful_extractions /
                max(1, self.performance_metrics.total_extractions)
            ),
            "average_processing_time": self.performance_metrics.average_processing_time,
            "extraction_types": self.performance_metrics.extraction_types,
            "error_types": self.performance_metrics.error_types
        }

    def log_fir_row_extraction(
        self,
        source_file: str,
        row_text: str,
        extracted_data: Dict[str, Any],
        confidence_scores: Dict[str, float],
        processing_time: float,
        success: bool = True,
        error_message: str = "",
        model_version: str = "1.0",
        processing_metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Convenience method for logging FIR row extractions.

        Returns:
            Training example ID if data collection is enabled
        """
        return self.log_extraction(
            source_file=source_file,
            extraction_type="fir_row",
            input_text=row_text,
            extracted_data=extracted_data,
            confidence_scores=confidence_scores,
            processing_time=processing_time,
            success=success,
            error_message=error_message,
            model_version=model_version,
            processing_metadata=processing_metadata
        )

    def log_full_document_extraction(
        self,
        source_file: str,
        document_text: str,
        extracted_data: Dict[str, Any],
        confidence_scores: Dict[str, float],
        processing_time: float,
        success: bool = True,
        error_message: str = "",
        model_version: str = "1.0",
        processing_metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Convenience method for logging full document extractions.

        Returns:
            Training example ID if data collection is enabled
        """
        return self.log_extraction(
            source_file=source_file,
            extraction_type="full_document",
            input_text=document_text,
            extracted_data=extracted_data,
            confidence_scores=confidence_scores,
            processing_time=processing_time,
            success=success,
            error_message=error_message,
            model_version=model_version,
            processing_metadata=processing_metadata
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure flush on exit."""
        self.flush()

# Global logger instance
_training_logger = None

def get_training_logger(
    log_directory: str = "models/training_data/logs",
    enable_data_collection: bool = True
) -> TrainingLogger:
    """Get or create global training logger instance."""
    global _training_logger
    if _training_logger is None:
        _training_logger = TrainingLogger(log_directory, enable_data_collection)
    return _training_logger

def log_fir_extraction(
    source_file: str,
    extraction_type: str,
    input_text: str,
    extracted_data: Dict[str, Any],
    confidence_scores: Dict[str, float],
    processing_time: float,
    success: bool = True,
    error_message: str = "",
    model_version: str = "1.0",
    processing_metadata: Dict[str, Any] = None
) -> Optional[str]:
    """
    Convenience function for logging FIR extractions.

    Returns:
        Training example ID if data collection is enabled
    """
    logger_instance = get_training_logger()
    return logger_instance.log_extraction(
        source_file=source_file,
        extraction_type=extraction_type,
        input_text=input_text,
        extracted_data=extracted_data,
        confidence_scores=confidence_scores,
        processing_time=processing_time,
        success=success,
        error_message=error_message,
        model_version=model_version,
        processing_metadata=processing_metadata
    )

# Decorator for automatic logging
def log_extraction(extraction_type: str, model_version: str = "1.0"):
    """
    Decorator for automatic extraction logging.

    Usage:
        @log_extraction("fir_row", "1.0")
        def extract_fir_row(row_text):
            # extraction logic here
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                # Execute the function
                result = func(*args, **kwargs)

                # Extract relevant information
                extracted_data = result if isinstance(result, dict) else {}
                confidence_scores = extracted_data.get('confidence_scores', {})

                processing_time = time.time() - start_time

                # Get source file from kwargs or use default
                source_file = kwargs.get('source_file', 'unknown')

                # Log successful extraction
                logger_instance = get_training_logger()
                logger_instance.log_extraction(
                    source_file=source_file,
                    extraction_type=extraction_type,
                    input_text=str(args[0] if args else ""),
                    extracted_data=extracted_data,
                    confidence_scores=confidence_scores,
                    processing_time=processing_time,
                    success=True,
                    model_version=model_version,
                    processing_metadata={"function": func.__name__}
                )

                return result

            except Exception as e:
                processing_time = time.time() - start_time

                # Log failed extraction
                logger_instance = get_training_logger()
                logger_instance.log_extraction(
                    source_file=kwargs.get('source_file', 'unknown'),
                    extraction_type=extraction_type,
                    input_text=str(args[0] if args else ""),
                    extracted_data={},
                    confidence_scores={},
                    processing_time=processing_time,
                    success=False,
                    error_message=str(e),
                    model_version=model_version,
                    processing_metadata={"function": func.__name__}
                )

                raise

        return wrapper
    return decorator