"""
Annotation Interface Framework for Training Data Collection

This module provides a comprehensive framework for manual annotation and
correction of FIR document extraction results. It supports various annotation
workflows and integrates with the training data collection system.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import webbrowser
import http.server
import socketserver
import threading
from urllib.parse import urlparse, parse_qs
import html

# Import data collector
from utils.data_collector import get_data_collector, TrainingExample

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AnnotationTask:
    """Represents an annotation task for manual review."""
    id: str
    example_id: str
    annotator_id: str
    status: str  # 'pending', 'in_progress', 'completed', 'skipped'
    priority: int  # 1-10, higher number = higher priority
    created_at: str
    assigned_at: str = ""
    completed_at: str = ""
    estimated_time: int = 0  # estimated minutes
    actual_time: int = 0  # actual minutes spent
    notes: str = ""

@dataclass
class AnnotationCorrection:
    """Represents a correction made during annotation."""
    field_name: str
    original_value: str
    corrected_value: str
    correction_type: str  # 'correction', 'addition', 'deletion'
    confidence: float
    reasoning: str = ""

@dataclass
class AnnotationSession:
    """Represents an annotation session."""
    id: str
    annotator_id: str
    start_time: str
    end_time: str = ""
    tasks_completed: int = 0
    total_tasks: int = 0
    accuracy_score: float = 0.0
    feedback: str = ""

class AnnotationInterface:
    """
    Framework for manual annotation and correction of extraction results.
    """

    def __init__(
        self,
        storage_path: str = "models/training_data/annotations",
        auto_create_tasks: bool = True,
        quality_threshold: float = 0.8
    ):
        """
        Initialize the annotation interface.

        Args:
            storage_path: Directory for storing annotation data
            auto_create_tasks: Whether to automatically create tasks for low-quality examples
            quality_threshold: Quality threshold below which examples need annotation
        """
        self.storage_path = Path(storage_path)
        self.auto_create_tasks = auto_create_tasks
        self.quality_threshold = quality_threshold

        # Create storage directories
        self._create_storage_structure()

        # Initialize components
        self.tasks: Dict[str, AnnotationTask] = {}
        self.sessions: Dict[str, AnnotationSession] = {}
        self.corrections: Dict[str, List[AnnotationCorrection]] = {}

        # Data collector integration
        self.data_collector = get_data_collector()

        # Load existing data
        self._load_existing_data()

        # Web interface components
        self.web_server = None
        self.server_thread = None

        logger.info(f"AnnotationInterface initialized with storage path: {storage_path}")

    def _create_storage_structure(self) -> None:
        """Create necessary directory structure."""
        directories = [
            self.storage_path,
            self.storage_path / "tasks",
            self.storage_path / "corrections",
            self.storage_path / "sessions",
            self.storage_path / "web_interface"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _load_existing_data(self) -> None:
        """Load existing annotation data."""
        # Load tasks
        tasks_file = self.storage_path / "tasks" / "annotation_tasks.json"
        if tasks_file.exists():
            try:
                with open(tasks_file, 'r', encoding='utf-8') as f:
                    tasks_data = json.load(f)
                    for task_data in tasks_data:
                        task = AnnotationTask(**task_data)
                        self.tasks[task.id] = task
            except Exception as e:
                logger.error(f"Error loading annotation tasks: {e}")

        # Load corrections
        corrections_file = self.storage_path / "corrections" / "corrections.json"
        if corrections_file.exists():
            try:
                with open(corrections_file, 'r', encoding='utf-8') as f:
                    self.corrections = json.load(f)
            except Exception as e:
                logger.error(f"Error loading corrections: {e}")

    def create_annotation_task(
        self,
        example_id: str,
        annotator_id: str,
        priority: int = 5,
        estimated_time: int = 10
    ) -> str:
        """
        Create an annotation task for manual review.

        Args:
            example_id: ID of the training example to annotate
            annotator_id: ID of the annotator
            priority: Priority level (1-10)
            estimated_time: Estimated time in minutes

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())

        task = AnnotationTask(
            id=task_id,
            example_id=example_id,
            annotator_id=annotator_id,
            status="pending",
            priority=priority,
            created_at=datetime.now(timezone.utc).isoformat(),
            estimated_time=estimated_time
        )

        self.tasks[task_id] = task
        self._save_tasks()

        logger.info(f"Created annotation task {task_id} for example {example_id}")
        return task_id

    def get_examples_needing_annotation(self, limit: int = 100) -> List[TrainingExample]:
        """Get training examples that need manual annotation."""
        low_quality_examples = self.data_collector.get_examples_needing_correction(
            threshold=self.quality_threshold
        )

        # Sort by quality score (lowest first)
        low_quality_examples.sort(key=lambda x: x.data_quality_score)

        return low_quality_examples[:limit]

    def auto_create_annotation_tasks(
        self,
        annotator_id: str,
        limit: int = 50,
        priority_boost: int = 5
    ) -> List[str]:
        """
        Automatically create annotation tasks for low-quality examples.

        Args:
            annotator_id: ID of the annotator to assign tasks to
            limit: Maximum number of tasks to create
            priority_boost: Base priority for tasks

        Returns:
            List of created task IDs
        """
        examples = self.get_examples_needing_annotation(limit)
        task_ids = []

        for example in examples:
            # Calculate priority based on quality score (lower quality = higher priority)
            quality_priority = int((1 - example.data_quality_score) * 10)
            priority = min(10, priority_boost + quality_priority)

            task_id = self.create_annotation_task(
                example_id=example.id,
                annotator_id=annotator_id,
                priority=priority,
                estimated_time=15  # Assume 15 minutes per annotation
            )

            task_ids.append(task_id)

        logger.info(f"Auto-created {len(task_ids)} annotation tasks for annotator {annotator_id}")
        return task_ids

    def submit_annotation(
        self,
        task_id: str,
        annotator_id: str,
        corrections: List[AnnotationCorrection],
        notes: str = ""
    ) -> bool:
        """
        Submit annotation corrections for a task.

        Args:
            task_id: ID of the annotation task
            annotator_id: ID of the annotator
            corrections: List of corrections made
            notes: Additional notes from annotator

        Returns:
            Success status
        """
        if task_id not in self.tasks:
            logger.error(f"Annotation task {task_id} not found")
            return False

        task = self.tasks[task_id]

        if task.annotator_id != annotator_id:
            logger.error(f"Task {task_id} not assigned to annotator {annotator_id}")
            return False

        if task.status != "in_progress":
            logger.error(f"Task {task_id} is not in progress")
            return False

        try:
            # Update task status
            task.status = "completed"
            task.completed_at = datetime.now(timezone.utc).isoformat()
            task.notes = notes

            # Store corrections
            self.corrections[task.example_id] = corrections

            # Update training example with corrections
            self._apply_corrections_to_example(task.example_id, corrections)

            # Save data
            self._save_tasks()
            self._save_corrections()

            logger.info(f"Annotation submitted for task {task_id}")
            return True

        except Exception as e:
            logger.error(f"Error submitting annotation for task {task_id}: {e}")
            return False

    def _apply_corrections_to_example(
        self,
        example_id: str,
        corrections: List[AnnotationCorrection]
    ) -> None:
        """Apply corrections to the training example."""
        # Get the example from data collector
        example = None
        for ex in self.data_collector.examples:
            if ex.id == example_id:
                example = ex
                break

        if not example:
            logger.error(f"Training example {example_id} not found")
            return

        # Apply corrections to expected output
        corrected_output = example.expected_output.copy()

        for correction in corrections:
            if correction.correction_type == "correction":
                corrected_output[correction.field_name] = correction.corrected_value
            elif correction.correction_type == "addition":
                corrected_output[correction.field_name] = correction.corrected_value
            elif correction.correction_type == "deletion":
                corrected_output.pop(correction.field_name, None)

        # Update the example
        example.expected_output = corrected_output
        example.is_corrected = True
        example.correction_notes = f"Annotated on {datetime.now(timezone.utc).isoformat()}"

        # Recalculate quality score
        example.data_quality_score = self.data_collector._calculate_data_quality(
            example.expected_output,
            example.actual_output,
            example.confidence_scores
        )

        # Save updated example
        self.data_collector.save_example(example)

    def start_annotation_session(self, annotator_id: str, total_tasks: int) -> str:
        """Start a new annotation session."""
        session_id = str(uuid.uuid4())

        session = AnnotationSession(
            id=session_id,
            annotator_id=annotator_id,
            start_time=datetime.now(timezone.utc).isoformat(),
            total_tasks=total_tasks
        )

        self.sessions[session_id] = session
        self._save_sessions()

        logger.info(f"Started annotation session {session_id} for annotator {annotator_id}")
        return session_id

    def end_annotation_session(self, session_id: str, feedback: str = "") -> None:
        """End an annotation session."""
        if session_id not in self.sessions:
            logger.error(f"Annotation session {session_id} not found")
            return

        session = self.sessions[session_id]
        session.end_time = datetime.now(timezone.utc).isoformat()
        session.feedback = feedback

        # Calculate accuracy score based on completed tasks
        completed_tasks = [
            task for task in self.tasks.values()
            if task.annotator_id == session.annotator_id and task.status == "completed"
        ]
        session.tasks_completed = len(completed_tasks)

        self._save_sessions()
        logger.info(f"Ended annotation session {session_id}")

    def get_pending_tasks(self, annotator_id: str, limit: int = 20) -> List[AnnotationTask]:
        """Get pending tasks for an annotator."""
        pending_tasks = [
            task for task in self.tasks.values()
            if task.annotator_id == annotator_id and task.status == "pending"
        ]

        # Sort by priority (highest first) and creation time (oldest first)
        pending_tasks.sort(key=lambda x: (-x.priority, x.created_at))

        return pending_tasks[:limit]

    def assign_task_to_annotator(self, task_id: str, annotator_id: str) -> bool:
        """Assign a task to an annotator."""
        if task_id not in self.tasks:
            logger.error(f"Task {task_id} not found")
            return False

        task = self.tasks[task_id]
        task.annotator_id = annotator_id
        task.assigned_at = datetime.now(timezone.utc).isoformat()
        task.status = "in_progress"

        self._save_tasks()
        logger.info(f"Assigned task {task_id} to annotator {annotator_id}")
        return True

    def _save_tasks(self) -> None:
        """Save annotation tasks to file."""
        try:
            tasks_file = self.storage_path / "tasks" / "annotation_tasks.json"
            with open(tasks_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(task) for task in self.tasks.values()], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving annotation tasks: {e}")

    def _save_corrections(self) -> None:
        """Save corrections to file."""
        try:
            corrections_file = self.storage_path / "corrections" / "corrections.json"
            with open(corrections_file, 'w', encoding='utf-8') as f:
                json.dump(self.corrections, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving corrections: {e}")

    def _save_sessions(self) -> None:
        """Save annotation sessions to file."""
        try:
            sessions_file = self.storage_path / "sessions" / "annotation_sessions.json"
            with open(sessions_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(session) for session in self.sessions.values()], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving annotation sessions: {e}")

    def generate_annotation_report(self, annotator_id: str = None) -> Dict[str, Any]:
        """Generate a report of annotation activities."""
        report = {
            "total_tasks": len(self.tasks),
            "completed_tasks": len([t for t in self.tasks.values() if t.status == "completed"]),
            "pending_tasks": len([t for t in self.tasks.values() if t.status == "pending"]),
            "in_progress_tasks": len([t for t in self.tasks.values() if t.status == "in_progress"]),
            "total_corrections": sum(len(corrections) for corrections in self.corrections.values()),
            "annotator_stats": {}
        }

        # Annotator-specific stats
        if annotator_id:
            annotator_tasks = [t for t in self.tasks.values() if t.annotator_id == annotator_id]
            report["annotator_stats"][annotator_id] = {
                "total_assigned": len(annotator_tasks),
                "completed": len([t for t in annotator_tasks if t.status == "completed"]),
                "pending": len([t for t in annotator_tasks if t.status == "pending"]),
                "in_progress": len([t for t in annotator_tasks if t.status == "in_progress"])
            }
        else:
            # Stats for all annotators
            for task in self.tasks.values():
                if task.annotator_id not in report["annotator_stats"]:
                    report["annotator_stats"][task.annotator_id] = {
                        "total_assigned": 0,
                        "completed": 0,
                        "pending": 0,
                        "in_progress": 0
                    }

                report["annotator_stats"][task.annotator_id]["total_assigned"] += 1
                if task.status == "completed":
                    report["annotator_stats"][task.annotator_id]["completed"] += 1
                elif task.status == "pending":
                    report["annotator_stats"][task.annotator_id]["pending"] += 1
                elif task.status == "in_progress":
                    report["annotator_stats"][task.annotator_id]["in_progress"] += 1

        return report

    def launch_web_interface(self, host: str = "localhost", port: int = 8080) -> None:
        """Launch web interface for annotation."""
        if self.web_server:
            logger.warning("Web interface already running")
            return

        # Create web interface handler
        handler = AnnotationWebHandler
        handler.annotation_interface = self

        try:
            with socketserver.TCPServer((host, port), handler) as httpd:
                self.web_server = httpd
                logger.info(f"Web interface started at http://{host}:{port}")

                # Open browser
                webbrowser.open(f"http://{host}:{port}")

                # Start server in thread
                self.server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
                self.server_thread.start()

        except Exception as e:
            logger.error(f"Error starting web interface: {e}")
            self.web_server = None

    def stop_web_interface(self) -> None:
        """Stop the web interface."""
        if self.web_server:
            self.web_server.shutdown()
            self.web_server = None
            self.server_thread = None
            logger.info("Web interface stopped")

class AnnotationWebHandler(http.server.SimpleHTTPRequestHandler):
    """Simple web handler for annotation interface."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent / "annotation_interface" / "web_interface"), **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/":
            self.path = "/index.html"
        elif self.path == "/api/tasks":
            self.handle_get_tasks()
        elif self.path == "/api/example":
            self.handle_get_example()
        elif self.path.startswith("/api/submit"):
            self.handle_submit_annotation()
        else:
            super().do_GET()

    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/api/submit":
            self.handle_submit_annotation()

    def handle_get_tasks(self):
        """Get pending tasks for annotation."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        # Get annotator ID from query parameters
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)
        annotator_id = query_params.get('annotator_id', ['default'])[0]

        if hasattr(self, 'annotation_interface'):
            tasks = self.annotation_interface.get_pending_tasks(annotator_id)
            task_data = [asdict(task) for task in tasks]

            response = json.dumps({
                "tasks": task_data,
                "total": len(task_data)
            })
        else:
            response = json.dumps({"error": "Annotation interface not available"})

        self.wfile.write(response.encode('utf-8'))

    def handle_get_example(self):
        """Get a specific training example for annotation."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)
        example_id = query_params.get('example_id', [None])[0]

        if not example_id:
            response = json.dumps({"error": "Example ID required"})
        elif hasattr(self, 'annotation_interface'):
            # Find the example
            example = None
            for ex in self.annotation_interface.data_collector.examples:
                if ex.id == example_id:
                    example = ex
                    break

            if example:
                response = json.dumps({
                    "example": asdict(example),
                    "success": True
                })
            else:
                response = json.dumps({"error": "Example not found"})
        else:
            response = json.dumps({"error": "Annotation interface not available"})

        self.wfile.write(response.encode('utf-8'))

    def handle_submit_annotation(self):
        """Handle annotation submission."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        if hasattr(self, 'annotation_interface'):
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))

                task_id = data.get('task_id')
                annotator_id = data.get('annotator_id')
                corrections = [
                    AnnotationCorrection(**correction)
                    for correction in data.get('corrections', [])
                ]
                notes = data.get('notes', '')

                success = self.annotation_interface.submit_annotation(
                    task_id, annotator_id, corrections, notes
                )

                response = json.dumps({"success": success})

            except Exception as e:
                response = json.dumps({"success": False, "error": str(e)})
        else:
            response = json.dumps({"success": False, "error": "Annotation interface not available"})

        self.wfile.write(response.encode('utf-8'))

# Global annotation interface instance
_annotation_interface = None

def get_annotation_interface(storage_path: str = "models/training_data/annotations") -> AnnotationInterface:
    """Get or create global annotation interface instance."""
    global _annotation_interface
    if _annotation_interface is None:
        _annotation_interface = AnnotationInterface(storage_path)
    return _annotation_interface

def create_annotation_task(
    example_id: str,
    annotator_id: str,
    priority: int = 5
) -> str:
    """Convenience function to create an annotation task."""
    interface = get_annotation_interface()
    return interface.create_annotation_task(example_id, annotator_id, priority)