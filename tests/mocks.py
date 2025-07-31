"""Mock implementations for testing Claude Flutter Firebase Agent."""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from unittest.mock import Mock, MagicMock

from claude_code_agent_farm.models import AgentStatus, AgentEvent, UsageLimitInfo


class MockFlutterAgentMonitor:
    """Mock implementation of FlutterAgentMonitor for testing."""
    
    def __init__(self, settings):
        self.settings = settings
        self.session = Mock()
        self.session.status = AgentStatus.READY
        self.session.runs = 0
        self.session.events = []
        self.session.usage_limit_hits = 0
        self.session.usage_limit_info = None
        
        self.running = False
        self.tmux_session_exists = False
        self._response_queue = []
        self._response_index = 0
        
    def add_response(self, response: str):
        """Add a response to the queue."""
        self._response_queue.append(response)
    
    def start_agent(self):
        """Mock agent startup."""
        self.tmux_session_exists = True
        self.session.start_time = datetime.now()
        self.session.events.append(
            AgentEvent(event_type="started", message="Mock agent started")
        )
    
    def send_prompt(self):
        """Mock sending prompt."""
        self.session.status = AgentStatus.WORKING
        self.session.events.append(
            AgentEvent(event_type="prompt_sent", message=self.settings.prompt_text)
        )
    
    def capture_pane_content(self) -> str:
        """Mock capturing tmux pane content."""
        if self._response_index < len(self._response_queue):
            response = self._response_queue[self._response_index]
            self._response_index += 1
            return response
        return ">> "  # Default ready prompt
    
    def check_agent_status(self) -> AgentStatus:
        """Mock status checking."""
        content = self.capture_pane_content()
        
        if "usage limit" in content.lower():
            self.session.status = AgentStatus.USAGE_LIMIT
            # Parse retry time
            self.session.usage_limit_info = self._parse_usage_limit(content)
            self.session.usage_limit_hits += 1
        elif "error" in content.lower() or "failed" in content.lower():
            self.session.status = AgentStatus.ERROR
        elif content.strip().endswith(">> "):
            self.session.status = AgentStatus.READY
        else:
            self.session.status = AgentStatus.WORKING
        
        return self.session.status
    
    def restart_agent(self):
        """Mock agent restart."""
        self.session.runs += 1
        self.session.status = AgentStatus.READY
        self.session.events.append(
            AgentEvent(event_type="restarted", message=f"Restart #{self.session.runs}")
        )
        self._response_index = 0  # Reset responses
    
    def _parse_usage_limit(self, content: str) -> Optional[UsageLimitInfo]:
        """Mock parsing usage limit info."""
        # Simple mock implementation
        retry_time = datetime.now() + timedelta(hours=2)
        return UsageLimitInfo(
            message=content,
            retry_time=retry_time
        )


class MockClaudeAPI:
    """Mock Claude API for testing interactions."""
    
    def __init__(self):
        self.conversation_history = []
        self.response_templates = {
            "greeting": "Hello! I'm ready to help with your Carenji development.",
            "error": "I encountered an error: {error_message}",
            "working": "Working on {task}...",
            "complete": "✓ Task completed successfully!"
        }
        self.custom_responses = {}
        
    def send_message(self, message: str) -> str:
        """Send message and get response."""
        self.conversation_history.append({"role": "user", "content": message})
        
        # Check for custom responses
        for pattern, response in self.custom_responses.items():
            if pattern in message.lower():
                self.conversation_history.append({"role": "assistant", "content": response})
                return response
        
        # Generate contextual response
        response = self._generate_response(message)
        self.conversation_history.append({"role": "assistant", "content": response})
        return response
    
    def _generate_response(self, message: str) -> str:
        """Generate appropriate response based on message."""
        message_lower = message.lower()
        
        if "error" in message_lower:
            return self.response_templates["error"].format(
                error_message="Mock error for testing"
            )
        elif any(word in message_lower for word in ["implement", "create", "build", "fix"]):
            task = message.split()[1] if len(message.split()) > 1 else "the task"
            return self.response_templates["working"].format(task=task)
        elif "test" in message_lower:
            return self._generate_test_response()
        elif "firebase" in message_lower:
            return self._generate_firebase_response()
        else:
            return self.response_templates["greeting"]
    
    def _generate_test_response(self) -> str:
        """Generate test-related response."""
        return """
Running tests for Carenji...

✓ Unit tests: 45 passed
✓ Widget tests: 23 passed  
✓ Integration tests: 12 passed

Test coverage: 83.5%
All tests passed!
>> """
    
    def _generate_firebase_response(self) -> str:
        """Generate Firebase-related response."""
        return """
Setting up Firebase emulators...

✓ Auth emulator started on port 9099
✓ Firestore emulator started on port 8080
✓ Storage emulator started on port 9199

Firebase emulators ready for development.
>> """
    
    def set_custom_response(self, pattern: str, response: str):
        """Set custom response for pattern matching."""
        self.custom_responses[pattern] = response
    
    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []
        self.custom_responses = {}


class MockFirebaseEmulatorSuite:
    """Mock Firebase emulator suite with realistic behavior."""
    
    def __init__(self):
        self.emulators = {
            "auth": MockAuthEmulator(),
            "firestore": MockFirestoreEmulator(),
            "storage": MockStorageEmulator(),
            "functions": MockFunctionsEmulator()
        }
        self.is_running = False
        self.start_time = None
        
    def start(self, project_id: str = "test-project") -> bool:
        """Start all emulators."""
        try:
            for emulator in self.emulators.values():
                emulator.start()
            self.is_running = True
            self.start_time = datetime.now()
            return True
        except Exception:
            return False
    
    def stop(self):
        """Stop all emulators."""
        for emulator in self.emulators.values():
            emulator.stop()
        self.is_running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get emulator status."""
        return {
            "running": self.is_running,
            "uptime": (datetime.now() - self.start_time).seconds if self.start_time else 0,
            "emulators": {
                name: emulator.get_status() 
                for name, emulator in self.emulators.items()
            }
        }


class MockFirestoreEmulator:
    """Mock Firestore emulator."""
    
    def __init__(self):
        self.collections = {}
        self.is_running = False
        self.listeners = {}
        
    def start(self):
        """Start Firestore emulator."""
        self.is_running = True
        # Initialize with some test data
        self._init_test_data()
    
    def stop(self):
        """Stop Firestore emulator."""
        self.is_running = False
        self.collections.clear()
        self.listeners.clear()
    
    def _init_test_data(self):
        """Initialize with Carenji test data."""
        self.collections = {
            "patients": {
                "patient_001": {
                    "firstName": "John",
                    "lastName": "Doe",
                    "roomNumber": "205",
                    "conditions": ["Diabetes", "Hypertension"]
                }
            },
            "medications": {
                "med_001": {
                    "patientId": "patient_001",
                    "name": "Metformin",
                    "dosage": "500mg",
                    "frequency": "Twice daily"
                }
            }
        }
    
    def collection(self, name: str) -> 'MockCollection':
        """Get collection reference."""
        if name not in self.collections:
            self.collections[name] = {}
        return MockCollection(name, self.collections[name])
    
    def add_listener(self, collection: str, callback: Callable):
        """Add real-time listener."""
        if collection not in self.listeners:
            self.listeners[collection] = []
        self.listeners[collection].append(callback)
    
    def trigger_update(self, collection: str, doc_id: str, data: dict):
        """Trigger real-time update."""
        if collection in self.collections:
            self.collections[collection][doc_id] = data
            
            # Notify listeners
            if collection in self.listeners:
                for callback in self.listeners[collection]:
                    callback({"type": "modified", "doc_id": doc_id, "data": data})
    
    def get_status(self) -> dict:
        """Get emulator status."""
        return {
            "running": self.is_running,
            "collections": list(self.collections.keys()),
            "document_count": sum(len(docs) for docs in self.collections.values())
        }


class MockCollection:
    """Mock Firestore collection."""
    
    def __init__(self, name: str, data: dict):
        self.name = name
        self.data = data
    
    def doc(self, doc_id: str) -> 'MockDocument':
        """Get document reference."""
        return MockDocument(doc_id, self.data.get(doc_id))
    
    def add(self, data: dict) -> str:
        """Add new document."""
        doc_id = f"{self.name}_{len(self.data) + 1:03d}"
        self.data[doc_id] = data
        return doc_id
    
    def where(self, field: str, op: str, value: Any) -> List[dict]:
        """Query documents."""
        results = []
        for doc_id, doc_data in self.data.items():
            if field in doc_data:
                if op == "==" and doc_data[field] == value:
                    results.append({"id": doc_id, **doc_data})
                elif op == ">" and doc_data[field] > value:
                    results.append({"id": doc_id, **doc_data})
                elif op == "<" and doc_data[field] < value:
                    results.append({"id": doc_id, **doc_data})
        return results


class MockDocument:
    """Mock Firestore document."""
    
    def __init__(self, doc_id: str, data: Optional[dict]):
        self.id = doc_id
        self.data = data
    
    def get(self) -> Optional[dict]:
        """Get document data."""
        return self.data
    
    def set(self, data: dict):
        """Set document data."""
        self.data = data
    
    def update(self, updates: dict):
        """Update document fields."""
        if self.data:
            self.data.update(updates)
        else:
            self.data = updates
    
    def delete(self):
        """Delete document."""
        self.data = None


class MockAuthEmulator:
    """Mock Firebase Auth emulator."""
    
    def __init__(self):
        self.users = {}
        self.is_running = False
        self.current_user = None
        
    def start(self):
        """Start auth emulator."""
        self.is_running = True
        self._create_test_users()
    
    def stop(self):
        """Stop auth emulator."""
        self.is_running = False
        self.users.clear()
        self.current_user = None
    
    def _create_test_users(self):
        """Create test users for Carenji."""
        test_users = [
            {
                "uid": "nurse_001",
                "email": "nurse@carenji.com",
                "role": "nurse",
                "displayName": "Nurse Johnson"
            },
            {
                "uid": "admin_001",
                "email": "admin@carenji.com",
                "role": "admin",
                "displayName": "Admin Smith"
            },
            {
                "uid": "family_001",
                "email": "family@carenji.com",
                "role": "family",
                "displayName": "John's Family"
            }
        ]
        
        for user in test_users:
            self.users[user["email"]] = user
    
    def sign_in(self, email: str, password: str) -> Optional[dict]:
        """Sign in user."""
        if email in self.users:
            self.current_user = self.users[email]
            return self.current_user
        return None
    
    def sign_out(self):
        """Sign out current user."""
        self.current_user = None
    
    def get_current_user(self) -> Optional[dict]:
        """Get current user."""
        return self.current_user
    
    def create_user(self, email: str, password: str, **kwargs) -> dict:
        """Create new user."""
        uid = f"user_{len(self.users) + 1:03d}"
        user = {
            "uid": uid,
            "email": email,
            **kwargs
        }
        self.users[email] = user
        return user
    
    def get_status(self) -> dict:
        """Get emulator status."""
        return {
            "running": self.is_running,
            "user_count": len(self.users),
            "current_user": self.current_user["email"] if self.current_user else None
        }


class MockStorageEmulator:
    """Mock Firebase Storage emulator."""
    
    def __init__(self):
        self.files = {}
        self.is_running = False
        
    def start(self):
        """Start storage emulator."""
        self.is_running = True
    
    def stop(self):
        """Stop storage emulator."""
        self.is_running = False
        self.files.clear()
    
    def upload(self, path: str, data: bytes, metadata: Optional[dict] = None) -> dict:
        """Upload file."""
        file_info = {
            "path": path,
            "size": len(data),
            "metadata": metadata or {},
            "uploaded_at": datetime.now().isoformat()
        }
        self.files[path] = file_info
        return file_info
    
    def download(self, path: str) -> Optional[dict]:
        """Download file info."""
        return self.files.get(path)
    
    def delete(self, path: str) -> bool:
        """Delete file."""
        if path in self.files:
            del self.files[path]
            return True
        return False
    
    def list_files(self, prefix: str = "") -> List[dict]:
        """List files with prefix."""
        return [
            info for path, info in self.files.items()
            if path.startswith(prefix)
        ]
    
    def get_status(self) -> dict:
        """Get emulator status."""
        return {
            "running": self.is_running,
            "file_count": len(self.files),
            "total_size": sum(f["size"] for f in self.files.values())
        }


class MockFunctionsEmulator:
    """Mock Firebase Functions emulator."""
    
    def __init__(self):
        self.functions = {}
        self.is_running = False
        self.call_history = []
        
    def start(self):
        """Start functions emulator."""
        self.is_running = True
        self._register_carenji_functions()
    
    def stop(self):
        """Stop functions emulator."""
        self.is_running = False
        self.functions.clear()
        self.call_history.clear()
    
    def _register_carenji_functions(self):
        """Register Carenji cloud functions."""
        self.functions = {
            "scheduleGenerator": self._schedule_generator,
            "medicationReminder": self._medication_reminder,
            "vitalAlertsProcessor": self._vital_alerts,
            "familyNotification": self._family_notification
        }
    
    def call(self, function_name: str, data: dict) -> dict:
        """Call cloud function."""
        if function_name in self.functions:
            result = self.functions[function_name](data)
            self.call_history.append({
                "function": function_name,
                "data": data,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            return result
        else:
            raise ValueError(f"Function {function_name} not found")
    
    def _schedule_generator(self, data: dict) -> dict:
        """Mock schedule generation using OR-Tools."""
        return {
            "success": True,
            "schedule": {
                "date": data.get("date"),
                "shifts": [
                    {"staff_id": "nurse_001", "start": "08:00", "end": "16:00"},
                    {"staff_id": "nurse_002", "start": "16:00", "end": "00:00"}
                ]
            }
        }
    
    def _medication_reminder(self, data: dict) -> dict:
        """Mock medication reminder."""
        return {
            "success": True,
            "reminder_sent": True,
            "patient_id": data.get("patient_id"),
            "medication": data.get("medication")
        }
    
    def _vital_alerts(self, data: dict) -> dict:
        """Mock vital signs alert processing."""
        vitals = data.get("vitals", {})
        alerts = []
        
        if vitals.get("blood_pressure_systolic", 0) > 140:
            alerts.append("High blood pressure detected")
        if vitals.get("temperature", 0) > 38.5:
            alerts.append("Fever detected")
        
        return {
            "success": True,
            "alerts": alerts,
            "notify_staff": len(alerts) > 0
        }
    
    def _family_notification(self, data: dict) -> dict:
        """Mock family notification."""
        return {
            "success": True,
            "notification_sent": True,
            "family_members_notified": data.get("family_ids", [])
        }
    
    def get_status(self) -> dict:
        """Get emulator status."""
        return {
            "running": self.is_running,
            "functions": list(self.functions.keys()),
            "call_count": len(self.call_history)
        }


class MockTmuxManager:
    """Mock tmux session manager for testing."""
    
    def __init__(self):
        self.sessions = {}
        
    def create_session(self, name: str) -> bool:
        """Create tmux session."""
        if name not in self.sessions:
            self.sessions[name] = MockTmuxSession(name)
            return True
        return False
    
    def kill_session(self, name: str) -> bool:
        """Kill tmux session."""
        if name in self.sessions:
            self.sessions[name].kill()
            del self.sessions[name]
            return True
        return False
    
    def send_keys(self, session_name: str, keys: str) -> bool:
        """Send keys to session."""
        if session_name in self.sessions:
            self.sessions[session_name].send_keys(keys)
            return True
        return False
    
    def capture_pane(self, session_name: str) -> Optional[str]:
        """Capture pane content."""
        if session_name in self.sessions:
            return self.sessions[session_name].capture_pane()
        return None
    
    def list_sessions(self) -> List[str]:
        """List active sessions."""
        return list(self.sessions.keys())


# Advanced mock scenarios

class MockScenarioRunner:
    """Run complex test scenarios with mocks."""
    
    def __init__(self):
        self.monitor = None
        self.firebase = None
        self.claude = None
        
    def setup_carenji_development_scenario(self):
        """Setup complete Carenji development scenario."""
        # Initialize mocks
        settings = Mock()
        settings.prompt_text = "Implement medication reminder feature"
        
        self.monitor = MockFlutterAgentMonitor(settings)
        self.firebase = MockFirebaseEmulatorSuite()
        self.claude = MockClaudeAPI()
        
        # Configure responses
        self.claude.set_custom_response(
            "medication reminder",
            self._generate_medication_feature_response()
        )
        
        return self.monitor, self.firebase, self.claude
    
    def _generate_medication_feature_response(self) -> str:
        """Generate realistic medication feature implementation."""
        return """
Implementing medication reminder feature for Carenji...

1. Creating data models...
✓ Created MedicationReminder model
✓ Added reminder frequency enum
✓ Added notification preferences

2. Implementing service layer...
✓ Created MedicationReminderService
✓ Added scheduling logic
✓ Integrated with notification system

3. Building UI components...
✓ Created reminder settings view
✓ Added medication schedule widget
✓ Implemented reminder notifications

4. Writing tests...
✓ Unit tests: 15 added
✓ Widget tests: 8 added
✓ Integration tests: 3 added

5. Running all tests...
✓ All 26 new tests passed
✓ Total coverage: 85.2%

Feature implementation complete!
>> """
    
    def simulate_usage_limit_scenario(self, retry_minutes: int = 30):
        """Simulate hitting usage limit and recovery."""
        retry_time = datetime.now() + timedelta(minutes=retry_minutes)
        
        # Configure monitor to hit usage limit
        self.monitor.add_response(
            f"You've reached your usage limit. Please try again at {retry_time.strftime('%-I:%M %p')} PST"
        )
        
        # Add recovery response
        self.monitor.add_response("Ready to continue with the task!\n>> ")
        
        return self.monitor


# Test data generators

def generate_patient_data(count: int = 10) -> List[dict]:
    """Generate random patient data."""
    first_names = ["John", "Jane", "Robert", "Mary", "William", "Patricia", "James", "Jennifer"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
    conditions = ["Diabetes", "Hypertension", "Arthritis", "Dementia", "COPD", "Heart Disease"]
    
    patients = []
    for i in range(count):
        patients.append({
            "id": f"patient_{i+1:03d}",
            "firstName": random.choice(first_names),
            "lastName": random.choice(last_names),
            "roomNumber": f"{random.randint(100, 300)}",
            "conditions": random.sample(conditions, random.randint(1, 3)),
            "dateOfBirth": f"{random.randint(1930, 1960)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        })
    
    return patients


def generate_medication_data(patient_count: int = 10) -> List[dict]:
    """Generate medication data for patients."""
    medications_db = [
        ("Metformin", "500mg", "Twice daily"),
        ("Lisinopril", "10mg", "Once daily"),
        ("Aspirin", "81mg", "Once daily"),
        ("Atorvastatin", "20mg", "Once daily at bedtime"),
        ("Omeprazole", "20mg", "Once daily before breakfast"),
        ("Levothyroxine", "50mcg", "Once daily on empty stomach")
    ]
    
    medications = []
    for i in range(patient_count):
        patient_id = f"patient_{i+1:03d}"
        med_count = random.randint(2, 5)
        
        for j, med_info in enumerate(random.sample(medications_db, med_count)):
            name, dosage, frequency = med_info
            medications.append({
                "id": f"med_{i+1:03d}_{j+1:02d}",
                "patientId": patient_id,
                "name": name,
                "dosage": dosage,
                "frequency": frequency,
                "startDate": datetime.now().isoformat(),
                "prescribedBy": f"Dr. {random.choice(['Smith', 'Johnson', 'Williams'])}"
            })
    
    return medications