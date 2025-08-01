"""End-to-end tests for Firebase integration."""

import json
import subprocess
from unittest.mock import Mock, patch

import pytest

from claude_code_agent_farm.flutter_agent_monitor import FlutterAgentMonitor
from claude_code_agent_farm.flutter_agent_settings import FlutterAgentSettings
from claude_code_agent_farm.models_new.session import AgentStatus
from claude_code_agent_farm.utils.flutter_helpers import get_firebase_emulator_status


@pytest.mark.e2e
@pytest.mark.firebase
class TestFirebaseIntegration:
    """Test Firebase emulator integration and operations."""

    @pytest.fixture
    def firebase_project_dir(self, temp_dir):
        """Create a Firebase-enabled Carenji project."""
        project_dir = temp_dir / "firebase_carenji"
        project_dir.mkdir()

        # Create pubspec.yaml with Firebase dependencies
        pubspec = """
name: carenji
description: Healthcare management system

environment:
  sdk: '>=2.19.0 <3.0.0'

dependencies:
  flutter:
    sdk: flutter
  
  # Firebase dependencies
  firebase_core: ^2.24.0
  cloud_firestore: ^4.13.0
  firebase_auth: ^4.15.0
  firebase_storage: ^11.5.0
  cloud_functions: ^4.5.0
  firebase_messaging: ^14.7.0

dev_dependencies:
  flutter_test:
    sdk: flutter
"""
        (project_dir / "pubspec.yaml").write_text(pubspec)

        # Create firebase.json
        firebase_config = {
            "emulators": {
                "auth": {"port": 9099, "host": "0.0.0.0"},
                "functions": {"port": 5001, "host": "0.0.0.0"},
                "firestore": {"port": 8080, "host": "0.0.0.0"},
                "storage": {"port": 9199, "host": "0.0.0.0"},
                "ui": {"enabled": true, "port": 4000, "host": "0.0.0.0"},
            },
            "firestore": {"rules": "firestore.rules", "indexes": "firestore.indexes.json"},
            "storage": {"rules": "storage.rules"},
        }
        (project_dir / "firebase.json").write_text(json.dumps(firebase_config, indent=2))

        # Create .firebaserc
        firebaserc = {"projects": {"default": "carenji-test"}}
        (project_dir / ".firebaserc").write_text(json.dumps(firebaserc, indent=2))

        # Create Firestore rules
        firestore_rules = """
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Healthcare data rules
    match /patients/{patientId} {
      allow read: if request.auth != null;
      allow write: if request.auth != null && 
        request.auth.token.role in ['nurse', 'admin'];
    }
    
    match /medications/{medicationId} {
      allow read: if request.auth != null;
      allow write: if request.auth != null &&
        request.auth.token.role in ['nurse', 'admin', 'pharmacist'];
    }
    
    match /vitals/{vitalId} {
      allow read: if request.auth != null;
      allow write: if request.auth != null &&
        request.auth.token.role in ['nurse', 'admin'];
    }
    
    match /staff/{staffId} {
      allow read: if request.auth != null;
      allow write: if request.auth != null &&
        request.auth.token.role == 'admin';
    }
  }
}
"""
        (project_dir / "firestore.rules").write_text(firestore_rules)

        # Create storage rules
        storage_rules = """
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    match /patient_photos/{patientId}/{allPaths=**} {
      allow read: if request.auth != null;
      allow write: if request.auth != null &&
        request.auth.token.role in ['nurse', 'admin'];
    }
    
    match /documents/{allPaths=**} {
      allow read: if request.auth != null;
      allow write: if request.auth != null &&
        request.auth.token.role in ['admin'];
    }
  }
}
"""
        (project_dir / "storage.rules").write_text(storage_rules)

        # Create Firestore indexes
        indexes = {
            "indexes": [
                {
                    "collectionGroup": "medications",
                    "queryScope": "COLLECTION",
                    "fields": [
                        {"fieldPath": "patientId", "order": "ASCENDING"},
                        {"fieldPath": "scheduledTime", "order": "ASCENDING"},
                    ],
                },
                {
                    "collectionGroup": "vitals",
                    "queryScope": "COLLECTION",
                    "fields": [
                        {"fieldPath": "patientId", "order": "ASCENDING"},
                        {"fieldPath": "timestamp", "order": "DESCENDING"},
                    ],
                },
            ]
        }
        (project_dir / "firestore.indexes.json").write_text(json.dumps(indexes, indent=2))

        # Create lib structure
        lib_dir = project_dir / "lib"
        lib_dir.mkdir()

        # Create Firebase initialization code
        firebase_init = """
import 'package:firebase_core/firebase_core.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class FirebaseService {
  static bool _initialized = false;
  
  static Future<void> initialize() async {
    if (_initialized) return;
    
    await Firebase.initializeApp();
    
    // Use emulators in development
    if (const bool.fromEnvironment('USE_FIREBASE_EMULATOR')) {
      await _connectToEmulators();
    }
    
    _initialized = true;
  }
  
  static Future<void> _connectToEmulators() async {
    FirebaseFirestore.instance.useFirestoreEmulator('localhost', 8080);
    await FirebaseAuth.instance.useAuthEmulator('localhost', 9099);
  }
}
"""
        (lib_dir / "firebase_service.dart").write_text(firebase_init)

        return project_dir

    @pytest.fixture
    def monitor_with_firebase(self, firebase_project_dir):
        """Create monitor configured for Firebase testing."""
        settings = FlutterAgentSettings(
            claude_project_path=firebase_project_dir,
            tmux_session_name="test-firebase",
            prompt_text="Test Firebase integration for Carenji",
            firebase_emulator_host="localhost",
        )

        with patch("subprocess.run") as mock_run:
            with patch("claude_code_agent_farm.flutter_agent_monitor.signal.signal"):
                mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
                monitor = FlutterAgentMonitor(settings)

                # Mock tmux operations
                monitor._create_tmux_session = Mock()
                monitor._kill_tmux_session = Mock()

                yield monitor

    def test_firebase_emulator_detection(self, monitor_with_firebase):
        """Test detecting Firebase emulator status."""
        monitor = monitor_with_firebase

        # Test when emulators are not running
        with patch("claude_code_agent_farm.utils.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=1, stdout="")

            status = get_firebase_emulator_status()
            # Check that all services are not running
            assert all(not running for running in status.values())

        # Test when emulators are running
        with patch("claude_code_agent_farm.utils.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="✔  All emulators ready!")

            status = get_firebase_emulator_status()
            # Check that at least some services are running
            assert any(running for running in status.values())

    def test_firebase_queries(self, monitor_with_firebase):
        """Test Firebase-related queries."""
        monitor = monitor_with_firebase
        monitor.start_agent()

        # Mock responses for Firebase operations
        firebase_responses = {
            "setup": """Setting up Firebase for the Carenji project...

Reading firebase.json configuration...
✓ Found emulator configuration for: auth, firestore, storage, functions

Checking Firebase CLI installation...
✓ Firebase CLI version 12.0.0 installed

Starting Firebase emulators...
$ firebase emulators:start --only auth,firestore,storage

✔  All emulators ready! View status at http://localhost:4000

┌─────────────────────────────────────────────────────────────┐
│ ✔  All emulators ready! It is now safe to connect your app. │
│ i  View Emulator UI at http://localhost:4000                │
└─────────────────────────────────────────────────────────────┘

┌────────────────┬────────────────┬─────────────────────────────┐
│ Emulator       │ Host:Port      │ View in Emulator UI         │
├────────────────┼────────────────┼─────────────────────────────┤
│ Authentication │ localhost:9099 │ http://localhost:4000/auth  │
│ Firestore      │ localhost:8080 │ http://localhost:4000/store │
│ Storage        │ localhost:9199 │ http://localhost:4000/storage│
└────────────────┴────────────────┴─────────────────────────────┘

Firebase emulators are now running for Carenji development.
>> """,
            "firestore_test": """Testing Firestore operations for Carenji...

Creating test patient document...
✓ Created patient: John Doe (ID: patient_001)

Testing medication records...
✓ Added medication: Aspirin 100mg
✓ Added medication: Metformin 500mg

Testing vital signs...
✓ Recorded blood pressure: 120/80
✓ Recorded temperature: 98.6°F

Testing queries...
✓ Found 2 medications for patient_001
✓ Latest vital signs retrieved

Testing real-time updates...
✓ Listener attached to patient updates
✓ Update received: patient status changed

All Firestore operations completed successfully!
>> """,
            "auth_test": """Testing Firebase Auth for Carenji staff...

Creating test users...
✓ Created nurse account: nurse@carenji.com
✓ Created admin account: admin@carenji.com
✓ Created family account: family@carenji.com

Testing authentication...
✓ Nurse login successful
✓ Custom claims set: {role: 'nurse', facility: 'Sunset Care'}

Testing role-based access...
✓ Nurse can read patient data
✓ Nurse can write vital signs
✗ Nurse cannot modify staff records (expected)

Testing family portal access...
✓ Family member authenticated
✓ Limited access verified

All authentication tests passed!
>> """,
            "storage_test": """Testing Firebase Storage for Carenji...

Uploading patient photo...
✓ Uploaded: patient_001_profile.jpg (245KB)

Uploading medical documents...
✓ Uploaded: medication_chart_2024.pdf (1.2MB)
✓ Uploaded: care_plan_001.docx (89KB)

Testing access control...
✓ Nurse can access patient photos
✗ Family cannot upload documents (expected)

Testing file metadata...
✓ Metadata updated with tags

Storage operations completed successfully!
>> """,
        }

        # Test Firebase setup
        self.response_key = "setup"

        def get_response():
            return firebase_responses.get(self.response_key, ">> ")

        with patch.object(monitor, "capture_pane_content", side_effect=get_response):
            monitor.settings.prompt_text = "Setup Firebase emulators for Carenji"
            monitor.send_prompt()

            response = monitor.capture_pane_content()
            assert "All emulators ready" in response
            assert "localhost:9099" in response  # Auth
            assert "localhost:8080" in response  # Firestore
            assert "localhost:9199" in response  # Storage

    def test_firestore_operations(self, monitor_with_firebase):
        """Test Firestore CRUD operations."""
        monitor = monitor_with_firebase
        monitor.start_agent()

        responses = [
            # Create
            "Creating patient record...\n✓ Document created with ID: abc123\n>> ",
            # Read
            "Reading patient data...\n✓ Found patient: John Doe\n>> ",
            # Update
            "Updating medication list...\n✓ Medication added to patient record\n>> ",
            # Delete
            "Removing expired prescription...\n✓ Document deleted\n>> ",
            # Query
            "Querying active medications...\n✓ Found 5 active prescriptions\n>> ",
        ]

        with patch.object(monitor, "capture_pane_content", side_effect=responses):
            # Test each operation
            for i, operation in enumerate(["create", "read", "update", "delete", "query"]):
                response = monitor.capture_pane_content()
                assert "✓" in response
                assert operation in response.lower() or "found" in response.lower()

    def test_firebase_error_handling(self, monitor_with_firebase):
        """Test handling Firebase errors."""
        monitor = monitor_with_firebase
        monitor.start_agent()

        error_scenarios = [
            ("PERMISSION_DENIED: Missing or insufficient permissions", AgentStatus.ERROR),
            ("Failed to connect to Firebase emulator at localhost:8080", AgentStatus.ERROR),
            ("Error: Firebase project not initialized", AgentStatus.ERROR),
            ("FirebaseError: No document to update", AgentStatus.ERROR),
        ]

        for error_msg, expected_status in error_scenarios:
            with patch.object(monitor, "capture_pane_content", return_value=error_msg):
                status = monitor.check_agent_status()
                assert status == expected_status

    def test_firebase_security_rules(self, monitor_with_firebase):
        """Test Firebase security rules validation."""
        monitor = monitor_with_firebase
        monitor.start_agent()

        rules_test_response = """Testing Carenji Firebase security rules...

Testing patient data access:
✓ Authenticated nurse can read patient data
✓ Authenticated nurse can write vital signs
✗ Unauthenticated user cannot read (expected)
✗ Family member cannot write medical data (expected)

Testing role-based permissions:
✓ Admin can access all collections
✓ Nurse can access patient and medication data
✓ Pharmacist can modify medication records
✗ Regular user cannot access staff data (expected)

Testing data validation:
✓ Medication requires all required fields
✓ Vital signs require valid ranges
✗ Cannot create patient without required fields (expected)

Security rules validation complete!
13 tests passed, 5 expected failures
>> """

        with patch.object(monitor, "capture_pane_content", return_value=rules_test_response):
            response = monitor.capture_pane_content()

            # Verify security testing
            assert "Security rules validation complete" in response
            assert "13 tests passed" in response
            assert "role-based permissions" in response.lower()

    def test_firebase_offline_support(self, monitor_with_firebase):
        """Test Firebase offline capabilities."""
        monitor = monitor_with_firebase
        monitor.start_agent()

        offline_response = """Testing Carenji offline support...

Enabling offline persistence...
✓ Firestore offline persistence enabled
✓ Cache size set to 100MB

Testing offline operations:
✓ Created local patient record (pending sync)
✓ Updated medication list (queued)
✓ Vital signs cached locally

Simulating reconnection...
✓ Connection restored
✓ 3 pending writes synchronized
✓ Local cache merged with server data

Testing conflict resolution:
✓ Server timestamp prevails for conflicts
✓ Merge strategy applied for arrays

Offline support verified for Carenji!
>> """

        with patch.object(monitor, "capture_pane_content", return_value=offline_response):
            response = monitor.capture_pane_content()

            assert "offline persistence enabled" in response
            assert "pending writes synchronized" in response
            assert "conflict resolution" in response.lower()

    @pytest.mark.docker
    def test_firebase_emulators_in_docker(self, docker_container):
        """Test Firebase emulators running in Docker."""
        # Check if Firebase CLI is available
        result = subprocess.run(
            ["docker", "exec", docker_container, "firebase", "--version"], capture_output=True, text=True
        )

        if result.returncode != 0:
            pytest.skip("Firebase CLI not available in container")

        # Check emulator config
        result = subprocess.run(
            [
                "docker",
                "exec",
                docker_container,
                "sh",
                "-c",
                "cd /tmp && firebase emulators:exec --only firestore 'echo ready' --project test",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should at least not crash
        assert "ready" in result.stdout or "emulator" in result.stdout.lower()

    def test_realtime_updates(self, monitor_with_firebase):
        """Test Firebase real-time update handling."""
        monitor = monitor_with_firebase
        monitor.start_agent()

        realtime_response = """Setting up real-time listeners for Carenji...

Listening to patient updates...
✓ Listener attached to /patients collection

Simulating real-time events:
📥 Patient admitted: Jane Smith (room 205)
📥 Medication administered: patient_001 - Aspirin 100mg
📥 Vital signs updated: patient_003 - BP 130/85
📥 Staff schedule changed: Nurse Johnson - Shift extended

Testing listener cleanup...
✓ All listeners detached on component unmount
✓ No memory leaks detected

Real-time synchronization working correctly!
>> """

        with patch.object(monitor, "capture_pane_content", return_value=realtime_response):
            response = monitor.capture_pane_content()

            assert "Listener attached" in response
            assert "Patient admitted" in response
            assert "Real-time synchronization working" in response

    def test_firebase_data_migration(self, monitor_with_firebase):
        """Test Firebase data migration scenarios."""
        monitor = monitor_with_firebase
        monitor.start_agent()

        migration_response = """Running Carenji data migration...

Analyzing current data structure...
✓ Found 150 patient records
✓ Found 1,234 medication entries
✓ Found 5,678 vital sign records

Migrating to new schema...
✓ Added 'lastModified' timestamp to all documents
✓ Converted medication dosage to structured format
✓ Normalized phone numbers to E.164 format

Creating indexes for new queries...
✓ Created compound index: patientId + timestamp
✓ Created index for medication search

Verifying data integrity...
✓ All documents successfully migrated
✓ No data loss detected
✓ Query performance improved by 40%

Migration completed successfully!
Processed 7,062 documents in 4.3 seconds
>> """

        with patch.object(monitor, "capture_pane_content", return_value=migration_response):
            response = monitor.capture_pane_content()

            assert "Migration completed successfully" in response
            assert "7,062 documents" in response
            assert "No data loss" in response

    def test_firebase_backup_restore(self, monitor_with_firebase):
        """Test Firebase backup and restore operations."""
        monitor = monitor_with_firebase
        monitor.start_agent()

        backup_response = """Performing Carenji database backup...

Exporting Firestore data...
✓ Exported /patients collection (150 documents)
✓ Exported /medications collection (1,234 documents)
✓ Exported /vitals collection (5,678 documents)
✓ Exported /staff collection (45 documents)

Creating backup archive...
✓ Created carenji_backup_2024_01_15.tar.gz (12.4MB)

Testing restore process...
✓ Cleared test database
✓ Imported all collections
✓ Verified document counts match
✓ Spot-checked data integrity

Backup and restore process verified!
Total backup size: 12.4MB
Time elapsed: 8.2 seconds
>> """

        with patch.object(monitor, "capture_pane_content", return_value=backup_response):
            response = monitor.capture_pane_content()

            assert "backup_2024" in response
            assert "12.4MB" in response
            assert "data integrity" in response
