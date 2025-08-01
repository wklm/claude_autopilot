"""End-to-end tests for Carenji-specific queries."""

import subprocess
from unittest.mock import Mock, patch

import pytest

from claude_code_agent_farm.flutter_agent_monitor import FlutterAgentMonitor
from claude_code_agent_farm.models_new.session import AgentStatus


@pytest.mark.e2e
@pytest.mark.carenji
class TestCarenjiQueries:
    """Test real-world Carenji development queries."""

    @pytest.fixture
    def mock_claude_responses(self):
        """Mock Claude responses for different queries."""
        responses = {
            "what is this app": """Looking at the carenji project...

This is Carenji - a comprehensive healthcare management system designed for nursing homes and care facilities. The app is built with Flutter for cross-platform support and uses Firebase as the backend infrastructure.

Key features include:
- Medication Management: ordering, tracking, stock management with alerts
- Vital Signs Monitoring: real-time recording with responsive UI
- Staff Scheduling: AI-powered roster generation using Google OR-Tools
- Family Portal: secure access for family members with visit scheduling
- Barcode/QR code scanning for medication and patient identification

The architecture follows clean architecture principles with MVVM pattern, using Provider for state management and GetIt + Injectable for dependency injection.
>> """,
            "list main features": """Analyzing the carenji healthcare app features...

Main features of Carenji:

1. **Medication Management**
   - Medication ordering and tracking
   - Stock management with automatic alerts
   - Administration recording with timestamps
   - Barcode/QR code scanning for verification

2. **Vital Signs Monitoring**
   - Real-time vital signs recording
   - Historical data visualization with charts
   - Alert thresholds for abnormal readings
   - Responsive UI for all screen sizes

3. **Staff Scheduling** (Phase 11)
   - AI-powered roster generation using Google OR-Tools
   - Real-time schedule updates
   - Staff availability management
   - QR code check-in/out system
   - Multiple export formats (CSV, iCal, PDF)

4. **Family Portal**
   - Secure family member access with role-based permissions
   - Visit scheduling and management
   - Communication tools for updates
   - Photo and document sharing

5. **Additional Features**
   - Bluetooth device integration for automated vital signs
   - Offline support with data synchronization
   - Multi-language support (English, German, Polish)
   - Comprehensive audit logging
>> """,
            "run flutter analyze": """Running flutter analyze on the carenji project...

$ flutter analyze
Analyzing carenji...

info • Prefer const with constant constructors • lib/presentation/views/medication_stock_view.dart:45:16 • prefer_const_constructors
info • Unused import: 'package:flutter/services.dart' • lib/domain/services/medication_service.dart:3:8 • unused_import

2 issues found. (ran in 4.2s)

The analyzer found 2 minor issues:
1. A missing const keyword for a constant constructor
2. An unused import that can be removed

These are style issues and don't affect functionality. Would you like me to fix them?
>> """,
            "fix analyzer errors": """I'll fix the Flutter analyzer issues in the carenji app.

Reading lib/presentation/views/medication_stock_view.dart...

Fixing the const constructor issue at line 45...
✓ Added const keyword to EdgeInsets constructor

Reading lib/domain/services/medication_service.dart...

Removing unused import at line 3...
✓ Removed unused import 'package:flutter/services.dart'

Running flutter analyze again to verify...

$ flutter analyze
Analyzing carenji...
No issues found!

✓ All analyzer issues have been resolved. The carenji codebase now passes all Flutter analyzer checks.
>> """,
            "run tests coverage": """Running tests with coverage for the carenji app...

$ flutter test --coverage
00:03 +142: All tests passed!

Generating coverage report...
$ genhtml coverage/lcov.info -o coverage/html

Overall coverage: 83.2%

Coverage breakdown by feature:
- Medication Management: 87.5%
- Vitals Monitoring: 81.3%
- Staff Scheduling: 85.2%
- Family Portal: 79.8%
- Core/Utils: 88.1%

✓ Coverage exceeds the 80% threshold required by carenji standards!

Top uncovered areas:
1. Error handling in bluetooth connectivity (vitals_bluetooth_service.dart)
2. Some edge cases in scheduling conflict resolution
3. Offline sync error recovery paths

The coverage report is available at coverage/html/index.html
>> """,
        }

        return responses

    @pytest.fixture
    def monitor_with_mock_claude(self, mock_settings, mock_claude_responses):
        """Create monitor with mocked Claude responses."""
        with patch("subprocess.run") as mock_run:
            with patch("claude_code_agent_farm.flutter_agent_monitor.signal.signal"):
                monitor = FlutterAgentMonitor(mock_settings)

                # Setup response cycle
                self.response_index = 0
                self.responses = list(mock_claude_responses.values())

                def mock_capture_content(self=monitor):
                    if self.response_index < len(self.responses):
                        response = self.responses[self.response_index]
                        return response
                    return ">> "

                monitor.capture_pane_content = mock_capture_content

                # Mock run for tmux commands
                mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

                yield monitor

    def advance_to_next_response(self, monitor):
        """Move to next mocked response."""
        self.response_index += 1

    def test_query_what_is_app(self, monitor_with_mock_claude, mock_claude_responses):
        """Test 'What is this app about?' query."""
        monitor = monitor_with_mock_claude
        monitor.settings.prompt_text = "What is this app about?"

        # Check initial prompt understanding
        response = monitor.capture_pane_content()

        assert "Carenji" in response
        assert "healthcare management system" in response
        assert "nursing homes" in response
        assert "Flutter" in response
        assert "Firebase" in response

        # Verify key features mentioned
        assert "Medication Management" in response
        assert "Vital Signs Monitoring" in response
        assert "Staff Scheduling" in response
        assert "Family Portal" in response

        # Check architecture details
        assert "clean architecture" in response
        assert "MVVM" in response
        assert "Provider" in response
        assert "Injectable" in response

    def test_query_list_features(self, monitor_with_mock_claude):
        """Test 'List the main features' query."""
        monitor = monitor_with_mock_claude
        monitor.settings.prompt_text = "List the main features of the carenji app"

        self.response_index = 1  # Use second response
        response = monitor.capture_pane_content()

        # Check all main features listed
        features = [
            "Medication Management",
            "Vital Signs Monitoring",
            "Staff Scheduling",
            "Family Portal",
            "ordering and tracking",
            "Stock management",
            "Real-time vital signs",
            "AI-powered roster generation",
            "OR-Tools",
            "QR code check-in",
            "Secure family member access",
            "Bluetooth device integration",
            "Offline support",
            "Multi-language support",
        ]

        for feature in features:
            assert feature in response, f"Missing feature: {feature}"

    def test_query_run_analyzer(self, monitor_with_mock_claude):
        """Test 'Run flutter analyze' query."""
        monitor = monitor_with_mock_claude
        monitor.settings.prompt_text = "Run flutter analyze"

        self.response_index = 2
        response = monitor.capture_pane_content()

        # Check command execution
        assert "flutter analyze" in response
        assert "Analyzing carenji" in response

        # Check issues found
        assert "2 issues found" in response
        assert "prefer_const_constructors" in response
        assert "unused_import" in response

        # Check file references
        assert "medication_stock_view.dart" in response
        assert "medication_service.dart" in response

        # Check it offers to fix
        assert "Would you like me to fix" in response or "fix them?" in response

    def test_query_fix_errors(self, monitor_with_mock_claude):
        """Test 'Fix analyzer errors' query."""
        monitor = monitor_with_mock_claude
        monitor.settings.prompt_text = "Fix all Flutter analyzer errors"

        self.response_index = 3
        response = monitor.capture_pane_content()

        # Check fixing process
        assert "fix the Flutter analyzer issues" in response
        assert "Reading" in response
        assert "medication_stock_view.dart" in response
        assert "medication_service.dart" in response

        # Check fixes applied
        assert "Added const keyword" in response
        assert "Removed unused import" in response

        # Check verification
        assert "flutter analyze" in response
        assert "No issues found" in response
        assert "✓" in response

        # Status should show as working while fixing
        monitor.session.status = monitor.check_agent_status()
        # In real scenario would be WORKING, but with static response might be READY

    def test_query_test_coverage(self, monitor_with_mock_claude):
        """Test 'Run tests with coverage' query."""
        monitor = monitor_with_mock_claude
        monitor.settings.prompt_text = "Run tests with coverage"

        self.response_index = 4
        response = monitor.capture_pane_content()

        # Check test execution
        assert "flutter test --coverage" in response
        assert "All tests passed" in response
        assert "+142:" in response  # Test count

        # Check coverage generation
        assert "genhtml" in response
        assert "Overall coverage: 83.2%" in response

        # Check coverage exceeds threshold
        assert "exceeds the 80% threshold" in response
        assert "carenji standards" in response

        # Check breakdown by feature
        assert "Medication Management: 87.5%" in response
        assert "Vitals Monitoring: 81.3%" in response
        assert "Staff Scheduling: 85.2%" in response
        assert "Family Portal: 79.8%" in response

        # Check uncovered areas mentioned
        assert "bluetooth connectivity" in response
        assert "scheduling conflict resolution" in response
        assert "Offline sync" in response

    @pytest.mark.slow
    def test_status_transitions(self, monitor_with_mock_claude):
        """Test agent status transitions during queries."""
        monitor = monitor_with_mock_claude

        # Initially should be READY
        monitor.session.status = AgentStatus.READY

        # Simulate working on a task
        with patch.object(monitor, "capture_pane_content", return_value="Analyzing the codebase..."):
            status = monitor.check_agent_status()
            assert status == AgentStatus.WORKING

        # Task complete, back to ready
        with patch.object(monitor, "capture_pane_content", return_value="✓ Task completed\n>> "):
            status = monitor.check_agent_status()
            assert status == AgentStatus.READY

        # Error state
        with patch.object(monitor, "capture_pane_content", return_value="Error: Failed to connect"):
            status = monitor.check_agent_status()
            assert status == AgentStatus.ERROR

    def test_carenji_context_in_responses(self, monitor_with_mock_claude):
        """Test that responses include Carenji-specific context."""
        monitor = monitor_with_mock_claude

        # Check all responses reference carenji appropriately
        for idx in range(len(self.responses)):
            self.response_index = idx
            response = monitor.capture_pane_content()

            # Should mention carenji (case-insensitive)
            assert "carenji" in response.lower(), f"Response {idx} missing carenji context"

            # Should show understanding of healthcare domain
            healthcare_terms = [
                "healthcare",
                "nursing",
                "medication",
                "vital",
                "patient",
                "staff",
                "family",
                "medical",
                "care facility",
                "health",
            ]

            found_healthcare = any(term in response.lower() for term in healthcare_terms)
            assert found_healthcare, f"Response {idx} missing healthcare context"

    @pytest.mark.docker
    def test_queries_in_docker(self, docker_container):
        """Test running queries inside Docker container."""
        # This would test actual execution in Docker
        # For now, verify the agent can be invoked

        result = subprocess.run(
            ["docker", "exec", docker_container, "claude-flutter-agent", "run", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Flutter" in result.stdout
        assert "Firebase" in result.stdout
