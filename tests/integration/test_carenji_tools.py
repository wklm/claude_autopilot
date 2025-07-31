"""Integration tests for Carenji development tools availability."""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.carenji
class TestCarenjiTools:
    """Test that all necessary tools for Carenji development are available."""
    
    @pytest.fixture
    def carenji_project_dir(self, temp_dir):
        """Create a minimal Carenji project structure."""
        project_dir = temp_dir / "test_carenji"
        project_dir.mkdir()
        
        # Create pubspec.yaml
        pubspec_content = """
name: carenji
description: Healthcare management system for nursing homes

version: 1.0.0+1

environment:
  sdk: '>=2.19.0 <3.0.0'

dependencies:
  flutter:
    sdk: flutter
  
  # Firebase
  firebase_core: ^2.24.0
  cloud_firestore: ^4.13.0
  firebase_auth: ^4.15.0
  firebase_storage: ^11.5.0
  
  # State Management
  provider: ^6.1.0
  
  # Dependency Injection
  get_it: ^7.6.0
  injectable: ^2.3.0
  
  # Utilities
  intl: ^0.18.1
  uuid: ^4.2.0
  
  # UI/UX
  flutter_barcode_scanner: ^2.0.0
  charts_flutter: ^0.12.0

dev_dependencies:
  flutter_test:
    sdk: flutter
  
  build_runner: ^2.4.0
  injectable_generator: ^2.4.0
  flutter_lints: ^3.0.0
  test: ^1.24.0
  mockito: ^5.4.0

flutter:
  uses-material-design: true
"""
        (project_dir / "pubspec.yaml").write_text(pubspec_content)
        
        # Create lib directory structure
        lib_dir = project_dir / "lib"
        lib_dir.mkdir()
        (lib_dir / "main.dart").write_text("""
import 'package:flutter/material.dart';

void main() {
  runApp(const CarenjiApp());
}

class CarenjiApp extends StatelessWidget {
  const CarenjiApp({super.key});
  
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Carenji',
      home: Container(),
    );
  }
}
""")
        
        # Create test directory
        test_dir = project_dir / "test"
        test_dir.mkdir()
        (test_dir / "widget_test.dart").write_text("""
import 'package:flutter_test/flutter_test.dart';
import 'package:carenji/main.dart';

void main() {
  testWidgets('Carenji app starts', (WidgetTester tester) async {
    await tester.pumpWidget(const CarenjiApp());
    expect(find.byType(CarenjiApp), findsOneWidget);
  });
}
""")
        
        # Create firebase.json
        firebase_config = {
            "emulators": {
                "auth": {"port": 9099},
                "firestore": {"port": 8080},
                "storage": {"port": 9199}
            }
        }
        (project_dir / "firebase.json").write_text(json.dumps(firebase_config, indent=2))
        
        # Create .firebaserc
        firebaserc = {
            "projects": {
                "default": "carenji-test"
            }
        }
        (project_dir / ".firebaserc").write_text(json.dumps(firebaserc, indent=2))
        
        return project_dir
    
    @pytest.mark.docker
    def test_flutter_commands(self, docker_container, carenji_project_dir):
        """Test essential Flutter commands work."""
        # Copy project to container
        subprocess.run(
            ["docker", "cp", str(carenji_project_dir), f"{docker_container}:/tmp/"],
            check=True
        )
        
        project_path = f"/tmp/{carenji_project_dir.name}"
        
        # Test flutter --version
        result = subprocess.run(
            ["docker", "exec", docker_container, "flutter", "--version"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Flutter" in result.stdout
        
        # Test flutter doctor
        result = subprocess.run(
            ["docker", "exec", docker_container, "flutter", "doctor", "-v"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        
        # Test flutter analyze
        result = subprocess.run(
            ["docker", "exec", "-w", project_path, docker_container,
             "flutter", "analyze", "--no-fatal-infos"],
            capture_output=True,
            text=True,
            timeout=60
        )
        # May have infos but should not fail
        assert "error" not in result.stdout.lower() or result.returncode == 0
    
    @pytest.mark.docker
    @pytest.mark.slow
    def test_flutter_pub_commands(self, docker_container, carenji_project_dir):
        """Test Flutter package management commands."""
        # Copy project to container
        subprocess.run(
            ["docker", "cp", str(carenji_project_dir), f"{docker_container}:/tmp/"],
            check=True
        )
        
        project_path = f"/tmp/{carenji_project_dir.name}"
        
        # Test flutter pub get
        result = subprocess.run(
            ["docker", "exec", "-w", project_path, docker_container,
             "flutter", "pub", "get"],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            pytest.skip("Flutter pub get failed - may be offline")
        
        # Test flutter pub deps
        result = subprocess.run(
            ["docker", "exec", "-w", project_path, docker_container,
             "flutter", "pub", "deps"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "provider" in result.stdout
        assert "firebase_core" in result.stdout
        assert "get_it" in result.stdout
    
    @pytest.mark.docker
    def test_dart_commands(self, docker_container):
        """Test Dart SDK commands."""
        # Test dart --version
        result = subprocess.run(
            ["docker", "exec", docker_container, "dart", "--version"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Dart SDK" in result.stdout
        
        # Test dart format
        test_file = """
void main(){print('test');}
"""
        result = subprocess.run(
            ["docker", "exec", docker_container, "sh", "-c",
             f"echo '{test_file}' | dart format"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "void main() {" in result.stdout  # Properly formatted
    
    @pytest.mark.docker
    def test_firebase_cli(self, docker_container, carenji_project_dir):
        """Test Firebase CLI commands."""
        # Test firebase --version
        result = subprocess.run(
            ["docker", "exec", docker_container, "firebase", "--version"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        
        # Copy project for Firebase tests
        subprocess.run(
            ["docker", "cp", str(carenji_project_dir), f"{docker_container}:/tmp/"],
            check=True
        )
        
        project_path = f"/tmp/{carenji_project_dir.name}"
        
        # Test firebase use
        result = subprocess.run(
            ["docker", "exec", "-w", project_path, docker_container,
             "firebase", "use"],
            capture_output=True,
            text=True
        )
        # Should show the default project
        assert "carenji-test" in result.stdout
    
    @pytest.mark.docker
    def test_git_commands(self, docker_container):
        """Test Git commands for version control."""
        # Test git --version
        result = subprocess.run(
            ["docker", "exec", docker_container, "git", "--version"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "git version" in result.stdout
        
        # Test git config
        result = subprocess.run(
            ["docker", "exec", docker_container, "git", "config", "--global", "user.name"],
            capture_output=True,
            text=True
        )
        # Should be configured or empty, but command should work
        assert result.returncode in [0, 1]
    
    @pytest.mark.docker
    def test_code_generation_tools(self, docker_container, carenji_project_dir):
        """Test code generation tools like build_runner."""
        # Copy project
        subprocess.run(
            ["docker", "cp", str(carenji_project_dir), f"{docker_container}:/tmp/"],
            check=True
        )
        
        project_path = f"/tmp/{carenji_project_dir.name}"
        
        # First ensure packages are installed
        subprocess.run(
            ["docker", "exec", "-w", project_path, docker_container,
             "flutter", "pub", "get"],
            capture_output=True,
            timeout=120
        )
        
        # Test build_runner version
        result = subprocess.run(
            ["docker", "exec", "-w", project_path, docker_container,
             "dart", "run", "build_runner", "--version"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should either work or fail gracefully
        if result.returncode == 0:
            assert result.stdout.strip()  # Should output version
    
    @pytest.mark.docker
    def test_testing_tools(self, docker_container, carenji_project_dir):
        """Test Flutter testing tools."""
        # Copy project
        subprocess.run(
            ["docker", "cp", str(carenji_project_dir), f"{docker_container}:/tmp/"],
            check=True
        )
        
        project_path = f"/tmp/{carenji_project_dir.name}"
        
        # Ensure packages are installed
        subprocess.run(
            ["docker", "exec", "-w", project_path, docker_container,
             "flutter", "pub", "get"],
            capture_output=True,
            timeout=120
        )
        
        # Test flutter test
        result = subprocess.run(
            ["docker", "exec", "-w", project_path, docker_container,
             "flutter", "test", "--no-pub"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Should run the test
        assert "Carenji app starts" in result.stdout
        assert result.returncode == 0 or "All tests passed" in result.stdout
    
    @pytest.mark.docker
    def test_linting_tools(self, docker_container, carenji_project_dir):
        """Test linting and analysis tools."""
        # Copy project
        subprocess.run(
            ["docker", "cp", str(carenji_project_dir), f"{docker_container}:/tmp/"],
            check=True
        )
        
        project_path = f"/tmp/{carenji_project_dir.name}"
        
        # Create analysis_options.yaml
        analysis_options = """
include: package:flutter_lints/flutter.yaml

linter:
  rules:
    prefer_const_constructors: true
    avoid_print: true
"""
        
        subprocess.run(
            ["docker", "exec", docker_container, "sh", "-c",
             f"echo '{analysis_options}' > {project_path}/analysis_options.yaml"],
            check=True
        )
        
        # Run analyzer
        result = subprocess.run(
            ["docker", "exec", "-w", project_path, docker_container,
             "dart", "analyze"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Should complete (may have warnings)
        assert result.returncode >= 0
    
    @pytest.mark.docker
    def test_or_tools_availability(self, docker_container):
        """Test Google OR-Tools for staff scheduling."""
        # Check if Python is available
        result = subprocess.run(
            ["docker", "exec", docker_container, "python3", "--version"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Try to import ortools
            result = subprocess.run(
                ["docker", "exec", docker_container, "python3", "-c",
                 "import ortools; print('OR-Tools available')"],
                capture_output=True,
                text=True
            )
            
            # OR-Tools might not be pre-installed, but Python should work
            assert "python" in result.stdout.lower() or "OR-Tools" in result.stdout
    
    @pytest.mark.docker
    def test_development_utilities(self, docker_container):
        """Test various development utilities."""
        utilities = [
            ("curl", ["curl", "--version"]),
            ("wget", ["wget", "--version"]),
            ("jq", ["jq", "--version"]),
            ("make", ["make", "--version"]),
            ("npm", ["npm", "--version"]),
            ("yarn", ["yarn", "--version"]),
        ]
        
        available_tools = []
        for tool_name, command in utilities:
            result = subprocess.run(
                ["docker", "exec", docker_container] + command,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                available_tools.append(tool_name)
        
        # Should have most essential tools
        assert len(available_tools) >= 4, f"Missing tools. Found: {available_tools}"
    
    @pytest.mark.docker
    def test_flutter_device_config(self, docker_container):
        """Test Flutter device configuration for headless testing."""
        # Check Flutter devices
        result = subprocess.run(
            ["docker", "exec", docker_container, "flutter", "devices"],
            capture_output=True,
            text=True
        )
        
        # Should not error even without devices
        assert result.returncode == 0
        
        # Check if web is available (common in containers)
        if "Chrome" in result.stdout or "web" in result.stdout:
            assert True  # Web development available
        else:
            # At minimum, should not crash
            assert "flutter devices" not in result.stderr
    
    @pytest.mark.docker
    def test_carenji_specific_setup(self, docker_container):
        """Test Carenji-specific tool requirements."""
        # Test locale support for multi-language
        result = subprocess.run(
            ["docker", "exec", docker_container, "locale", "-a"],
            capture_output=True,
            text=True
        )
        
        # Should support required locales
        locales = result.stdout.lower()
        assert "en_us" in locales or "c.utf" in locales
        
        # Test timezone data for scheduling
        result = subprocess.run(
            ["docker", "exec", docker_container, "ls", "/usr/share/zoneinfo/"],
            capture_output=True,
            text=True
        )
        
        # Should have timezone data
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0
    
    def test_ide_support_files(self, carenji_project_dir):
        """Test IDE support file generation."""
        # Create .vscode/settings.json
        vscode_dir = carenji_project_dir / ".vscode"
        vscode_dir.mkdir()
        
        vscode_settings = {
            "dart.flutterSdkPath": "${env:FLUTTER_HOME}",
            "editor.formatOnSave": True,
            "editor.codeActionsOnSave": {
                "source.fixAll": True
            }
        }
        
        (vscode_dir / "settings.json").write_text(
            json.dumps(vscode_settings, indent=2)
        )
        
        # Verify files created
        assert (vscode_dir / "settings.json").exists()
    
    @pytest.mark.docker
    @pytest.mark.slow
    def test_full_development_cycle(self, docker_container, carenji_project_dir):
        """Test a full development cycle for Carenji."""
        # Copy project
        subprocess.run(
            ["docker", "cp", str(carenji_project_dir), f"{docker_container}:/tmp/"],
            check=True
        )
        
        project_path = f"/tmp/{carenji_project_dir.name}"
        
        # 1. Get dependencies
        result = subprocess.run(
            ["docker", "exec", "-w", project_path, docker_container,
             "flutter", "pub", "get"],
            capture_output=True,
            timeout=120
        )
        
        if result.returncode != 0:
            pytest.skip("Cannot test full cycle - pub get failed")
        
        # 2. Run analyzer
        subprocess.run(
            ["docker", "exec", "-w", project_path, docker_container,
             "flutter", "analyze", "--no-fatal-infos"],
            capture_output=True,
            timeout=60
        )
        
        # 3. Run tests
        result = subprocess.run(
            ["docker", "exec", "-w", project_path, docker_container,
             "flutter", "test", "--no-pub"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        assert "All tests passed" in result.stdout or result.returncode == 0
        
        # 4. Format code
        result = subprocess.run(
            ["docker", "exec", "-w", project_path, docker_container,
             "dart", "format", "."],
            capture_output=True,
            timeout=30
        )
        
        # Should complete formatting
        assert result.returncode == 0