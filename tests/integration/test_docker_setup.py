"""Integration tests for Docker container setup."""

import os
import subprocess
import time
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.docker
class TestDockerSetup:
    """Test Docker container setup and tools availability."""
    
    @pytest.fixture(scope="class")
    def docker_image_built(self):
        """Ensure Docker image is built."""
        # Check if image exists
        result = subprocess.run(
            ["docker", "images", "-q", "claude-flutter-firebase-agent:latest"],
            capture_output=True,
            text=True
        )
        
        if not result.stdout.strip():
            # Build image
            build_result = subprocess.run(
                ["docker", "build", "-t", "claude-flutter-firebase-agent:latest", "."],
                cwd=Path(__file__).parent.parent.parent,
                capture_output=True,
                text=True
            )
            
            if build_result.returncode != 0:
                pytest.fail(f"Failed to build Docker image: {build_result.stderr}")
        
        return True
    
    @pytest.fixture
    def docker_container(self, docker_image_built, temp_dir):
        """Run a test Docker container."""
        container_name = f"test-carenji-agent-{os.getpid()}"
        
        # Create a mock workspace
        workspace = temp_dir / "workspace"
        workspace.mkdir()
        (workspace / "pubspec.yaml").write_text("name: test_app\n")
        
        # Start container
        run_cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "-v", f"{workspace}:/workspace",
            "-e", "CLAUDE_PROJECT_PATH=/workspace",
            "-e", "CLAUDE_PROMPT_TEXT=test",
            "claude-flutter-firebase-agent:latest",
            "sleep", "3600"  # Keep container running for tests
        ]
        
        result = subprocess.run(run_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            pytest.fail(f"Failed to start container: {result.stderr}")
        
        # Wait for container to be ready
        time.sleep(2)
        
        yield container_name
        
        # Cleanup
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)
    
    def test_flutter_sdk_installed(self, docker_container):
        """Test that Flutter SDK is installed and working."""
        result = subprocess.run(
            ["docker", "exec", docker_container, "flutter", "--version"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Flutter" in result.stdout
        assert "Dart" in result.stdout
    
    def test_flutter_doctor(self, docker_container):
        """Test Flutter doctor output."""
        result = subprocess.run(
            ["docker", "exec", docker_container, "flutter", "doctor", "-v"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Flutter" in result.stdout
        # Should have Flutter installed
        assert "✓" in result.stdout or "[✓]" in result.stdout
    
    def test_firebase_cli_installed(self, docker_container):
        """Test that Firebase CLI is installed."""
        result = subprocess.run(
            ["docker", "exec", docker_container, "firebase", "--version"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert result.stdout.strip()  # Should output version
    
    def test_tmux_installed(self, docker_container):
        """Test that tmux is installed and configured."""
        # Check tmux version
        result = subprocess.run(
            ["docker", "exec", docker_container, "tmux", "-V"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "tmux" in result.stdout
        
        # Check tmux config exists
        result = subprocess.run(
            ["docker", "exec", docker_container, "test", "-f", "/home/claude/.tmux.conf"],
            capture_output=True
        )
        
        assert result.returncode == 0
    
    def test_python_agent_installed(self, docker_container):
        """Test that the Python agent is installed."""
        # Check if claude-flutter-agent command exists
        result = subprocess.run(
            ["docker", "exec", docker_container, "which", "claude-flutter-agent"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert result.stdout.strip()
        
        # Test help command
        result = subprocess.run(
            ["docker", "exec", docker_container, "claude-flutter-agent", "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Claude Flutter Firebase Agent" in result.stdout
    
    def test_git_installed(self, docker_container):
        """Test that git is installed."""
        result = subprocess.run(
            ["docker", "exec", docker_container, "git", "--version"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "git version" in result.stdout
    
    def test_node_installed(self, docker_container):
        """Test that Node.js is installed."""
        result = subprocess.run(
            ["docker", "exec", docker_container, "node", "--version"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert result.stdout.startswith("v")
        
        # Check npm
        result = subprocess.run(
            ["docker", "exec", docker_container, "npm", "--version"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
    
    def test_build_tools_installed(self, docker_container):
        """Test that build tools are installed."""
        tools = ["make", "gcc", "cmake", "ninja"]
        
        for tool in tools:
            result = subprocess.run(
                ["docker", "exec", docker_container, "which", tool],
                capture_output=True
            )
            
            assert result.returncode == 0, f"{tool} not found"
    
    def test_user_permissions(self, docker_container):
        """Test that claude user has correct permissions."""
        # Check user
        result = subprocess.run(
            ["docker", "exec", docker_container, "whoami"],
            capture_output=True,
            text=True
        )
        
        assert result.stdout.strip() == "claude"
        
        # Check sudo access
        result = subprocess.run(
            ["docker", "exec", docker_container, "sudo", "whoami"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert result.stdout.strip() == "root"
    
    def test_workspace_mounted(self, docker_container):
        """Test that workspace is properly mounted."""
        # Check if /workspace exists
        result = subprocess.run(
            ["docker", "exec", docker_container, "test", "-d", "/workspace"],
            capture_output=True
        )
        
        assert result.returncode == 0
        
        # Check if pubspec.yaml is accessible
        result = subprocess.run(
            ["docker", "exec", docker_container, "test", "-f", "/workspace/pubspec.yaml"],
            capture_output=True
        )
        
        assert result.returncode == 0
    
    def test_environment_variables(self, docker_container):
        """Test that environment variables are set correctly."""
        # Check Flutter home
        result = subprocess.run(
            ["docker", "exec", docker_container, "printenv", "FLUTTER_HOME"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert result.stdout.strip()
        
        # Check PATH includes Flutter
        result = subprocess.run(
            ["docker", "exec", docker_container, "printenv", "PATH"],
            capture_output=True,
            text=True
        )
        
        assert "flutter/bin" in result.stdout
    
    def test_entrypoint_script(self, docker_container):
        """Test that entrypoint script exists and is executable."""
        result = subprocess.run(
            ["docker", "exec", docker_container, "test", "-x", "/home/claude/entrypoint.sh"],
            capture_output=True
        )
        
        assert result.returncode == 0
    
    @pytest.mark.slow
    def test_flutter_pub_get(self, docker_container):
        """Test that Flutter can download packages."""
        # Create a simple Flutter project
        result = subprocess.run(
            ["docker", "exec", docker_container, "sh", "-c",
             "cd /tmp && flutter create test_app --project-name test_app"],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            pytest.skip("Flutter create failed - may be offline")
        
        # Run pub get
        result = subprocess.run(
            ["docker", "exec", docker_container, "sh", "-c",
             "cd /tmp/test_app && flutter pub get"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        assert result.returncode == 0
    
    def test_dart_analyzer(self, docker_container):
        """Test that Dart analyzer is available."""
        # Create a simple Dart file
        result = subprocess.run(
            ["docker", "exec", docker_container, "sh", "-c",
             "echo 'void main() { print(\"test\"); }' > /tmp/test.dart"],
            capture_output=True
        )
        
        # Run analyzer
        result = subprocess.run(
            ["docker", "exec", docker_container, "dart", "analyze", "/tmp/test.dart"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "No issues found" in result.stdout
    
    def test_ports_exposed(self):
        """Test that necessary ports are exposed in the image."""
        result = subprocess.run(
            ["docker", "image", "inspect", "claude-flutter-firebase-agent:latest",
             "--format", "{{json .Config.ExposedPorts}}"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        exposed_ports = result.stdout
        
        # Check key ports
        assert "8181" in exposed_ports  # Flutter DDS
        assert "8182" in exposed_ports  # Flutter VM Service
        assert "9100" in exposed_ports  # Flutter DevTools
        assert "4000" in exposed_ports or "4001" in exposed_ports  # Firebase UI