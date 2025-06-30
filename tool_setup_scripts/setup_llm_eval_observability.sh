#!/usr/bin/env bash

# Setup script for LLM Evaluation and Observability environment
# Installs: Evaluation frameworks, observability tools, monitoring, benchmarking

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions
PYTHON_VERSION="3.11"

main() {
    show_banner "LLM Evaluation and Observability Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "python3" "--version"
    show_tool_status "pip3" "--version"
    show_tool_status "docker" "--version"
    show_tool_status "prometheus" "--version 2>&1 | head -n 1"
    show_tool_status "grafana-server" "-v"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "build-essential" "Build Essential"
    install_apt_package "curl" "cURL"
    install_apt_package "git" "Git"
    install_apt_package "python3-dev" "Python 3 Dev"
    install_apt_package "python3-pip" "Python 3 pip"
    install_apt_package "python3-venv" "Python 3 venv"
    
    # Check Python version
    log_step "Checking Python version"
    if command_exists python3; then
        current_python=$(python3 --version | cut -d' ' -f2 | cut -d. -f1,2)
        log_info "Current Python version: $current_python"
        
        if [[ $(echo "$current_python < $PYTHON_VERSION" | bc) -eq 1 ]]; then
            log_warning "Python $current_python is older than recommended $PYTHON_VERSION"
            log_info "Consider upgrading Python for better compatibility"
        else
            log_success "Python version is sufficient"
        fi
    fi
    
    # Install Docker (required for some observability tools)
    log_step "Installing Docker"
    if ! command_exists docker; then
        if confirm "Install Docker (required for some observability tools)?"; then
            install_docker
        fi
    else
        log_info "Docker is already installed"
    fi
    
    # Create virtual environment for eval tools
    log_step "Setting up LLM Evaluation Environment"
    log_info "Creating virtual environment for evaluation tools..."
    EVAL_VENV="$HOME/.llm-eval-tools"
    if [[ ! -d "$EVAL_VENV" ]]; then
        python3 -m venv "$EVAL_VENV"
    fi
    
    # Activate virtual environment
    source "$EVAL_VENV/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install evaluation frameworks
    log_step "Installing LLM Evaluation Frameworks"
    if confirm "Install core evaluation frameworks?"; then
        # HELM (Holistic Evaluation of Language Models)
        pip install crfm-helm
        
        # Eleuther AI LM Evaluation Harness
        pip install lm-eval
        
        # RAGAS (Retrieval Augmented Generation Assessment)
        pip install ragas
        
        # DeepEval
        pip install deepeval
        
        # TruLens
        pip install trulens-eval
        
        # Promptfoo
        npm install -g promptfoo || log_warning "npm not found, skipping promptfoo"
        
        log_success "Evaluation frameworks installed"
    fi
    
    # Install observability tools
    log_step "Installing Observability Tools"
    if confirm "Install LLM observability tools?"; then
        # LangFuse (open source LLM observability)
        pip install langfuse
        
        # Phoenix by Arize
        pip install arize-phoenix
        
        # Weights & Biases
        pip install wandb
        
        # MLflow
        pip install mlflow
        
        # OpenTelemetry
        pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation
        
        # LangSmith client
        pip install langsmith
        
        log_success "Observability tools installed"
    fi
    
    # Install benchmarking tools
    log_step "Installing Benchmarking Tools"
    if confirm "Install LLM benchmarking tools?"; then
        # BigBench
        pip install bigbench
        
        # MMLU evaluation
        pip install datasets transformers
        
        # HumanEval for code generation
        pip install human-eval
        
        # Cost tracking
        pip install tiktoken litellm
        
        log_success "Benchmarking tools installed"
    fi
    
    # Install monitoring infrastructure
    log_step "Installing Monitoring Infrastructure"
    if confirm "Install Prometheus and Grafana for metrics?"; then
        install_prometheus_grafana
    fi
    
    # Install Jaeger for tracing
    if confirm "Install Jaeger for distributed tracing?"; then
        install_jaeger
    fi
    
    # Create evaluation scripts
    log_step "Creating evaluation helper scripts"
    if confirm "Create LLM evaluation helper scripts?"; then
        create_eval_scripts
    fi
    
    # Deactivate virtual environment
    deactivate
    
    # Setup dashboard templates
    log_step "Setting up monitoring dashboards"
    if confirm "Download LLM monitoring dashboard templates?"; then
        setup_dashboards
    fi
    
    # Setup shell aliases
    log_step "Setting up shell aliases"
    if confirm "Add evaluation aliases to shell?"; then
        setup_eval_aliases
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "python3" "--version"
    show_tool_status "$EVAL_VENV/bin/deepeval" "--version 2>/dev/null || echo 'Installed in venv'"
    show_tool_status "$EVAL_VENV/bin/ragas" "--version 2>/dev/null || echo 'Installed in venv'"
    show_tool_status "promptfoo" "--version"
    show_tool_status "docker" "--version"
    
    echo
    log_success "LLM Evaluation and Observability environment is ready!"
    log_info "Useful commands:"
    echo -e "  ${CYAN}source ~/.llm-eval-tools/bin/activate${RESET} - Activate evaluation tools"
    echo -e "  ${CYAN}deepeval test run${RESET} - Run DeepEval tests"
    echo -e "  ${CYAN}ragas evaluate${RESET} - Run RAGAS evaluation"
    echo -e "  ${CYAN}lm_eval --model hf --model_args pretrained=gpt2 --tasks hellaswag${RESET} - Run LM Eval Harness"
    echo -e "  ${CYAN}phoenix serve${RESET} - Start Phoenix observability server"
    echo -e "  ${CYAN}mlflow ui${RESET} - Start MLflow tracking UI"
    echo
    log_info "Monitoring URLs (if installed):"
    echo -e "  Prometheus: ${CYAN}http://localhost:9090${RESET}"
    echo -e "  Grafana: ${CYAN}http://localhost:3000${RESET} (admin/admin)"
    echo -e "  Jaeger: ${CYAN}http://localhost:16686${RESET}"
}

install_docker() {
    log_info "Installing Docker..."
    
    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Add Docker repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Update apt and install Docker
    update_apt
    install_apt_package "docker-ce" "Docker CE"
    install_apt_package "docker-ce-cli" "Docker CE CLI"
    install_apt_package "containerd.io" "containerd"
    install_apt_package "docker-compose-plugin" "Docker Compose"
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    log_success "Docker installed. You may need to log out and back in for group changes to take effect."
}

install_prometheus_grafana() {
    log_info "Setting up Prometheus and Grafana..."
    
    # Create monitoring directory
    mkdir -p "$HOME/llm-monitoring"
    cd "$HOME/llm-monitoring"
    
    # Create docker-compose.yml
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
EOF
    
    # Create Prometheus config
    cat > prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'llm-metrics'
    static_configs:
      - targets: ['localhost:8000']  # Your LLM app metrics endpoint
EOF
    
    # Create Grafana directories
    mkdir -p grafana/dashboards grafana/datasources
    
    # Create Grafana datasource config
    cat > grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
    
    log_success "Prometheus and Grafana configuration created"
    log_info "Start with: cd ~/llm-monitoring && docker compose up -d"
    
    cd "$SCRIPT_DIR"
}

install_jaeger() {
    log_info "Installing Jaeger for distributed tracing..."
    
    # Run Jaeger all-in-one via Docker
    if command_exists docker; then
        cat > "$HOME/start-jaeger.sh" << 'EOF'
#!/usr/bin/env bash
docker run -d --name jaeger \
  -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
  -p 5775:5775/udp \
  -p 6831:6831/udp \
  -p 6832:6832/udp \
  -p 5778:5778 \
  -p 16686:16686 \
  -p 14268:14268 \
  -p 14250:14250 \
  -p 9411:9411 \
  jaegertracing/all-in-one:latest
EOF
        chmod +x "$HOME/start-jaeger.sh"
        log_success "Jaeger setup script created at ~/start-jaeger.sh"
    else
        log_warning "Docker not available, skipping Jaeger installation"
    fi
}

create_eval_scripts() {
    log_info "Creating evaluation helper scripts..."
    
    # Create scripts directory
    mkdir -p "$HOME/bin"
    
    # LLM benchmark runner
    cat > "$HOME/bin/llm-benchmark" << 'EOF'
#!/usr/bin/env bash
# Run standard LLM benchmarks

source ~/.llm-eval-tools/bin/activate

if [[ $# -eq 0 ]]; then
    echo "Usage: llm-benchmark <benchmark> <model> [args...]"
    echo "Benchmarks: mmlu, humaneval, hellaswag, arc, truthfulqa"
    exit 1
fi

benchmark=$1
model=$2
shift 2

case $benchmark in
    mmlu)
        lm_eval --model $model --tasks mmlu "$@"
        ;;
    humaneval)
        lm_eval --model $model --tasks humaneval "$@"
        ;;
    hellaswag)
        lm_eval --model $model --tasks hellaswag "$@"
        ;;
    arc)
        lm_eval --model $model --tasks arc_challenge "$@"
        ;;
    truthfulqa)
        lm_eval --model $model --tasks truthfulqa "$@"
        ;;
    *)
        echo "Unknown benchmark: $benchmark"
        exit 1
        ;;
esac
EOF
    chmod +x "$HOME/bin/llm-benchmark"
    
    # LLM cost calculator
    cat > "$HOME/bin/llm-cost" << 'EOF'
#!/usr/bin/env bash
# Calculate LLM usage costs

source ~/.llm-eval-tools/bin/activate

python3 << 'PYTHON_SCRIPT'
import sys
import tiktoken
from litellm import completion_cost

def calculate_cost(model, input_tokens, output_tokens):
    try:
        cost = completion_cost(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens
        )
        return cost
    except Exception as e:
        return f"Error: {e}"

if len(sys.argv) < 4:
    print("Usage: llm-cost <model> <input_tokens> <output_tokens>")
    print("Example: llm-cost gpt-4 1000 500")
    sys.exit(1)

model = sys.argv[1]
input_tokens = int(sys.argv[2])
output_tokens = int(sys.argv[3])

cost = calculate_cost(model, input_tokens, output_tokens)
print(f"Model: {model}")
print(f"Input tokens: {input_tokens}")
print(f"Output tokens: {output_tokens}")
print(f"Estimated cost: ${cost:.4f}")
PYTHON_SCRIPT
EOF
    chmod +x "$HOME/bin/llm-cost"
    
    # Add ~/bin to PATH if not already there
    if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
        echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
        export PATH="$HOME/bin:$PATH"
    fi
    
    log_success "Evaluation helper scripts created in ~/bin"
}

setup_dashboards() {
    log_info "Setting up monitoring dashboard templates..."
    
    # Create dashboards directory
    mkdir -p "$HOME/llm-monitoring/grafana/dashboards"
    
    # Download or create LLM monitoring dashboard
    cat > "$HOME/llm-monitoring/grafana/dashboards/llm-metrics.json" << 'EOF'
{
  "dashboard": {
    "title": "LLM Metrics Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(llm_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Token Usage",
        "targets": [
          {
            "expr": "rate(llm_tokens_total[5m])"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(llm_errors_total[5m])"
          }
        ]
      }
    ]
  }
}
EOF
    
    log_success "Dashboard templates created"
}

setup_eval_aliases() {
    log_info "Setting up evaluation aliases..."
    
    eval_aliases='
# LLM Evaluation aliases
alias eval-activate="source ~/.llm-eval-tools/bin/activate"
alias eval-deepeval="source ~/.llm-eval-tools/bin/activate && deepeval"
alias eval-ragas="source ~/.llm-eval-tools/bin/activate && ragas"
alias eval-phoenix="source ~/.llm-eval-tools/bin/activate && phoenix serve"
alias eval-mlflow="source ~/.llm-eval-tools/bin/activate && mlflow ui"
alias eval-wandb="source ~/.llm-eval-tools/bin/activate && wandb"

# Quick evaluation commands
alias llm-eval-quick="source ~/.llm-eval-tools/bin/activate && deepeval test run"
alias llm-monitor-start="cd ~/llm-monitoring && docker compose up -d"
alias llm-monitor-stop="cd ~/llm-monitoring && docker compose down"
alias llm-jaeger="~/start-jaeger.sh"

# Evaluation project setup
eval-new-project() {
    if [[ -z "$1" ]]; then
        echo "Usage: eval-new-project <project-name>"
        return 1
    fi
    mkdir -p "$1" && cd "$1"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install deepeval ragas langfuse mlflow pytest
    
    # Create test structure
    mkdir -p tests/evaluation
    cat > tests/evaluation/test_llm_quality.py << "TEST_FILE"
import deepeval
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

def test_answer_relevancy():
    metric = AnswerRelevancyMetric(threshold=0.7)
    test_case = LLMTestCase(
        input="What is machine learning?",
        actual_output="Machine learning is a subset of AI...",
        expected_output="Machine learning is a field of artificial intelligence..."
    )
    assert metric.measure(test_case)
TEST_FILE
    
    echo "# $1 - LLM Evaluation Project" > README.md
    echo ".venv/" > .gitignore
    echo "__pycache__/" >> .gitignore
    echo "*.pyc" >> .gitignore
    echo ".env" >> .gitignore
    echo "mlruns/" >> .gitignore
    
    git init
    echo "Evaluation project $1 created!"
}'
    
    if [[ -f "$HOME/.bashrc" ]]; then
        add_to_file_if_missing "$HOME/.bashrc" "$eval_aliases" "LLM evaluation aliases"
    fi
    
    if [[ -f "$HOME/.zshrc" ]]; then
        add_to_file_if_missing "$HOME/.zshrc" "$eval_aliases" "LLM evaluation aliases"
    fi
    
    log_success "Evaluation aliases added to shell"
}

# Run main function
main "$@" 