#!/usr/bin/env bash

# Setup script for GenAI/LLM Ops development environment
# Installs: Python 3.13+, uv, CUDA tools, vLLM, LangGraph, vector databases, and ML tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions and requirements
PYTHON_MIN_VERSION="3.13"
UV_VERSION="latest"
CUDA_MIN_VERSION="12.1"
OLLAMA_VERSION="latest"
DOCKER_COMPOSE_VERSION="2.29.7"

main() {
    show_banner "GenAI/LLM Ops Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "python3" "--version"
    show_tool_status "uv" "--version"
    show_tool_status "nvidia-smi" "--version 2>/dev/null | grep 'Driver Version'"
    show_tool_status "docker" "--version"
    show_tool_status "ollama" "--version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "build-essential" "Build Essential"
    install_apt_package "curl" "cURL"
    install_apt_package "git" "Git"
    install_apt_package "python3-dev" "Python 3 Development Headers"
    install_apt_package "python3-pip" "pip"
    install_apt_package "python3-venv" "Python venv"
    install_apt_package "libssl-dev" "SSL Development Libraries"
    install_apt_package "libffi-dev" "FFI Development Libraries"
    
    # Check Python version
    log_step "Checking Python version"
    if command_exists python3; then
        current_python=$(python3 --version | awk '{print $2}')
        log_info "Current Python version: $current_python"
        
        # Compare versions
        if [[ "$(printf '%s\n' "$PYTHON_MIN_VERSION" "$current_python" | sort -V | head -n1)" != "$PYTHON_MIN_VERSION" ]]; then
            log_warning "Python $current_python is older than recommended $PYTHON_MIN_VERSION"
            
            if confirm "Install Python $PYTHON_MIN_VERSION from deadsnakes PPA?"; then
                # Add deadsnakes PPA
                sudo add-apt-repository -y ppa:deadsnakes/ppa
                sudo apt-get update
                install_apt_package "python${PYTHON_MIN_VERSION}" "Python ${PYTHON_MIN_VERSION}"
                install_apt_package "python${PYTHON_MIN_VERSION}-dev" "Python ${PYTHON_MIN_VERSION} Dev"
                install_apt_package "python${PYTHON_MIN_VERSION}-venv" "Python ${PYTHON_MIN_VERSION} venv"
                
                # Update alternatives
                if confirm "Set Python ${PYTHON_MIN_VERSION} as default python3?"; then
                    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_MIN_VERSION} 1
                    sudo update-alternatives --config python3
                fi
            fi
        else
            log_success "Python version is sufficient"
        fi
    else
        log_error "Python 3 is not installed"
        install_apt_package "python3" "Python 3"
    fi
    
    # Check for NVIDIA GPU and CUDA
    log_step "Checking GPU and CUDA support"
    if command_exists nvidia-smi; then
        log_info "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        
        # Check CUDA version
        if command_exists nvcc; then
            cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d, -f1)
            log_info "CUDA version: $cuda_version"
            
            if [[ "$(printf '%s\n' "$CUDA_MIN_VERSION" "$cuda_version" | sort -V | head -n1)" != "$CUDA_MIN_VERSION" ]]; then
                log_warning "CUDA $cuda_version is older than recommended $CUDA_MIN_VERSION"
                log_info "Please update CUDA from: https://developer.nvidia.com/cuda-downloads"
            fi
        else
            log_warning "CUDA toolkit not found"
            log_info "For GPU acceleration, install CUDA from: https://developer.nvidia.com/cuda-downloads"
        fi
    else
        log_warning "No NVIDIA GPU detected. CPU inference will be slower."
        if confirm "Continue with CPU-only setup?"; then
            log_info "Continuing with CPU-only setup"
        else
            exit 1
        fi
    fi
    
    # Install Docker for vector databases and services
    log_step "Setting up Docker"
    if ! command_exists docker; then
        if confirm "Install Docker for running vector databases and services?"; then
            # Add Docker's official GPG key
            sudo apt-get install -y ca-certificates gnupg
            sudo install -m 0755 -d /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
            sudo chmod a+r /etc/apt/keyrings/docker.gpg
            
            # Add the repository
            echo \
              "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
              $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
              sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
            
            # Install Docker
            sudo apt-get update
            sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
            
            # Add user to docker group
            sudo usermod -aG docker $USER
            log_success "Docker installed. Please log out and back in for group changes to take effect."
        fi
    else
        log_info "Docker is already installed"
    fi
    
    # Install Docker Compose standalone
    if ! command_exists docker-compose; then
        if confirm "Install Docker Compose standalone?"; then
            sudo curl -L "https://github.com/docker/compose/releases/download/v${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            sudo chmod +x /usr/local/bin/docker-compose
            log_success "Docker Compose installed"
        fi
    fi
    
    # Install Ollama for local model serving
    log_step "Setting up Ollama"
    if ! command_exists ollama; then
        if confirm "Install Ollama for local model serving?"; then
            curl -fsSL https://ollama.com/install.sh | sh
            
            # Start Ollama service
            sudo systemctl enable ollama
            sudo systemctl start ollama
            
            log_success "Ollama installed"
            log_info "Ollama service is running"
            
            # Pull popular models
            if confirm "Pull popular models (llama3.2, phi3)?"; then
                ollama pull llama3.2
                ollama pull phi3
                log_success "Models downloaded"
            fi
        fi
    else
        log_info "Ollama is already installed"
    fi
    
    # Install uv
    log_step "Installing uv (Fast Python package manager)"
    if command_exists uv; then
        log_info "uv is already installed"
        if confirm "Update uv to latest version?"; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            log_success "uv updated"
        fi
    else
        if confirm "Install uv (recommended for modern Python development)?"; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            
            # Add uv to PATH for different shells
            uv_path_line='export PATH="$HOME/.local/bin:$PATH"'
            
            if [[ -f "$HOME/.bashrc" ]]; then
                add_to_file_if_missing "$HOME/.bashrc" "$uv_path_line" "uv PATH"
            fi
            
            if [[ -f "$HOME/.zshrc" ]]; then
                add_to_file_if_missing "$HOME/.zshrc" "$uv_path_line" "uv PATH"
            fi
            
            # Source for current session
            export PATH="$HOME/.local/bin:$PATH"
            log_success "uv installed"
            log_info "You may need to restart your shell or run: source ~/.bashrc"
        fi
    fi
    
    # Install Python LLM tools using uv
    if command_exists uv; then
        log_step "Setting up LLM Ops Python environment"
        
        # Create project directory
        if confirm "Create LLM Ops project environment?"; then
            llm_project="$HOME/llm-ops-env"
            create_directory "$llm_project" "LLM Ops project"
            
            # Copy pyproject.toml template
            cp "$SCRIPT_DIR/llm_ops_pyproject.toml" "$llm_project/pyproject.toml"
            
            # Create README
            cat > "$llm_project/README.md" << 'EOF'
# LLM Ops Development Environment

This environment includes all tools for LLM/GenAI development including:
- LangChain, LangGraph, LlamaIndex
- OpenAI, Anthropic, Google AI SDKs
- Vector databases (ChromaDB, Qdrant, Weaviate)
- FastAPI for building APIs
- Development tools (ruff, mypy, pre-commit)

## Usage

Activate the environment:
```bash
source .venv/bin/activate
```

## GPU Support

If you have CUDA installed, install GPU packages:
```bash
uv sync --extra gpu
```

## Development Tools

Install development tools:
```bash
uv sync --extra dev
```

## Interactive Tools

Install Gradio, Streamlit, and Jupyter:
```bash
uv sync --extra interactive
```
EOF
            
            cd "$llm_project"
            
            # Create virtual environment with Python 3.13
            log_info "Creating virtual environment with Python 3.13..."
            if command_exists "python${PYTHON_MIN_VERSION}"; then
                uv venv --python "${PYTHON_MIN_VERSION}"
            else
                uv venv --python python3
            fi
            
            # Install base dependencies
            log_info "Installing LLM/GenAI packages (this may take a while)..."
            uv sync
            
            # Check if GPU is available and install GPU packages
            if command_exists nvidia-smi; then
                log_info "GPU detected. Installing GPU-accelerated packages..."
                uv sync --extra gpu
            else
                log_info "No GPU detected. Skipping GPU packages."
            fi
            
            # Install development tools
            if confirm "Install development tools (ruff, mypy, pre-commit)?"; then
                uv sync --extra dev
                log_success "Development tools installed"
            fi
            
            # Install interactive tools
            if confirm "Install interactive tools (Gradio, Streamlit, Jupyter)?"; then
                uv sync --extra interactive
                log_success "Interactive tools installed"
            fi
            
            # Create activation helper
            activate_script="$HOME/activate-llm-env.sh"
            cat > "$activate_script" << EOF
#!/bin/bash
# Activate LLM Ops environment
source $llm_project/.venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$llm_project:\$PYTHONPATH"

# Set CUDA visibility if GPU available
if command -v nvidia-smi &> /dev/null; then
    export CUDA_VISIBLE_DEVICES=0
fi

echo "LLM Ops environment activated!"
echo "Project directory: $llm_project"
echo "Python version: \$(python --version)"
echo ""
echo "Available frameworks:"
echo "  - LangChain, LangGraph, LlamaIndex"
echo "  - Vector DBs: ChromaDB, Qdrant, Weaviate"
echo "  - Model providers: OpenAI, Anthropic, Google, Ollama"
echo ""
echo "Commands:"
echo "  - ruff check .    # Lint code"
echo "  - mypy .         # Type check"
echo "  - python         # Start Python REPL"
EOF
            chmod +x "$activate_script"
            
            # Create .envrc for direnv
            cat > "$llm_project/.envrc" << 'EOF'
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
EOF
            
            # Create example .env file
            cat > "$llm_project/.env.example" << 'EOF'
# API Keys
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
GOOGLE_API_KEY=your-google-key-here

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/llmops

# Redis
REDIS_URL=redis://localhost:6379

# Vector Database URLs
QDRANT_URL=http://localhost:6333
WEAVIATE_URL=http://localhost:8080
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Monitoring
WANDB_API_KEY=your-wandb-key-here
MLFLOW_TRACKING_URI=http://localhost:5000

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO
EOF
            
            cd - > /dev/null
            
            log_success "LLM Ops environment created at $llm_project"
            log_info "To activate: source ~/activate-llm-env.sh"
        fi
        
        # Install global uv tools
        log_step "Installing global development tools"
        if confirm "Install global tools (ruff, mypy, pre-commit)?"; then
            uv tool install ruff
            uv tool install mypy
            uv tool install pre-commit
            log_success "Global tools installed"
        fi
    fi
    
    # Setup vector databases with Docker Compose
    log_step "Setting up Vector Database Stack"
    if command_exists docker && confirm "Create docker-compose for vector databases?"; then
        vdb_dir="$HOME/vector-databases"
        create_directory "$vdb_dir" "Vector database directory"
        
        # Create docker-compose.yml
        cat > "$vdb_dir/docker-compose.yml" << 'EOF'
version: '3.8'

services:
  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334

  # Weaviate Vector Database
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
      - "50051:50051"
    volumes:
      - ./weaviate_data:/var/lib/weaviate
    environment:
      - QUERY_DEFAULTS_LIMIT=20
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
      - PERSISTENCE_DATA_PATH=/var/lib/weaviate
      - DEFAULT_VECTORIZER_MODULE=text2vec-transformers
      - ENABLE_MODULES=text2vec-transformers
      - TRANSFORMERS_INFERENCE_API=http://t2v-transformers:8080

  # Text2Vec Transformers for Weaviate
  t2v-transformers:
    image: cr.weaviate.io/semitechnologies/transformers-inference:sentence-transformers-multi-qa-MiniLM-L6-cos-v1
    environment:
      - ENABLE_CUDA=0  # Set to 1 if GPU available

  # ChromaDB Vector Database
  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - ./chroma_data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - PERSIST_DIRECTORY=/chroma/chroma
      - ANONYMIZED_TELEMETRY=FALSE

  # MinIO for S3-compatible storage
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - ./minio_data:/data
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    command: server /data --console-address ":9001"

  # PostgreSQL for metadata and pgvector
  postgres:
    image: pgvector/pgvector:pg16
    ports:
      - "5432:5432"
    volumes:
      - ./postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=llmops
      - POSTGRES_PASSWORD=llmops
      - POSTGRES_DB=llmops
EOF
        
        # Create start/stop scripts
        cat > "$vdb_dir/start-databases.sh" << 'EOF'
#!/bin/bash
echo "Starting vector databases..."
docker-compose up -d
echo "Services started:"
echo "  - Qdrant: http://localhost:6333"
echo "  - Weaviate: http://localhost:8080"
echo "  - ChromaDB: http://localhost:8000"
echo "  - MinIO: http://localhost:9001 (minioadmin/minioadmin)"
echo "  - PostgreSQL: localhost:5432 (llmops/llmops)"
EOF
        chmod +x "$vdb_dir/start-databases.sh"
        
        cat > "$vdb_dir/stop-databases.sh" << 'EOF'
#!/bin/bash
echo "Stopping vector databases..."
docker-compose down
echo "Services stopped."
EOF
        chmod +x "$vdb_dir/stop-databases.sh"
        
        log_success "Vector database stack configured"
        log_info "To start databases: cd ~/vector-databases && ./start-databases.sh"
    fi
    
    # Configure Git for LLM development
    log_step "Configuring Git for LLM development"
    if confirm "Add LLM/ML patterns to global .gitignore?"; then
        gitignore_file="$HOME/.gitignore_global"
        touch "$gitignore_file"
        
        # LLM/ML patterns
        ml_patterns=(
            "*.ckpt"
            "*.pt"
            "*.pth"
            "*.pkl"
            "*.h5"
            "*.safetensors"
            "*.gguf"
            "*.ggml"
            "models/"
            "checkpoints/"
            ".chroma/"
            ".qdrant/"
            ".weaviate/"
            ".cache/"
            "wandb/"
            "mlruns/"
            ".ollama/"
            "*.log"
            ".ipynb_checkpoints/"
            "__pycache__/"
            ".pytest_cache/"
            ".mypy_cache/"
            ".ruff_cache/"
            ".env"
            ".env.local"
            "*.db"
            "*.sqlite"
        )
        
        for pattern in "${ml_patterns[@]}"; do
            if ! grep -Fxq "$pattern" "$gitignore_file"; then
                echo "$pattern" >> "$gitignore_file"
            fi
        done
        
        git config --global core.excludesfile "$gitignore_file"
        log_success "Global .gitignore configured for LLM development"
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "python3" "--version"
    show_tool_status "uv" "--version"
    show_tool_status "nvidia-smi" "--version 2>/dev/null | grep 'Driver Version'"
    show_tool_status "docker" "--version"
    show_tool_status "ollama" "--version"
    show_tool_status "ruff" "--version"
    show_tool_status "mypy" "--version"
    
    echo
    log_success "GenAI/LLM Ops environment is ready!"
    log_info "Key commands:"
    echo -e "  ${CYAN}source ~/activate-llm-env.sh${RESET} - Activate LLM environment"
    echo -e "  ${CYAN}cd ~/llm-ops-env${RESET} - Go to project directory"
    echo -e "  ${CYAN}uv sync --all-extras${RESET} - Install all packages"
    echo -e "  ${CYAN}ollama list${RESET} - List downloaded models"
    echo -e "  ${CYAN}ollama run llama3.2${RESET} - Chat with Llama 3.2"
    echo -e "  ${CYAN}cd ~/vector-databases && ./start-databases.sh${RESET} - Start vector DBs"
    
    if [[ -d "$HOME/llm-ops-env" ]]; then
        echo
        log_info "Project structure:"
        echo -e "  ${CYAN}~/llm-ops-env/${RESET}"
        echo -e "    ├── pyproject.toml    # Package configuration"
        echo -e "    ├── .venv/           # Virtual environment"
        echo -e "    ├── .env.example     # Environment variables template"
        echo -e "    └── README.md        # Project documentation"
    fi
    
    if ! command_exists nvidia-smi; then
        echo
        log_warning "Remember: You're running in CPU-only mode. GPU acceleration requires NVIDIA drivers and CUDA."
    fi
}

# Run main function
main "$@" 