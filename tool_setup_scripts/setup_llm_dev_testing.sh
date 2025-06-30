#!/usr/bin/env bash

# Setup script for LLM Development and Testing environment
# Installs: Python 3.11+, LangChain, LlamaIndex, evaluation frameworks, testing tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions
PYTHON_VERSION="3.11"

main() {
    show_banner "LLM Development and Testing Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "python3" "--version"
    show_tool_status "pip3" "--version"
    show_tool_status "uv" "--version"
    show_tool_status "poetry" "--version"
    show_tool_status "ollama" "--version"
    show_tool_status "litellm" "--version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "build-essential" "Build Essential"
    install_apt_package "curl" "cURL"
    install_apt_package "git" "Git"
    install_apt_package "software-properties-common" "Software Properties Common"
    install_apt_package "python3-dev" "Python 3 Dev"
    install_apt_package "python3-pip" "Python 3 pip"
    install_apt_package "python3-venv" "Python 3 venv"
    
    # Install Python 3.11+ if needed
    log_step "Checking Python version"
    if command_exists python3; then
        current_python=$(python3 --version | cut -d' ' -f2 | cut -d. -f1,2)
        log_info "Current Python version: $current_python"
        
        if [[ $(echo "$current_python < $PYTHON_VERSION" | bc) -eq 1 ]]; then
            log_warning "Python $current_python is older than recommended $PYTHON_VERSION"
            if confirm "Install Python $PYTHON_VERSION?"; then
                install_python
            fi
        else
            log_success "Python version is sufficient"
        fi
    else
        if confirm "Install Python $PYTHON_VERSION?"; then
            install_python
        else
            log_error "Python is required to continue"
            exit 1
        fi
    fi
    
    # Install uv (fast Python package manager)
    log_step "Installing uv"
    if ! command_exists uv; then
        if confirm "Install uv (fast Python package manager)?"; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.cargo/bin:$PATH"
            log_success "uv installed"
        fi
    else
        log_info "uv is already installed"
    fi
    
    # Install Poetry
    log_step "Installing Poetry"
    if ! command_exists poetry; then
        if confirm "Install Poetry (dependency management)?"; then
            curl -sSL https://install.python-poetry.org | python3 -
            export PATH="$HOME/.local/bin:$PATH"
            log_success "Poetry installed"
        fi
    else
        log_info "Poetry is already installed"
    fi
    
    # Install LLM frameworks
    log_step "Installing LLM Development Frameworks"
    
    # Create a virtual environment for global tools
    log_info "Creating virtual environment for LLM tools..."
    LLM_VENV="$HOME/.llm-tools"
    if [[ ! -d "$LLM_VENV" ]]; then
        python3 -m venv "$LLM_VENV"
    fi
    
    # Activate virtual environment
    source "$LLM_VENV/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Core LLM frameworks
    if confirm "Install core LLM frameworks (LangChain, LlamaIndex, etc.)?"; then
        pip install langchain langchain-community langchain-core
        pip install llama-index llama-index-core
        pip install openai anthropic cohere
        pip install transformers datasets
        pip install sentence-transformers
        pip install chromadb faiss-cpu
        pip install tiktoken
        log_success "Core LLM frameworks installed"
    fi
    
    # Evaluation frameworks
    if confirm "Install LLM evaluation frameworks?"; then
        pip install ragas
        pip install deepeval
        pip install promptfoo
        pip install langfuse
        pip install phoenix-arize
        pip install trulens-eval
        log_success "Evaluation frameworks installed"
    fi
    
    # Testing and debugging tools
    if confirm "Install LLM testing and debugging tools?"; then
        pip install pytest pytest-asyncio pytest-mock
        pip install hypothesis
        pip install responses
        pip install langsmith
        pip install wandb
        pip install mlflow
        log_success "Testing tools installed"
    fi
    
    # Prompt engineering tools
    if confirm "Install prompt engineering tools?"; then
        pip install guidance
        pip install prompttools
        pip install dspy-ai
        log_success "Prompt engineering tools installed"
    fi
    
    # Install Ollama (local LLM runtime)
    log_step "Installing Ollama"
    if ! command_exists ollama; then
        if confirm "Install Ollama (run LLMs locally)?"; then
            curl -fsSL https://ollama.ai/install.sh | sh
            log_success "Ollama installed"
            log_info "Start Ollama with: ollama serve"
            log_info "Pull models with: ollama pull llama2"
        fi
    else
        log_info "Ollama is already installed"
    fi
    
    # Install LiteLLM (unified LLM interface)
    if confirm "Install LiteLLM (unified API for 100+ LLMs)?"; then
        pip install litellm
        log_success "LiteLLM installed"
    fi
    
    # Development tools
    log_step "Installing development tools"
    
    # Jupyter and extensions
    if confirm "Install Jupyter Lab with LLM extensions?"; then
        pip install jupyterlab
        pip install jupyter-ai
        pip install ipywidgets
        log_success "Jupyter Lab installed"
    fi
    
    # Code quality tools
    if confirm "Install code quality tools (mypy, ruff, black)?"; then
        pip install mypy ruff black isort
        pip install pre-commit
        log_success "Code quality tools installed"
    fi
    
    # Create convenient scripts
    log_step "Creating convenience scripts"
    if confirm "Create LLM development helper scripts?"; then
        create_llm_scripts
    fi
    
    # Deactivate virtual environment
    deactivate
    
    # Setup shell aliases
    log_step "Setting up shell aliases"
    if confirm "Add LLM development aliases to shell?"; then
        setup_llm_aliases
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "python3" "--version"
    show_tool_status "pip3" "--version"
    show_tool_status "uv" "--version"
    show_tool_status "poetry" "--version"
    show_tool_status "ollama" "--version"
    show_tool_status "$LLM_VENV/bin/langchain" "--version 2>/dev/null || echo 'Installed in venv'"
    
    echo
    log_success "LLM Development and Testing environment is ready!"
    log_info "Useful commands:"
    echo -e "  ${CYAN}source ~/.llm-tools/bin/activate${RESET} - Activate LLM tools environment"
    echo -e "  ${CYAN}ollama serve${RESET} - Start Ollama server"
    echo -e "  ${CYAN}ollama pull llama2${RESET} - Download Llama 2 model"
    echo -e "  ${CYAN}litellm --model ollama/llama2${RESET} - Start LiteLLM proxy"
    echo -e "  ${CYAN}jupyter lab${RESET} - Start Jupyter Lab"
    echo
    log_info "Example project setup:"
    echo -e "  ${CYAN}mkdir my-llm-project && cd my-llm-project${RESET}"
    echo -e "  ${CYAN}uv venv && source .venv/bin/activate${RESET}"
    echo -e "  ${CYAN}uv pip install langchain openai pytest${RESET}"
}

install_python() {
    log_info "Installing Python $PYTHON_VERSION..."
    
    # Add deadsnakes PPA
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    update_apt
    
    # Install Python
    install_apt_package "python${PYTHON_VERSION}" "Python $PYTHON_VERSION"
    install_apt_package "python${PYTHON_VERSION}-venv" "Python $PYTHON_VERSION venv"
    install_apt_package "python${PYTHON_VERSION}-dev" "Python $PYTHON_VERSION dev"
    
    # Update alternatives
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
    
    log_success "Python $PYTHON_VERSION installed"
}

create_llm_scripts() {
    log_info "Creating LLM development helper scripts..."
    
    # Create scripts directory
    mkdir -p "$HOME/bin"
    
    # LLM test runner script
    cat > "$HOME/bin/llm-test" << 'EOF'
#!/usr/bin/env bash
# Run LLM tests with proper configuration

source ~/.llm-tools/bin/activate
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

pytest -v --tb=short "$@"
EOF
    chmod +x "$HOME/bin/llm-test"
    
    # LLM evaluation script
    cat > "$HOME/bin/llm-eval" << 'EOF'
#!/usr/bin/env bash
# Run LLM evaluation suite

source ~/.llm-tools/bin/activate

if [[ $# -eq 0 ]]; then
    echo "Usage: llm-eval <framework> [args...]"
    echo "Frameworks: ragas, deepeval, trulens"
    exit 1
fi

framework=$1
shift

case $framework in
    ragas)
        python -m ragas "$@"
        ;;
    deepeval)
        deepeval "$@"
        ;;
    trulens)
        python -m trulens_eval "$@"
        ;;
    *)
        echo "Unknown framework: $framework"
        exit 1
        ;;
esac
EOF
    chmod +x "$HOME/bin/llm-eval"
    
    # Add ~/bin to PATH if not already there
    if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
        echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
        export PATH="$HOME/bin:$PATH"
    fi
    
    log_success "Helper scripts created in ~/bin"
}

setup_llm_aliases() {
    log_info "Setting up LLM development aliases..."
    
    llm_aliases='
# LLM Development aliases
alias llm-activate="source ~/.llm-tools/bin/activate"
alias llm-jupyter="source ~/.llm-tools/bin/activate && jupyter lab"
alias llm-test="~/.llm-tools/bin/pytest -v --tb=short"
alias llm-format="~/.llm-tools/bin/black . && ~/.llm-tools/bin/isort ."
alias llm-lint="~/.llm-tools/bin/ruff check ."
alias llm-typecheck="~/.llm-tools/bin/mypy ."
alias ollama-llama2="ollama run llama2"
alias ollama-codellama="ollama run codellama"
alias litellm-proxy="source ~/.llm-tools/bin/activate && litellm --model"

# Quick LLM project setup
llm-new-project() {
    if [[ -z "$1" ]]; then
        echo "Usage: llm-new-project <project-name>"
        return 1
    fi
    mkdir -p "$1" && cd "$1"
    uv venv
    source .venv/bin/activate
    uv pip install langchain openai pytest black ruff mypy
    echo "# $1" > README.md
    echo ".venv/" > .gitignore
    echo "__pycache__/" >> .gitignore
    echo "*.pyc" >> .gitignore
    echo ".env" >> .gitignore
    git init
    echo "LLM project $1 created and initialized!"
}'
    
    if [[ -f "$HOME/.bashrc" ]]; then
        add_to_file_if_missing "$HOME/.bashrc" "$llm_aliases" "LLM aliases"
    fi
    
    if [[ -f "$HOME/.zshrc" ]]; then
        add_to_file_if_missing "$HOME/.zshrc" "$llm_aliases" "LLM aliases"
    fi
    
    log_success "LLM development aliases added to shell"
}

# Run main function
main "$@" 