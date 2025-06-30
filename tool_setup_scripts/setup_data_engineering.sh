#!/usr/bin/env bash

# Setup script for Data Engineering & Analytics development environment
# Installs: Python 3.12+, Spark, DuckDB, Great Expectations, and data engineering tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions and requirements
PYTHON_MIN_VERSION="3.12"
SPARK_VERSION="3.5.3"
JAVA_VERSION="11"
DUCKDB_VERSION="1.1.3"

main() {
    show_banner "Data Engineering & Analytics Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "python3" "--version"
    show_tool_status "java" "-version 2>&1 | head -n1"
    show_tool_status "spark-submit" "--version 2>&1 | grep version"
    show_tool_status "duckdb" "--version"
    show_tool_status "great_expectations" "--version"
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
    install_apt_package "libpq-dev" "PostgreSQL Development Libraries"
    install_apt_package "unzip" "Unzip"
    
    # Install Java for Spark
    log_step "Setting up Java for Spark"
    if ! command_exists java || ! java -version 2>&1 | grep -q "openjdk version \"$JAVA_VERSION"; then
        if confirm "Install OpenJDK $JAVA_VERSION for Spark?"; then
            install_apt_package "openjdk-${JAVA_VERSION}-jdk" "OpenJDK $JAVA_VERSION"
            
            # Set JAVA_HOME
            java_home="/usr/lib/jvm/java-${JAVA_VERSION}-openjdk-amd64"
            java_home_line="export JAVA_HOME=$java_home"
            path_line='export PATH=$JAVA_HOME/bin:$PATH'
            
            if [[ -f "$HOME/.bashrc" ]]; then
                add_to_file_if_missing "$HOME/.bashrc" "$java_home_line" "JAVA_HOME"
                add_to_file_if_missing "$HOME/.bashrc" "$path_line" "Java PATH"
            fi
            
            if [[ -f "$HOME/.zshrc" ]]; then
                add_to_file_if_missing "$HOME/.zshrc" "$java_home_line" "JAVA_HOME"
                add_to_file_if_missing "$HOME/.zshrc" "$path_line" "Java PATH"
            fi
            
            export JAVA_HOME=$java_home
            export PATH=$JAVA_HOME/bin:$PATH
            log_success "Java $JAVA_VERSION installed"
        fi
    else
        log_info "Java is already installed"
    fi
    
    # Install Apache Spark
    log_step "Setting up Apache Spark"
    spark_home="/opt/spark"
    if [[ ! -d "$spark_home" ]] || ! command_exists spark-submit; then
        if confirm "Install Apache Spark $SPARK_VERSION?"; then
            spark_url="https://dlcdn.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz"
            
            log_info "Downloading Spark..."
            curl -L "$spark_url" -o /tmp/spark.tgz
            
            log_info "Extracting Spark..."
            sudo tar -xzf /tmp/spark.tgz -C /opt/
            sudo mv "/opt/spark-${SPARK_VERSION}-bin-hadoop3" "$spark_home"
            rm /tmp/spark.tgz
            
            # Set Spark environment variables
            spark_home_line="export SPARK_HOME=$spark_home"
            spark_path_line='export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH'
            pyspark_python_line="export PYSPARK_PYTHON=python3"
            
            for rcfile in "$HOME/.bashrc" "$HOME/.zshrc"; do
                if [[ -f "$rcfile" ]]; then
                    add_to_file_if_missing "$rcfile" "$spark_home_line" "SPARK_HOME"
                    add_to_file_if_missing "$rcfile" "$spark_path_line" "Spark PATH"
                    add_to_file_if_missing "$rcfile" "$pyspark_python_line" "PySpark Python"
                fi
            done
            
            export SPARK_HOME=$spark_home
            export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
            export PYSPARK_PYTHON=python3
            
            log_success "Apache Spark $SPARK_VERSION installed"
        fi
    else
        log_info "Apache Spark is already installed"
    fi
    
    # Install DuckDB
    log_step "Installing DuckDB"
    if ! command_exists duckdb; then
        if confirm "Install DuckDB $DUCKDB_VERSION?"; then
            duckdb_url="https://github.com/duckdb/duckdb/releases/download/v${DUCKDB_VERSION}/duckdb_cli-linux-amd64.zip"
            
            log_info "Downloading DuckDB..."
            curl -L "$duckdb_url" -o /tmp/duckdb.zip
            
            log_info "Installing DuckDB..."
            sudo unzip -o /tmp/duckdb.zip -d /usr/local/bin/
            sudo chmod +x /usr/local/bin/duckdb
            rm /tmp/duckdb.zip
            
            log_success "DuckDB $DUCKDB_VERSION installed"
        fi
    else
        log_info "DuckDB is already installed"
    fi
    
    # Install uv for Python package management
    log_step "Setting up Python package manager"
    if ! command_exists uv; then
        if confirm "Install uv (Fast Python package manager)?"; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            
            # Add uv to PATH
            uv_path_line='export PATH="$HOME/.local/bin:$PATH"'
            
            for rcfile in "$HOME/.bashrc" "$HOME/.zshrc"; do
                if [[ -f "$rcfile" ]]; then
                    add_to_file_if_missing "$rcfile" "$uv_path_line" "uv PATH"
                fi
            done
            
            export PATH="$HOME/.local/bin:$PATH"
            log_success "uv installed"
        fi
    else
        log_info "uv is already installed"
    fi
    
    # Install Python data engineering tools
    if command_exists uv; then
        log_step "Installing Python data engineering tools"
        
        # Create a virtual environment for data tools
        if confirm "Create virtual environment for data engineering tools?"; then
            data_env="$HOME/.data-engineering-env"
            if [[ ! -d "$data_env" ]]; then
                uv venv "$data_env"
                log_success "Virtual environment created at $data_env"
            fi
            
            # Install tools in the virtual environment
            log_info "Installing data engineering packages..."
            
            # Core packages
            "$data_env/bin/pip" install --upgrade pip
            "$data_env/bin/pip" install \
                "pyspark==$SPARK_VERSION" \
                "duckdb==$DUCKDB_VERSION" \
                "great-expectations>=0.19.0" \
                "pandas>=2.0.0" \
                "pyarrow>=14.0.0" \
                "polars>=0.20.0" \
                "dbt-core>=1.9.0" \
                "dbt-duckdb>=1.9.0" \
                "apache-airflow>=2.10.0" \
                "delta-spark>=3.2.0" \
                "pyiceberg[pyarrow,duckdb,pandas]>=0.7.0" \
                "confluent-kafka>=2.5.0" \
                "sqlalchemy>=2.0.0" \
                "jupyter>=1.0.0" \
                "ipython>=8.0.0"
            
            # Create activation helper
            activate_script="$HOME/activate-data-env.sh"
            cat > "$activate_script" << EOF
#!/bin/bash
# Activate data engineering environment
source $data_env/bin/activate
export SPARK_HOME=$spark_home
export JAVA_HOME=/usr/lib/jvm/java-${JAVA_VERSION}-openjdk-amd64
export PATH=\$SPARK_HOME/bin:\$SPARK_HOME/sbin:\$PATH
export PYSPARK_PYTHON=python3
echo "Data engineering environment activated!"
echo "Available tools: pyspark, duckdb, great_expectations, dbt, airflow"
EOF
            chmod +x "$activate_script"
            
            log_success "Data engineering tools installed"
            log_info "To activate the environment, run: source ~/activate-data-env.sh"
        fi
        
        # Install development tools globally
        if confirm "Install development tools (ruff, mypy)?"; then
            uv tool install ruff
            uv tool install mypy
            log_success "Development tools installed"
        fi
    fi
    
    # Setup MinIO for local S3 testing
    log_step "Setting up MinIO (Local S3)"
    if ! command_exists minio; then
        if confirm "Install MinIO for local S3-compatible storage?"; then
            curl -L https://dl.min.io/server/minio/release/linux-amd64/minio -o /tmp/minio
            sudo mv /tmp/minio /usr/local/bin/
            sudo chmod +x /usr/local/bin/minio
            
            # Create MinIO directories
            create_directory "$HOME/minio/data" "MinIO data directory"
            
            # Create MinIO service script
            minio_script="$HOME/start-minio.sh"
            cat > "$minio_script" << 'EOF'
#!/bin/bash
# Start MinIO server
export MINIO_ROOT_USER=minioadmin
export MINIO_ROOT_PASSWORD=minioadmin
minio server ~/minio/data --console-address :9001
EOF
            chmod +x "$minio_script"
            
            log_success "MinIO installed"
            log_info "To start MinIO: ~/start-minio.sh"
            log_info "Console: http://localhost:9001 (minioadmin/minioadmin)"
        fi
    else
        log_info "MinIO is already installed"
    fi
    
    # Configure Git for data engineering
    log_step "Configuring Git for data engineering"
    if confirm "Add data engineering patterns to global .gitignore?"; then
        gitignore_file="$HOME/.gitignore_global"
        touch "$gitignore_file"
        
        # Data engineering patterns
        data_patterns=(
            "*.parquet"
            "*.avro"
            "*.orc"
            "*.csv"
            "*.json"
            "*.db"
            "*.duckdb"
            ".ipynb_checkpoints/"
            "spark-warehouse/"
            "metastore_db/"
            "derby.log"
            ".great_expectations/uncommitted/"
            "logs/"
            "airflow.db"
            "airflow-webserver.pid"
            ".dbt/"
            "target/"
            "dbt_packages/"
            "minio/data/"
        )
        
        for pattern in "${data_patterns[@]}"; do
            if ! grep -Fxq "$pattern" "$gitignore_file"; then
                echo "$pattern" >> "$gitignore_file"
            fi
        done
        
        git config --global core.excludesfile "$gitignore_file"
        log_success "Global .gitignore configured for data engineering"
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "python3" "--version"
    show_tool_status "java" "-version 2>&1 | head -n1"
    show_tool_status "spark-submit" "--version 2>&1 | grep version"
    show_tool_status "duckdb" "--version"
    show_tool_status "minio" "--version"
    
    echo
    log_success "Data Engineering environment is ready!"
    log_info "Key commands:"
    echo -e "  ${CYAN}source ~/activate-data-env.sh${RESET} - Activate data environment"
    echo -e "  ${CYAN}pyspark${RESET} - Start PySpark shell"
    echo -e "  ${CYAN}duckdb${RESET} - Start DuckDB CLI"
    echo -e "  ${CYAN}great_expectations init${RESET} - Initialize Great Expectations"
    echo -e "  ${CYAN}dbt init${RESET} - Initialize dbt project"
    echo -e "  ${CYAN}~/start-minio.sh${RESET} - Start MinIO S3 server"
}

# Run main function
main "$@" 