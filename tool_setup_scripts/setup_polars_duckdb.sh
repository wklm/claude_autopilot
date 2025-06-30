#!/usr/bin/env bash

# Setup script for Polars/DuckDB Data Engineering environment
# Installs: Python 3.11+, Polars, DuckDB, PyArrow, data tools, notebooks

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions
PYTHON_VERSION="3.11"

main() {
    show_banner "Polars/DuckDB Data Engineering Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "python3" "--version"
    show_tool_status "pip3" "--version"
    show_tool_status "duckdb" "--version 2>/dev/null || echo 'Not installed'"
    show_tool_status "rust" "--version"
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
    install_apt_package "cmake" "CMake"
    install_apt_package "libarrow-dev" "Apache Arrow Dev"
    install_apt_package "libparquet-dev" "Parquet Dev"
    
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
    
    # Install Rust (optional, for building Polars from source)
    log_step "Installing Rust (optional for Polars development)"
    if ! command_exists rustc; then
        if confirm "Install Rust (optional, for building Polars from source)?"; then
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source "$HOME/.cargo/env"
            log_success "Rust installed"
        fi
    else
        log_info "Rust is already installed"
    fi
    
    # Install DuckDB CLI
    log_step "Installing DuckDB CLI"
    if ! command_exists duckdb; then
        if confirm "Install DuckDB CLI?"; then
            install_duckdb_cli
        fi
    else
        log_info "DuckDB CLI is already installed"
        if confirm "Update DuckDB CLI to latest version?"; then
            install_duckdb_cli
        fi
    fi
    
    # Create virtual environment for data engineering
    log_step "Setting up Data Engineering Environment"
    log_info "Creating virtual environment for data engineering tools..."
    DATA_VENV="$HOME/.data-engineering"
    if [[ ! -d "$DATA_VENV" ]]; then
        python3 -m venv "$DATA_VENV"
    fi
    
    # Activate virtual environment
    source "$DATA_VENV/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install core data libraries
    log_step "Installing Core Data Libraries"
    if confirm "Install Polars and DuckDB Python libraries?"; then
        # Polars with all optional dependencies
        pip install "polars[all]"
        
        # DuckDB with extensions
        pip install duckdb duckdb-engine
        
        # PyArrow for interoperability
        pip install pyarrow pyarrow-hotfix
        
        log_success "Core data libraries installed"
    fi
    
    # Install complementary data tools
    if confirm "Install complementary data tools (pandas, numpy, etc.)?"; then
        pip install pandas numpy
        pip install scipy scikit-learn
        pip install fastparquet  # Alternative Parquet implementation
        pip install connectorx  # Fast data loading
        pip install deltalake  # Delta Lake support
        log_success "Complementary tools installed"
    fi
    
    # Install visualization libraries
    log_step "Installing Visualization Libraries"
    if confirm "Install data visualization libraries?"; then
        pip install matplotlib seaborn plotly
        pip install altair vega_datasets
        pip install hvplot holoviews panel
        log_success "Visualization libraries installed"
    fi
    
    # Install notebook environments
    log_step "Installing Notebook Environments"
    if confirm "Install Jupyter Lab and extensions?"; then
        pip install jupyterlab
        pip install ipywidgets jupyterlab-lsp
        pip install jupyter-duckdb  # DuckDB Jupyter magic
        
        # Install useful Jupyter extensions
        pip install jupyterlab-git jupyterlab-github
        pip install jupyterlab-execute-time
        
        log_success "Jupyter Lab installed"
    fi
    
    # Install profiling and debugging tools
    if confirm "Install profiling and debugging tools?"; then
        pip install memory_profiler line_profiler
        pip install py-spy snakeviz
        pip install pandas-profiling  # Now called ydata-profiling
        pip install great-expectations  # Data validation
        log_success "Profiling tools installed"
    fi
    
    # Install data quality and testing tools
    log_step "Installing Data Quality Tools"
    if confirm "Install data quality and testing tools?"; then
        pip install pytest pytest-benchmark
        pip install hypothesis  # Property-based testing
        pip install pandera  # Data validation
        pip install dbt-duckdb  # dbt for DuckDB
        log_success "Data quality tools installed"
    fi
    
    # Create example projects
    log_step "Creating example projects"
    if confirm "Create example data engineering projects?"; then
        create_data_examples
    fi
    
    # Setup DuckDB extensions
    log_step "Setting up DuckDB extensions"
    if confirm "Install popular DuckDB extensions?"; then
        setup_duckdb_extensions
    fi
    
    # Deactivate virtual environment
    deactivate
    
    # Setup shell aliases
    log_step "Setting up shell aliases"
    if confirm "Add data engineering aliases to shell?"; then
        setup_data_aliases
    fi
    
    # Install VS Code extensions
    log_step "VS Code Extensions"
    if command_exists code; then
        if confirm "Install recommended VS Code extensions?"; then
            code --install-extension ms-python.python
            code --install-extension ms-toolsai.jupyter
            code --install-extension RandomFractalsInc.duckdb-sql-tools
            code --install-extension mechatroner.rainbow-csv
            log_success "VS Code extensions installed"
        fi
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "python3" "--version"
    show_tool_status "duckdb" "--version"
    show_tool_status "$DATA_VENV/bin/python" "--version"
    
    echo
    log_success "Polars/DuckDB Data Engineering environment is ready!"
    log_info "Useful commands:"
    echo -e "  ${CYAN}source ~/.data-engineering/bin/activate${RESET} - Activate environment"
    echo -e "  ${CYAN}jupyter lab${RESET} - Start Jupyter Lab"
    echo -e "  ${CYAN}duckdb${RESET} - Start DuckDB CLI"
    echo -e "  ${CYAN}data-profile <file>${RESET} - Profile a data file"
    echo -e "  ${CYAN}polars-bench${RESET} - Run Polars vs Pandas benchmark"
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

install_duckdb_cli() {
    log_info "Installing DuckDB CLI..."
    
    # Download latest DuckDB
    DUCKDB_VERSION=$(curl -s https://api.github.com/repos/duckdb/duckdb/releases/latest | grep -oP '"tag_name": "\K[^"]+')
    wget -q "https://github.com/duckdb/duckdb/releases/download/${DUCKDB_VERSION}/duckdb_cli-linux-amd64.zip"
    
    # Extract and install
    unzip -q duckdb_cli-linux-amd64.zip
    sudo mv duckdb /usr/local/bin/
    rm duckdb_cli-linux-amd64.zip
    
    # Make executable
    sudo chmod +x /usr/local/bin/duckdb
    
    log_success "DuckDB CLI installed"
}

create_data_examples() {
    log_info "Creating example data engineering projects..."
    
    # Create examples directory
    mkdir -p "$HOME/data-engineering-examples"
    
    # Polars example
    cat > "$HOME/data-engineering-examples/polars_example.py" << 'EOF'
#!/usr/bin/env python3
"""Example showcasing Polars capabilities"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
import time

# Generate sample data
np.random.seed(42)
n_rows = 1_000_000

print(f"Generating {n_rows:,} rows of sample data...")

# Create sample dataframe
df = pl.DataFrame({
    "id": range(n_rows),
    "timestamp": [datetime.now() - timedelta(days=x) for x in range(n_rows)],
    "category": np.random.choice(["A", "B", "C", "D"], n_rows),
    "value": np.random.randn(n_rows) * 100,
    "quantity": np.random.randint(1, 100, n_rows),
})

print("\nDataFrame info:")
print(f"Shape: {df.shape}")
print(f"Schema: {df.schema}")

# Demonstrate lazy evaluation
print("\n--- Lazy Evaluation Example ---")
start = time.time()

result = (
    df.lazy()
    .filter(pl.col("value") > 0)
    .group_by("category")
    .agg([
        pl.col("value").mean().alias("avg_value"),
        pl.col("quantity").sum().alias("total_quantity"),
        pl.count().alias("count")
    ])
    .sort("avg_value", descending=True)
    .collect()
)

print(f"Execution time: {time.time() - start:.3f}s")
print(result)

# Window functions
print("\n--- Window Functions Example ---")
window_result = df.with_columns([
    pl.col("value").rolling_mean(window_size=10).alias("rolling_avg"),
    pl.col("value").rank().over("category").alias("rank_in_category")
]).head(20)

print(window_result)

# Save to various formats
print("\n--- Saving to different formats ---")
df.write_parquet("sample_data.parquet")
print("✓ Saved to Parquet")

df.write_csv("sample_data.csv")
print("✓ Saved to CSV")

# Demonstrate reading back with lazy evaluation
print("\n--- Reading with lazy evaluation ---")
lazy_df = pl.scan_parquet("sample_data.parquet")
filtered = lazy_df.filter(pl.col("category") == "A").collect()
print(f"Filtered rows (category A): {len(filtered):,}")
EOF
    
    # DuckDB example
    cat > "$HOME/data-engineering-examples/duckdb_example.py" << 'EOF'
#!/usr/bin/env python3
"""Example showcasing DuckDB capabilities"""

import duckdb
import pandas as pd
import polars as pl
import time

# Connect to DuckDB (in-memory)
con = duckdb.connect(':memory:')

print("--- DuckDB Example ---\n")

# Create sample data
print("Creating sample tables...")
con.execute("""
    CREATE TABLE sales AS
    SELECT 
        range AS sale_id,
        '2024-01-01'::DATE + INTERVAL (range % 365) DAY AS sale_date,
        (range % 100) + 1 AS customer_id,
        (range % 50) + 1 AS product_id,
        random() * 1000 AS amount,
        ['Online', 'Store', 'Phone'][1 + range % 3] AS channel
    FROM range(1000000);
""")

con.execute("""
    CREATE TABLE products AS
    SELECT 
        range + 1 AS product_id,
        'Product ' || (range + 1) AS product_name,
        ['Electronics', 'Clothing', 'Food', 'Books'][1 + range % 4] AS category,
        random() * 100 AS unit_price
    FROM range(50);
""")

print("✓ Tables created")

# Complex analytical query
print("\n--- Running analytical queries ---")
start = time.time()

result = con.execute("""
    WITH monthly_sales AS (
        SELECT 
            DATE_TRUNC('month', sale_date) AS month,
            p.category,
            SUM(s.amount) AS total_sales,
            COUNT(DISTINCT s.customer_id) AS unique_customers,
            AVG(s.amount) AS avg_sale
        FROM sales s
        JOIN products p ON s.product_id = p.product_id
        GROUP BY DATE_TRUNC('month', sale_date), p.category
    )
    SELECT 
        month,
        category,
        total_sales,
        unique_customers,
        avg_sale,
        SUM(total_sales) OVER (
            PARTITION BY category 
            ORDER BY month 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_sales
    FROM monthly_sales
    ORDER BY category, month
    LIMIT 20;
""").fetchdf()

print(f"Query execution time: {time.time() - start:.3f}s")
print("\nResults:")
print(result)

# Demonstrate DuckDB's ability to query Pandas DataFrames directly
print("\n--- Querying Pandas DataFrame directly ---")
pandas_df = pd.DataFrame({
    'x': range(1000),
    'y': [i**2 for i in range(1000)]
})

result = con.execute("SELECT * FROM pandas_df WHERE y > 500000").fetchdf()
print(f"Rows where y > 500000: {len(result)}")

# Demonstrate DuckDB's ability to query Polars DataFrames
print("\n--- Querying Polars DataFrame directly ---")
polars_df = pl.DataFrame({
    'a': range(1000),
    'b': [i**3 for i in range(1000)]
})

result = con.execute("SELECT * FROM polars_df WHERE b > 1000000 LIMIT 5").fetchdf()
print("First 5 rows where b > 1000000:")
print(result)

# Export to Parquet
print("\n--- Exporting to Parquet ---")
con.execute("COPY (SELECT * FROM sales LIMIT 100000) TO 'sales_sample.parquet' (FORMAT PARQUET);")
print("✓ Exported to sales_sample.parquet")

# Close connection
con.close()
EOF
    
    # Benchmark script
    cat > "$HOME/data-engineering-examples/benchmark.py" << 'EOF'
#!/usr/bin/env python3
"""Benchmark comparing Polars, DuckDB, and Pandas"""

import polars as pl
import pandas as pd
import duckdb
import numpy as np
import time
from datetime import datetime

def generate_data(n_rows=1_000_000):
    """Generate sample data"""
    print(f"Generating {n_rows:,} rows of data...")
    return {
        'id': range(n_rows),
        'group': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),
        'value1': np.random.randn(n_rows),
        'value2': np.random.randn(n_rows) * 100,
        'value3': np.random.randint(1, 1000, n_rows)
    }

def benchmark_groupby_agg(data):
    """Benchmark group by aggregation"""
    results = {}
    
    # Pandas
    print("\n--- Pandas ---")
    pandas_df = pd.DataFrame(data)
    start = time.time()
    result = pandas_df.groupby('group').agg({
        'value1': 'mean',
        'value2': 'sum',
        'value3': ['min', 'max', 'std']
    })
    results['Pandas'] = time.time() - start
    print(f"Time: {results['Pandas']:.3f}s")
    
    # Polars
    print("\n--- Polars ---")
    polars_df = pl.DataFrame(data)
    start = time.time()
    result = polars_df.group_by('group').agg([
        pl.col('value1').mean(),
        pl.col('value2').sum(),
        pl.col('value3').min(),
        pl.col('value3').max(),
        pl.col('value3').std()
    ])
    results['Polars'] = time.time() - start
    print(f"Time: {results['Polars']:.3f}s")
    
    # DuckDB
    print("\n--- DuckDB ---")
    con = duckdb.connect(':memory:')
    con.register('df', pandas_df)
    start = time.time()
    result = con.execute("""
        SELECT 
            "group",
            AVG(value1) as value1_mean,
            SUM(value2) as value2_sum,
            MIN(value3) as value3_min,
            MAX(value3) as value3_max,
            STDDEV(value3) as value3_std
        FROM df
        GROUP BY "group"
    """).fetchdf()
    results['DuckDB'] = time.time() - start
    print(f"Time: {results['DuckDB']:.3f}s")
    
    return results

def benchmark_filter_sort(data):
    """Benchmark filtering and sorting"""
    results = {}
    
    # Pandas
    print("\n--- Pandas ---")
    pandas_df = pd.DataFrame(data)
    start = time.time()
    result = pandas_df[pandas_df['value1'] > 0].sort_values(['group', 'value2'], ascending=[True, False])
    results['Pandas'] = time.time() - start
    print(f"Time: {results['Pandas']:.3f}s")
    
    # Polars
    print("\n--- Polars ---")
    polars_df = pl.DataFrame(data)
    start = time.time()
    result = polars_df.filter(pl.col('value1') > 0).sort(['group', 'value2'], descending=[False, True])
    results['Polars'] = time.time() - start
    print(f"Time: {results['Polars']:.3f}s")
    
    # DuckDB
    print("\n--- DuckDB ---")
    con = duckdb.connect(':memory:')
    con.register('df', pandas_df)
    start = time.time()
    result = con.execute("""
        SELECT * FROM df 
        WHERE value1 > 0 
        ORDER BY "group" ASC, value2 DESC
    """).fetchdf()
    results['DuckDB'] = time.time() - start
    print(f"Time: {results['DuckDB']:.3f}s")
    
    return results

if __name__ == "__main__":
    print("=" * 60)
    print("Data Processing Benchmark: Polars vs DuckDB vs Pandas")
    print("=" * 60)
    
    data = generate_data()
    
    print("\n### Benchmark 1: Group By Aggregation ###")
    groupby_results = benchmark_groupby_agg(data)
    
    print("\n### Benchmark 2: Filter and Sort ###")
    filter_results = benchmark_filter_sort(data)
    
    print("\n### Summary ###")
    print("\nGroup By Aggregation:")
    for lib, time_taken in sorted(groupby_results.items(), key=lambda x: x[1]):
        print(f"  {lib}: {time_taken:.3f}s")
    
    print("\nFilter and Sort:")
    for lib, time_taken in sorted(filter_results.items(), key=lambda x: x[1]):
        print(f"  {lib}: {time_taken:.3f}s")
EOF
    
    # Make scripts executable
    chmod +x "$HOME/data-engineering-examples/"*.py
    
    log_success "Example projects created in ~/data-engineering-examples/"
}

setup_duckdb_extensions() {
    log_info "Installing DuckDB extensions..."
    
    # Create a script to install extensions
    cat > "$HOME/install_duckdb_extensions.py" << 'EOF'
#!/usr/bin/env python3
import duckdb

# Connect to DuckDB
con = duckdb.connect()

extensions = [
    'httpfs',      # Read files over HTTP(S)
    'postgres',    # PostgreSQL connector
    'sqlite',      # SQLite connector
    'parquet',     # Parquet support (usually built-in)
    'json',        # JSON support
    'excel',       # Excel file support
    'spatial',     # Spatial/GIS functions
    'inet',        # IP address functions
]

print("Installing DuckDB extensions...")
for ext in extensions:
    try:
        con.execute(f"INSTALL {ext};")
        con.execute(f"LOAD {ext};")
        print(f"✓ {ext}")
    except Exception as e:
        print(f"✗ {ext}: {e}")

print("\nInstalled extensions:")
result = con.execute("SELECT * FROM duckdb_extensions();").fetchdf()
print(result[['extension_name', 'loaded', 'installed']].to_string(index=False))

con.close()
EOF
    
    python3 "$HOME/install_duckdb_extensions.py"
    rm "$HOME/install_duckdb_extensions.py"
    
    log_success "DuckDB extensions configured"
}

setup_data_aliases() {
    log_info "Setting up data engineering aliases..."
    
    data_aliases='
# Data Engineering aliases
alias data-env="source ~/.data-engineering/bin/activate"
alias data-jupyter="source ~/.data-engineering/bin/activate && jupyter lab"
alias duckdb-mem="duckdb :memory:"
alias polars-repl="source ~/.data-engineering/bin/activate && python -c \"import polars as pl; print(\'Polars\', pl.__version__, \'loaded\')\""

# Quick data profiling
data-profile() {
    if [[ -z "$1" ]]; then
        echo "Usage: data-profile <file>"
        return 1
    fi
    
    source ~/.data-engineering/bin/activate
    python -c "
import polars as pl
import sys

file = sys.argv[1]
if file.endswith(\'.parquet\'):
    df = pl.read_parquet(file)
elif file.endswith(\'.csv\'):
    df = pl.read_csv(file)
else:
    print(\'Unsupported file type\')
    sys.exit(1)

print(f\'File: {file}\')
print(f\'Shape: {df.shape}\')
print(f\'Memory usage: {df.estimated_size() / 1024 / 1024:.2f} MB\')
print(f\'\\nSchema:\')
for col, dtype in zip(df.columns, df.dtypes):
    print(f\'  {col}: {dtype}\')
print(f\'\\nFirst 5 rows:\')
print(df.head())
print(f\'\\nSummary statistics:\')
print(df.describe())
" "$1"
}

# DuckDB quick query
duckdb-query() {
    if [[ -z "$1" ]]; then
        echo "Usage: duckdb-query <file> [query]"
        return 1
    fi
    
    file="$1"
    query="${2:-SELECT * FROM data LIMIT 10}"
    
    duckdb :memory: << EOF
.echo off
CREATE TABLE data AS SELECT * FROM read_csv_auto('"$file"');
$query;
EOF
}

# Convert between formats
data-convert() {
    if [[ $# -lt 2 ]]; then
        echo "Usage: data-convert <input> <output>"
        echo "Supports: .csv, .parquet, .json"
        return 1
    fi
    
    source ~/.data-engineering/bin/activate
    python -c "
import polars as pl
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

# Read input
if input_file.endswith(\'.csv\'):
    df = pl.read_csv(input_file)
elif input_file.endswith(\'.parquet\'):
    df = pl.read_parquet(input_file)
elif input_file.endswith(\'.json\'):
    df = pl.read_json(input_file)
else:
    print(f\'Unsupported input format: {input_file}\')
    sys.exit(1)

# Write output
if output_file.endswith(\'.csv\'):
    df.write_csv(output_file)
elif output_file.endswith(\'.parquet\'):
    df.write_parquet(output_file)
elif output_file.endswith(\'.json\'):
    df.write_json(output_file)
else:
    print(f\'Unsupported output format: {output_file}\')
    sys.exit(1)

print(f\'Converted {input_file} -> {output_file}\')
print(f\'Rows: {len(df):,}\')
" "$1" "$2"
}

# Run benchmarks
alias polars-bench="source ~/.data-engineering/bin/activate && python ~/data-engineering-examples/benchmark.py"

# Memory profiling
mem-profile() {
    if [[ -z "$1" ]]; then
        echo "Usage: mem-profile <python_script>"
        return 1
    fi
    source ~/.data-engineering/bin/activate
    python -m memory_profiler "$1"
}'
    
    if [[ -f "$HOME/.bashrc" ]]; then
        add_to_file_if_missing "$HOME/.bashrc" "$data_aliases" "Data engineering aliases"
    fi
    
    if [[ -f "$HOME/.zshrc" ]]; then
        add_to_file_if_missing "$HOME/.zshrc" "$data_aliases" "Data engineering aliases"
    fi
    
    log_success "Data engineering aliases added to shell"
}

# Run main function
main "$@" 