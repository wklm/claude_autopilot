#!/usr/bin/env bash

# Setup script for Excel Automation with Python and Azure
# Installs: Python 3.11+, openpyxl, xlwings, pandas, Azure SDK, Office integration tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions
PYTHON_VERSION="3.11"

main() {
    show_banner "Excel Automation with Python and Azure Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "python3" "--version"
    show_tool_status "pip3" "--version"
    show_tool_status "az" "--version 2>&1 | head -n 1"
    show_tool_status "node" "--version"
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
    install_apt_package "libxml2-dev" "LibXML2 Dev"
    install_apt_package "libxslt1-dev" "LibXSLT Dev"
    
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
    
    # Install Azure CLI
    log_step "Installing Azure CLI"
    if ! command_exists az; then
        if confirm "Install Azure CLI?"; then
            install_azure_cli
        fi
    else
        log_info "Azure CLI is already installed"
        if confirm "Update Azure CLI to latest version?"; then
            az upgrade --yes
            log_success "Azure CLI updated"
        fi
    fi
    
    # Create virtual environment for Excel tools
    log_step "Setting up Excel Automation Environment"
    log_info "Creating virtual environment for Excel automation tools..."
    EXCEL_VENV="$HOME/.excel-automation"
    if [[ ! -d "$EXCEL_VENV" ]]; then
        python3 -m venv "$EXCEL_VENV"
    fi
    
    # Activate virtual environment
    source "$EXCEL_VENV/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Excel manipulation libraries
    log_step "Installing Excel Manipulation Libraries"
    if confirm "Install core Excel libraries (openpyxl, xlsxwriter, xlrd)?"; then
        pip install openpyxl xlsxwriter xlrd xlwt
        pip install python-docx python-pptx  # For Word and PowerPoint
        log_success "Core Excel libraries installed"
    fi
    
    # Install data analysis libraries
    if confirm "Install data analysis libraries (pandas, numpy)?"; then
        pip install pandas numpy
        pip install matplotlib seaborn plotly  # For charts
        pip install scipy statsmodels  # For statistics
        log_success "Data analysis libraries installed"
    fi
    
    # Install xlwings (Excel COM automation)
    if confirm "Install xlwings (for Excel COM automation)?"; then
        pip install xlwings
        log_success "xlwings installed"
        log_info "Note: xlwings requires Excel to be installed on Windows/Mac for full functionality"
    fi
    
    # Install Azure SDK components
    log_step "Installing Azure SDK Components"
    if confirm "Install Azure SDK for Python?"; then
        pip install azure-identity azure-storage-blob azure-keyvault-secrets
        pip install azure-mgmt-resource azure-mgmt-storage
        pip install azure-functions
        log_success "Azure SDK components installed"
    fi
    
    # Install Office 365 integration
    if confirm "Install Office 365 integration libraries?"; then
        pip install O365 python-dotenv
        pip install msgraph-core msgraph-sdk
        log_success "Office 365 integration libraries installed"
    fi
    
    # Install SharePoint integration
    if confirm "Install SharePoint integration libraries?"; then
        pip install sharepoint Office365-REST-Python-Client
        log_success "SharePoint libraries installed"
    fi
    
    # Install automation and scheduling tools
    log_step "Installing Automation Tools"
    if confirm "Install automation tools (schedule, watchdog)?"; then
        pip install schedule watchdog
        pip install python-crontab  # For cron job management
        pip install pyautogui  # For GUI automation
        log_success "Automation tools installed"
    fi
    
    # Install reporting libraries
    if confirm "Install reporting libraries (Jinja2, WeasyPrint)?"; then
        pip install Jinja2 weasyprint
        pip install reportlab  # For PDF generation
        pip install python-docx-template  # For template-based documents
        log_success "Reporting libraries installed"
    fi
    
    # Install development tools
    log_step "Installing Development Tools"
    if confirm "Install development tools (pytest, black, mypy)?"; then
        pip install pytest pytest-cov pytest-mock
        pip install black isort mypy
        pip install jupyterlab ipykernel
        log_success "Development tools installed"
    fi
    
    # Create helper scripts
    log_step "Creating helper scripts"
    if confirm "Create Excel automation helper scripts?"; then
        create_excel_scripts
    fi
    
    # Deactivate virtual environment
    deactivate
    
    # Setup shell aliases
    log_step "Setting up shell aliases"
    if confirm "Add Excel automation aliases to shell?"; then
        setup_excel_aliases
    fi
    
    # Install VS Code extensions
    log_step "VS Code Extensions"
    if command_exists code; then
        if confirm "Install recommended VS Code extensions for Excel automation?"; then
            code --install-extension ms-python.python
            code --install-extension ms-vscode.powershell
            code --install-extension ms-azuretools.vscode-azurefunctions
            code --install-extension GrapeCity.gc-excelviewer
            log_success "VS Code extensions installed"
        fi
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "python3" "--version"
    show_tool_status "pip3" "--version"
    show_tool_status "az" "--version 2>&1 | head -n 1"
    show_tool_status "$EXCEL_VENV/bin/python" "--version"
    
    echo
    log_success "Excel Automation environment is ready!"
    log_info "Useful commands:"
    echo -e "  ${CYAN}source ~/.excel-automation/bin/activate${RESET} - Activate Excel automation environment"
    echo -e "  ${CYAN}excel-new-project <name>${RESET} - Create new Excel automation project"
    echo -e "  ${CYAN}excel-read-write${RESET} - Run example Excel read/write script"
    echo -e "  ${CYAN}az login${RESET} - Login to Azure"
    echo -e "  ${CYAN}jupyter lab${RESET} - Start Jupyter Lab for interactive development"
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

install_azure_cli() {
    log_info "Installing Azure CLI..."
    
    # Install dependencies
    install_apt_package "ca-certificates" "CA Certificates"
    install_apt_package "curl" "cURL"
    install_apt_package "apt-transport-https" "APT Transport HTTPS"
    install_apt_package "lsb-release" "LSB Release"
    install_apt_package "gnupg" "GnuPG"
    
    # Add Microsoft signing key
    curl -sL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/microsoft.gpg > /dev/null
    
    # Add Azure CLI repository
    AZ_REPO=$(lsb_release -cs)
    echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ $AZ_REPO main" | sudo tee /etc/apt/sources.list.d/azure-cli.list
    
    # Update and install
    update_apt
    install_apt_package "azure-cli" "Azure CLI"
    
    log_success "Azure CLI installed"
}

create_excel_scripts() {
    log_info "Creating Excel automation helper scripts..."
    
    # Create scripts directory
    mkdir -p "$HOME/bin"
    
    # Excel project creator
    cat > "$HOME/bin/excel-new-project" << 'EOF'
#!/usr/bin/env bash
# Create new Excel automation project

if [[ -z "$1" ]]; then
    echo "Usage: excel-new-project <project-name>"
    exit 1
fi

project_name="$1"
mkdir -p "$project_name"
cd "$project_name"

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install openpyxl pandas numpy matplotlib azure-storage-blob python-dotenv

# Create project structure
mkdir -p src tests data output templates

# Create .env template
cat > .env.template << 'ENV'
# Azure Configuration
AZURE_STORAGE_CONNECTION_STRING=
AZURE_STORAGE_ACCOUNT_NAME=
AZURE_STORAGE_ACCOUNT_KEY=
AZURE_CONTAINER_NAME=

# Office 365 Configuration
O365_CLIENT_ID=
O365_CLIENT_SECRET=
O365_TENANT_ID=
ENV

# Create example script
cat > src/excel_automation.py << 'PYTHON'
import pandas as pd
from openpyxl import Workbook, load_workbook
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

def read_excel_file(file_path):
    """Read Excel file and return DataFrame"""
    df = pd.read_excel(file_path)
    return df

def create_report(data, output_path):
    """Create Excel report with formatting"""
    wb = Workbook()
    ws = wb.active
    ws.title = "Report"
    
    # Write headers
    headers = list(data.columns)
    for col, header in enumerate(headers, 1):
        ws.cell(row=1, column=col, value=header)
    
    # Write data
    for row_idx, row_data in enumerate(data.values, 2):
        for col_idx, value in enumerate(row_data, 1):
            ws.cell(row=row_idx, column=col_idx, value=value)
    
    # Save workbook
    wb.save(output_path)
    print(f"Report saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    print("Excel Automation Example")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Date': [datetime.now().date()],
        'Sales': [1000],
        'Region': ['North']
    })
    
    # Create report
    create_report(sample_data, 'output/sample_report.xlsx')
PYTHON

# Create README
cat > README.md << 'README'
# $project_name

Excel automation project using Python and Azure.

## Setup

1. Create virtual environment: `python3 -m venv .venv`
2. Activate: `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.template` to `.env` and fill in values

## Structure

- `src/` - Source code
- `tests/` - Unit tests
- `data/` - Input data files
- `output/` - Generated reports
- `templates/` - Excel templates
README

# Create requirements.txt
pip freeze > requirements.txt

# Initialize git
git init
echo ".venv/" > .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".env" >> .gitignore
echo "output/*.xlsx" >> .gitignore

echo "Excel automation project '$project_name' created successfully!"
echo "Next steps:"
echo "1. cd $project_name"
echo "2. source .venv/bin/activate"
echo "3. python src/excel_automation.py"
EOF
    chmod +x "$HOME/bin/excel-new-project"
    
    # Excel read/write example
    cat > "$HOME/bin/excel-read-write" << 'EOF'
#!/usr/bin/env bash
# Example Excel read/write operations

source ~/.excel-automation/bin/activate

python3 << 'PYTHON'
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.chart import BarChart, Reference

# Create sample data
data = {
    'Product': ['A', 'B', 'C', 'D'],
    'Q1': [100, 150, 120, 180],
    'Q2': [110, 160, 140, 190],
    'Q3': [120, 170, 130, 200],
    'Q4': [130, 180, 150, 210]
}

df = pd.DataFrame(data)

# Write to Excel with formatting
with pd.ExcelWriter('sample_report.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Sales Data', index=False)
    
    # Get the workbook and worksheet
    workbook = writer.book
    worksheet = writer.sheets['Sales Data']
    
    # Apply formatting
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    
    for cell in worksheet[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
    
    # Auto-adjust column widths
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2) * 1.2
        worksheet.column_dimensions[column_letter].width = adjusted_width
    
    # Add a chart
    chart = BarChart()
    chart.title = "Quarterly Sales"
    chart.x_axis.title = "Product"
    chart.y_axis.title = "Sales"
    
    data_ref = Reference(worksheet, min_col=2, min_row=1, max_col=5, max_row=5)
    categories = Reference(worksheet, min_col=1, min_row=2, max_row=5)
    
    chart.add_data(data_ref, titles_from_data=True)
    chart.set_categories(categories)
    
    worksheet.add_chart(chart, "G2")

print("Sample Excel report created: sample_report.xlsx")

# Read the Excel file back
df_read = pd.read_excel('sample_report.xlsx', sheet_name='Sales Data')
print("\nData read from Excel:")
print(df_read)
print("\nTotal sales by product:")
print(df_read.set_index('Product').sum(axis=1))
PYTHON
EOF
    chmod +x "$HOME/bin/excel-read-write"
    
    # Add ~/bin to PATH if not already there
    if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
        echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
        export PATH="$HOME/bin:$PATH"
    fi
    
    log_success "Helper scripts created in ~/bin"
}

setup_excel_aliases() {
    log_info "Setting up Excel automation aliases..."
    
    excel_aliases='
# Excel Automation aliases
alias excel-env="source ~/.excel-automation/bin/activate"
alias excel-jupyter="source ~/.excel-automation/bin/activate && jupyter lab"
alias excel-format="source ~/.excel-automation/bin/activate && black . && isort ."
alias excel-test="source ~/.excel-automation/bin/activate && pytest -v"

# Azure shortcuts
alias az-storage-list="az storage account list --output table"
alias az-blob-list="az storage blob list --container-name"
alias az-blob-upload="az storage blob upload --container-name"
alias az-blob-download="az storage blob download --container-name"

# Quick Excel operations
excel-to-csv() {
    if [[ -z "$1" ]]; then
        echo "Usage: excel-to-csv <excel-file>"
        return 1
    fi
    source ~/.excel-automation/bin/activate
    python -c "import pandas as pd; df = pd.read_excel('$1'); df.to_csv('${1%.xlsx}.csv', index=False); print('Converted to ${1%.xlsx}.csv')"
}

csv-to-excel() {
    if [[ -z "$1" ]]; then
        echo "Usage: csv-to-excel <csv-file>"
        return 1
    fi
    source ~/.excel-automation/bin/activate
    python -c "import pandas as pd; df = pd.read_csv('$1'); df.to_excel('${1%.csv}.xlsx', index=False); print('Converted to ${1%.csv}.xlsx')"
}

excel-merge() {
    if [[ $# -lt 2 ]]; then
        echo "Usage: excel-merge <output.xlsx> <input1.xlsx> <input2.xlsx> ..."
        return 1
    fi
    output=$1
    shift
    source ~/.excel-automation/bin/activate
    python -c "
import pandas as pd
import sys
dfs = [pd.read_excel(f) for f in sys.argv[1:]]
merged = pd.concat(dfs, ignore_index=True)
merged.to_excel('$output', index=False)
print(f'Merged {len(sys.argv[1:])} files into $output')
" "$@"
}'
    
    if [[ -f "$HOME/.bashrc" ]]; then
        add_to_file_if_missing "$HOME/.bashrc" "$excel_aliases" "Excel automation aliases"
    fi
    
    if [[ -f "$HOME/.zshrc" ]]; then
        add_to_file_if_missing "$HOME/.zshrc" "$excel_aliases" "Excel automation aliases"
    fi
    
    log_success "Excel automation aliases added to shell"
}

# Run main function
main "$@" 