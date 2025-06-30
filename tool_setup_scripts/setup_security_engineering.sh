#!/usr/bin/env bash

# Setup script for Security Engineering environment
# Installs: Security scanning tools, penetration testing frameworks, analysis tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

main() {
    show_banner "Security Engineering Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "nmap" "--version | head -n 1"
    show_tool_status "metasploit" "version 2>/dev/null | head -n 1"
    show_tool_status "burpsuite" "--version 2>/dev/null || echo 'Not installed'"
    show_tool_status "python3" "--version"
    show_tool_status "go" "version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "build-essential" "Build Essential"
    install_apt_package "git" "Git"
    install_apt_package "curl" "cURL"
    install_apt_package "wget" "Wget"
    install_apt_package "python3" "Python 3"
    install_apt_package "python3-pip" "Python 3 pip"
    install_apt_package "python3-venv" "Python 3 venv"
    install_apt_package "libssl-dev" "OpenSSL Dev"
    install_apt_package "libffi-dev" "FFI Dev"
    install_apt_package "net-tools" "Net Tools"
    
    # Install Go (required for many security tools)
    log_step "Installing Go"
    if ! command_exists go; then
        if confirm "Install Go (required for many security tools)?"; then
            install_go
        fi
    else
        log_info "Go is already installed"
        go_version=$(go version | awk '{print $3}')
        log_info "Current version: $go_version"
    fi
    
    # Create security tools environment
    log_step "Setting up Security Tools Environment"
    log_info "Creating virtual environment for Python security tools..."
    SEC_VENV="$HOME/.security-tools"
    if [[ ! -d "$SEC_VENV" ]]; then
        python3 -m venv "$SEC_VENV"
    fi
    
    # Activate virtual environment
    source "$SEC_VENV/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install network scanning tools
    log_step "Installing Network Scanning Tools"
    if confirm "Install network scanning tools?"; then
        # Nmap
        install_apt_package "nmap" "Nmap"
        
        # Masscan
        if confirm "Install Masscan (fast port scanner)?"; then
            install_masscan
        fi
        
        # Zmap
        if confirm "Install ZMap (internet-wide scanner)?"; then
            install_apt_package "zmap" "ZMap"
        fi
        
        log_success "Network scanning tools installed"
    fi
    
    # Install vulnerability scanners
    log_step "Installing Vulnerability Scanners"
    if confirm "Install vulnerability scanning tools?"; then
        # Nikto
        install_apt_package "nikto" "Nikto"
        
        # OWASP ZAP
        if confirm "Install OWASP ZAP?"; then
            install_owasp_zap
        fi
        
        # Nuclei
        if confirm "Install Nuclei (template-based scanner)?"; then
            go install -v github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest
            log_success "Nuclei installed"
        fi
        
        log_success "Vulnerability scanners installed"
    fi
    
    # Install penetration testing frameworks
    log_step "Installing Penetration Testing Frameworks"
    if confirm "Install Metasploit Framework?"; then
        install_metasploit
    fi
    
    # Install web application testing tools
    log_step "Installing Web Application Testing Tools"
    if confirm "Install web app testing tools?"; then
        # Burp Suite Community
        if confirm "Download Burp Suite Community Edition?"; then
            install_burpsuite
        fi
        
        # SQLMap
        pip install sqlmap
        
        # Gobuster
        go install github.com/OJ/gobuster/v3@latest
        
        # Dirb
        install_apt_package "dirb" "DIRB"
        
        # Hydra
        install_apt_package "hydra" "Hydra"
        
        log_success "Web app testing tools installed"
    fi
    
    # Install wireless security tools
    log_step "Installing Wireless Security Tools"
    if confirm "Install wireless security tools?"; then
        install_apt_package "aircrack-ng" "Aircrack-ng"
        install_apt_package "kismet" "Kismet"
        install_apt_package "reaver" "Reaver"
        log_success "Wireless security tools installed"
    fi
    
    # Install forensics tools
    log_step "Installing Forensics Tools"
    if confirm "Install forensics tools?"; then
        install_apt_package "foremost" "Foremost"
        install_apt_package "binwalk" "Binwalk"
        install_apt_package "volatility" "Volatility"
        install_apt_package "steghide" "Steghide"
        
        # Ghidra
        if confirm "Install Ghidra (reverse engineering)?"; then
            install_ghidra
        fi
        
        log_success "Forensics tools installed"
    fi
    
    # Install Python security libraries
    log_step "Installing Python Security Libraries"
    if confirm "Install Python security libraries?"; then
        pip install scapy
        pip install impacket
        pip install pwntools
        pip install cryptography
        pip install requests
        pip install beautifulsoup4
        pip install paramiko
        pip install pyshark
        pip install python-nmap
        pip install shodan
        
        log_success "Python security libraries installed"
    fi
    
    # Install exploitation tools
    log_step "Installing Exploitation Tools"
    if confirm "Install exploitation tools?"; then
        # John the Ripper
        install_apt_package "john" "John the Ripper"
        
        # Hashcat
        install_apt_package "hashcat" "Hashcat"
        
        # Radare2
        install_apt_package "radare2" "Radare2"
        
        # GDB with peda
        install_apt_package "gdb" "GDB"
        if confirm "Install GDB PEDA (Python Exploit Development Assistance)?"; then
            git clone https://github.com/longld/peda.git ~/peda
            echo "source ~/peda/peda.py" >> ~/.gdbinit
            log_success "GDB PEDA installed"
        fi
        
        log_success "Exploitation tools installed"
    fi
    
    # Install container security tools
    log_step "Installing Container Security Tools"
    if confirm "Install container security tools?"; then
        # Trivy
        if confirm "Install Trivy (container scanner)?"; then
            wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
            echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
            update_apt
            install_apt_package "trivy" "Trivy"
        fi
        
        # Grype
        if confirm "Install Grype (vulnerability scanner)?"; then
            curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin
            log_success "Grype installed"
        fi
        
        log_success "Container security tools installed"
    fi
    
    # Create security workspace
    log_step "Creating security workspace"
    if confirm "Create security engineering workspace?"; then
        create_security_workspace
    fi
    
    # Deactivate virtual environment
    deactivate
    
    # Setup shell aliases
    log_step "Setting up shell aliases"
    if confirm "Add security engineering aliases to shell?"; then
        setup_security_aliases
    fi
    
    # Install VS Code extensions
    log_step "VS Code Extensions"
    if command_exists code; then
        if confirm "Install VS Code extensions for security?"; then
            code --install-extension ms-python.python
            code --install-extension golang.go
            code --install-extension coolbear.systemd-unit-file
            log_success "VS Code extensions installed"
        fi
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "nmap" "--version | head -n 1"
    show_tool_status "nikto" "-Version 2>&1 | grep Version"
    show_tool_status "sqlmap" "--version"
    show_tool_status "go" "version"
    
    echo
    log_success "Security Engineering environment is ready!"
    log_info "Useful commands:"
    echo -e "  ${CYAN}source ~/.security-tools/bin/activate${RESET} - Activate Python environment"
    echo -e "  ${CYAN}nmap -sV <target>${RESET} - Service version scan"
    echo -e "  ${CYAN}nikto -h <target>${RESET} - Web vulnerability scan"
    echo -e "  ${CYAN}nuclei -u <target>${RESET} - Template-based scanning"
    echo -e "  ${CYAN}gobuster dir -u <url> -w <wordlist>${RESET} - Directory bruteforce"
    echo
    log_warning "Remember: Only use these tools on systems you own or have permission to test!"
}

install_go() {
    log_info "Installing Go..."
    
    GO_VERSION="1.21.5"
    wget -q https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz
    sudo tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz
    rm go${GO_VERSION}.linux-amd64.tar.gz
    
    # Add Go to PATH
    go_path_config='
# Go configuration
export PATH=$PATH:/usr/local/go/bin
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin'
    
    if [[ -f "$HOME/.bashrc" ]]; then
        add_to_file_if_missing "$HOME/.bashrc" "$go_path_config" "Go configuration"
    fi
    
    if [[ -f "$HOME/.zshrc" ]]; then
        add_to_file_if_missing "$HOME/.zshrc" "$go_path_config" "Go configuration"
    fi
    
    # Source for current session
    export PATH=$PATH:/usr/local/go/bin
    export GOPATH=$HOME/go
    export PATH=$PATH:$GOPATH/bin
    
    log_success "Go installed"
}

install_masscan() {
    log_info "Installing Masscan..."
    
    cd /tmp
    git clone https://github.com/robertdavidgraham/masscan
    cd masscan
    make -j$(nproc)
    sudo make install
    cd "$SCRIPT_DIR"
    rm -rf /tmp/masscan
    
    log_success "Masscan installed"
}

install_metasploit() {
    log_info "Installing Metasploit Framework..."
    
    # Add Metasploit repository
    curl https://raw.githubusercontent.com/rapid7/metasploit-omnibus/master/config/templates/metasploit-framework-wrappers/msfupdate.erb > /tmp/msfinstall
    chmod 755 /tmp/msfinstall
    /tmp/msfinstall
    rm /tmp/msfinstall
    
    # Initialize database
    sudo msfdb init
    
    log_success "Metasploit installed"
}

install_owasp_zap() {
    log_info "Downloading OWASP ZAP..."
    
    # Download ZAP
    ZAP_VERSION="2.14.0"
    wget -q "https://github.com/zaproxy/zaproxy/releases/download/v${ZAP_VERSION}/ZAP_${ZAP_VERSION}_Linux.tar.gz" -O /tmp/zap.tar.gz
    
    # Extract to /opt
    sudo tar -xzf /tmp/zap.tar.gz -C /opt/
    sudo mv /opt/ZAP_${ZAP_VERSION} /opt/zap
    rm /tmp/zap.tar.gz
    
    # Create symlink
    sudo ln -sf /opt/zap/zap.sh /usr/local/bin/zaproxy
    
    log_success "OWASP ZAP installed"
}

install_burpsuite() {
    log_info "Burp Suite Community Edition download instructions:"
    echo
    echo "1. Download from: https://portswigger.net/burp/communitydownload"
    echo "2. Run installer: sh burpsuite_community_linux_*.sh"
    echo "3. Follow installation wizard"
    echo
    log_info "Burp Suite requires manual download due to license agreement"
}

install_ghidra() {
    log_info "Installing Ghidra..."
    
    # Check Java
    if ! command_exists java; then
        log_error "Java is required for Ghidra. Install OpenJDK 17 first."
        return
    fi
    
    # Download Ghidra
    GHIDRA_VERSION="11.0_PUBLIC"
    GHIDRA_DATE="20231222"
    wget -q "https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_${GHIDRA_VERSION}/ghidra_${GHIDRA_VERSION}_${GHIDRA_DATE}.zip" -O /tmp/ghidra.zip
    
    # Extract to /opt
    sudo unzip -q /tmp/ghidra.zip -d /opt/
    sudo mv /opt/ghidra_${GHIDRA_VERSION}_PUBLIC /opt/ghidra
    rm /tmp/ghidra.zip
    
    # Create launcher script
    cat > /tmp/ghidra << 'EOF'
#!/usr/bin/env bash
/opt/ghidra/ghidraRun
EOF
    sudo mv /tmp/ghidra /usr/local/bin/
    sudo chmod +x /usr/local/bin/ghidra
    
    log_success "Ghidra installed"
}

create_security_workspace() {
    log_info "Creating security engineering workspace..."
    
    WORKSPACE="$HOME/security-workspace"
    mkdir -p "$WORKSPACE"/{tools,wordlists,reports,scripts,targets}
    
    # Download common wordlists
    log_info "Downloading SecLists wordlists..."
    cd "$WORKSPACE/wordlists"
    if [[ ! -d "SecLists" ]]; then
        git clone --depth 1 https://github.com/danielmiessler/SecLists.git
    fi
    
    # Create useful scripts
    cat > "$WORKSPACE/scripts/quick-scan.sh" << 'EOF'
#!/usr/bin/env bash
# Quick security scan script

if [[ -z "$1" ]]; then
    echo "Usage: $0 <target>"
    exit 1
fi

TARGET="$1"
REPORT_DIR="../reports/$(date +%Y%m%d_%H%M%S)_${TARGET//\//_}"
mkdir -p "$REPORT_DIR"

echo "Starting security scan of $TARGET"
echo "Reports will be saved to: $REPORT_DIR"

# Nmap scan
echo "[*] Running Nmap scan..."
nmap -sV -sC -oA "$REPORT_DIR/nmap" "$TARGET"

# If web server detected, run web scans
if grep -q "80/tcp\|443/tcp\|8080/tcp\|8443/tcp" "$REPORT_DIR/nmap.nmap"; then
    echo "[*] Web server detected, running additional scans..."
    
    # Nikto scan
    echo "[*] Running Nikto scan..."
    nikto -h "$TARGET" -o "$REPORT_DIR/nikto.txt"
    
    # Gobuster
    echo "[*] Running directory enumeration..."
    gobuster dir -u "http://$TARGET" -w ../wordlists/SecLists/Discovery/Web-Content/common.txt -o "$REPORT_DIR/gobuster.txt"
fi

echo "[*] Scan complete! Results saved to $REPORT_DIR"
EOF
    chmod +x "$WORKSPACE/scripts/quick-scan.sh"
    
    # Create Python security script template
    cat > "$WORKSPACE/scripts/security_template.py" << 'EOF'
#!/usr/bin/env python3
"""
Security testing script template
"""

import argparse
import sys
import requests
from scapy.all import *
import nmap

def banner():
    print("""
    ╔═══════════════════════════════════╗
    ║   Security Testing Script         ║
    ╚═══════════════════════════════════╝
    """)

def nmap_scan(target):
    """Perform nmap scan"""
    nm = nmap.PortScanner()
    print(f"[*] Scanning {target}...")
    nm.scan(target, '1-1000')
    
    for host in nm.all_hosts():
        print(f"\nHost: {host} ({nm[host].hostname()})")
        print(f"State: {nm[host].state()}")
        
        for proto in nm[host].all_protocols():
            ports = nm[host][proto].keys()
            for port in ports:
                state = nm[host][proto][port]['state']
                print(f"  {port}/{proto}: {state}")

def main():
    parser = argparse.ArgumentParser(description='Security testing script')
    parser.add_argument('target', help='Target to scan')
    parser.add_argument('-p', '--port', type=int, help='Specific port to scan')
    args = parser.parse_args()
    
    banner()
    nmap_scan(args.target)

if __name__ == '__main__':
    main()
EOF
    
    cd "$SCRIPT_DIR"
    log_success "Security workspace created at $WORKSPACE"
}

setup_security_aliases() {
    log_info "Setting up security engineering aliases..."
    
    security_aliases='
# Security engineering aliases
export SECURITY_WORKSPACE="$HOME/security-workspace"
alias sec-env="source ~/.security-tools/bin/activate"

# Navigation
alias cdsec="cd $SECURITY_WORKSPACE"
alias cdtools="cd $SECURITY_WORKSPACE/tools"
alias cdreports="cd $SECURITY_WORKSPACE/reports"

# Nmap shortcuts
alias nmap-quick="nmap -sV -sC"
alias nmap-full="nmap -sS -sV -sC -A -p-"
alias nmap-udp="sudo nmap -sU"
alias nmap-vulns="nmap --script vuln"

# Web scanning
alias nikto-ssl="nikto -ssl"
alias gobuster-common="gobuster dir -w $SECURITY_WORKSPACE/wordlists/SecLists/Discovery/Web-Content/common.txt"
alias gobuster-big="gobuster dir -w $SECURITY_WORKSPACE/wordlists/SecLists/Discovery/Web-Content/big.txt"

# Quick functions
port-scan() {
    if [[ -z "$1" ]]; then
        echo "Usage: port-scan <target>"
        return 1
    fi
    nmap -sV -sC "$1" | grep -E "^[0-9]+/(tcp|udp)"
}

web-headers() {
    if [[ -z "$1" ]]; then
        echo "Usage: web-headers <url>"
        return 1
    fi
    curl -I -s "$1" | grep -E "^[A-Z]"
}

ssl-check() {
    if [[ -z "$1" ]]; then
        echo "Usage: ssl-check <domain>"
        return 1
    fi
    echo | openssl s_client -connect "$1:443" -servername "$1" 2>/dev/null | openssl x509 -noout -dates -subject -issuer
}

# Wordlist shortcuts
export WORDLIST_DIR="$SECURITY_WORKSPACE/wordlists/SecLists"
export WL_COMMON="$WORDLIST_DIR/Discovery/Web-Content/common.txt"
export WL_BIG="$WORDLIST_DIR/Discovery/Web-Content/big.txt"
export WL_USERS="$WORDLIST_DIR/Usernames/top-usernames-shortlist.txt"
export WL_PASS="$WORDLIST_DIR/Passwords/Common-Credentials/10k-most-common.txt"

# Security reminders
alias sec-disclaimer="echo \"⚠️  Only test systems you own or have written permission to test!\""'
    
    if [[ -f "$HOME/.bashrc" ]]; then
        add_to_file_if_missing "$HOME/.bashrc" "$security_aliases" "Security engineering aliases"
    fi
    
    if [[ -f "$HOME/.zshrc" ]]; then
        add_to_file_if_missing "$HOME/.zshrc" "$security_aliases" "Security engineering aliases"
    fi
    
    log_success "Security engineering aliases added to shell"
}

# Run main function
main "$@" 