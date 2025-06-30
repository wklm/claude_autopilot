#!/usr/bin/env bash

# Setup script for Hardware Development environment
# Installs: Arduino IDE, PlatformIO, hardware debugging tools, simulators

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions
ARDUINO_VERSION="2.2.1"

main() {
    show_banner "Hardware Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "arduino-cli" "version"
    show_tool_status "pio" "--version"
    show_tool_status "avrdude" "-v 2>&1 | head -n 1"
    show_tool_status "openocd" "--version 2>&1 | head -n 1"
    show_tool_status "python3" "--version"
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
    install_apt_package "libusb-1.0-0-dev" "libusb Dev"
    install_apt_package "libusb-dev" "libusb Legacy Dev"
    
    # Install Arduino IDE
    log_step "Installing Arduino IDE"
    if ! command_exists arduino-ide && ! command_exists arduino; then
        if confirm "Install Arduino IDE $ARDUINO_VERSION?"; then
            install_arduino_ide
        fi
    else
        log_info "Arduino IDE is already installed"
    fi
    
    # Install Arduino CLI
    log_step "Installing Arduino CLI"
    if ! command_exists arduino-cli; then
        if confirm "Install Arduino CLI (command line interface)?"; then
            install_arduino_cli
        fi
    else
        log_info "Arduino CLI is already installed"
        if confirm "Update Arduino CLI to latest version?"; then
            install_arduino_cli
        fi
    fi
    
    # Install PlatformIO
    log_step "Installing PlatformIO"
    if ! command_exists pio; then
        if confirm "Install PlatformIO (professional embedded development)?"; then
            install_platformio
        fi
    else
        log_info "PlatformIO is already installed"
        if confirm "Update PlatformIO to latest version?"; then
            pio upgrade
            log_success "PlatformIO updated"
        fi
    fi
    
    # Install hardware debugging tools
    log_step "Installing hardware debugging tools"
    
    # AVR tools
    if confirm "Install AVR development tools?"; then
        install_apt_package "gcc-avr" "AVR GCC"
        install_apt_package "binutils-avr" "AVR Binutils"
        install_apt_package "avr-libc" "AVR libc"
        install_apt_package "avrdude" "AVRDUDE"
        log_success "AVR tools installed"
    fi
    
    # ARM tools
    if confirm "Install ARM development tools?"; then
        install_apt_package "gcc-arm-none-eabi" "ARM GCC"
        install_apt_package "gdb-multiarch" "GDB Multiarch"
        install_apt_package "openocd" "OpenOCD"
        log_success "ARM tools installed"
    fi
    
    # ESP tools
    if confirm "Install ESP32/ESP8266 development tools?"; then
        install_esp_tools
    fi
    
    # Install serial communication tools
    log_step "Installing serial communication tools"
    if confirm "Install serial communication tools?"; then
        install_apt_package "minicom" "Minicom"
        install_apt_package "picocom" "Picocom"
        install_apt_package "screen" "Screen"
        install_apt_package "putty" "PuTTY"
        
        # Add user to dialout group for serial port access
        sudo usermod -a -G dialout $USER
        log_success "Serial tools installed (logout/login for group changes)"
    fi
    
    # Install logic analyzer and oscilloscope software
    log_step "Installing analysis tools"
    if confirm "Install logic analyzer software (sigrok/PulseView)?"; then
        install_apt_package "sigrok" "Sigrok"
        install_apt_package "pulseview" "PulseView"
        log_success "Logic analyzer tools installed"
    fi
    
    # Install circuit design software
    log_step "Installing circuit design software"
    if confirm "Install KiCad (PCB design)?"; then
        install_apt_package "kicad" "KiCad"
        install_apt_package "kicad-libraries" "KiCad Libraries"
        log_success "KiCad installed"
    fi
    
    if confirm "Install Fritzing (breadboard design)?"; then
        install_fritzing
    fi
    
    # Install simulators
    log_step "Installing simulators"
    if confirm "Install circuit simulators?"; then
        install_simulators
    fi
    
    # Create hardware workspace
    log_step "Creating hardware workspace"
    if confirm "Create hardware development workspace?"; then
        create_hardware_workspace
    fi
    
    # Install VS Code extensions
    log_step "VS Code Extensions"
    if command_exists code; then
        if confirm "Install VS Code extensions for hardware development?"; then
            code --install-extension platformio.platformio-ide
            code --install-extension vsciot-vscode.vscode-arduino
            code --install-extension ms-vscode.cpptools
            code --install-extension webfreak.debug
            log_success "VS Code extensions installed"
        fi
    fi
    
    # Setup shell aliases
    log_step "Setting up shell aliases"
    if confirm "Add hardware development aliases to shell?"; then
        setup_hardware_aliases
    fi
    
    # Configure udev rules
    log_step "Configuring device permissions"
    if confirm "Configure udev rules for common hardware?"; then
        configure_udev_rules
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "arduino-cli" "version"
    show_tool_status "pio" "--version"
    show_tool_status "avrdude" "-v 2>&1 | head -n 1"
    show_tool_status "openocd" "--version 2>&1 | head -n 1"
    
    echo
    log_success "Hardware development environment is ready!"
    log_info "Useful commands:"
    echo -e "  ${CYAN}arduino-cli board list${RESET} - List connected boards"
    echo -e "  ${CYAN}pio device list${RESET} - List serial devices"
    echo -e "  ${CYAN}pio run${RESET} - Build PlatformIO project"
    echo -e "  ${CYAN}pio run -t upload${RESET} - Upload to device"
    echo -e "  ${CYAN}minicom -D /dev/ttyUSB0${RESET} - Serial monitor"
    echo -e "  ${CYAN}pulseview${RESET} - Logic analyzer"
    echo
    log_info "Remember to logout/login for serial port access!"
}

install_arduino_ide() {
    log_info "Installing Arduino IDE $ARDUINO_VERSION..."
    
    # Download Arduino IDE AppImage
    cd /tmp
    wget -q "https://downloads.arduino.cc/arduino-ide/arduino-ide_${ARDUINO_VERSION}_Linux_64bit.AppImage"
    
    # Make executable and move to /opt
    chmod +x "arduino-ide_${ARDUINO_VERSION}_Linux_64bit.AppImage"
    sudo mkdir -p /opt/arduino
    sudo mv "arduino-ide_${ARDUINO_VERSION}_Linux_64bit.AppImage" /opt/arduino/arduino-ide.AppImage
    
    # Create desktop entry
    cat > ~/.local/share/applications/arduino-ide.desktop << EOF
[Desktop Entry]
Name=Arduino IDE
Comment=Arduino IDE
Exec=/opt/arduino/arduino-ide.AppImage
Icon=arduino
Terminal=false
Type=Application
Categories=Development;IDE;Electronics;
EOF
    
    # Create symlink
    sudo ln -sf /opt/arduino/arduino-ide.AppImage /usr/local/bin/arduino-ide
    
    cd "$SCRIPT_DIR"
    log_success "Arduino IDE installed"
}

install_arduino_cli() {
    log_info "Installing Arduino CLI..."
    
    # Install using the install script
    curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
    
    # Move to system path
    sudo mv bin/arduino-cli /usr/local/bin/
    rmdir bin
    
    # Initialize configuration
    arduino-cli config init
    
    # Update core index
    arduino-cli core update-index
    
    log_success "Arduino CLI installed"
}

install_platformio() {
    log_info "Installing PlatformIO..."
    
    # Install using pip
    pip3 install --user platformio
    
    # Add to PATH if needed
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    # Initialize PlatformIO
    pio --version
    
    log_success "PlatformIO installed"
}

install_esp_tools() {
    log_info "Installing ESP32/ESP8266 tools..."
    
    # Install esptool
    pip3 install --user esptool
    
    # Arduino CLI - add ESP32 board support
    if command_exists arduino-cli; then
        arduino-cli config add board_manager.additional_urls https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
        arduino-cli core update-index
        arduino-cli core install esp32:esp32
        log_success "ESP32 support added to Arduino CLI"
    fi
    
    # PlatformIO - ESP platforms will be installed per project
    log_info "ESP32/ESP8266 platforms will be installed by PlatformIO per project"
    
    log_success "ESP tools installed"
}

install_fritzing() {
    log_info "Installing Fritzing..."
    
    # Fritzing is not in standard repos, download from website
    log_info "Fritzing must be downloaded from: https://fritzing.org/download/"
    log_info "Consider supporting the project with a donation"
    
    # Alternative: Install from flatpak if available
    if command_exists flatpak; then
        if confirm "Install Fritzing via Flatpak?"; then
            flatpak install -y flathub org.fritzing.Fritzing
            log_success "Fritzing installed via Flatpak"
        fi
    else
        log_info "Install Flatpak first to install Fritzing automatically"
    fi
}

install_simulators() {
    log_info "Installing circuit simulators..."
    
    # SimulIDE
    if confirm "Install SimulIDE (real-time circuit simulator)?"; then
        # Download latest version
        cd /tmp
        wget -q "https://launchpad.net/simulide/+download" -O simulide-download.html
        SIMULIDE_URL=$(grep -oP 'https://[^"]+simulide_[^"]+\.tar\.gz' simulide-download.html | head -1)
        
        if [[ -n "$SIMULIDE_URL" ]]; then
            wget -q "$SIMULIDE_URL" -O simulide.tar.gz
            sudo tar -xzf simulide.tar.gz -C /opt/
            sudo ln -sf /opt/simulide_*/bin/simulide /usr/local/bin/simulide
            rm simulide.tar.gz simulide-download.html
            log_success "SimulIDE installed"
        else
            log_warning "Could not download SimulIDE automatically"
        fi
        cd "$SCRIPT_DIR"
    fi
    
    # LTspice (via Wine)
    if confirm "Install LTspice (via Wine)?"; then
        install_apt_package "wine" "Wine"
        log_info "Download LTspice from: https://www.analog.com/en/design-center/design-tools-and-calculators/ltspice-simulator.html"
        log_info "Run with: wine LTspice.exe"
    fi
    
    # ngspice
    if confirm "Install ngspice (open source SPICE simulator)?"; then
        install_apt_package "ngspice" "ngspice"
        log_success "ngspice installed"
    fi
}

create_hardware_workspace() {
    log_info "Creating hardware development workspace..."
    
    WORKSPACE="$HOME/hardware-projects"
    mkdir -p "$WORKSPACE"/{arduino,platformio,kicad,documentation,libraries}
    
    # Create Arduino project template
    cat > "$WORKSPACE/arduino/template.ino" << 'EOF'
/*
 * Arduino Project Template
 * 
 * Created: $(date)
 * Author: $USER
 */

// Pin definitions
const int LED_PIN = 13;

void setup() {
    // Initialize serial communication
    Serial.begin(115200);
    Serial.println("Arduino initialized!");
    
    // Configure pins
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    // Main program loop
    digitalWrite(LED_PIN, HIGH);
    delay(1000);
    digitalWrite(LED_PIN, LOW);
    delay(1000);
}
EOF
    
    # Create PlatformIO project template
    cat > "$WORKSPACE/platformio/platformio-template.ini" << 'EOF'
; PlatformIO Project Configuration File
;
; Build options: build flags, source filter
; Upload options: custom upload port, speed and extra flags
; Library options: dependencies, extra library storages
; Advanced options: extra scripting
;
; Please visit documentation for the other options and examples
; https://docs.platformio.org/page/projectconf.html

[env:arduino_uno]
platform = atmelavr
board = uno
framework = arduino
monitor_speed = 115200

[env:esp32]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 115200
upload_speed = 921600

[env:stm32]
platform = ststm32
board = bluepill_f103c8
framework = arduino
upload_protocol = stlink
debug_tool = stlink
EOF
    
    # Create useful scripts
    cat > "$WORKSPACE/new-arduino-project.sh" << 'EOF'
#!/usr/bin/env bash
# Create new Arduino project

if [[ -z "$1" ]]; then
    echo "Usage: $0 <project-name>"
    exit 1
fi

PROJECT="$1"
mkdir -p "$PROJECT"
cp template.ino "$PROJECT/$PROJECT.ino"
echo "Arduino project '$PROJECT' created!"
EOF
    chmod +x "$WORKSPACE/new-arduino-project.sh"
    
    cat > "$WORKSPACE/new-platformio-project.sh" << 'EOF'
#!/usr/bin/env bash
# Create new PlatformIO project

if [[ -z "$1" ]] || [[ -z "$2" ]]; then
    echo "Usage: $0 <project-name> <board>"
    echo "Example: $0 my-project esp32"
    echo "Boards: uno, esp32, esp8266, stm32, etc."
    exit 1
fi

PROJECT="$1"
BOARD="$2"

cd platformio
pio project init --board "$BOARD" --project-dir "$PROJECT"
echo "PlatformIO project '$PROJECT' created for board '$BOARD'!"
EOF
    chmod +x "$WORKSPACE/new-platformio-project.sh"
    
    log_success "Hardware workspace created at $WORKSPACE"
}

configure_udev_rules() {
    log_info "Configuring udev rules for hardware devices..."
    
    # Create udev rules file
    cat > /tmp/99-hardware-dev.rules << 'EOF'
# Arduino boards
SUBSYSTEM=="usb", ATTR{idVendor}=="2341", MODE="0666", GROUP="dialout"
SUBSYSTEM=="usb", ATTR{idVendor}=="2a03", MODE="0666", GROUP="dialout"

# ESP32/ESP8266 (CP2102, CH340, FTDI)
SUBSYSTEM=="usb", ATTR{idVendor}=="10c4", ATTR{idProduct}=="ea60", MODE="0666", GROUP="dialout"
SUBSYSTEM=="usb", ATTR{idVendor}=="1a86", ATTR{idProduct}=="7523", MODE="0666", GROUP="dialout"
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6001", MODE="0666", GROUP="dialout"

# STM32 (ST-Link)
SUBSYSTEM=="usb", ATTR{idVendor}=="0483", ATTR{idProduct}=="3748", MODE="0666", GROUP="dialout"
SUBSYSTEM=="usb", ATTR{idVendor}=="0483", ATTR{idProduct}=="374b", MODE="0666", GROUP="dialout"

# Atmel SAM-BA
SUBSYSTEM=="usb", ATTR{idVendor}=="03eb", ATTR{idProduct}=="6124", MODE="0666", GROUP="dialout"

# Logic Analyzers
SUBSYSTEM=="usb", ATTR{idVendor}=="0925", ATTR{idProduct}=="3881", MODE="0666", GROUP="dialout"
SUBSYSTEM=="usb", ATTR{idVendor}=="21a9", ATTR{idProduct}=="1001", MODE="0666", GROUP="dialout"
EOF
    
    sudo cp /tmp/99-hardware-dev.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    
    log_success "udev rules configured"
}

setup_hardware_aliases() {
    log_info "Setting up hardware development aliases..."
    
    hardware_aliases='
# Hardware development aliases
export HARDWARE_WORKSPACE="$HOME/hardware-projects"

# Arduino CLI shortcuts
alias ard-list="arduino-cli board list"
alias ard-search="arduino-cli board listall"
alias ard-compile="arduino-cli compile --fqbn"
alias ard-upload="arduino-cli upload -p"
alias ard-monitor="arduino-cli monitor -p"
alias ard-lib-search="arduino-cli lib search"
alias ard-lib-install="arduino-cli lib install"

# PlatformIO shortcuts
alias pio-init="pio project init"
alias pio-build="pio run"
alias pio-upload="pio run -t upload"
alias pio-monitor="pio device monitor"
alias pio-clean="pio run -t clean"
alias pio-lib="pio lib"

# Serial shortcuts
serial-list() {
    echo "Available serial ports:"
    ls -la /dev/tty* | grep -E "(USB|ACM)" || echo "No USB serial devices found"
}

serial-monitor() {
    if [[ -z "$1" ]]; then
        echo "Usage: serial-monitor <port> [baudrate]"
        echo "Example: serial-monitor /dev/ttyUSB0 115200"
        return 1
    fi
    minicom -D "$1" -b "${2:-115200}"
}

# Quick Arduino upload
arduino-upload() {
    if [[ -z "$1" ]] || [[ -z "$2" ]]; then
        echo "Usage: arduino-upload <sketch.ino> <board>"
        echo "Example: arduino-upload blink.ino arduino:avr:uno"
        return 1
    fi
    arduino-cli compile --fqbn "$2" "$1"
    arduino-cli upload -p $(arduino-cli board list | grep "$2" | awk '{print $1}') --fqbn "$2" "$1"
}

# Quick PlatformIO project
pio-new() {
    if [[ -z "$1" ]] || [[ -z "$2" ]]; then
        echo "Usage: pio-new <project-name> <board>"
        echo "Example: pio-new my-project esp32dev"
        return 1
    fi
    mkdir -p "$1" && cd "$1"
    pio project init --board "$2"
    echo "PlatformIO project created in $(pwd)"
}

# Workspace navigation
alias cdhw="cd $HARDWARE_WORKSPACE"
alias cdarduino="cd $HARDWARE_WORKSPACE/arduino"
alias cdpio="cd $HARDWARE_WORKSPACE/platformio"'
    
    if [[ -f "$HOME/.bashrc" ]]; then
        add_to_file_if_missing "$HOME/.bashrc" "$hardware_aliases" "Hardware development aliases"
    fi
    
    if [[ -f "$HOME/.zshrc" ]]; then
        add_to_file_if_missing "$HOME/.zshrc" "$hardware_aliases" "Hardware development aliases"
    fi
    
    log_success "Hardware development aliases added to shell"
}

# Run main function
main "$@" 