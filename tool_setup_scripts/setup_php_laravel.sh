#!/usr/bin/env bash

# Setup script for PHP and Laravel development environment
# Installs: PHP 8.2+, Composer, Laravel, and related tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions and requirements
PHP_VERSION="8.2"
COMPOSER_VERSION="latest"

main() {
    show_banner "PHP & Laravel Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "php" "--version"
    show_tool_status "composer" "--version"
    show_tool_status "laravel" "--version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Add PHP PPA
    log_step "Adding PHP repository"
    if ! grep -q "ondrej/php" /etc/apt/sources.list.d/*.list 2>/dev/null; then
        if confirm "Add Ondřej's PHP PPA for latest PHP versions?"; then
            sudo add-apt-repository -y ppa:ondrej/php
            sudo apt-get update
            log_success "PHP repository added"
        fi
    else
        log_info "PHP repository already added"
    fi
    
    # Install PHP and extensions
    log_step "Installing PHP $PHP_VERSION and extensions"
    php_packages=(
        "php$PHP_VERSION"
        "php$PHP_VERSION-cli"
        "php$PHP_VERSION-common"
        "php$PHP_VERSION-mysql"
        "php$PHP_VERSION-pgsql"
        "php$PHP_VERSION-sqlite3"
        "php$PHP_VERSION-xml"
        "php$PHP_VERSION-curl"
        "php$PHP_VERSION-zip"
        "php$PHP_VERSION-bcmath"
        "php$PHP_VERSION-mbstring"
        "php$PHP_VERSION-gd"
        "php$PHP_VERSION-intl"
        "php$PHP_VERSION-redis"
        "php$PHP_VERSION-opcache"
        "php$PHP_VERSION-readline"
        "php$PHP_VERSION-dev"
    )
    
    for package in "${php_packages[@]}"; do
        install_apt_package "$package" "$package"
    done
    
    # Install additional system dependencies
    log_step "Installing additional dependencies"
    install_apt_package "git" "Git"
    install_apt_package "curl" "cURL"
    install_apt_package "unzip" "Unzip"
    install_apt_package "nodejs" "Node.js"
    install_apt_package "npm" "npm"
    
    # Install Composer
    log_step "Installing Composer"
    if command_exists composer; then
        log_info "Composer is already installed"
        if confirm "Update Composer to latest version?"; then
            sudo composer self-update
            log_success "Composer updated"
        fi
    else
        if confirm "Install Composer (PHP package manager)?"; then
            cd /tmp
            curl -sS https://getcomposer.org/installer | php
            sudo mv composer.phar /usr/local/bin/composer
            sudo chmod +x /usr/local/bin/composer
            log_success "Composer installed"
        fi
    fi
    
    # Install Laravel Installer
    if command_exists composer; then
        log_step "Installing Laravel Installer"
        if ! command_exists laravel; then
            if confirm "Install Laravel Installer globally?"; then
                composer global require laravel/installer
                
                # Add Composer global bin to PATH
                composer_path_line='export PATH="$HOME/.composer/vendor/bin:$HOME/.config/composer/vendor/bin:$PATH"'
                
                if [[ -f "$HOME/.bashrc" ]]; then
                    add_to_file_if_missing "$HOME/.bashrc" "$composer_path_line" "Composer PATH"
                fi
                
                if [[ -f "$HOME/.zshrc" ]]; then
                    add_to_file_if_missing "$HOME/.zshrc" "$composer_path_line" "Composer PATH"
                fi
                
                # Source for current session
                export PATH="$HOME/.composer/vendor/bin:$HOME/.config/composer/vendor/bin:$PATH"
                
                log_success "Laravel Installer installed"
            fi
        else
            log_info "Laravel Installer is already installed"
        fi
        
        # Install global PHP tools
        log_step "Installing PHP development tools"
        
        # PHP CS Fixer
        if ! command_exists php-cs-fixer; then
            if confirm "Install PHP CS Fixer (code style fixer)?"; then
                composer global require friendsofphp/php-cs-fixer
                log_success "PHP CS Fixer installed"
            fi
        else
            log_info "PHP CS Fixer is already installed"
        fi
        
        # PHPStan
        if ! command_exists phpstan; then
            if confirm "Install PHPStan (static analysis tool)?"; then
                composer global require phpstan/phpstan
                log_success "PHPStan installed"
            fi
        else
            log_info "PHPStan is already installed"
        fi
        
        # Laravel Pint
        if ! command_exists pint; then
            if confirm "Install Laravel Pint (code style fixer)?"; then
                composer global require laravel/pint
                log_success "Laravel Pint installed"
            fi
        else
            log_info "Laravel Pint is already installed"
        fi
    fi
    
    # Configure PHP
    log_step "Configuring PHP"
    if confirm "Optimize PHP configuration for development?"; then
        # Create custom PHP configuration
        cat << EOF | sudo tee /etc/php/$PHP_VERSION/cli/conf.d/99-development.ini
; Development settings
display_errors = On
display_startup_errors = On
error_reporting = E_ALL
memory_limit = 512M
max_execution_time = 300
upload_max_filesize = 100M
post_max_size = 100M
EOF
        log_success "PHP configured for development"
    fi
    
    # Install database (MySQL/MariaDB)
    log_step "Database setup"
    if ! command_exists mysql; then
        if confirm "Install MariaDB (MySQL compatible database)?"; then
            install_apt_package "mariadb-server" "MariaDB Server"
            sudo systemctl start mariadb
            sudo systemctl enable mariadb
            
            if confirm "Run mysql_secure_installation to secure MariaDB?"; then
                sudo mysql_secure_installation
            fi
            log_success "MariaDB installed"
        fi
    else
        log_info "MySQL/MariaDB is already installed"
    fi
    
    # Install Redis (optional)
    if confirm "Install Redis (for caching/queues)?"; then
        install_apt_package "redis-server" "Redis Server"
        sudo systemctl start redis-server
        sudo systemctl enable redis-server
        log_success "Redis installed"
    fi
    
    # VS Code extensions
    if command_exists code; then
        log_step "VS Code PHP/Laravel extensions"
        if confirm "Install PHP and Laravel VS Code extensions?"; then
            code --install-extension bmewburn.vscode-intelephense-client
            code --install-extension onecentlin.laravel-blade
            code --install-extension amiralizadeh9480.laravel-extra-intellisense
            log_success "VS Code extensions installed"
        fi
    fi
    
    # Create Laravel projects directory
    log_step "Setting up workspace"
    LARAVEL_WORKSPACE="$HOME/laravel-projects"
    if confirm "Create Laravel workspace directory at $LARAVEL_WORKSPACE?"; then
        create_directory "$LARAVEL_WORKSPACE" "Laravel workspace"
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "php" "--version"
    show_tool_status "composer" "--version"
    show_tool_status "laravel" "--version"
    show_tool_status "phpstan" "--version"
    show_tool_status "mysql" "--version"
    
    echo
    log_success "PHP & Laravel development environment is ready!"
    log_info "To create a new Laravel project, run:"
    echo -e "  ${CYAN}laravel new my-project${RESET}"
    echo -e "  ${CYAN}# OR${RESET}"
    echo -e "  ${CYAN}composer create-project laravel/laravel my-project${RESET}"
    echo -e "  ${CYAN}cd my-project${RESET}"
    echo -e "  ${CYAN}php artisan serve${RESET}"
}

# Run main function
main "$@" 