#!/usr/bin/env bash

# Setup script for Java Enterprise development environment
# Installs: Java 21 LTS, Gradle, Maven, and enterprise dev tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions
JAVA_VERSION="21"
GRADLE_VERSION="8.11"
MAVEN_VERSION="3.9.10"

main() {
    show_banner "Java Enterprise Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "java" "-version"
    show_tool_status "javac" "-version"
    show_tool_status "gradle" "--version"
    show_tool_status "mvn" "--version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "build-essential" "Build Essential"
    install_apt_package "curl" "cURL"
    install_apt_package "git" "Git"
    install_apt_package "wget" "wget"
    install_apt_package "unzip" "unzip"
    
    # Install Java
    log_step "Installing Java"
    install_java
    
    # Install SDKMAN for managing Java tools
    log_step "Installing SDKMAN"
    if [[ -f "$HOME/.sdkman/bin/sdkman-init.sh" ]]; then
        log_info "SDKMAN is already installed"
        source "$HOME/.sdkman/bin/sdkman-init.sh"
        
        if confirm "Update SDKMAN to latest version?"; then
            sdk selfupdate
            log_success "SDKMAN updated"
        fi
    else
        if confirm "Install SDKMAN (Software Development Kit Manager)?"; then
            curl -s "https://get.sdkman.io" | bash
            source "$HOME/.sdkman/bin/sdkman-init.sh"
            
            # Add to shell configs
            log_info "SDKMAN installed. Restart your shell or run: source ~/.sdkman/bin/sdkman-init.sh"
        fi
    fi
    
    # Install Gradle via SDKMAN
    if [[ -f "$HOME/.sdkman/bin/sdkman-init.sh" ]]; then
        source "$HOME/.sdkman/bin/sdkman-init.sh"
        
        log_step "Installing Gradle"
        if ! command_exists gradle; then
            if confirm "Install Gradle $GRADLE_VERSION?"; then
                sdk install gradle $GRADLE_VERSION
                log_success "Gradle $GRADLE_VERSION installed"
            fi
        else
            log_info "Gradle is already installed"
            current_gradle=$(gradle --version | grep "Gradle" | awk '{print $2}')
            log_info "Current Gradle version: $current_gradle"
        fi
        
        # Install Maven via SDKMAN
        log_step "Installing Maven"
        if ! command_exists mvn; then
            if confirm "Install Maven $MAVEN_VERSION?"; then
                sdk install maven $MAVEN_VERSION
                log_success "Maven $MAVEN_VERSION installed"
            fi
        else
            log_info "Maven is already installed"
            current_maven=$(mvn --version | grep "Apache Maven" | awk '{print $3}')
            log_info "Current Maven version: $current_maven"
        fi
    fi
    
    # Install development tools
    log_step "Installing development tools"
    
    # Install JBang for scripting
    if ! command_exists jbang; then
        if confirm "Install JBang (Java scripting tool)?"; then
            curl -Ls https://sh.jbang.dev | bash -s - app setup
            export PATH="$HOME/.jbang/bin:$PATH"
            log_success "JBang installed"
        fi
    else
        log_info "JBang is already installed"
    fi
    
    # Install native-image for GraalVM
    if confirm "Install GraalVM native-image support?"; then
        if command_exists gu; then
            gu install native-image
            log_success "native-image installed"
        else
            log_warning "GraalVM not detected. Install GraalVM first for native-image support"
        fi
    fi
    
    # Setup Docker for Testcontainers
    log_step "Setting up Docker for Testcontainers"
    if ! command_exists docker; then
        if confirm "Install Docker (required for Testcontainers)?"; then
            install_docker
        fi
    else
        log_info "Docker is already installed"
        docker --version
    fi
    
    # Create Gradle wrapper script
    log_step "Creating development helpers"
    if confirm "Create Gradle init script for enterprise projects?"; then
        create_gradle_init
    fi
    
    # Setup global gitignore for Java
    log_step "Configuring Git for Java development"
    if confirm "Add Java patterns to global .gitignore?"; then
        gitignore_file="$HOME/.gitignore_global"
        touch "$gitignore_file"
        
        # Java patterns
        java_patterns=(
            "# Compiled class files"
            "*.class"
            ""
            "# Log files"
            "*.log"
            ""
            "# BlueJ files"
            "*.ctxt"
            ""
            "# Mobile Tools for Java (J2ME)"
            ".mtj.tmp/"
            ""
            "# Package Files"
            "*.jar"
            "*.war"
            "*.nar"
            "*.ear"
            "*.zip"
            "*.tar.gz"
            "*.rar"
            ""
            "# Virtual machine crash logs"
            "hs_err_pid*"
            ""
            "# Build directories"
            "target/"
            "build/"
            "out/"
            ""
            "# Gradle"
            ".gradle/"
            "gradle-app.setting"
            "!gradle-wrapper.jar"
            ".gradletasknamecache"
            ""
            "# Maven"
            "pom.xml.tag"
            "pom.xml.releaseBackup"
            "pom.xml.versionsBackup"
            "pom.xml.next"
            "release.properties"
            "dependency-reduced-pom.xml"
            ""
            "# IDE files"
            ".idea/"
            "*.iml"
            "*.iws"
            "*.ipr"
            ".vscode/"
            ".settings/"
            ".project"
            ".classpath"
            ""
            "# OS files"
            ".DS_Store"
            "Thumbs.db"
            ""
            "# Environment"
            ".env"
            ".env.local"
        )
        
        for pattern in "${java_patterns[@]}"; do
            if [[ -n "$pattern" ]] && ! grep -Fxq "$pattern" "$gitignore_file" 2>/dev/null; then
                echo "$pattern" >> "$gitignore_file"
            elif [[ -z "$pattern" ]]; then
                echo "" >> "$gitignore_file"
            fi
        done
        
        git config --global core.excludesfile "$gitignore_file"
        log_success "Global .gitignore configured for Java"
    fi
    
    # Configure Maven settings
    log_step "Configuring Maven"
    if confirm "Create optimized Maven settings.xml?"; then
        create_maven_settings
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "java" "-version"
    show_tool_status "javac" "-version"
    show_tool_status "gradle" "--version"
    show_tool_status "mvn" "--version"
    show_tool_status "jbang" "--version"
    show_tool_status "docker" "--version"
    
    echo
    log_success "Java Enterprise development environment is ready!"
    log_info "To create a new Spring Boot project:"
    echo -e "  ${CYAN}curl https://start.spring.io/starter.zip \\${RESET}"
    echo -e "  ${CYAN}  -d dependencies=web,data-jpa,security,actuator \\${RESET}"
    echo -e "  ${CYAN}  -d type=gradle-project \\${RESET}"
    echo -e "  ${CYAN}  -d language=java \\${RESET}"
    echo -e "  ${CYAN}  -d javaVersion=21 \\${RESET}"
    echo -e "  ${CYAN}  -d groupId=com.example \\${RESET}"
    echo -e "  ${CYAN}  -d artifactId=demo \\${RESET}"
    echo -e "  ${CYAN}  -o demo.zip${RESET}"
    echo
    log_info "Or use Spring CLI: spring init --list"
}

install_java() {
    if command_exists java; then
        current_java=$(java -version 2>&1 | head -n 1 | awk -F '"' '{print $2}' | cut -d. -f1)
        log_info "Current Java version: $(java -version 2>&1 | head -n 1)"
        
        if [[ "$current_java" -ge "$JAVA_VERSION" ]]; then
            log_success "Java $current_java is sufficient"
            return 0
        fi
    fi
    
    log_warning "Java $JAVA_VERSION or higher is required"
    
    # Offer different Java distributions
    echo
    echo "Choose Java distribution to install:"
    echo "1) OpenJDK (Ubuntu default)"
    echo "2) Eclipse Temurin (Adoptium)"
    echo "3) Amazon Corretto"
    echo "4) Liberica JDK (with JavaFX)"
    echo "5) Skip Java installation"
    echo
    
    read -p "Enter choice (1-5): " choice
    
    case $choice in
        1)
            install_apt_package "openjdk-${JAVA_VERSION}-jdk" "OpenJDK ${JAVA_VERSION}"
            ;;
        2)
            install_temurin
            ;;
        3)
            install_corretto
            ;;
        4)
            install_liberica
            ;;
        5)
            log_warning "Skipping Java installation"
            ;;
        *)
            log_error "Invalid choice"
            exit 1
            ;;
    esac
}

install_temurin() {
    log_info "Installing Eclipse Temurin $JAVA_VERSION..."
    
    # Add Adoptium repository
    wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | sudo apt-key add -
    echo "deb https://packages.adoptium.net/artifactory/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/adoptium.list
    
    sudo apt-get update
    install_apt_package "temurin-${JAVA_VERSION}-jdk" "Eclipse Temurin ${JAVA_VERSION}"
}

install_corretto() {
    log_info "Installing Amazon Corretto $JAVA_VERSION..."
    
    wget -O- https://apt.corretto.aws/corretto.key | sudo apt-key add -
    sudo add-apt-repository 'deb https://apt.corretto.aws stable main'
    
    sudo apt-get update
    install_apt_package "java-${JAVA_VERSION}-amazon-corretto-jdk" "Amazon Corretto ${JAVA_VERSION}"
}

install_liberica() {
    log_info "Installing Liberica JDK $JAVA_VERSION..."
    
    wget -q -O - https://download.bell-sw.com/pki/GPG-KEY-bellsoft | sudo apt-key add -
    echo "deb [arch=amd64] https://apt.bell-sw.com/ stable main" | sudo tee /etc/apt/sources.list.d/bellsoft.list
    
    sudo apt-get update
    install_apt_package "bellsoft-java${JAVA_VERSION}-full" "Liberica JDK ${JAVA_VERSION} Full"
}

install_docker() {
    log_info "Installing Docker..."
    
    # Remove old versions
    sudo apt-get remove -y docker docker-engine docker.io containerd runc || true
    
    # Install prerequisites
    install_apt_package "ca-certificates" "CA certificates"
    install_apt_package "gnupg" "GnuPG"
    install_apt_package "lsb-release" "LSB Release"
    
    # Add Docker's official GPG key
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    
    # Add Docker repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker
    sudo apt-get update
    install_apt_package "docker-ce" "Docker CE"
    install_apt_package "docker-ce-cli" "Docker CLI"
    install_apt_package "containerd.io" "containerd"
    install_apt_package "docker-buildx-plugin" "Docker Buildx"
    install_apt_package "docker-compose-plugin" "Docker Compose"
    
    # Add user to docker group
    if confirm "Add current user to docker group (requires logout/login)?"; then
        sudo usermod -aG docker $USER
        log_success "User added to docker group. Please log out and back in for changes to take effect."
    fi
}

create_gradle_init() {
    mkdir -p ~/.gradle
    cat > ~/.gradle/init.gradle.kts << 'EOF'
// Global Gradle initialization script

allprojects {
    repositories {
        mavenCentral()
        google()
        gradlePluginPortal()
    }
}

gradle.projectsLoaded {
    rootProject.allprojects {
        tasks.withType<JavaCompile> {
            options.encoding = "UTF-8"
            options.compilerArgs.add("-parameters")
            
            // Enable preview features if using latest Java
            // options.compilerArgs.add("--enable-preview")
        }
        
        tasks.withType<Test> {
            useJUnitPlatform()
            
            // Parallel test execution
            maxParallelForks = (Runtime.getRuntime().availableProcessors() / 2).coerceAtLeast(1)
            
            testLogging {
                events("passed", "skipped", "failed")
                exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
            }
        }
    }
}
EOF
    
    log_success "Created ~/.gradle/init.gradle.kts"
}

create_maven_settings() {
    mkdir -p ~/.m2
    cat > ~/.m2/settings.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<settings xmlns="http://maven.apache.org/SETTINGS/1.2.0"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.2.0
                              https://maven.apache.org/xsd/settings-1.2.0.xsd">
    
    <!-- Parallel artifact downloads -->
    <offline>false</offline>
    <interactiveMode>true</interactiveMode>
    <usePluginRegistry>false</usePluginRegistry>
    
    <profiles>
        <profile>
            <id>default</id>
            <properties>
                <!-- Parallel builds -->
                <maven.build.threads>1C</maven.build.threads>
                
                <!-- Memory settings -->
                <maven.compiler.maxmem>1024m</maven.compiler.maxmem>
                
                <!-- Encoding -->
                <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
                <project.reporting.outputEncoding>UTF-8</project.reporting.outputEncoding>
            </properties>
        </profile>
    </profiles>
    
    <activeProfiles>
        <activeProfile>default</activeProfile>
    </activeProfiles>
    
    <!-- Mirror for faster downloads (optional) -->
    <!--
    <mirrors>
        <mirror>
            <id>central</id>
            <mirrorOf>central</mirrorOf>
            <url>https://repo1.maven.org/maven2</url>
        </mirror>
    </mirrors>
    -->
</settings>
EOF
    
    log_success "Created ~/.m2/settings.xml"
}

# Run main function
main "$@" 