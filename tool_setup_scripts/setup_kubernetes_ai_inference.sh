#!/usr/bin/env bash

# Setup script for Kubernetes AI Inference development environment
# Installs: kubectl, Helm, k3s/minikube, NVIDIA GPU operator, monitoring tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions
KUBECTL_VERSION="v1.28.3"
HELM_VERSION="v3.13.2"
K9S_VERSION="v0.27.4"

main() {
    show_banner "Kubernetes AI Inference Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "kubectl" "version --client --short 2>/dev/null | head -n 1"
    show_tool_status "helm" "version --short"
    show_tool_status "docker" "--version"
    show_tool_status "k3s" "--version"
    show_tool_status "minikube" "version --short"
    show_tool_status "k9s" "version --short"
    show_tool_status "kustomize" "version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "curl" "cURL"
    install_apt_package "wget" "Wget"
    install_apt_package "git" "Git"
    install_apt_package "apt-transport-https" "APT Transport HTTPS"
    install_apt_package "ca-certificates" "CA Certificates"
    install_apt_package "gnupg" "GnuPG"
    install_apt_package "lsb-release" "LSB Release"
    
    # Install Docker (required for some Kubernetes distributions)
    log_step "Installing Docker"
    if ! command_exists docker; then
        if confirm "Install Docker (required for some Kubernetes setups)?"; then
            install_docker
        fi
    else
        log_info "Docker is already installed"
    fi
    
    # Install kubectl
    log_step "Installing kubectl"
    if ! command_exists kubectl; then
        if confirm "Install kubectl?"; then
            install_kubectl
        fi
    else
        log_info "kubectl is already installed"
        if confirm "Update kubectl to $KUBECTL_VERSION?"; then
            install_kubectl
        fi
    fi
    
    # Install Helm
    log_step "Installing Helm"
    if ! command_exists helm; then
        if confirm "Install Helm package manager?"; then
            install_helm
        fi
    else
        log_info "Helm is already installed"
        if confirm "Update Helm to latest version?"; then
            install_helm
        fi
    fi
    
    # Choose Kubernetes distribution
    log_step "Kubernetes Distribution"
    log_info "Choose a local Kubernetes distribution:"
    echo "1) k3s (Lightweight, production-ready)"
    echo "2) minikube (Full-featured, development-focused)"
    echo "3) kind (Kubernetes in Docker)"
    echo "4) Skip local Kubernetes installation"
    read -p "Enter your choice (1-4): " k8s_choice
    
    case $k8s_choice in
        1) install_k3s ;;
        2) install_minikube ;;
        3) install_kind ;;
        4) log_info "Skipping local Kubernetes installation" ;;
        *) log_warning "Invalid choice, skipping" ;;
    esac
    
    # Install k9s (Kubernetes CLI UI)
    log_step "Installing k9s"
    if ! command_exists k9s; then
        if confirm "Install k9s (Terminal UI for Kubernetes)?"; then
            install_k9s
        fi
    else
        log_info "k9s is already installed"
    fi
    
    # Install kustomize
    log_step "Installing kustomize"
    if ! command_exists kustomize; then
        if confirm "Install kustomize?"; then
            install_kustomize
        fi
    else
        log_info "kustomize is already installed"
    fi
    
    # Install AI/ML specific tools
    log_step "Installing AI/ML tools for Kubernetes"
    
    # KubeFlow
    if confirm "Install KubeFlow CLI (Machine Learning toolkit for Kubernetes)?"; then
        install_kubeflow_cli
    fi
    
    # Seldon Core CLI
    if confirm "Install Seldon Core CLI (ML deployment platform)?"; then
        pip3 install seldon-core --user
        log_success "Seldon Core CLI installed"
    fi
    
    # NVIDIA GPU Operator setup instructions
    log_step "GPU Support"
    if confirm "Show instructions for NVIDIA GPU Operator setup?"; then
        show_gpu_setup_instructions
    fi
    
    # Monitoring tools
    log_step "Monitoring tools"
    if confirm "Install kubectx and kubens (context/namespace switchers)?"; then
        install_kubectx
    fi
    
    # Setup kubectl aliases and completion
    log_step "Setting up kubectl convenience features"
    if confirm "Setup kubectl aliases and auto-completion?"; then
        setup_kubectl_convenience
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "kubectl" "version --client --short 2>/dev/null | head -n 1"
    show_tool_status "helm" "version --short"
    show_tool_status "docker" "--version"
    show_tool_status "k3s" "--version"
    show_tool_status "minikube" "version --short"
    show_tool_status "k9s" "version --short"
    show_tool_status "kustomize" "version"
    show_tool_status "kubectx" "--help 2>&1 | head -n 1"
    
    echo
    log_success "Kubernetes AI Inference environment is ready!"
    log_info "Useful commands:"
    echo -e "  ${CYAN}kubectl get nodes${RESET} - List cluster nodes"
    echo -e "  ${CYAN}kubectl get pods -A${RESET} - List all pods"
    echo -e "  ${CYAN}k9s${RESET} - Launch Kubernetes terminal UI"
    echo -e "  ${CYAN}helm repo add stable https://charts.helm.sh/stable${RESET} - Add Helm stable repo"
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
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    log_success "Docker installed. You may need to log out and back in for group changes to take effect."
}

install_kubectl() {
    log_info "Installing kubectl $KUBECTL_VERSION..."
    
    # Download kubectl
    curl -LO "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl"
    
    # Verify checksum
    curl -LO "https://dl.k8s.io/${KUBECTL_VERSION}/bin/linux/amd64/kubectl.sha256"
    echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check
    
    # Install
    sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
    rm kubectl kubectl.sha256
    
    log_success "kubectl installed"
}

install_helm() {
    log_info "Installing Helm..."
    
    # Download Helm installation script
    curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
    chmod 700 get_helm.sh
    ./get_helm.sh
    rm get_helm.sh
    
    log_success "Helm installed"
}

install_k3s() {
    log_info "Installing k3s..."
    
    # Install k3s
    curl -sfL https://get.k3s.io | sh -
    
    # Copy kubeconfig for user
    mkdir -p ~/.kube
    sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
    sudo chown $USER:$USER ~/.kube/config
    
    log_success "k3s installed"
    log_info "k3s service will start automatically"
}

install_minikube() {
    log_info "Installing minikube..."
    
    # Download and install minikube
    curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
    sudo install minikube-linux-amd64 /usr/local/bin/minikube
    rm minikube-linux-amd64
    
    log_success "minikube installed"
    log_info "Start minikube with: minikube start"
}

install_kind() {
    log_info "Installing kind..."
    
    # Download and install kind
    curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
    chmod +x ./kind
    sudo mv ./kind /usr/local/bin/kind
    
    log_success "kind installed"
    log_info "Create a cluster with: kind create cluster"
}

install_k9s() {
    log_info "Installing k9s..."
    
    # Download k9s
    wget -q https://github.com/derailed/k9s/releases/download/${K9S_VERSION}/k9s_Linux_amd64.tar.gz
    tar xzf k9s_Linux_amd64.tar.gz
    sudo mv k9s /usr/local/bin/
    rm k9s_Linux_amd64.tar.gz README.md LICENSE
    
    log_success "k9s installed"
}

install_kustomize() {
    log_info "Installing kustomize..."
    
    # Download and install kustomize
    curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
    sudo mv kustomize /usr/local/bin/
    
    log_success "kustomize installed"
}

install_kubeflow_cli() {
    log_info "Installing KubeFlow CLI..."
    
    # Download kfctl
    wget -q https://github.com/kubeflow/kfctl/releases/download/v1.2.0/kfctl_v1.2.0-0-gbc038f9_linux.tar.gz
    tar xzf kfctl_v1.2.0-0-gbc038f9_linux.tar.gz
    sudo mv kfctl /usr/local/bin/
    rm kfctl_v1.2.0-0-gbc038f9_linux.tar.gz
    
    log_success "KubeFlow CLI installed"
}

install_kubectx() {
    log_info "Installing kubectx and kubens..."
    
    # Clone repository
    git clone https://github.com/ahmetb/kubectx /tmp/kubectx
    sudo mv /tmp/kubectx/kubectx /usr/local/bin/
    sudo mv /tmp/kubectx/kubens /usr/local/bin/
    rm -rf /tmp/kubectx
    
    log_success "kubectx and kubens installed"
}

setup_kubectl_convenience() {
    log_info "Setting up kubectl convenience features..."
    
    # Add kubectl completion
    kubectl_completion='source <(kubectl completion bash)'
    if [[ -f "$HOME/.bashrc" ]]; then
        add_to_file_if_missing "$HOME/.bashrc" "$kubectl_completion" "kubectl completion"
    fi
    
    # Add common aliases
    kubectl_aliases='
# Kubectl aliases
alias k=kubectl
alias kgp="kubectl get pods"
alias kgs="kubectl get svc"
alias kgd="kubectl get deployment"
alias kaf="kubectl apply -f"
alias kdel="kubectl delete"
alias klog="kubectl logs"
alias kexec="kubectl exec -it"'
    
    if [[ -f "$HOME/.bashrc" ]]; then
        add_to_file_if_missing "$HOME/.bashrc" "$kubectl_aliases" "kubectl aliases"
    fi
    
    if [[ -f "$HOME/.zshrc" ]]; then
        add_to_file_if_missing "$HOME/.zshrc" "$kubectl_aliases" "kubectl aliases"
    fi
    
    log_success "kubectl convenience features configured"
}

show_gpu_setup_instructions() {
    log_info "NVIDIA GPU Operator Setup Instructions:"
    echo
    echo "1. Ensure NVIDIA drivers are installed on your nodes"
    echo "2. Add the NVIDIA Helm repository:"
    echo -e "   ${CYAN}helm repo add nvidia https://helm.ngc.nvidia.com/nvidia${RESET}"
    echo -e "   ${CYAN}helm repo update${RESET}"
    echo
    echo "3. Install the GPU Operator:"
    echo -e "   ${CYAN}helm install --wait --generate-name \\
     -n gpu-operator --create-namespace \\
     nvidia/gpu-operator${RESET}"
    echo
    echo "4. Verify GPU nodes:"
    echo -e "   ${CYAN}kubectl get nodes -l nvidia.com/gpu.present=true${RESET}"
    echo
    echo "For more details, visit: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html"
}

# Run main function
main "$@" 