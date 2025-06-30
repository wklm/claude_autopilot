#!/usr/bin/env bash

# Setup script for Cloud Native DevOps development environment
# Installs: kubectl, k3s/kind, Argo CD, Terraform/OpenTofu, and cloud-native tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions
KUBECTL_VERSION="1.31.0"
KIND_VERSION="0.24.0"
K3S_VERSION="v1.31.0+k3s1"
ARGOCD_VERSION="2.13.0"
TERRAFORM_VERSION="1.10.0"
OPENTOFU_VERSION="1.8.4"
HELM_VERSION="3.16.2"
FLUX_VERSION="2.4.0"

main() {
    show_banner "Cloud Native DevOps Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "docker" "--version"
    show_tool_status "kubectl" "version --client --short 2>/dev/null"
    show_tool_status "kind" "--version"
    show_tool_status "k3s" "--version 2>/dev/null"
    show_tool_status "helm" "version --short"
    show_tool_status "argocd" "version --client --short"
    show_tool_status "terraform" "--version | head -n1"
    show_tool_status "tofu" "--version | head -n1"
    show_tool_status "flux" "--version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "curl" "cURL"
    install_apt_package "git" "Git"
    install_apt_package "jq" "jq (JSON processor)"
    install_apt_package "unzip" "Unzip"
    install_apt_package "gnupg" "GnuPG"
    install_apt_package "lsb-release" "LSB Release"
    install_apt_package "ca-certificates" "CA Certificates"
    install_apt_package "apt-transport-https" "APT HTTPS Transport"
    install_apt_package "software-properties-common" "Software Properties Common"
    
    # Install Docker (required for kind and many tools)
    install_docker
    
    # Install kubectl
    install_kubectl
    
    # Install kind (Kubernetes in Docker)
    install_kind
    
    # Install k3s (alternative to kind)
    install_k3s
    
    # Install Helm
    install_helm
    
    # Install Argo CD CLI
    install_argocd_cli
    
    # Install Flux CLI
    install_flux_cli
    
    # Install Terraform
    install_terraform
    
    # Install OpenTofu (Terraform alternative)
    install_opentofu
    
    # Install additional cloud-native tools
    install_additional_tools
    
    # Create helper scripts
    create_helper_scripts
    
    # Configure Git for IaC
    configure_git_for_iac
    
    # Final status
    show_final_status
}

install_docker() {
    log_step "Setting up Docker"
    if ! command_exists docker; then
        if confirm "Install Docker (required for kind and local development)?"; then
            # Add Docker's official GPG key
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
}

install_kubectl() {
    log_step "Installing kubectl"
    if ! command_exists kubectl || [[ $(kubectl version --client --short 2>/dev/null | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' | sed 's/v//') != "$KUBECTL_VERSION" ]]; then
        if confirm "Install kubectl $KUBECTL_VERSION?"; then
            curl -LO "https://dl.k8s.io/release/v${KUBECTL_VERSION}/bin/linux/amd64/kubectl"
            sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
            rm kubectl
            log_success "kubectl $KUBECTL_VERSION installed"
            
            # Enable kubectl autocompletion
            kubectl_completion='source <(kubectl completion bash)'
            kubectl_alias='alias k=kubectl'
            kubectl_complete='complete -o default -F __start_kubectl k'
            
            for rcfile in "$HOME/.bashrc" "$HOME/.zshrc"; do
                if [[ -f "$rcfile" ]]; then
                    add_to_file_if_missing "$rcfile" "$kubectl_completion" "kubectl completion"
                    add_to_file_if_missing "$rcfile" "$kubectl_alias" "kubectl alias"
                    add_to_file_if_missing "$rcfile" "$kubectl_complete" "kubectl alias completion"
                fi
            done
        fi
    else
        log_info "kubectl is already at version $KUBECTL_VERSION"
    fi
    
    # Install kind (Kubernetes in Docker)
    log_step "Installing kind"
    if ! command_exists kind; then
        if confirm "Install kind $KIND_VERSION for local Kubernetes clusters?"; then
            curl -Lo ./kind "https://kind.sigs.k8s.io/dl/v${KIND_VERSION}/kind-linux-amd64"
            sudo install -o root -g root -m 0755 kind /usr/local/bin/kind
            rm kind
            log_success "kind $KIND_VERSION installed"
            
            # Create kind config template
            kind_config="$HOME/.kube/kind-config.yaml"
            mkdir -p "$HOME/.kube"
            cat > "$kind_config" << 'EOF'
# Kind cluster configuration
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
  extraPortMappings:
  - containerPort: 80
    hostPort: 80
    protocol: TCP
  - containerPort: 443
    hostPort: 443
    protocol: TCP
- role: worker
- role: worker
EOF
            log_info "Kind config template created at $kind_config"
        fi
    else
        log_info "kind is already installed"
    fi
    
    # Install k3s (alternative to kind)
    log_step "Installing k3s"
    if ! command_exists k3s; then
        if confirm "Install k3s $K3S_VERSION (lightweight Kubernetes)?"; then
            curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="$K3S_VERSION" sh -s - --write-kubeconfig-mode 644
            
            # Add k3s kubeconfig to environment
            k3s_kubeconfig='export KUBECONFIG=/etc/rancher/k3s/k3s.yaml'
            for rcfile in "$HOME/.bashrc" "$HOME/.zshrc"; do
                if [[ -f "$rcfile" ]]; then
                    add_to_file_if_missing "$rcfile" "$k3s_kubeconfig" "k3s kubeconfig"
                fi
            done
            
            log_success "k3s $K3S_VERSION installed"
            log_info "k3s is running as a system service"
        fi
    else
        log_info "k3s is already installed"
    fi
    
    # Install Helm
    log_step "Installing Helm"
    if ! command_exists helm; then
        if confirm "Install Helm $HELM_VERSION?"; then
            curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
            chmod 700 get_helm.sh
            ./get_helm.sh --version "v$HELM_VERSION"
            rm get_helm.sh
            
            # Add Helm repos
            helm repo add stable https://charts.helm.sh/stable
            helm repo add bitnami https://charts.bitnami.com/bitnami
            helm repo update
            
            log_success "Helm $HELM_VERSION installed"
        fi
    else
        log_info "Helm is already installed"
    fi
    
    # Install Argo CD CLI
    log_step "Installing Argo CD CLI"
    if ! command_exists argocd; then
        if confirm "Install Argo CD CLI $ARGOCD_VERSION?"; then
            curl -sSL -o /tmp/argocd "https://github.com/argoproj/argo-cd/releases/download/v${ARGOCD_VERSION}/argocd-linux-amd64"
            sudo install -m 555 /tmp/argocd /usr/local/bin/argocd
            rm /tmp/argocd
            
            # Enable Argo CD CLI completion
            argocd_completion='source <(argocd completion bash)'
            for rcfile in "$HOME/.bashrc"; do
                if [[ -f "$rcfile" ]]; then
                    add_to_file_if_missing "$rcfile" "$argocd_completion" "argocd completion"
                fi
            done
            
            log_success "Argo CD CLI $ARGOCD_VERSION installed"
        fi
    else
        log_info "Argo CD CLI is already installed"
    fi
    
    # Install Flux CLI
    log_step "Installing Flux CLI"
    if ! command_exists flux; then
        if confirm "Install Flux CLI $FLUX_VERSION?"; then
            curl -s https://fluxcd.io/install.sh | sudo bash
            
            # Enable Flux CLI completion
            flux_completion='source <(flux completion bash)'
            for rcfile in "$HOME/.bashrc"; do
                if [[ -f "$rcfile" ]]; then
                    add_to_file_if_missing "$rcfile" "$flux_completion" "flux completion"
                fi
            done
            
            log_success "Flux CLI installed"
        fi
    else
        log_info "Flux CLI is already installed"
    fi
    
    # Install Terraform
    log_step "Installing Terraform"
    if ! command_exists terraform; then
        if confirm "Install Terraform $TERRAFORM_VERSION?"; then
            # Add HashiCorp GPG key
            wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
            echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
            sudo apt-get update
            sudo apt-get install -y terraform=$TERRAFORM_VERSION-*
            
            log_success "Terraform $TERRAFORM_VERSION installed"
        fi
    else
        log_info "Terraform is already installed"
    fi
    
    # Install OpenTofu (Terraform alternative)
    log_step "Installing OpenTofu"
    if ! command_exists tofu; then
        if confirm "Install OpenTofu $OPENTOFU_VERSION (open-source Terraform)?"; then
            # Download OpenTofu
            curl -Lo /tmp/tofu.zip "https://github.com/opentofu/opentofu/releases/download/v${OPENTOFU_VERSION}/tofu_${OPENTOFU_VERSION}_linux_amd64.zip"
            sudo unzip -o /tmp/tofu.zip -d /usr/local/bin/
            rm /tmp/tofu.zip
            
            log_success "OpenTofu $OPENTOFU_VERSION installed"
        fi
    else
        log_info "OpenTofu is already installed"
    fi
    
    # Install additional cloud-native tools
    log_step "Installing additional cloud-native tools"
    
    # Install kubectx and kubens
    if ! command_exists kubectx; then
        if confirm "Install kubectx and kubens for easier context/namespace switching?"; then
            sudo git clone https://github.com/ahmetb/kubectx /opt/kubectx
            sudo ln -s /opt/kubectx/kubectx /usr/local/bin/kubectx
            sudo ln -s /opt/kubectx/kubens /usr/local/bin/kubens
            log_success "kubectx and kubens installed"
        fi
    fi
    
    # Install k9s
    if ! command_exists k9s; then
        if confirm "Install k9s (terminal UI for Kubernetes)?"; then
            K9S_VERSION=$(curl -s https://api.github.com/repos/derailed/k9s/releases/latest | grep -Po '"tag_name": "\K.*?(?=")')
            curl -Lo /tmp/k9s.tar.gz "https://github.com/derailed/k9s/releases/download/${K9S_VERSION}/k9s_Linux_amd64.tar.gz"
            tar -xzf /tmp/k9s.tar.gz -C /tmp/
            sudo install -o root -g root -m 0755 /tmp/k9s /usr/local/bin/k9s
            rm /tmp/k9s.tar.gz /tmp/k9s
            log_success "k9s installed"
        fi
    fi
    
    # Install kustomize
    if ! command_exists kustomize; then
        if confirm "Install kustomize?"; then
            curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
            sudo mv kustomize /usr/local/bin/
            log_success "kustomize installed"
        fi
    fi
    
    # Install yamllint for YAML validation
    if ! command_exists yamllint; then
        if confirm "Install yamllint for YAML validation?"; then
            sudo apt-get install -y yamllint
            log_success "yamllint installed"
        fi
    fi
    
    # Install kube-ps1 for shell prompt
    log_step "Setting up Kubernetes shell prompt"
    if confirm "Install kube-ps1 for Kubernetes context in shell prompt?"; then
        git clone https://github.com/jonmosco/kube-ps1.git ~/.kube-ps1
        
        # Add to shell configs
        kube_ps1_source='source ~/.kube-ps1/kube-ps1.sh'
        kube_ps1_prompt='PS1="[\$(kube_ps1)] $PS1"'
        
        for rcfile in "$HOME/.bashrc" "$HOME/.zshrc"; do
            if [[ -f "$rcfile" ]]; then
                add_to_file_if_missing "$rcfile" "$kube_ps1_source" "kube-ps1 source"
                add_to_file_if_missing "$rcfile" "$kube_ps1_prompt" "kube-ps1 prompt"
            fi
        done
        
        log_success "kube-ps1 installed"
    fi
    
    # Create helper scripts directory
    log_step "Creating helper scripts"
    scripts_dir="$HOME/.k8s-scripts"
    create_directory "$scripts_dir" "Kubernetes scripts directory"
    
    # Create kind cluster creation script
    cat > "$scripts_dir/create-kind-cluster.sh" << 'EOF'
#!/bin/bash
# Create a kind cluster with ingress support
CLUSTER_NAME=${1:-kind}
echo "Creating kind cluster: $CLUSTER_NAME"
kind create cluster --name $CLUSTER_NAME --config ~/.kube/kind-config.yaml
echo "Installing NGINX Ingress..."
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
echo "Waiting for ingress to be ready..."
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=90s
echo "Cluster $CLUSTER_NAME is ready!"
EOF
    chmod +x "$scripts_dir/create-kind-cluster.sh"
    
    # Create Argo CD installation script
    cat > "$scripts_dir/install-argocd.sh" << 'EOF'
#!/bin/bash
# Install Argo CD in the current cluster
echo "Installing Argo CD..."
kubectl create namespace argocd 2>/dev/null || true
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
echo "Waiting for Argo CD to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/argocd-server -n argocd
echo "Argo CD installed!"
echo "To access Argo CD:"
echo "1. Port forward: kubectl port-forward svc/argocd-server -n argocd 8080:443"
echo "2. Get admin password: kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d"
EOF
    chmod +x "$scripts_dir/install-argocd.sh"
    
    # Create monitoring stack installation script
    cat > "$scripts_dir/install-monitoring.sh" << 'EOF'
#!/bin/bash
# Install Prometheus and Grafana using Helm
echo "Adding prometheus-community Helm repo..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
echo "Installing kube-prometheus-stack..."
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set grafana.adminPassword=admin
echo "Monitoring stack installed!"
echo "To access Grafana:"
echo "1. Port forward: kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80"
echo "2. Login with admin/admin"
EOF
    chmod +x "$scripts_dir/install-monitoring.sh"
    
    # Add scripts directory to PATH
    scripts_path_line="export PATH=\$PATH:$scripts_dir"
    for rcfile in "$HOME/.bashrc" "$HOME/.zshrc"; do
        if [[ -f "$rcfile" ]]; then
            add_to_file_if_missing "$rcfile" "$scripts_path_line" "k8s scripts PATH"
        fi
    done
    
    # Configure Git for IaC
    log_step "Configuring Git for Infrastructure as Code"
    if confirm "Add IaC patterns to global .gitignore?"; then
        gitignore_file="$HOME/.gitignore_global"
        touch "$gitignore_file"
        
        # IaC patterns
        iac_patterns=(
            # Terraform
            "*.tfstate"
            "*.tfstate.*"
            ".terraform/"
            ".terraform.lock.hcl"
            "terraform.tfvars"
            "*.auto.tfvars"
            "override.tf"
            "override.tf.json"
            "*_override.tf"
            "*_override.tf.json"
            ".terraformrc"
            "terraform.rc"
            # Kubernetes
            "kubeconfig"
            "*.kubeconfig"
            ".kube/cache/"
            ".kube/http-cache/"
            # Helm
            "charts/*.tgz"
            "requirements.lock"
            # Cloud credentials
            ".aws/"
            ".azure/"
            ".gcloud/"
            "*.pem"
            "*.key"
            "*.crt"
            # General
            ".env"
            ".env.*"
            "*.log"
            ".DS_Store"
            "*.swp"
            "*.swo"
        )
        
        for pattern in "${iac_patterns[@]}"; do
            if ! grep -Fxq "$pattern" "$gitignore_file"; then
                echo "$pattern" >> "$gitignore_file"
            fi
        done
        
        git config --global core.excludesfile "$gitignore_file"
        log_success "Global .gitignore configured for IaC"
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "docker" "--version"
    show_tool_status "kubectl" "version --client --short 2>/dev/null"
    show_tool_status "kind" "--version"
    show_tool_status "k3s" "--version 2>/dev/null"
    show_tool_status "helm" "version --short"
    show_tool_status "argocd" "version --client --short"
    show_tool_status "terraform" "--version | head -n1"
    show_tool_status "tofu" "--version | head -n1"
    show_tool_status "flux" "--version"
    show_tool_status "k9s" "version --short"
    
    echo
    log_success "Cloud Native DevOps environment is ready!"
    log_info "Key commands:"
    echo -e "  ${CYAN}create-kind-cluster.sh${RESET} - Create a local kind cluster"
    echo -e "  ${CYAN}install-argocd.sh${RESET} - Install Argo CD in current cluster"
    echo -e "  ${CYAN}install-monitoring.sh${RESET} - Install Prometheus & Grafana"
    echo -e "  ${CYAN}k9s${RESET} - Launch Kubernetes terminal UI"
    echo -e "  ${CYAN}kubectx${RESET} - Switch between Kubernetes contexts"
    echo -e "  ${CYAN}kubens${RESET} - Switch between namespaces"
    
    if groups | grep -q docker; then
        echo
        log_info "You're in the docker group. If you just added, log out and back in."
    else
        echo
        log_warning "You're not in the docker group. Run: sudo usermod -aG docker $USER"
        log_warning "Then log out and back in for Docker to work without sudo."
    fi
}

# Run main function
main "$@" 