# The Definitive Guide to Terraform, Azure DevOps, and Cloud Infrastructure for Modern Applications (2025)

This guide synthesizes battle-tested patterns for building secure, scalable, and cost-efficient infrastructure on Azure using Terraform 1.10+ and Azure DevOps. It covers web applications, AI/ML workloads, and database systems with a focus on production-grade practices.

### Prerequisites & Core Requirements
- **Terraform 1.10.2+** (with native WASM provider support)
- **Azure CLI 2.65+** with bicep extensions
- **Azure DevOps** with Terraform extension v4+
- **Git** with git-crypt or SOPS for secrets
- **VS Code** with HashiCorp Terraform extension

### Key Toolchain Configuration
```bash
# Install Terraform via official HashiCorp tap
brew tap hashicorp/tap
brew install hashicorp/tap/terraform

# Azure CLI with ML extension
brew install azure-cli
az extension add --name ml
az extension add --name containerapp

# Terraform tools
brew install terraform-docs tflint tfsec terrascan
brew install infracost # Cost estimation
brew install terragrunt # DRY wrapper (optional)

# Azure DevOps CLI
az extension add --name azure-devops
```

---

## 1. Project Structure: The Foundation of Maintainable Infrastructure

A well-organized Terraform project is crucial for team collaboration and long-term maintenance. Use a modular, environment-based structure.

### ✅ DO: Use a Scalable Multi-Environment Layout

```
infrastructure/
├── environments/
│   ├── dev/
│   │   ├── main.tf           # Environment-specific root module
│   │   ├── variables.tf      # Environment variables
│   │   ├── terraform.tfvars  # Variable values (git-ignored)
│   │   └── backend.tf        # Backend configuration
│   ├── staging/
│   └── production/
├── modules/
│   ├── compute/
│   │   ├── app-service/      # Web app infrastructure
│   │   ├── container-apps/   # Microservices platform
│   │   └── aks/              # Kubernetes infrastructure
│   ├── data/
│   │   ├── cosmos-db/        # NoSQL database
│   │   ├── sql-database/     # Azure SQL
│   │   └── storage/          # Blob/File storage
│   ├── ml/
│   │   ├── ml-workspace/     # Azure ML workspace
│   │   ├── databricks/       # Spark workloads
│   │   └── cognitive/        # AI services
│   ├── networking/
│   │   ├── vnet/             # Virtual networks
│   │   ├── firewall/         # Azure Firewall
│   │   └── frontdoor/        # Global load balancer
│   └── shared/
│       ├── monitoring/       # Log Analytics, App Insights
│       ├── security/         # Key Vault, Managed Identity
│       └── governance/       # Policies, RBAC
├── scripts/
│   ├── init-backend.sh       # Backend initialization
│   └── validate-costs.sh     # Cost validation
├── policies/                 # OPA/Sentinel policies
├── tests/                    # Terratest files
└── .azure-pipelines/         # Azure DevOps YAML
```

### ✅ DO: Use Terraform Workspaces Sparingly

Workspaces are best for temporary environments, not permanent env separation:

```hcl
# Bad - Using workspaces for permanent environments
terraform workspace select production  # Risky!

# Good - Use separate directories with distinct state files
cd environments/production
terraform apply
```

---

## 2. State Management: The Critical Foundation

State management is the most critical aspect of Terraform. A corrupted or lost state file can be catastrophic.

### ✅ DO: Use Azure Storage with Proper Configuration

```hcl
# environments/production/backend.tf
terraform {
  backend "azurerm" {
    resource_group_name  = "terraform-state-prod"
    storage_account_name = "tfstateprodeastus2001"
    container_name       = "tfstate"
    key                  = "production.terraform.tfstate"
    
    # New in Terraform 1.10: Native state encryption
    encryption_key = var.state_encryption_key
    
    # Enable state locking
    use_azuread_auth = true  # Use Azure AD instead of keys
    
    # Enable versioning and soft delete in Azure Storage
    # Configure via Azure Portal or CLI - critical for recovery
  }
}
```

### State Storage Best Practices

**Configure the storage account properly:**
```bash
# Create state storage with maximum protection
az storage account create \
  --name tfstateprodeastus2001 \
  --resource-group terraform-state-prod \
  --sku Standard_GRS \
  --encryption-services blob \
  --https-only true \
  --allow-blob-public-access false \
  --min-tls-version TLS1_2

# Enable versioning
az storage blob service-properties update \
  --account-name tfstateprodeastus2001 \
  --enable-versioning true

# Enable soft delete with 30-day retention
az storage blob service-properties delete-policy update \
  --account-name tfstateprodeastus2001 \
  --enable true \
  --days-retained 30

# Enable point-in-time restore
az storage account blob-service-properties update \
  --account-name tfstateprodeastus2001 \
  --enable-restore-policy true \
  --restore-days 7
```

### ❌ DON'T: Store State in Git or Unencrypted Locations

```hcl
# NEVER commit state files
# .gitignore
*.tfstate
*.tfstate.*
.terraform/
.terraform.lock.hcl  # This should be committed though!
```

---

## 3. Module Design: Building Reusable Infrastructure

Modules are the building blocks of maintainable Terraform code. Design them to be composable, versioned, and well-documented.

### ✅ DO: Create Semantic, Versioned Modules

```hcl
# modules/compute/app-service/main.tf
terraform {
  required_version = ">= 1.10"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.10"
    }
    azapi = {
      source  = "azure/azapi"
      version = "~> 2.1"
    }
  }
}

locals {
  # Consistent tagging strategy
  default_tags = {
    Module           = "app-service"
    TerraformManaged = "true"
    LastUpdated      = timestamp()
  }
  
  tags = merge(local.default_tags, var.tags)
}

# Resource naming convention
resource "azurerm_service_plan" "this" {
  name                = "${var.name_prefix}-asp-${var.environment}-${var.location_short}"
  resource_group_name = var.resource_group_name
  location            = var.location
  
  os_type  = "Linux"
  sku_name = var.sku_name
  
  # Zone redundancy for production
  zone_balancing_enabled = var.environment == "production" ? true : false
  
  tags = local.tags
}

resource "azurerm_linux_web_app" "this" {
  name                = "${var.name_prefix}-app-${var.environment}-${var.location_short}"
  resource_group_name = var.resource_group_name
  location            = var.location
  service_plan_id     = azurerm_service_plan.this.id
  
  # Managed identity by default
  identity {
    type = "SystemAssigned"
  }
  
  site_config {
    always_on              = var.environment == "production" ? true : false
    http2_enabled          = true
    minimum_tls_version    = "1.2"
    ftps_state             = "Disabled"
    vnet_route_all_enabled = var.enable_vnet_integration
    
    # Health check for production
    health_check_path = var.health_check_path
    
    # Modern app stack
    application_stack {
      node_version = var.node_version # "22-lts" for 2025
    }
    
    # Security headers
    dynamic "cors" {
      for_each = var.cors_origins != null ? [1] : []
      content {
        allowed_origins = var.cors_origins
      }
    }
  }
  
  # Application Insights integration
  app_settings = merge(
    {
      "APPLICATIONINSIGHTS_CONNECTION_STRING" = var.app_insights_connection_string
      "ApplicationInsightsAgent_EXTENSION_VERSION" = "~3"
    },
    var.app_settings
  )
  
  # Deployment slots for zero-downtime deployments
  dynamic "sticky_settings" {
    for_each = var.environment == "production" ? [1] : []
    content {
      app_setting_names = ["ASPNETCORE_ENVIRONMENT", "NODE_ENV"]
    }
  }
  
  tags = local.tags
  
  lifecycle {
    ignore_changes = [
      app_settings["WEBSITE_RUN_FROM_PACKAGE"], # Managed by deployment
      site_config[0].application_stack,          # Prevent drift from deployments
    ]
  }
}

# Autoscaling for production
resource "azurerm_monitor_autoscale_setting" "this" {
  count = var.enable_autoscale ? 1 : 0
  
  name                = "${var.name_prefix}-autoscale-${var.environment}"
  resource_group_name = var.resource_group_name
  location            = var.location
  target_resource_id  = azurerm_service_plan.this.id
  
  profile {
    name = "default"
    
    capacity {
      default = var.autoscale_min_capacity
      minimum = var.autoscale_min_capacity
      maximum = var.autoscale_max_capacity
    }
    
    # CPU-based scaling
    rule {
      metric_trigger {
        metric_name        = "CpuPercentage"
        metric_resource_id = azurerm_service_plan.this.id
        time_grain         = "PT1M"
        statistic          = "Average"
        time_window        = "PT5M"
        time_aggregation   = "Average"
        operator           = "GreaterThan"
        threshold          = 70
      }
      
      scale_action {
        direction = "Increase"
        type      = "ChangeCount"
        value     = "1"
        cooldown  = "PT5M"
      }
    }
    
    # Scale down rule
    rule {
      metric_trigger {
        metric_name        = "CpuPercentage"
        metric_resource_id = azurerm_service_plan.this.id
        time_grain         = "PT1M"
        statistic          = "Average"
        time_window        = "PT10M"
        time_aggregation   = "Average"
        operator           = "LessThan"
        threshold          = 30
      }
      
      scale_action {
        direction = "Decrease"
        type      = "ChangeCount"
        value     = "1"
        cooldown  = "PT10M"
      }
    }
  }
  
  tags = local.tags
}
```

### Module Interface Design

```hcl
# modules/compute/app-service/variables.tf
variable "name_prefix" {
  description = "Prefix for all resource names"
  type        = string
  
  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.name_prefix))
    error_message = "Name prefix must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "sku_name" {
  description = "App Service Plan SKU"
  type        = string
  default     = "P1v3"  # Premium v3 for production workloads
}

# modules/compute/app-service/outputs.tf
output "app_service_id" {
  description = "ID of the App Service"
  value       = azurerm_linux_web_app.this.id
}

output "default_hostname" {
  description = "Default hostname of the App Service"
  value       = azurerm_linux_web_app.this.default_hostname
}

output "managed_identity_principal_id" {
  description = "Principal ID of the system-assigned managed identity"
  value       = azurerm_linux_web_app.this.identity[0].principal_id
}

output "outbound_ip_addresses" {
  description = "Outbound IP addresses of the App Service"
  value       = split(",", azurerm_linux_web_app.this.outbound_ip_addresses)
}
```

---

## 4. Azure DevOps Pipeline Architecture

Modern CI/CD for Infrastructure as Code requires sophisticated pipelines with proper gates, approvals, and cost controls.

### ✅ DO: Use Multi-Stage YAML Pipelines

```yaml
# .azure-pipelines/terraform-deploy.yml
trigger:
  branches:
    include:
      - main
      - develop
  paths:
    include:
      - infrastructure/*

pool:
  vmImage: 'ubuntu-latest'

variables:
  - group: terraform-secrets  # Azure DevOps variable group
  - name: TF_VERSION
    value: '1.10.2'
  - name: TFLINT_VERSION
    value: '0.55.0'
  - name: TFSEC_VERSION
    value: '1.29.0'

stages:
  - stage: Validate
    displayName: 'Validate Infrastructure'
    jobs:
      - job: SecurityScan
        displayName: 'Security and Compliance Scan'
        steps:
          - task: TerraformInstaller@2
            inputs:
              terraformVersion: $(TF_VERSION)
          
          - script: |
              # Install tools
              curl -L https://github.com/terraform-linters/tflint/releases/download/v$(TFLINT_VERSION)/tflint_linux_amd64.zip -o tflint.zip
              unzip tflint.zip && chmod +x tflint && sudo mv tflint /usr/local/bin/
              
              curl -L https://github.com/aquasecurity/tfsec/releases/download/v$(TFSEC_VERSION)/tfsec-linux-amd64 -o tfsec
              chmod +x tfsec && sudo mv tfsec /usr/local/bin/
            displayName: 'Install Security Tools'
          
          - script: |
              # Run security scans
              cd infrastructure/environments/$(Environment)
              terraform init -backend=false
              
              # Format check
              terraform fmt -check -recursive
              
              # Validate syntax
              terraform validate
              
              # Lint for best practices
              tflint --init
              tflint
              
              # Security scan
              tfsec . --format junit --out tfsec-report.xml
            displayName: 'Run Security Scans'
          
          - task: PublishTestResults@2
            inputs:
              testResultsFormat: 'JUnit'
              testResultsFiles: '**/tfsec-report.xml'
              failTaskOnFailedTests: true
      
      - job: CostEstimation
        displayName: 'Estimate Infrastructure Costs'
        steps:
          - task: InfracostSetup@2
            inputs:
              version: 'latest'
          
          - task: AzureCLI@2
            inputs:
              azureSubscription: 'terraform-service-connection'
              scriptType: 'bash'
              scriptLocation: 'inlineScript'
              inlineScript: |
                cd infrastructure/environments/$(Environment)
                terraform init
                
                # Generate plan for cost estimation
                terraform plan -out=tfplan.binary
                terraform show -json tfplan.binary > tfplan.json
                
                # Run Infracost
                infracost breakdown --path tfplan.json \
                  --format json \
                  --out-file infracost.json
                
                # Generate diff for PR comments
                if [ "$BUILD_REASON" == "PullRequest" ]; then
                  infracost diff --path tfplan.json \
                    --format json \
                    --compare-to infracost-base.json \
                    --out-file infracost-diff.json
                  
                  # Post to PR
                  infracost comment github \
                    --path infracost-diff.json \
                    --repo $BUILD_REPOSITORY_NAME \
                    --pull-request $SYSTEM_PULLREQUEST_PULLREQUESTNUMBER \
                    --github-token $GITHUB_TOKEN \
                    --behavior update
                fi
                
                # Fail if costs exceed threshold
                MONTHLY_COST=$(cat infracost.json | jq '.projects[0].breakdown.totalMonthlyCost')
                if (( $(echo "$MONTHLY_COST > 5000" | bc -l) )); then
                  echo "ERROR: Estimated monthly cost ($MONTHLY_COST) exceeds threshold"
                  exit 1
                fi
  
  - stage: Plan
    displayName: 'Plan Infrastructure Changes'
    dependsOn: Validate
    condition: succeeded()
    jobs:
      - deployment: PlanInfrastructure
        displayName: 'Generate Terraform Plan'
        environment: '$(Environment)-plan'  # Requires approval for production
        strategy:
          runOnce:
            deploy:
              steps:
                - checkout: self
                
                - task: AzureCLI@2
                  displayName: 'Terraform Plan'
                  inputs:
                    azureSubscription: 'terraform-service-connection'
                    scriptType: 'bash'
                    scriptLocation: 'inlineScript'
                    addSpnToEnvironment: true
                    inlineScript: |
                      # Configure Azure Provider auth
                      export ARM_CLIENT_ID=$servicePrincipalId
                      export ARM_CLIENT_SECRET=$servicePrincipalKey
                      export ARM_SUBSCRIPTION_ID=$(az account show --query id -o tsv)
                      export ARM_TENANT_ID=$tenantId
                      
                      cd infrastructure/environments/$(Environment)
                      
                      # Initialize with backend
                      terraform init \
                        -backend-config="resource_group_name=$(TF_STATE_RG)" \
                        -backend-config="storage_account_name=$(TF_STATE_SA)" \
                        -backend-config="container_name=$(TF_STATE_CONTAINER)" \
                        -backend-config="key=$(Environment).terraform.tfstate"
                      
                      # Create detailed plan
                      terraform plan \
                        -var-file="terraform.tfvars" \
                        -var="environment=$(Environment)" \
                        -out=tfplan \
                        -detailed-exitcode
                      
                      # Convert to JSON for analysis
                      terraform show -json tfplan > tfplan.json
                      
                      # Analyze plan for risky changes
                      python3 $(Build.SourcesDirectory)/scripts/analyze-plan.py tfplan.json
                
                - task: PublishPipelineArtifact@1
                  inputs:
                    targetPath: 'infrastructure/environments/$(Environment)/tfplan'
                    artifact: 'terraform-plan-$(Environment)'
  
  - stage: Deploy
    displayName: 'Deploy Infrastructure'
    dependsOn: Plan
    condition: succeeded()
    jobs:
      - deployment: DeployInfrastructure
        displayName: 'Apply Terraform Changes'
        environment: '$(Environment)'  # Production requires manual approval
        strategy:
          runOnce:
            deploy:
              steps:
                - checkout: self
                
                - task: DownloadPipelineArtifact@2
                  inputs:
                    artifact: 'terraform-plan-$(Environment)'
                    path: 'infrastructure/environments/$(Environment)'
                
                - task: AzureCLI@2
                  displayName: 'Terraform Apply'
                  inputs:
                    azureSubscription: 'terraform-service-connection'
                    scriptType: 'bash'
                    scriptLocation: 'inlineScript'
                    addSpnToEnvironment: true
                    inlineScript: |
                      export ARM_CLIENT_ID=$servicePrincipalId
                      export ARM_CLIENT_SECRET=$servicePrincipalKey
                      export ARM_SUBSCRIPTION_ID=$(az account show --query id -o tsv)
                      export ARM_TENANT_ID=$tenantId
                      
                      cd infrastructure/environments/$(Environment)
                      
                      # Re-initialize to ensure backend connection
                      terraform init \
                        -backend-config="resource_group_name=$(TF_STATE_RG)" \
                        -backend-config="storage_account_name=$(TF_STATE_SA)" \
                        -backend-config="container_name=$(TF_STATE_CONTAINER)" \
                        -backend-config="key=$(Environment).terraform.tfstate"
                      
                      # Apply the saved plan
                      terraform apply -auto-approve tfplan
                      
                      # Export outputs for downstream jobs
                      terraform output -json > outputs.json
                      echo "##vso[task.uploadsummary]$(pwd)/outputs.json"
                
                - task: AzureCLI@2
                  displayName: 'Validate Deployment'
                  inputs:
                    azureSubscription: 'terraform-service-connection'
                    scriptType: 'bash'
                    scriptLocation: 'inlineScript'
                    inlineScript: |
                      # Run smoke tests against deployed infrastructure
                      cd $(Build.SourcesDirectory)/tests
                      npm install
                      npm run test:$(Environment)
  
  - stage: PostDeploy
    displayName: 'Post-Deployment Tasks'
    dependsOn: Deploy
    condition: succeeded()
    jobs:
      - job: UpdateDocumentation
        displayName: 'Update Infrastructure Documentation'
        steps:
          - script: |
              cd infrastructure
              terraform-docs markdown table . > INFRASTRUCTURE.md
              
              # Generate architecture diagram
              pip install diagrams
              python3 scripts/generate-diagram.py
            displayName: 'Generate Documentation'
          
          - task: GitHubComment@0
            condition: eq(variables['Build.Reason'], 'PullRequest')
            inputs:
              gitHubConnection: 'github-connection'
              repositoryName: '$(Build.Repository.Name)'
              comment: |
                ### 🚀 Infrastructure Deployed Successfully
                
                **Environment:** $(Environment)
                **Build:** $(Build.BuildNumber)
                **Commit:** $(Build.SourceVersion)
                
                View the [deployment summary]($(System.CollectionUri)$(System.TeamProject)/_build/results?buildId=$(Build.BuildId))
```

### Advanced Pipeline Features

```yaml
# .azure-pipelines/templates/terraform-test.yml
parameters:
  - name: testType
    type: string
    default: 'integration'
  - name: terraformVersion
    type: string
    default: '1.10.2'

steps:
  - task: GoTool@0
    inputs:
      version: '1.23'
    displayName: 'Install Go for Terratest'
  
  - script: |
      cd tests
      go mod download
      
      # Run Terratest with proper Azure auth
      export ARM_SUBSCRIPTION_ID=$(SUBSCRIPTION_ID)
      export ARM_CLIENT_ID=$(CLIENT_ID)
      export ARM_CLIENT_SECRET=$(CLIENT_SECRET)
      export ARM_TENANT_ID=$(TENANT_ID)
      
      if [ "${{ parameters.testType }}" == "unit" ]; then
        go test -v -timeout 30m -tags=unit ./...
      else
        go test -v -timeout 90m -tags=integration ./...
      fi
    displayName: 'Run Terratest ${{ parameters.testType }} Tests'
```

---

## 5. AI/ML Infrastructure Patterns

Building infrastructure for AI/ML workloads requires special consideration for compute, storage, and orchestration.

### ✅ DO: Create Comprehensive ML Platform Infrastructure

```hcl
# modules/ml/ml-workspace/main.tf
resource "azurerm_machine_learning_workspace" "this" {
  name                          = "${var.name_prefix}-mlw-${var.environment}"
  resource_group_name           = var.resource_group_name
  location                      = var.location
  application_insights_id       = var.application_insights_id
  key_vault_id                  = var.key_vault_id
  storage_account_id            = azurerm_storage_account.ml_storage.id
  container_registry_id         = var.container_registry_id
  
  # Managed virtual network for security
  public_network_access_enabled = false
  managed_network {
    isolation_mode = "AllowOnlyApprovedOutbound"
  }
  
  identity {
    type = "SystemAssigned"
  }
  
  # High business impact for compliance
  sku_name = "Basic"
  high_business_impact = var.environment == "production" ? true : false
  
  tags = local.tags
}

# Compute instances for development
resource "azapi_resource" "compute_instance" {
  for_each = var.compute_instances
  
  type      = "Microsoft.MachineLearningServices/workspaces/computes@2024-10-01"
  name      = each.key
  parent_id = azurerm_machine_learning_workspace.this.id
  
  body = {
    properties = {
      computeType = "ComputeInstance"
      properties = {
        vmSize                          = each.value.vm_size
        enableNodePublicIp              = false
        sshSettings = {
          sshPublicAccess = "Disabled"
        }
        subnet = {
          id = var.subnet_id
        }
        applicationSharingPolicy        = "Personal"
        idleTimeBeforeShutdown         = "PT${each.value.idle_minutes}M"
        
        # Assign to specific user
        assignedUser = {
          objectId = each.value.user_object_id
          tenantId = data.azurerm_client_config.current.tenant_id
        }
      }
    }
  }
}

# GPU Compute Clusters for Training
resource "azapi_resource" "gpu_compute_cluster" {
  count = var.enable_gpu_cluster ? 1 : 0
  
  type      = "Microsoft.MachineLearningServices/workspaces/computes@2024-10-01"
  name      = "${var.name_prefix}-gpu-cluster"
  parent_id = azurerm_machine_learning_workspace.this.id
  
  body = {
    properties = {
      computeType = "AmlCompute"
      properties = {
        scaleSettings = {
          minNodeCount                = var.gpu_cluster_min_nodes
          maxNodeCount                = var.gpu_cluster_max_nodes
          nodeIdleTimeBeforeScaleDown = "PT30M"
        }
        
        vmSize     = var.gpu_vm_size  # "Standard_NC96ads_A100_v4" for A100 GPUs
        vmPriority = var.environment == "production" ? "Dedicated" : "LowPriority"
        
        subnet = {
          id = var.subnet_id
        }
        
        # Managed identity for ACR access
        identity = {
          type = "SystemAssigned"
        }
        
        # Enable SSH for debugging (dev only)
        sshSettings = {
          sshPublicAccess = var.environment == "dev" ? "Enabled" : "Disabled"
          adminUserName   = var.environment == "dev" ? "azureuser" : null
          adminPublicKey  = var.environment == "dev" ? var.ssh_public_key : null
        }
      }
    }
  }
}

# Inference Endpoints with Managed Online Endpoints
resource "azapi_resource" "online_endpoint" {
  for_each = var.inference_endpoints
  
  type      = "Microsoft.MachineLearningServices/workspaces/onlineEndpoints@2024-10-01"
  name      = each.key
  parent_id = azurerm_machine_learning_workspace.this.id
  location  = var.location
  
  body = {
    properties = {
      authMode = "Key"  # Or "AMLToken" for Azure AD auth
      compute  = "Managed"
      
      traffic = {
        (each.value.deployment_name) = 100  # 100% traffic to single deployment
      }
      
      # Network isolation
      publicNetworkAccess = "Disabled"
    }
    
    identity = {
      type = "SystemAssigned"
    }
    
    sku = {
      name     = "Default"
      capacity = each.value.instance_count
    }
  }
  
  tags = merge(local.tags, { EndpointType = "RealTime" })
}

# Feature Store for MLOps
module "feature_store" {
  source = "../feature-store"
  
  name_prefix         = var.name_prefix
  resource_group_name = var.resource_group_name
  location            = var.location
  
  # Materialization infrastructure
  spark_cluster_config = {
    enabled      = true
    node_count   = 3
    node_vm_size = "Standard_E8s_v5"
  }
  
  # Online store (Redis)
  redis_config = {
    sku_name   = var.environment == "production" ? "Premium" : "Standard"
    family     = var.environment == "production" ? "P" : "C"
    capacity   = var.environment == "production" ? 1 : 0
    shard_count = var.environment == "production" ? 2 : 0
  }
  
  # Offline store (ADLS Gen2 + Delta Lake)
  storage_config = {
    account_tier             = "Standard"
    account_replication_type = var.environment == "production" ? "GZRS" : "LRS"
    enable_hns               = true  # Hierarchical namespace for Delta
  }
}

# Outputs for downstream configuration
output "ml_workspace_id" {
  value = azurerm_machine_learning_workspace.this.id
}

output "compute_targets" {
  value = {
    instances = { for k, v in azapi_resource.compute_instance : k => v.id }
    gpu_cluster = var.enable_gpu_cluster ? azapi_resource.gpu_compute_cluster[0].id : null
  }
}

output "inference_endpoints" {
  value = { for k, v in azapi_resource.online_endpoint : k => {
    id   = v.id
    name = v.name
    uri  = jsondecode(v.output).properties.scoringUri
  }}
}
```

### Databricks Infrastructure for Spark Workloads

```hcl
# modules/ml/databricks/main.tf
resource "azurerm_databricks_workspace" "this" {
  name                = "${var.name_prefix}-dbw-${var.environment}"
  resource_group_name = var.resource_group_name
  location            = var.location
  sku                 = var.environment == "production" ? "premium" : "standard"
  
  # Secure deployment with customer-managed VNet
  custom_parameters {
    no_public_ip                                         = true
    virtual_network_id                                   = var.vnet_id
    public_subnet_name                                   = var.public_subnet_name
    private_subnet_name                                  = var.private_subnet_name
    public_subnet_network_security_group_association_id  = var.public_subnet_nsg_id
    private_subnet_network_security_group_association_id = var.private_subnet_nsg_id
    storage_account_name                                 = var.managed_storage_account_name
    
    # Unity Catalog metastore
    managed_resource_group_name = "${var.name_prefix}-dbw-managed-${var.environment}"
  }
  
  # Compliance features
  customer_managed_key_enabled = var.environment == "production"
  infrastructure_encryption_enabled = var.environment == "production"
  
  # Managed identity for Azure services integration
  managed_services_cmk_key_vault_key_id = var.cmk_key_vault_key_id
  
  tags = local.tags
}

# Terraform Databricks provider configuration
terraform {
  required_providers {
    databricks = {
      source  = "databricks/databricks"
      version = "~> 1.58"
    }
  }
}

# Configure provider after workspace creation
provider "databricks" {
  host                = azurerm_databricks_workspace.this.workspace_url
  azure_workspace_resource_id = azurerm_databricks_workspace.this.id
}

# Create Unity Catalog metastore
resource "databricks_metastore" "unity" {
  count = var.environment == "production" ? 1 : 0
  
  name          = "${var.name_prefix}-metastore"
  storage_root  = "abfss://unity-catalog@${var.adls_account_name}.dfs.core.windows.net/"
  owner         = var.unity_catalog_admin_group
  
  # Cross-region disaster recovery
  delta_sharing_scope                         = "INTERNAL"
  delta_sharing_recipient_token_lifetime_in_seconds = 86400
}

# SQL Warehouse for BI workloads
resource "databricks_sql_endpoint" "this" {
  name       = "${var.name_prefix}-sql-endpoint"
  cluster_size = var.sql_warehouse_size  # "2X-Large" for production
  
  auto_stop_mins = var.environment == "production" ? 120 : 30
  
  # Photon for performance
  enable_photon = true
  
  # Serverless for better cold start (Premium only)
  enable_serverless_compute = var.environment == "production"
  
  warehouse_type = var.environment == "production" ? "PRO" : "CLASSIC"
  
  # Spot instances for cost savings in non-prod
  spot_instance_policy = var.environment != "production" ? "COST_OPTIMIZED" : "RELIABILITY_OPTIMIZED"
  
  tags {
    Environment = var.environment
    Purpose     = "Analytics"
  }
}

# ML-optimized cluster policies
resource "databricks_cluster_policy" "ml_gpu" {
  name = "${var.name_prefix}-ml-gpu-policy"
  
  definition = jsonencode({
    "spark_version": {
      "type": "fixed",
      "value": "15.4.x-ml-gpu-scala2.12"  # Latest ML runtime with GPU
    },
    "node_type_id": {
      "type": "allowlist",
      "values": [
        "Standard_NC96ads_A100_v4",   # A100 80GB
        "Standard_NC48ads_A100_v4",   # A100 80GB (smaller)
        "Standard_NC24ads_A100_v4"    # A100 80GB (smallest)
      ]
    },
    "driver_node_type_id": {
      "type": "fixed",
      "value": "Standard_DS5_v2"  # CPU driver for cost savings
    },
    "autoscale.min_workers": {
      "type": "range",
      "minValue": 0,
      "maxValue": 2,
      "defaultValue": 0
    },
    "autoscale.max_workers": {
      "type": "range",
      "minValue": 1,
      "maxValue": 10,
      "defaultValue": 4
    },
    "custom_tags.Purpose": {
      "type": "fixed",
      "value": "MLTraining"
    }
  })
}
```

---

## 6. Database Infrastructure Patterns

Modern applications need sophisticated database infrastructure with proper backup, security, and performance optimization.

### Azure SQL Database with Comprehensive Security

```hcl
# modules/data/sql-database/main.tf
resource "azurerm_mssql_server" "this" {
  name                         = "${var.name_prefix}-sql-${var.environment}"
  resource_group_name          = var.resource_group_name
  location                     = var.location
  version                      = "12.0"
  administrator_login          = "sqladmin"
  administrator_login_password = random_password.sql_admin.result
  
  # Azure AD authentication only for production
  azuread_administrator {
    login_username = var.aad_admin_group_name
    object_id      = var.aad_admin_group_object_id
    tenant_id      = data.azurerm_client_config.current.tenant_id
  }
  
  # Managed identity for Azure services
  identity {
    type = "SystemAssigned"
  }
  
  # Security settings
  minimum_tls_version = "1.2"
  public_network_access_enabled = false
  
  tags = local.tags
}

# Store admin password in Key Vault
resource "azurerm_key_vault_secret" "sql_admin_password" {
  name         = "${var.name_prefix}-sql-admin-password"
  value        = random_password.sql_admin.result
  key_vault_id = var.key_vault_id
  
  lifecycle {
    ignore_changes = [value]  # Prevent regeneration on every apply
  }
}

resource "azurerm_mssql_database" "this" {
  name         = var.database_name
  server_id    = azurerm_mssql_server.this.id
  collation    = "SQL_Latin1_General_CP1_CI_AS"
  license_type = "LicenseIncluded"
  
  # Flexible scaling
  sku_name = var.sku_name  # "S3", "GP_Gen5_4", "BC_Gen5_8"
  
  # Backup retention
  short_term_retention_policy {
    retention_days = var.environment == "production" ? 35 : 7
  }
  
  long_term_retention_policy {
    weekly_retention  = var.environment == "production" ? "P1W" : null
    monthly_retention = var.environment == "production" ? "P1M" : null
    yearly_retention  = var.environment == "production" ? "P1Y" : null
    week_of_year      = var.environment == "production" ? 1 : null
  }
  
  # Threat detection
  threat_detection_policy {
    state                      = "Enabled"
    disabled_alerts            = []
    email_account_admins       = true
    email_addresses            = var.security_alert_emails
    retention_days             = 30
    storage_endpoint           = var.storage_endpoint
    storage_account_access_key = var.storage_access_key
  }
  
  # Transparent data encryption with customer key
  transparent_data_encryption {
    state                 = "Enabled"
    key_vault_key_id      = var.environment == "production" ? var.tde_key_vault_key_id : null
  }
  
  # Zone redundancy for HA
  zone_redundant = var.environment == "production" ? true : false
  
  # Geo-replication for DR
  geo_backup_enabled = true
  
  tags = local.tags
}

# Advanced Threat Protection
resource "azurerm_mssql_server_security_alert_policy" "this" {
  server_name         = azurerm_mssql_server.this.name
  resource_group_name = var.resource_group_name
  state               = "Enabled"
  
  disabled_alerts = []
  
  email_account_admins = true
  email_addresses      = var.security_alert_emails
  
  retention_days = 30
  
  storage_endpoint           = azurerm_storage_account.audit.primary_blob_endpoint
  storage_account_access_key = azurerm_storage_account.audit.primary_access_key
}

# Vulnerability Assessment
resource "azurerm_mssql_server_vulnerability_assessment" "this" {
  server_security_alert_policy_id = azurerm_mssql_server_security_alert_policy.this.id
  
  storage_container_path   = "${azurerm_storage_account.audit.primary_blob_endpoint}vulnerability-scans/"
  storage_account_access_key = azurerm_storage_account.audit.primary_access_key
  
  recurring_scans {
    enabled                   = true
    email_subscription_admins = true
    emails                    = var.security_alert_emails
  }
}

# Failover group for HA/DR
resource "azurerm_mssql_failover_group" "this" {
  count = var.enable_failover_group ? 1 : 0
  
  name      = "${var.name_prefix}-fog"
  server_id = azurerm_mssql_server.this.id
  
  partner_server {
    id = azurerm_mssql_server.secondary[0].id
  }
  
  database_ids = [azurerm_mssql_database.this.id]
  
  read_write_endpoint_failover_policy {
    mode          = "Automatic"
    grace_minutes = 60
  }
  
  readonly_endpoint_failover_policy_enabled = true
  
  tags = local.tags
}
```

### Cosmos DB for Global Distribution

```hcl
# modules/data/cosmos-db/main.tf
resource "azurerm_cosmosdb_account" "this" {
  name                = "${var.name_prefix}-cosmos-${var.environment}"
  resource_group_name = var.resource_group_name
  location            = var.location
  offer_type          = "Standard"
  
  # Consistency for global apps
  consistency_policy {
    consistency_level       = var.consistency_level  # "BoundedStaleness" for balance
    max_interval_in_seconds = var.consistency_level == "BoundedStaleness" ? 300 : null
    max_staleness_prefix    = var.consistency_level == "BoundedStaleness" ? 100000 : null
  }
  
  # Multi-region for HA
  dynamic "geo_location" {
    for_each = var.geo_locations
    content {
      location          = geo_location.value.location
      failover_priority = geo_location.value.failover_priority
      zone_redundant    = var.environment == "production" ? true : false
    }
  }
  
  # Security
  is_virtual_network_filter_enabled = true
  public_network_access_enabled     = false
  local_authentication_disabled     = var.environment == "production"  # Azure AD only
  
  # Managed identity
  identity {
    type = "SystemAssigned"
  }
  
  # Backup
  backup {
    type                = "Continuous"  # Point-in-time restore
    tier                = var.environment == "production" ? "Continuous35Days" : "Continuous7Days"
  }
  
  # Advanced features
  capabilities {
    name = "EnableServerless"  # For dev/test cost savings
  }
  
  analytical_storage_enabled = var.enable_synapse_link
  
  # Customer-managed keys
  dynamic "key_vault_key_uri" {
    for_each = var.cmk_key_vault_uri != null ? [1] : []
    content {
      uri = var.cmk_key_vault_uri
    }
  }
  
  # Network restrictions
  dynamic "virtual_network_rule" {
    for_each = var.allowed_subnets
    content {
      id = virtual_network_rule.value
      ignore_missing_vnet_service_endpoint = false
    }
  }
  
  # CORS for web apps
  cors_rule {
    allowed_origins    = var.cors_allowed_origins
    allowed_methods    = ["GET", "POST", "PUT", "DELETE"]
    allowed_headers    = ["*"]
    exposed_headers    = ["*"]
    max_age_in_seconds = 3600
  }
  
  tags = local.tags
}

# SQL API Database
resource "azurerm_cosmosdb_sql_database" "this" {
  name                = var.database_name
  resource_group_name = var.resource_group_name
  account_name        = azurerm_cosmosdb_account.this.name
  
  # Autoscale for cost optimization
  autoscale_settings {
    max_throughput = var.max_autoscale_throughput  # 4000 for production
  }
}

# Containers with optimized partitioning
resource "azurerm_cosmosdb_sql_container" "users" {
  name                  = "users"
  resource_group_name   = var.resource_group_name
  account_name          = azurerm_cosmosdb_account.this.name
  database_name         = azurerm_cosmosdb_sql_database.this.name
  partition_key_path    = "/userId"
  partition_key_version = 2  # Large partition key support
  
  # Optimized indexing
  indexing_policy {
    indexing_mode = "consistent"
    
    included_path {
      path = "/*"
    }
    
    excluded_path {
      path = "/metadata/*"
    }
    
    # Composite indexes for complex queries
    composite_index {
      index {
        path  = "/country"
        order = "ascending"
      }
      index {
        path  = "/createdAt"
        order = "descending"
      }
    }
    
    # Spatial indexes
    spatial_index {
      path = "/location/*"
      types = ["Point", "Polygon", "LineString"]
    }
  }
  
  # TTL for automatic cleanup
  default_ttl = var.enable_ttl ? 2592000 : null  # 30 days
  
  # Analytical store for Synapse
  analytical_storage_ttl = var.enable_synapse_link ? -1 : null
  
  # Conflict resolution
  conflict_resolution_policy {
    mode                     = "LastWriterWins"
    conflict_resolution_path = "/timestamp"
  }
  
  # Unique keys for data integrity
  unique_key {
    paths = ["/email"]
  }
}

# Monitoring and alerts
resource "azurerm_monitor_metric_alert" "cosmos_ru_consumption" {
  name                = "${var.name_prefix}-cosmos-ru-alert"
  resource_group_name = var.resource_group_name
  scopes              = [azurerm_cosmosdb_account.this.id]
  
  description = "Alert when RU consumption is high"
  severity    = 2
  
  criteria {
    metric_namespace = "Microsoft.DocumentDB/databaseAccounts"
    metric_name      = "NormalizedRUConsumption"
    aggregation      = "Maximum"
    operator         = "GreaterThan"
    threshold        = 80
    
    dimension {
      name     = "CollectionName"
      operator = "Include"
      values   = ["*"]
    }
  }
  
  window_size        = "PT5M"
  frequency          = "PT1M"
  
  action {
    action_group_id = var.action_group_id
  }
}
```

---

## 7. Cost Optimization Strategies

Cost management requires proactive monitoring and automated policies.

### ✅ DO: Implement Comprehensive Cost Controls

```hcl
# modules/governance/cost-management/main.tf
# Budget with alerts
resource "azurerm_consumption_budget_subscription" "this" {
  name            = "${var.name_prefix}-budget-${var.environment}"
  subscription_id = data.azurerm_subscription.current.id
  
  amount     = var.monthly_budget
  time_grain = "Monthly"
  
  time_period {
    start_date = formatdate("YYYY-MM-01", timestamp())
    end_date   = formatdate("YYYY-MM-01", timeadd(timestamp(), "8760h"))  # 1 year
  }
  
  notification {
    enabled   = true
    threshold = 80
    operator  = "GreaterThan"
    
    contact_emails = var.budget_alert_emails
    contact_roles  = ["Owner", "Contributor"]
  }
  
  notification {
    enabled   = true
    threshold = 100
    operator  = "GreaterThan"
    
    contact_emails = var.budget_alert_emails
    contact_roles  = ["Owner"]
    
    # Webhook for automated response
    contact_webhooks = [var.cost_alert_webhook_url]
  }
  
  notification {
    enabled   = true
    threshold = 110  # Forecast
    operator  = "GreaterThan"
    threshold_type = "Forecasted"
    
    contact_emails = var.budget_alert_emails
  }
}

# Auto-shutdown policies for dev/test resources
resource "azurerm_dev_test_global_vm_shutdown_schedule" "this" {
  for_each = { for vm in var.dev_vms : vm.name => vm }
  
  virtual_machine_id = each.value.id
  location           = each.value.location
  enabled            = true
  
  daily_recurrence_time = "1900"  # 7 PM
  timezone              = "Pacific Standard Time"
  
  notification_settings {
    enabled         = true
    time_in_minutes = 30
    email           = each.value.owner_email
  }
  
  tags = merge(local.tags, { AutoShutdown = "Enabled" })
}

# Policy for enforcing cost tags
resource "azurerm_policy_definition" "require_cost_tags" {
  name         = "require-cost-center-tags"
  policy_type  = "Custom"
  mode         = "Indexed"
  display_name = "Require cost center tags"
  
  metadata = jsonencode({
    category = "Cost Management"
    version  = "1.0.0"
  })
  
  policy_rule = jsonencode({
    if = {
      allOf = [
        {
          field = "type"
          like  = "Microsoft.*/*"
        },
        {
          field = "tags['CostCenter']"
          exists = false
        }
      ]
    }
    then = {
      effect = "deny"
    }
  })
}

# Spot instance configuration for batch workloads
module "spot_vm_scale_set" {
  source = "../compute/vmss"
  
  name_prefix = "${var.name_prefix}-spot"
  
  # Spot configuration
  priority        = "Spot"
  eviction_policy = "Deallocate"
  max_bid_price   = 0.1  # Max price per hour
  
  # Scale based on price
  scale_rules = [{
    metric_name = "Price"
    operator    = "LessThan"
    threshold   = 0.05  # Scale up when price is low
    scale_type  = "ChangeCount"
    scale_value = 5
  }]
}

# Reserved instances automation
resource "azurerm_resource_group_template_deployment" "reserved_instances" {
  name                = "reserved-instances-${var.environment}"
  resource_group_name = var.resource_group_name
  deployment_mode     = "Incremental"
  
  template_content = jsonencode({
    "$schema" = "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#"
    contentVersion = "1.0.0.0"
    resources = [
      {
        type       = "Microsoft.Capacity/reservationOrders"
        apiVersion = "2022-03-01"
        name       = "reservation-${var.environment}"
        properties = {
          displayName      = "Production VMs - 3 Year"
          billingScopeId   = data.azurerm_subscription.current.id
          term             = "P3Y"
          billingPlan      = "Upfront"
          appliedScopeType = "Shared"
          
          reservedResourceProperties = {
            instanceFlexibility = "On"
          }
          
          skuName = "Standard_D8s_v5"
          quantity = var.reserved_instance_count
        }
      }
    ]
  })
}
```

### Cost Analysis Automation

```hcl
# scripts/analyze-costs.py
import os
from azure.mgmt.costmanagement import CostManagementClient
from azure.identity import DefaultAzureCredential
from datetime import datetime, timedelta
import pandas as pd

def analyze_terraform_costs():
    """Analyze costs by Terraform-managed resources"""
    
    credential = DefaultAzureCredential()
    client = CostManagementClient(credential)
    
    # Query for Terraform-tagged resources
    query = {
        "type": "ActualCost",
        "timeframe": "MonthToDate",
        "dataset": {
            "granularity": "Daily",
            "aggregation": {
                "totalCost": {
                    "name": "Cost",
                    "function": "Sum"
                }
            },
            "grouping": [
                {"type": "Dimension", "name": "ResourceGroup"},
                {"type": "Tag", "name": "TerraformManaged"}
            ],
            "filter": {
                "tags": {
                    "name": "TerraformManaged",
                    "operator": "In",
                    "values": ["true"]
                }
            }
        }
    }
    
    result = client.query.usage(
        scope=f"/subscriptions/{subscription_id}",
        parameters=query
    )
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(result.properties.rows)
    df.columns = ['Date', 'ResourceGroup', 'TerraformManaged', 'Cost']
    
    # Identify cost anomalies
    df['Cost'] = pd.to_numeric(df['Cost'])
    daily_avg = df.groupby('Date')['Cost'].sum().rolling(7).mean()
    current_cost = df[df['Date'] == df['Date'].max()]['Cost'].sum()
    
    if current_cost > daily_avg.iloc[-1] * 1.5:
        send_alert(f"Cost spike detected: ${current_cost:.2f} (50% above average)")
    
    return df

# Integration with Azure DevOps
def post_cost_summary_to_pr(df, pr_number):
    """Post cost impact summary to PR"""
    
    cost_diff = calculate_cost_difference(df)
    
    comment = f"""
    ## 💰 Infrastructure Cost Impact
    
    **Estimated Monthly Change:** ${cost_diff:.2f}
    
    | Resource Group | Current | Projected | Change |
    |----------------|---------|-----------|--------|
    """
    
    # Add details per resource group...
    
    # Post to Azure DevOps PR
    from azure.devops.connection import Connection
    from msrest.authentication import BasicAuthentication
    
    credentials = BasicAuthentication('', os.environ['AZURE_DEVOPS_PAT'])
    connection = Connection(base_url=org_url, creds=credentials)
    git_client = connection.clients.get_git_client()
    
    git_client.create_thread(
        comment_thread={
            "comments": [{"content": comment}],
            "status": "active"
        },
        repository_id=repo_id,
        pull_request_id=pr_number
    )
```

---

## 8. Security and Compliance Patterns

Security must be built into every layer of infrastructure.

### ✅ DO: Implement Defense in Depth

```hcl
# modules/security/baseline/main.tf
# Azure Policy for compliance
resource "azurerm_policy_set_definition" "security_baseline" {
  name         = "security-baseline"
  policy_type  = "Custom"
  display_name = "Security Baseline Policy Set"
  
  metadata = jsonencode({
    category = "Security"
    version  = "2.0.0"
  })
  
  # CIS Azure Foundations Benchmark
  policy_definition_reference {
    policy_definition_id = "/providers/Microsoft.Authorization/policyDefinitions/34c877ad-507e-4c82-993e-3452a6e0ad3c"
    reference_id         = "storage-secure-transfer"
  }
  
  # Require encryption at rest
  policy_definition_reference {
    policy_definition_id = azurerm_policy_definition.encryption_at_rest.id
    reference_id         = "encryption-at-rest"
    
    parameter_values = jsonencode({
      effect = { value = "Deny" }
    })
  }
  
  # Network security
  policy_definition_reference {
    policy_definition_id = azurerm_policy_definition.require_nsg.id
    reference_id         = "require-nsg"
  }
}

# Managed Identity for all resources
resource "azurerm_user_assigned_identity" "app_identity" {
  name                = "${var.name_prefix}-identity"
  resource_group_name = var.resource_group_name
  location            = var.location
  
  tags = local.tags
}

# Key Vault with RBAC
resource "azurerm_key_vault" "this" {
  name                = "${var.name_prefix}-kv-${random_string.suffix.result}"
  resource_group_name = var.resource_group_name
  location            = var.location
  tenant_id           = data.azurerm_client_config.current.tenant_id
  
  # Premium for HSM-backed keys
  sku_name = var.environment == "production" ? "premium" : "standard"
  
  # Security features
  enabled_for_disk_encryption     = true
  enabled_for_deployment          = false
  enabled_for_template_deployment = true
  enable_rbac_authorization       = true  # RBAC instead of access policies
  
  # Network restrictions
  network_acls {
    default_action = "Deny"
    bypass         = "AzureServices"
    
    ip_rules = var.allowed_ip_ranges
    
    virtual_network_subnet_ids = var.allowed_subnet_ids
  }
  
  # Soft delete and purge protection
  soft_delete_retention_days = 90
  purge_protection_enabled   = true
  
  tags = local.tags
}

# RBAC assignments with least privilege
resource "azurerm_role_assignment" "key_vault_reader" {
  scope                = azurerm_key_vault.this.id
  role_definition_name = "Key Vault Reader"
  principal_id         = azurerm_user_assigned_identity.app_identity.principal_id
}

resource "azurerm_role_assignment" "key_vault_secrets_user" {
  scope                = "${azurerm_key_vault.this.id}/secrets"
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.app_identity.principal_id
}

# Network security with WAF
module "application_gateway" {
  source = "../networking/app-gateway"
  
  name_prefix         = var.name_prefix
  resource_group_name = var.resource_group_name
  location            = var.location
  
  # WAF v2 configuration
  sku_name = "WAF_v2"
  sku_tier = "WAF_v2"
  
  waf_configuration = {
    enabled          = true
    firewall_mode    = var.environment == "production" ? "Prevention" : "Detection"
    rule_set_type    = "OWASP"
    rule_set_version = "3.2"
    
    # Custom rules
    custom_rules = [
      {
        name      = "BlockSuspiciousUserAgents"
        priority  = 1
        rule_type = "MatchRule"
        
        match_conditions = [{
          match_variables = [{
            variable_name = "RequestHeaders"
            selector      = "User-Agent"
          }]
          
          operator           = "Contains"
          negation_condition = false
          match_values       = ["bot", "crawler", "spider"]
        }]
        
        action = "Block"
      }
    ]
    
    # Exclusions for false positives
    exclusion = [
      {
        match_variable          = "RequestHeaderNames"
        selector_match_operator = "Equals"
        selector                = "x-custom-token"
      }
    ]
  }
  
  # Backend configuration
  backend_pools = [{
    name         = "app-backend"
    fqdns        = [azurerm_linux_web_app.this.default_hostname]
    probe_name   = "app-health-probe"
    
    health_probe = {
      interval            = 30
      timeout             = 30
      unhealthy_threshold = 3
      path                = "/health"
      match_status_codes  = ["200-399"]
    }
  }]
  
  # SSL/TLS configuration
  ssl_policy = {
    policy_type          = "Predefined"
    policy_name          = "AppGwSslPolicy20220101"  # TLS 1.2+ only
    min_protocol_version = "TLSv1_2"
  }
  
  # Managed certificates
  ssl_certificates = [{
    name                = "app-cert"
    key_vault_secret_id = azurerm_key_vault_certificate.app_cert.secret_id
  }]
}

# DDoS Protection
resource "azurerm_network_ddos_protection_plan" "this" {
  count = var.enable_ddos_protection ? 1 : 0
  
  name                = "${var.name_prefix}-ddos"
  resource_group_name = var.resource_group_name
  location            = var.location
  
  tags = local.tags
}

# Private Endpoints for PaaS services
resource "azurerm_private_endpoint" "sql" {
  name                = "${var.name_prefix}-sql-pe"
  resource_group_name = var.resource_group_name
  location            = var.location
  subnet_id           = var.private_endpoint_subnet_id
  
  private_service_connection {
    name                           = "${var.name_prefix}-sql-psc"
    private_connection_resource_id = azurerm_mssql_server.this.id
    is_manual_connection           = false
    subresource_names              = ["sqlServer"]
  }
  
  private_dns_zone_group {
    name                 = "sql-dns-zone-group"
    private_dns_zone_ids = [azurerm_private_dns_zone.sql.id]
  }
  
  tags = local.tags
}
```

### Security Monitoring and Compliance

```hcl
# modules/security/monitoring/main.tf
# Microsoft Defender for Cloud
resource "azurerm_security_center_subscription_pricing" "defender" {
  for_each = toset([
    "AppServices",
    "ContainerRegistry",
    "KeyVaults",
    "KubernetesService",
    "SqlServers",
    "SqlServerVirtualMachines",
    "StorageAccounts",
    "VirtualMachines",
    "OpenSourceRelationalDatabases",
    "CosmosDBs"
  ])
  
  tier          = "Standard"
  resource_type = each.value
}

# Sentinel for SIEM
resource "azurerm_sentinel_log_analytics_workspace_onboarding" "this" {
  workspace_id                 = azurerm_log_analytics_workspace.this.id
  customer_managed_key_enabled = var.environment == "production"
}

# Automated threat response
resource "azurerm_sentinel_automation_rule" "block_suspicious_ip" {
  name                       = "block-suspicious-ip"
  log_analytics_workspace_id = azurerm_log_analytics_workspace.this.id
  display_name               = "Block Suspicious IPs"
  order                      = 1
  enabled                    = true
  
  triggers {
    operator = "And"
    conditions {
      property = "IncidentSeverity"
      operator = "GreaterThan"
      values   = ["Medium"]
    }
  }
  
  actions {
    order = 1
    run_playbook {
      logic_app_id = azurerm_logic_app_workflow.block_ip.id
      tenant_id    = data.azurerm_client_config.current.tenant_id
    }
  }
}

# Compliance assessments
resource "azurerm_policy_assignment" "cis_benchmark" {
  name                 = "cis-azure-benchmark"
  scope                = data.azurerm_subscription.current.id
  policy_definition_id = "/providers/Microsoft.Authorization/policySetDefinitions/612b5213-9160-4969-8578-1518bd2a000c"
  
  identity {
    type = "SystemAssigned"
  }
  
  parameters = jsonencode({
    logAnalyticsWorkspaceIdForVMReporting = {
      value = azurerm_log_analytics_workspace.this.id
    }
  })
}
```

---

## 9. Monitoring and Observability

Comprehensive monitoring is essential for maintaining reliable infrastructure.

### ✅ DO: Implement Full-Stack Observability

```hcl
# modules/monitoring/observability/main.tf
# Centralized Log Analytics
resource "azurerm_log_analytics_workspace" "this" {
  name                = "${var.name_prefix}-law-${var.environment}"
  resource_group_name = var.resource_group_name
  location            = var.location
  sku                 = "PerGB2018"
  retention_in_days   = var.environment == "production" ? 90 : 30
  
  # Commitment tier for cost savings
  reservation_capacity_in_gb_per_day = var.environment == "production" ? 100 : null
  
  # Security features
  internet_ingestion_enabled = false
  internet_query_enabled     = false
  
  tags = local.tags
}

# Application Insights with workspace-based model
resource "azurerm_application_insights" "this" {
  name                = "${var.name_prefix}-ai-${var.environment}"
  resource_group_name = var.resource_group_name
  location            = var.location
  workspace_id        = azurerm_log_analytics_workspace.this.id
  application_type    = "web"
  
  # Sampling for cost control
  sampling_percentage = var.environment == "production" ? 10 : 100
  
  # Continuous export for long-term storage
  daily_data_cap_in_gb                     = var.environment == "production" ? 100 : 10
  daily_data_cap_notifications_disabled    = false
  disable_ip_masking                       = false
  internet_ingestion_enabled               = true
  internet_query_enabled                   = true
  retention_in_days                        = 90
  
  tags = local.tags
}

# Smart detection alerts
resource "azurerm_application_insights_smart_detection_rule" "failure_anomalies" {
  name                    = "failure-anomalies"
  application_insights_id = azurerm_application_insights.this.id
  enabled                 = true
  
  send_emails_to_subscription_owners = true
  additional_email_recipients        = var.alert_emails
}

# Custom availability tests
resource "azurerm_application_insights_standard_web_test" "homepage" {
  name                    = "${var.name_prefix}-availability-homepage"
  resource_group_name     = var.resource_group_name
  location                = var.location
  application_insights_id = azurerm_application_insights.this.id
  
  geo_locations = ["us-tx-sn1-azr", "us-il-ch1-azr", "us-ca-sjc-azr"]
  frequency     = 300
  timeout       = 30
  enabled       = true
  
  request {
    url                              = "https://${var.app_hostname}/health"
    http_verb                        = "GET"
    parse_dependent_requests_enabled = true
  }
  
  validation_rules {
    expected_status_code = 200
    ssl_check_enabled    = true
    ssl_cert_remaining_lifetime_check = 30
    
    content {
      content_match      = "healthy"
      ignore_case        = true
      pass_if_text_found = true
    }
  }
  
  tags = local.tags
}

# Workbooks for visualization
resource "azurerm_application_insights_workbook" "performance" {
  name                = "${var.name_prefix}-performance-workbook"
  resource_group_name = var.resource_group_name
  location            = var.location
  display_name        = "Application Performance Dashboard"
  
  data_json = jsonencode({
    version = "Notebook/1.0"
    items = [
      {
        type = 9
        content = {
          version = "KqlParameterItem/1.0"
          parameters = [
            {
              name  = "TimeRange"
              type  = 4
              value = {
                durationMs = 3600000
              }
            }
          ]
        }
      },
      {
        type = 3
        content = {
          version = "KqlItem/1.0"
          query   = <<-QUERY
            requests
            | summarize 
                RequestCount = count(), 
                AvgDuration = avg(duration),
                P95Duration = percentile(duration, 95),
                P99Duration = percentile(duration, 99)
                by bin(timestamp, 1m)
            | order by timestamp desc
          QUERY
          
          size         = 0
          queryType    = 0
          resourceType = "microsoft.insights/components"
          
          visualization = "timechart"
          chartSettings = {
            showMetrics = true
            showLegend  = true
          }
        }
      }
    ]
  })
  
  tags = local.tags
}

# Action groups for alerting
resource "azurerm_monitor_action_group" "critical" {
  name                = "${var.name_prefix}-ag-critical"
  resource_group_name = var.resource_group_name
  short_name          = "critical"
  
  email_receiver {
    name                    = "oncall-email"
    email_address           = var.oncall_email
    use_common_alert_schema = true
  }
  
  sms_receiver {
    name         = "oncall-sms"
    country_code = "1"
    phone_number = var.oncall_phone
  }
  
  webhook_receiver {
    name                    = "pagerduty"
    service_uri             = var.pagerduty_webhook_url
    use_common_alert_schema = true
  }
  
  azure_function_receiver {
    name                     = "remediation-function"
    function_app_resource_id = var.remediation_function_id
    function_name            = "AutoRemediate"
    http_trigger_url         = var.remediation_function_url
    use_common_alert_schema  = true
  }
}

# Metric alerts
resource "azurerm_monitor_metric_alert" "high_cpu" {
  name                = "${var.name_prefix}-high-cpu-alert"
  resource_group_name = var.resource_group_name
  scopes              = var.monitored_resource_ids
  
  description = "Alert when CPU usage is high"
  severity    = 2
  enabled     = true
  
  criteria {
    metric_namespace = "Microsoft.Compute/virtualMachines"
    metric_name      = "Percentage CPU"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 80
    
    dimension {
      name     = "Instance"
      operator = "Include"
      values   = ["*"]
    }
  }
  
  dynamic_criteria {
    metric_namespace = "Microsoft.Compute/virtualMachines"
    metric_name      = "Percentage CPU"
    aggregation      = "Average"
    operator         = "GreaterThan"
    
    alert_sensitivity = "Medium"
    
    dimension {
      name     = "Instance"
      operator = "Include"
      values   = ["*"]
    }
  }
  
  window_size        = "PT5M"
  frequency          = "PT1M"
  auto_mitigate      = true
  
  action {
    action_group_id = azurerm_monitor_action_group.critical.id
  }
  
  tags = local.tags
}

# Log alerts for errors
resource "azurerm_monitor_scheduled_query_rules_alert_v2" "application_errors" {
  name                = "${var.name_prefix}-app-errors-alert"
  resource_group_name = var.resource_group_name
  location            = var.location
  
  description         = "Alert on application error spike"
  severity            = 1
  enabled             = true
  
  scopes              = [azurerm_application_insights.this.id]
  evaluation_frequency = "PT5M"
  window_duration      = "PT15M"
  
  criteria {
    query                   = <<-QUERY
      exceptions
      | where timestamp > ago(15m)
      | summarize ErrorCount = count() by bin(timestamp, 5m)
      | where ErrorCount > 100
    QUERY
    
    time_aggregation_method = "Count"
    threshold               = 0
    operator                = "GreaterThan"
    
    failing_periods {
      minimum_failing_periods_to_trigger_alert = 1
      number_of_evaluation_periods             = 1
    }
  }
  
  auto_mitigation_enabled = true
  
  action {
    action_groups = [azurerm_monitor_action_group.critical.id]
  }
  
  tags = local.tags
}
```

---

## 10. Advanced Terraform Patterns

### Dynamic Provider Configuration

```hcl
# modules/providers/multi-region/main.tf
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.10"
      configuration_aliases = [
        azurerm.primary,
        azurerm.secondary,
        azurerm.dr
      ]
    }
  }
}

# Dynamic region deployment
variable "regions" {
  type = map(object({
    location       = string
    location_short = string
    is_primary     = bool
  }))
}

# Create resources in each region
resource "azurerm_resource_group" "regional" {
  for_each = var.regions
  provider = azurerm[each.value.is_primary ? "primary" : "secondary"]
  
  name     = "${var.name_prefix}-${each.key}-rg"
  location = each.value.location
  
  tags = merge(local.tags, {
    Region     = each.key
    IsPrimary  = each.value.is_primary
  })
}
```

### Complex Type Validation

```hcl
# Advanced variable validation
variable "network_config" {
  type = object({
    vnet_cidr = string
    subnets = map(object({
      cidr              = string
      service_endpoints = optional(list(string), [])
      delegation = optional(object({
        name    = string
        service = string
        actions = list(string)
      }))
    }))
  })
  
  validation {
    condition = can(cidrhost(var.network_config.vnet_cidr, 0))
    error_message = "VNet CIDR must be a valid IPv4 CIDR block."
  }
  
  validation {
    condition = alltrue([
      for subnet in var.network_config.subnets : 
      can(cidrsubnet(var.network_config.vnet_cidr, 
        ceil(log(pow(2, 32 - tonumber(split("/", subnet.cidr)[1])) / pow(2, 32 - tonumber(split("/", var.network_config.vnet_cidr)[1])), 2)), 
        0))
    ])
    error_message = "All subnet CIDRs must be within the VNet CIDR range."
  }
}
```

### State Migration Patterns

```hcl
# scripts/migrate-state.sh
#!/bin/bash
# Safe state migration with backup

set -euo pipefail

ENVIRONMENT=$1
OLD_BACKEND=$2
NEW_BACKEND=$3
BACKUP_DIR="state-backups/$(date +%Y%m%d-%H%M%S)"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Pull current state
cd "environments/$ENVIRONMENT"
terraform init -backend-config="$OLD_BACKEND"
terraform state pull > "$BACKUP_DIR/terraform.tfstate"

# Verify state integrity
if ! jq empty "$BACKUP_DIR/terraform.tfstate" 2>/dev/null; then
  echo "ERROR: State file is not valid JSON"
  exit 1
fi

# Initialize new backend
terraform init -backend-config="$NEW_BACKEND" -migrate-state

# Verify migration
NEW_STATE=$(terraform state pull)
OLD_STATE=$(cat "$BACKUP_DIR/terraform.tfstate")

if [ "$(echo "$NEW_STATE" | jq -r .serial)" -le "$(echo "$OLD_STATE" | jq -r .serial)" ]; then
  echo "ERROR: New state serial is not greater than old state"
  exit 1
fi

echo "State migration completed successfully"
echo "Backup saved to: $BACKUP_DIR"
```

---

## 11. Testing Infrastructure

### Terratest Implementation

```go
// tests/app_service_test.go
package test

import (
    "testing"
    "time"
    "fmt"
    
    "github.com/Azure/azure-sdk-for-go/sdk/azidentity"
    "github.com/Azure/azure-sdk-for-go/sdk/resourcemanager/appservice/armappservice/v4"
    "github.com/gruntwork-io/terratest/modules/terraform"
    "github.com/gruntwork-io/terratest/modules/http-helper"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestAppServiceModule(t *testing.T) {
    t.Parallel()
    
    // Unique suffix for resource names
    uniqueID := fmt.Sprintf("test-%d", time.Now().Unix())
    
    terraformOptions := &terraform.Options{
        TerraformDir: "../modules/compute/app-service",
        
        Vars: map[string]interface{}{
            "name_prefix":     uniqueID,
            "environment":     "test",
            "location":        "eastus2",
            "resource_group_name": fmt.Sprintf("%s-rg", uniqueID),
        },
        
        // Retry configuration for Azure eventual consistency
        MaxRetries:         3,
        TimeBetweenRetries: 30 * time.Second,
        RetryableTerraformErrors: map[string]string{
            ".*timeout.*": "Retry on timeout",
        },
    }
    
    // Clean up resources after test
    defer terraform.Destroy(t, terraformOptions)
    
    // Deploy infrastructure
    terraform.InitAndApply(t, terraformOptions)
    
    // Get outputs
    appServiceURL := terraform.Output(t, terraformOptions, "default_hostname")
    managedIdentityID := terraform.Output(t, terraformOptions, "managed_identity_principal_id")
    
    // Verify App Service is accessible
    url := fmt.Sprintf("https://%s/health", appServiceURL)
    http_helper.HttpGetWithRetryWithCustomValidation(
        t,
        url,
        nil,
        30,
        10*time.Second,
        func(statusCode int, body string) bool {
            return statusCode == 200
        },
    )
    
    // Verify managed identity was created
    assert.NotEmpty(t, managedIdentityID)
    
    // Additional Azure-specific validations
    t.Run("VerifySecurityConfiguration", func(t *testing.T) {
        ctx := context.Background()
        
        cred, err := azidentity.NewDefaultAzureCredential(nil)
        require.NoError(t, err)
        
        client, err := armappservice.NewWebAppsClient(
            os.Getenv("ARM_SUBSCRIPTION_ID"),
            cred,
            nil,
        )
        require.NoError(t, err)
        
        appName := terraform.Output(t, terraformOptions, "app_service_name")
        rgName := terraform.Output(t, terraformOptions, "resource_group_name")
        
        app, err := client.Get(ctx, rgName, appName, nil)
        require.NoError(t, err)
        
        // Verify security settings
        assert.Equal(t, "1.2", *app.Properties.SiteConfig.MinTLSVersion)
        assert.Equal(t, "Disabled", *app.Properties.SiteConfig.FtpsState)
        assert.True(t, *app.Properties.HTTPSOnly)
    })
}

// Integration test with real dependencies
func TestAppServiceIntegration(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping integration test in short mode")
    }
    
    terraformOptions := &terraform.Options{
        TerraformDir: "../environments/test",
        
        // Use remote state for integration tests
        BackendConfig: map[string]interface{}{
            "resource_group_name":  os.Getenv("TF_STATE_RG"),
            "storage_account_name": os.Getenv("TF_STATE_SA"),
            "container_name":       "tfstate",
            "key":                  "integration-test.terraform.tfstate",
        },
    }
    
    defer func() {
        // Only destroy if test created resources
        if !t.Failed() {
            terraform.Destroy(t, terraformOptions)
        }
    }()
    
    terraform.InitAndApply(t, terraformOptions)
    
    // Test end-to-end functionality
    t.Run("EndToEndDeployment", func(t *testing.T) {
        appURL := terraform.Output(t, terraformOptions, "app_url")
        
        // Deploy sample application
        deployApp(t, appURL)
        
        // Verify application functionality
        resp := http_helper.HttpGet(t, appURL+"/api/health", nil)
        assert.Contains(t, resp, `"status":"healthy"`)
    })
}
```

### Policy as Code Testing

```hcl
# policies/cost-tagging.sentinel
import "tfplan/v2" as tfplan

mandatory_tags = ["CostCenter", "Environment", "Owner", "Project"]

# Check all resources have required tags
main = rule {
    all tfplan.resource_changes as _, rc {
        rc.mode is "managed" and
        rc.change.actions contains "create" implies
        all mandatory_tags as tag {
            rc.change.after.tags contains tag
        }
    }
}

# Validate tag values
validate_cost_center = rule {
    all tfplan.resource_changes as _, rc {
        rc.mode is "managed" and
        rc.change.after.tags.CostCenter is not null implies
        rc.change.after.tags.CostCenter matches "^[0-9]{6}$"
    }
}
```

---

## 12. GitOps and Pipeline Automation

### Advanced Pipeline Patterns

```yaml
# .azure-pipelines/terraform-gitops.yml
trigger:
  branches:
    include:
      - main
      - develop
      - 'release/*'
  paths:
    include:
      - 'infrastructure/**'
      - '.azure-pipelines/**'

pr:
  branches:
    include:
      - main
      - develop
  paths:
    include:
      - 'infrastructure/**'

variables:
  - template: variables/global.yml
  - ${{ if eq(variables['Build.SourceBranch'], 'refs/heads/main') }}:
    - group: production-secrets
    - name: environment
      value: production
  - ${{ elseif eq(variables['Build.SourceBranch'], 'refs/heads/develop') }}:
    - group: staging-secrets
    - name: environment
      value: staging
  - ${{ else }}:
    - group: dev-secrets
    - name: environment
      value: dev

resources:
  repositories:
    - repository: terraform-modules
      type: github
      endpoint: github-connection
      name: myorg/terraform-modules
      ref: 'refs/tags/v2.0.0'

stages:
  - stage: SecurityScan
    displayName: 'Security and Compliance Scan'
    jobs:
      - template: templates/security-scan.yml
        parameters:
          scanType: 'full'
          
  - stage: CostAnalysis
    displayName: 'Cost Impact Analysis'
    dependsOn: []
    jobs:
      - job: AnalyzeCosts
        displayName: 'Analyze Cost Impact'
        pool:
          vmImage: 'ubuntu-latest'
        steps:
          - template: templates/cost-analysis.yml
            parameters:
              threshold: 1000
              
  - stage: Plan
    displayName: 'Plan Infrastructure Changes'
    dependsOn: 
      - SecurityScan
      - CostAnalysis
    condition: |
      and(
        succeeded(),
        or(
          eq(variables['Build.SourceBranch'], 'refs/heads/main'),
          eq(variables['Build.SourceBranch'], 'refs/heads/develop'),
          eq(variables['Build.Reason'], 'PullRequest')
        )
      )
    jobs:
      - deployment: PlanDeploy
        displayName: 'Plan Terraform Changes'
        environment: '$(environment)-plan'
        pool:
          vmImage: 'ubuntu-latest'
        strategy:
          runOnce:
            deploy:
              steps:
                - checkout: self
                - checkout: terraform-modules
                
                - task: AzureCLI@2
                  displayName: 'Setup Terraform Backend'
                  inputs:
                    azureSubscription: 'terraform-sp-$(environment)'
                    scriptType: 'bash'
                    scriptLocation: 'inlineScript'
                    inlineScript: |
                      # Dynamic backend configuration
                      cat > backend.tf <<EOF
                      terraform {
                        backend "azurerm" {
                          resource_group_name  = "$(TF_STATE_RG)"
                          storage_account_name = "$(TF_STATE_SA)"
                          container_name       = "tfstate"
                          key                  = "$(environment).terraform.tfstate"
                          use_azuread_auth     = true
                        }
                      }
                      EOF
                
                - task: TerraformTaskV4@4
                  displayName: 'Terraform Init'
                  inputs:
                    provider: 'azurerm'
                    command: 'init'
                    workingDirectory: '$(System.DefaultWorkingDirectory)/infrastructure/environments/$(environment)'
                    backendServiceArm: 'terraform-sp-$(environment)'
                
                - task: TerraformTaskV4@4
                  displayName: 'Terraform Plan'
                  inputs:
                    provider: 'azurerm'
                    command: 'plan'
                    workingDirectory: '$(System.DefaultWorkingDirectory)/infrastructure/environments/$(environment)'
                    commandOptions: '-out=tfplan -detailed-exitcode'
                    environmentServiceNameAzureRM: 'terraform-sp-$(environment)'
                    publishPlanResults: 'tfplan-$(environment)'
                
                - task: PublishPipelineArtifact@1
                  inputs:
                    targetPath: '$(System.DefaultWorkingDirectory)/infrastructure/environments/$(environment)/tfplan'
                    artifact: 'tfplan-$(environment)-$(Build.BuildId)'
                
                - task: PowerShell@2
                  displayName: 'Generate Plan Summary'
                  inputs:
                    targetType: 'inline'
                    script: |
                      $plan = terraform show -json tfplan | ConvertFrom-Json
                      
                      $summary = @{
                        ResourceChanges = @{
                          Create = ($plan.resource_changes | Where-Object { $_.change.actions -contains "create" }).Count
                          Update = ($plan.resource_changes | Where-Object { $_.change.actions -contains "update" }).Count
                          Delete = ($plan.resource_changes | Where-Object { $_.change.actions -contains "delete" }).Count
                        }
                      }
                      
                      $summary | ConvertTo-Json | Out-File plan-summary.json
                      
                      # Post to PR if applicable
                      if ($env:BUILD_REASON -eq "PullRequest") {
                        # Format markdown comment
                        $comment = @"
                      ## Terraform Plan Summary
                      
                      | Action | Count |
                      |--------|-------|
                      | Create | $($summary.ResourceChanges.Create) |
                      | Update | $($summary.ResourceChanges.Update) |
                      | Delete | $($summary.ResourceChanges.Delete) |
                      
                      [View detailed plan]($(System.CollectionUri)$(System.TeamProject)/_build/results?buildId=$(Build.BuildId))
                      "@
                        
                        # Post comment to PR
                        $uri = "$(System.CollectionUri)$(System.TeamProject)/_apis/git/repositories/$(Build.Repository.ID)/pullRequests/$(System.PullRequest.PullRequestId)/threads?api-version=7.0"
                        
                        $body = @{
                          comments = @(
                            @{
                              parentCommentId = 0
                              content = $comment
                              commentType = 1
                            }
                          )
                          status = "active"
                        } | ConvertTo-Json -Depth 10
                        
                        Invoke-RestMethod -Uri $uri -Method Post -Body $body -ContentType "application/json" -Headers @{
                          Authorization = "Bearer $(System.AccessToken)"
                        }
                      }
  
  - stage: Deploy
    displayName: 'Deploy Infrastructure'
    dependsOn: Plan
    condition: |
      and(
        succeeded(),
        ne(variables['Build.Reason'], 'PullRequest'),
        or(
          eq(variables['Build.SourceBranch'], 'refs/heads/main'),
          eq(variables['Build.SourceBranch'], 'refs/heads/develop')
        )
      )
    jobs:
      - deployment: ApplyChanges
        displayName: 'Apply Terraform Changes'
        environment: '$(environment)'
        pool:
          vmImage: 'ubuntu-latest'
        strategy:
          runOnce:
            deploy:
              steps:
                - checkout: self
                
                - task: DownloadPipelineArtifact@2
                  inputs:
                    artifact: 'tfplan-$(environment)-$(Build.BuildId)'
                    path: '$(System.DefaultWorkingDirectory)/infrastructure/environments/$(environment)'
                
                - task: AzureCLI@2
                  displayName: 'Pre-deployment Validation'
                  inputs:
                    azureSubscription: 'terraform-sp-$(environment)'
                    scriptType: 'bash'
                    scriptLocation: 'inlineScript'
                    inlineScript: |
                      # Verify critical resources exist
                      if [ "$(environment)" == "production" ]; then
                        # Check state storage is accessible
                        az storage account show \
                          --name "$(TF_STATE_SA)" \
                          --resource-group "$(TF_STATE_RG)" || exit 1
                        
                        # Verify backup exists
                        BACKUP_EXISTS=$(az storage blob exists \
                          --account-name "$(TF_STATE_SA)" \
                          --container-name "tfstate-backups" \
                          --name "$(environment)-$(Build.BuildId).tfstate" \
                          --query exists -o tsv)
                        
                        if [ "$BACKUP_EXISTS" != "true" ]; then
                          echo "Creating state backup..."
                          terraform state pull > temp.tfstate
                          
                          az storage blob upload \
                            --account-name "$(TF_STATE_SA)" \
                            --container-name "tfstate-backups" \
                            --name "$(environment)-$(Build.BuildId).tfstate" \
                            --file temp.tfstate
                        fi
                      fi
                
                - task: TerraformTaskV4@4
                  displayName: 'Terraform Apply'
                  inputs:
                    provider: 'azurerm'
                    command: 'apply'
                    workingDirectory: '$(System.DefaultWorkingDirectory)/infrastructure/environments/$(environment)'
                    commandOptions: 'tfplan'
                    environmentServiceNameAzureRM: 'terraform-sp-$(environment)'
                
                - task: AzureCLI@2
                  displayName: 'Post-deployment Validation'
                  inputs:
                    azureSubscription: 'terraform-sp-$(environment)'
                    scriptType: 'bash'
                    scriptLocation: 'inlineScript'
                    inlineScript: |
                      cd $(System.DefaultWorkingDirectory)/tests
                      
                      # Run smoke tests
                      python -m pytest smoke_tests/ \
                        --environment=$(environment) \
                        --junitxml=test-results.xml
                
                - task: PublishTestResults@2
                  inputs:
                    testResultsFormat: 'JUnit'
                    testResultsFiles: '**/test-results.xml'
                    failTaskOnFailedTests: true
  
  - stage: Monitoring
    displayName: 'Configure Monitoring'
    dependsOn: Deploy
    condition: succeeded()
    jobs:
      - job: SetupAlerts
        displayName: 'Configure Alerts and Dashboards'
        pool:
          vmImage: 'ubuntu-latest'
        steps:
          - task: AzureCLI@2
            displayName: 'Deploy Monitoring Configuration'
            inputs:
              azureSubscription: 'terraform-sp-$(environment)'
              scriptType: 'bash'
              scriptLocation: 'inlineScript'
              inlineScript: |
                # Deploy Azure Monitor alerts
                az deployment group create \
                  --resource-group "$(environment)-monitoring-rg" \
                  --template-file monitoring/alerts.bicep \
                  --parameters environment=$(environment)
                
                # Create Grafana dashboards
                if [ "$(environment)" == "production" ]; then
                  ./scripts/deploy-dashboards.sh production
                fi
```

---

## 13. Disaster Recovery and Business Continuity

### Multi-Region Failover Architecture

```hcl
# modules/dr/multi-region/main.tf
locals {
  regions = {
    primary = {
      location       = var.primary_location
      location_short = var.primary_location_short
    }
    secondary = {
      location       = var.secondary_location  
      location_short = var.secondary_location_short
    }
  }
}

# Traffic Manager for global load balancing
resource "azurerm_traffic_manager_profile" "global" {
  name                   = "${var.name_prefix}-tm-global"
  resource_group_name    = var.resource_group_name
  traffic_routing_method = "Priority"
  
  dns_config {
    relative_name = var.name_prefix
    ttl           = 30
  }
  
  monitor_config {
    protocol                    = "HTTPS"
    port                        = 443
    path                        = "/health"
    interval_in_seconds         = 10
    timeout_in_seconds          = 5
    tolerated_number_of_failures = 2
  }
  
  tags = local.tags
}

# Deploy resources in each region
module "regional_deployment" {
  source = "../regional-resources"
  
  for_each = local.regions
  
  name_prefix    = var.name_prefix
  location       = each.value.location
  location_short = each.value.location_short
  environment    = var.environment
  
  # Cross-region references
  peer_vnet_id = each.key == "primary" ? 
    module.regional_deployment["secondary"].vnet_id : 
    module.regional_deployment["primary"].vnet_id
  
  # Cosmos DB with multi-region writes
  cosmos_account_id = azurerm_cosmosdb_account.global.id
  
  providers = {
    azurerm = azurerm[each.key]
  }
}

# Add endpoints to Traffic Manager
resource "azurerm_traffic_manager_endpoint" "regional" {
  for_each = local.regions
  
  name                = "${each.key}-endpoint"
  profile_id          = azurerm_traffic_manager_profile.global.id
  type                = "azureEndpoints"
  target_resource_id  = module.regional_deployment[each.key].app_gateway_id
  priority            = each.key == "primary" ? 1 : 2
  
  custom_header {
    name  = "host"
    value = var.custom_domain
  }
}

# Cross-region VNet peering
resource "azurerm_virtual_network_peering" "primary_to_secondary" {
  name                      = "primary-to-secondary"
  resource_group_name       = module.regional_deployment["primary"].resource_group_name
  virtual_network_name      = module.regional_deployment["primary"].vnet_name
  remote_virtual_network_id = module.regional_deployment["secondary"].vnet_id
  
  allow_virtual_network_access = true
  allow_forwarded_traffic      = true
  allow_gateway_transit        = false
  use_remote_gateways          = false
}

# Automated failover runbook
resource "azurerm_automation_runbook" "failover" {
  name                    = "${var.name_prefix}-failover-runbook"
  location                = var.primary_location
  resource_group_name     = var.resource_group_name
  automation_account_name = azurerm_automation_account.dr.name
  log_verbose             = true
  log_progress            = true
  description             = "Automated DR failover runbook"
  runbook_type            = "PowerShell"
  
  content = file("${path.module}/scripts/failover.ps1")
  
  publish_content_link {
    uri = "https://raw.githubusercontent.com/myorg/runbooks/main/dr-failover.ps1"
  }
}

# Recovery Services Vault for backup
resource "azurerm_recovery_services_vault" "dr" {
  for_each = local.regions
  
  name                = "${var.name_prefix}-rsv-${each.value.location_short}"
  location            = each.value.location
  resource_group_name = module.regional_deployment[each.key].resource_group_name
  sku                 = "Standard"
  soft_delete_enabled = true
  
  encryption {
    key_id                       = module.regional_deployment[each.key].cmk_key_id
    infrastructure_encryption_enabled = true
  }
  
  tags = merge(local.tags, {
    Region = each.key
  })
}

# Cross-region backup policies
resource "azurerm_backup_policy_vm" "cross_region" {
  name                = "${var.name_prefix}-backup-policy-xr"
  resource_group_name = var.resource_group_name
  recovery_vault_name = azurerm_recovery_services_vault.dr["primary"].name
  
  backup {
    frequency = "Daily"
    time      = "23:00"
  }
  
  retention_daily {
    count = 30
  }
  
  retention_weekly {
    count    = 12
    weekdays = ["Sunday"]
  }
  
  retention_monthly {
    count    = 12
    weekdays = ["Sunday"]
    weeks    = ["First"]
  }
  
  retention_yearly {
    count    = 5
    weekdays = ["Sunday"]
    weeks    = ["First"]
    months   = ["January"]
  }
}

# Site Recovery for VM replication
resource "azurerm_site_recovery_fabric" "primary" {
  name                = "primary-fabric"
  resource_group_name = var.resource_group_name
  recovery_vault_name = azurerm_recovery_services_vault.dr["secondary"].name
  location            = var.primary_location
}

resource "azurerm_site_recovery_fabric" "secondary" {
  name                = "secondary-fabric"
  resource_group_name = var.resource_group_name
  recovery_vault_name = azurerm_recovery_services_vault.dr["secondary"].name
  location            = var.secondary_location
}

resource "azurerm_site_recovery_protection_container" "primary" {
  name                 = "primary-container"
  resource_group_name  = var.resource_group_name
  recovery_vault_name  = azurerm_recovery_services_vault.dr["secondary"].name
  recovery_fabric_name = azurerm_site_recovery_fabric.primary.name
}

resource "azurerm_site_recovery_protection_container" "secondary" {
  name                 = "secondary-container"
  resource_group_name  = var.resource_group_name
  recovery_vault_name  = azurerm_recovery_services_vault.dr["secondary"].name
  recovery_fabric_name = azurerm_site_recovery_fabric.secondary.name
}

resource "azurerm_site_recovery_protection_container_mapping" "mapping" {
  name                                      = "primary-to-secondary-mapping"
  resource_group_name                       = var.resource_group_name
  recovery_vault_name                       = azurerm_recovery_services_vault.dr["secondary"].name
  recovery_fabric_name                      = azurerm_site_recovery_fabric.primary.name
  recovery_source_protection_container_name = azurerm_site_recovery_protection_container.primary.name
  recovery_target_protection_container_id   = azurerm_site_recovery_protection_container.secondary.id
  
  recovery_replication_policy_id = azurerm_site_recovery_replication_policy.policy.id
}

# DR validation testing
resource "azurerm_automation_schedule" "dr_test" {
  name                    = "monthly-dr-test"
  resource_group_name     = var.resource_group_name
  automation_account_name = azurerm_automation_account.dr.name
  frequency               = "Month"
  interval                = 1
  start_time              = "${formatdate("YYYY-MM-DD", timeadd(timestamp(), "24h"))}T03:00:00Z"
  description             = "Monthly DR validation test"
  
  monthly_occurrence {
    day        = "Sunday"
    occurrence = 1
  }
}
```

---

## 14. Performance Optimization Patterns

### Terraform Performance Tuning

```hcl
# terraform.tf - Provider configuration for performance
terraform {
  required_version = ">= 1.10"
  
  # Experimental features for performance
  experiments = [
    module_variable_optional_attrs,
  ]
  
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.10"
    }
    
    # Lightweight provider for data operations
    azapi = {
      source  = "azure/azapi"
      version = "~> 2.1"
    }
  }
  
  # Provider plugin cache
  plugin_cache_dir = "$HOME/.terraform.d/plugin-cache"
}

# Configure providers for performance
provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false  # Faster destroys in dev
    }
    
    virtual_machine {
      delete_os_disk_on_deletion = true
      graceful_shutdown          = false  # Faster VM operations in dev
    }
  }
  
  # Increase timeouts for large deployments
  partner_id = "12345678-1234-1234-1234-123456789012"
  
  # Skip provider registration for faster init
  skip_provider_registration = true
  
  # Use MSI for faster auth
  use_msi = var.use_msi
}

# Parallelize operations
locals {
  # Increase parallelism for large deployments
  parallelism = var.environment == "production" ? 50 : 10
}
```

### Resource Deployment Optimization

```hcl
# Use azapi for new features without provider updates
resource "azapi_resource" "container_app_job" {
  type      = "Microsoft.App/jobs@2024-03-01"
  name      = "${var.name_prefix}-job"
  parent_id = azurerm_resource_group.this.id
  location  = var.location
  
  # Jobs API not yet in azurerm provider
  body = {
    properties = {
      configuration = {
        triggerType = "Schedule"
        scheduleTriggerConfig = {
          cronExpression = "0 0 * * *"
        }
        registries = [{
          server            = azurerm_container_registry.this.login_server
          identity          = azurerm_user_assigned_identity.this.id
        }]
      }
      
      template = {
        containers = [{
          name  = "job"
          image = "${azurerm_container_registry.this.login_server}/job:latest"
          
          resources = {
            cpu    = 0.5
            memory = "1Gi"
          }
        }]
      }
    }
  }
}

# Conditional resource creation for environments
resource "azurerm_application_gateway" "waf" {
  count = var.enable_waf ? 1 : 0
  # ... configuration
}

# Dynamic resource creation
resource "azurerm_network_security_rule" "dynamic" {
  for_each = { for rule in var.security_rules : rule.name => rule }
  
  name                        = each.key
  priority                    = each.value.priority
  direction                   = each.value.direction
  access                      = each.value.access
  protocol                    = each.value.protocol
  source_port_range           = each.value.source_port_range
  destination_port_range      = each.value.destination_port_range
  source_address_prefix       = each.value.source_address_prefix
  destination_address_prefix  = each.value.destination_address_prefix
  resource_group_name         = var.resource_group_name
  network_security_group_name = azurerm_network_security_group.this.name
}
```

---

## 15. Modern Architecture Patterns

### Container Apps for Microservices

```hcl
# modules/compute/container-apps/main.tf
resource "azurerm_container_app_environment" "this" {
  name                       = "${var.name_prefix}-cae-${var.environment}"
  location                   = var.location
  resource_group_name        = var.resource_group_name
  log_analytics_workspace_id = var.log_analytics_workspace_id
  
  # VNet integration for security
  infrastructure_subnet_id = var.infrastructure_subnet_id
  internal_load_balancer_enabled = true
  
  # Zone redundancy
  zone_redundancy_enabled = var.environment == "production"
  
  # Workload profiles for GPU/Memory optimized containers
  dynamic "workload_profile" {
    for_each = var.workload_profiles
    content {
      name                  = workload_profile.value.name
      workload_profile_type = workload_profile.value.type
      minimum_count         = workload_profile.value.min_count
      maximum_count         = workload_profile.value.max_count
    }
  }
  
  tags = local.tags
}

# Container Apps with Dapr
resource "azurerm_container_app" "api" {
  name                         = "${var.name_prefix}-api"
  container_app_environment_id = azurerm_container_app_environment.this.id
  resource_group_name          = var.resource_group_name
  revision_mode                = "Multiple"  # Blue-green deployments
  
  # Workload profile assignment
  workload_profile_name = var.api_workload_profile
  
  identity {
    type = "SystemAssigned, UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.app.id]
  }
  
  template {
    min_replicas = var.environment == "production" ? 2 : 0
    max_replicas = var.environment == "production" ? 100 : 10
    
    # Advanced scaling rules
    azure_queue_scale_rule {
      name         = "queue-scale"
      queue_name   = azurerm_storage_queue.tasks.name
      queue_length = 5
      
      authentication {
        secret_name       = "storage-connection"
        trigger_parameter = "connection"
      }
    }
    
    http_scale_rule {
      name                = "http-scale"
      concurrent_requests = 100
    }
    
    # Containers
    container {
      name   = "api"
      image  = "${var.container_registry}/api:${var.api_version}"
      cpu    = 1
      memory = "2Gi"
      
      # Probes
      readiness_probe {
        transport = "HTTP"
        path      = "/ready"
        port      = 8080
        
        initial_delay_seconds = 10
        period_seconds        = 5
        failure_threshold     = 3
        success_threshold     = 1
      }
      
      liveness_probe {
        transport = "HTTP"
        path      = "/health"
        port      = 8080
        
        initial_delay_seconds = 30
        period_seconds        = 10
        failure_threshold     = 3
      }
      
      # Environment variables
      env {
        name  = "AZURE_CLIENT_ID"
        value = azurerm_user_assigned_identity.app.client_id
      }
      
      env {
        name        = "DATABASE_URL"
        secret_name = "database-url"
      }
      
      # Volume mounts
      volume_mounts {
        name = "azure-files"
        path = "/data"
      }
    }
    
    # Init containers
    init_container {
      name   = "migration"
      image  = "${var.container_registry}/migration:${var.api_version}"
      cpu    = 0.5
      memory = "1Gi"
      
      command = ["/bin/sh", "-c", "npm run migrate:prod"]
      
      env {
        name        = "DATABASE_URL"
        secret_name = "database-url"
      }
    }
    
    # Volumes
    volume {
      name         = "azure-files"
      storage_type = "AzureFile"
      storage_name = azurerm_storage_share.app_data.name
    }
  }
  
  # Ingress configuration
  ingress {
    external_enabled = false  # Internal only
    target_port      = 8080
    transport        = "http2"
    
    traffic_weight {
      revision_suffix = "v1"
      percentage      = 90
    }
    
    traffic_weight {
      latest_revision = true
      percentage      = 10
    }
    
    # Custom domains
    custom_domain {
      name           = var.api_domain
      certificate_id = azurerm_container_app_environment_certificate.api.id
    }
    
    # CORS configuration
    cors_policy {
      allowed_origins = var.cors_origins
      allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
      allowed_headers = ["*"]
      max_age         = 86400
    }
  }
  
  # Dapr configuration
  dapr {
    enabled      = true
    app_id       = "api"
    app_port     = 8080
    app_protocol = "http"
    
    # Dapr components configured separately
  }
  
  # Secrets
  secret {
    name  = "database-url"
    value = var.database_connection_string
  }
  
  secret {
    name  = "storage-connection"
    value = azurerm_storage_account.app.primary_connection_string
  }
  
  registry {
    server               = var.container_registry
    username             = azurerm_container_registry.this.admin_username
    password_secret_name = "registry-password"
  }
  
  tags = local.tags
}

# Dapr Components
resource "azapi_resource" "dapr_state_store" {
  type      = "Microsoft.App/managedEnvironments/daprComponents@2024-03-01"
  name      = "statestore"
  parent_id = azurerm_container_app_environment.this.id
  
  body = {
    properties = {
      componentType = "state.azure.cosmosdb"
      version       = "v1"
      
      metadata = [
        {
          name  = "url"
          value = azurerm_cosmosdb_account.this.endpoint
        },
        {
          name  = "database"
          value = azurerm_cosmosdb_sql_database.this.name
        },
        {
          name  = "collection"
          value = "state"
        }
      ]
      
      secrets = [
        {
          name  = "masterkey"
          value = azurerm_cosmosdb_account.this.primary_key
        }
      ]
      
      scopes = ["api", "worker"]
    }
  }
}
```

### Event-Driven Architecture with Event Grid

```hcl
# modules/events/event-grid/main.tf
resource "azurerm_eventgrid_topic" "main" {
  name                = "${var.name_prefix}-egt-${var.environment}"
  location            = var.location
  resource_group_name = var.resource_group_name
  
  # Enhanced security
  public_network_access_enabled = false
  local_auth_enabled            = false  # Azure AD only
  
  input_schema = "CloudEventSchemaV1_0"
  
  # Managed identity for event delivery
  identity {
    type = "SystemAssigned"
  }
  
  # Advanced filtering
  inbound_ip_rule {
    ip_mask = var.allowed_ip_range
    action  = "Allow"
  }
  
  tags = local.tags
}

# Event subscriptions with dead letter and retry
resource "azurerm_eventgrid_event_subscription" "orders" {
  name  = "order-processor"
  scope = azurerm_eventgrid_topic.main.id
  
  # Delivery to Container App
  webhook_endpoint {
    url = "https://${azurerm_container_app.processor.latest_revision_fqdn}/events"
    
    # Authentication
    active_directory_tenant_id     = data.azurerm_client_config.current.tenant_id
    active_directory_app_id_or_uri = azurerm_user_assigned_identity.processor.client_id
  }
  
  # Event filtering
  advanced_filter {
    string_in {
      key    = "eventType"
      values = ["Order.Created", "Order.Updated"]
    }
  }
  
  # Retry policy
  retry_policy {
    max_delivery_attempts = 30
    event_time_to_live    = 1440  # 24 hours
  }
  
  # Dead lettering
  dead_letter_identity {
    type = "SystemAssigned"
  }
  
  storage_blob_dead_letter_destination {
    storage_account_id          = azurerm_storage_account.deadletter.id
    storage_blob_container_name = "deadletter-events"
  }
}

# Event Grid Domain for multi-tenant scenarios
resource "azurerm_eventgrid_domain" "multitenant" {
  count = var.enable_multi_tenant ? 1 : 0
  
  name                = "${var.name_prefix}-egd-${var.environment}"
  location            = var.location
  resource_group_name = var.resource_group_name
  
  # Partitioning for scale
  input_schema                  = "CloudEventSchemaV1_0"
  public_network_access_enabled = false
  local_auth_enabled            = false
  
  # Auto-create topics
  auto_create_topic_with_first_subscription = true
  auto_delete_topic_with_last_subscription  = true
  
  identity {
    type = "SystemAssigned"
  }
  
  tags = local.tags
}
```

---

## Conclusion

This guide represents battle-tested patterns for building production-grade infrastructure on Azure using Terraform and Azure DevOps. The key principles to remember:

1. **State is Sacred**: Always use remote state with proper versioning and backup
2. **Security First**: Implement defense in depth at every layer
3. **Cost Awareness**: Monitor and optimize costs continuously
4. **Automation Everything**: From testing to deployment to monitoring
5. **Plan for Failure**: Design for resilience and practice disaster recovery

Remember that infrastructure as code is a journey, not a destination. Continue to iterate, improve, and adapt these patterns to your specific needs while maintaining the core principles of security, reliability, and efficiency.

For the latest updates and community patterns, refer to:
- [Terraform Azure Provider Documentation](https://registry.terraform.io/providers/hashicorp/azurerm/latest)
- [Azure Architecture Center](https://docs.microsoft.com/azure/architecture/)
- [Azure DevOps Documentation](https://docs.microsoft.com/azure/devops/)

Stay current with the rapidly evolving cloud landscape, but always validate new features in non-production environments before adopting them in critical workloads.