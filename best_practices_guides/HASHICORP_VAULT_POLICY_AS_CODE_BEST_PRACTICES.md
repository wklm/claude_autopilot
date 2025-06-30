# The Definitive Guide to Policy-as-Code with OPA, Vault, and Boundary (2025)

This guide synthesizes modern best practices for building secure, scalable, and auditable infrastructure using Open Policy Agent (OPA) with Rego for Kubernetes admission control, and HashiCorp Vault 1.16 with Boundary for secrets management and secure session brokering. It moves beyond basic tutorials to provide production-grade architectural patterns.

### Prerequisites & Configuration
Ensure your environment uses **OPA 0.69+**, **Kubernetes 1.31+**, **Vault 1.16+**, and **Boundary 0.18+**.

Core configuration for production deployments:

```yaml
# values/opa-gatekeeper.yaml
replicas: 3  # HA configuration
podDisruptionBudget:
  minAvailable: 2
resources:
  limits:
    cpu: 1000m
    memory: 512Mi
  requests:
    cpu: 100m
    memory: 256Mi
# Enable mutation webhooks (stable in 3.17)
experimentalEnableMutation: true
# Policy library synchronization
enableExternalData: true
externaldataProviderResponseCacheTTLSeconds: 60
```

```hcl
# vault-config.hcl
ui = true
cluster_addr = "https://vault.internal:8201"
api_addr = "https://vault.internal:8200"

listener "tcp" {
  address = "0.0.0.0:8200"
  tls_disable = false
  tls_cert_file = "/vault/tls/server.crt"
  tls_key_file = "/vault/tls/server.key"
  tls_min_version = "tls13"  # Enforce TLS 1.3
}

storage "raft" {
  path = "/vault/data"
  node_id = "vault-0"
  
  # Performance tuning for 1.16
  performance_multiplier = 2
  max_entry_size = "2MB"
  
  # Enable integrated storage autopilot
  autopilot {
    cleanup_dead_servers = true
    last_contact_threshold = "10s"
    min_quorum = 3
  }
}

# Enable new 1.16 features
experiments = ["events.beta1", "workflow.beta1"]
```

---

## 1. Foundational Architecture & Directory Structure

A well-architected policy-as-code system separates concerns between policy definitions, test fixtures, deployment configurations, and monitoring. This structure scales from small teams to enterprise deployments.

### ✅ DO: Use a Scalable Repository Layout

```
/platform-policies
├── opa/                      # OPA/Gatekeeper policies
│   ├── policies/             # Rego policy definitions
│   │   ├── admission/        # K8s admission policies
│   │   │   ├── security/     # Security-focused policies
│   │   │   ├── resource/     # Resource management
│   │   │   └── compliance/   # Regulatory compliance
│   │   └── library/          # Shared Rego functions
│   ├── templates/            # Gatekeeper ConstraintTemplates
│   ├── constraints/          # Constraint instances
│   ├── mutations/            # Mutation policies (AssignMetadata, etc.)
│   └── tests/                # Rego unit tests
├── vault/                    # Vault configuration as code
│   ├── policies/             # Vault policy documents
│   ├── auth/                 # Auth method configurations
│   ├── secrets/              # Secret engine configurations
│   └── scripts/              # Automation scripts
├── boundary/                 # Boundary configuration
│   ├── terraform/            # Boundary resources via Terraform
│   ├── targets/              # Target configurations
│   └── workers/              # Worker configurations
├── integration/              # Cross-tool integrations
│   ├── opa-vault/            # OPA external data from Vault
│   └── boundary-vault/       # Credential brokering
└── .github/                  # CI/CD workflows
    ├── workflows/
    └── policy-tests/
```

### ✅ DO: Version Control Everything

Policy is code. Every policy decision, configuration change, and integration must be version controlled, peer-reviewed, and tested before deployment.

```yaml
# .github/workflows/policy-validation.yml
name: Policy Validation Pipeline
on:
  pull_request:
    paths:
      - 'opa/**'
      - 'vault/**'
      - 'boundary/**'

jobs:
  opa-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run OPA Tests
        run: |
          opa test opa/policies/ -v
          opa fmt --list opa/policies/
      
      - name: Validate Gatekeeper Templates
        run: |
          for template in opa/templates/*.yaml; do
            kubectl --dry-run=server apply -f "$template"
          done
      
      - name: Policy Coverage Report
        run: |
          opa test opa/policies/ --coverage | tee coverage.json
          coverage=$(jq '.coverage' coverage.json)
          echo "Policy coverage: ${coverage}%"
          if (( $(echo "$coverage < 80" | bc -l) )); then
            echo "Coverage below 80% threshold"
            exit 1
          fi
```

---

## 2. OPA + Rego: The Admission Control Paradigm

Modern Kubernetes security requires policy enforcement at the admission control layer. OPA Gatekeeper provides this capability with a declarative, testable approach.

### ✅ DO: Structure Policies for Reusability and Testing

Create modular policies that separate business logic from Kubernetes-specific details.

```rego
# opa/policies/library/security_standards.rego
package lib.security

import future.keywords.contains
import future.keywords.if
import future.keywords.in

# Reusable security predicates
is_privileged(container) if {
    container.securityContext.privileged == true
}

runs_as_root(container) if {
    container.securityContext.runAsUser == 0
}

runs_as_root(container) if {
    container.securityContext.runAsNonRoot == false
}

has_dangerous_capabilities(container) if {
    some capability in container.securityContext.capabilities.add
    capability in {"SYS_ADMIN", "NET_ADMIN", "SYS_TIME"}
}

# Image trust verification
is_trusted_registry(image) if {
    some registry in data.inventory.trusted_registries
    startswith(image, registry)
}

# CVE scanning integration
has_critical_cves(image) if {
    # External data provider integration
    response := http.send({
        "method": "GET",
        "url": sprintf("%s/scan/%s", [data.config.scanner_url, image]),
        "headers": {"Authorization": sprintf("Bearer %s", [data.config.scanner_token])}
    })
    response.status_code == 200
    count([cve | cve := response.body.vulnerabilities[_]; cve.severity == "CRITICAL"]) > 0
}
```

### ✅ DO: Create Comprehensive Admission Policies

Build admission policies that enforce security, compliance, and operational standards.

```rego
# opa/policies/admission/security/pod_security_standards.rego
package kubernetes.admission.security

import data.lib.security
import future.keywords.contains
import future.keywords.if
import future.keywords.in

# Enforce Pod Security Standards (Restricted profile)
violation[msg] {
    input.review.kind.kind == "Pod"
    container := input.review.object.spec.containers[_]
    security.is_privileged(container)
    msg := sprintf("Pod %s/%s: container %s runs in privileged mode", [
        input.review.namespace,
        input.review.object.metadata.name,
        container.name
    ])
}

violation[msg] {
    input.review.kind.kind == "Pod"
    container := input.review.object.spec.containers[_]
    security.runs_as_root(container)
    msg := sprintf("Pod %s/%s: container %s must not run as root", [
        input.review.namespace,
        input.review.object.metadata.name,
        container.name
    ])
}

violation[msg] {
    input.review.kind.kind == "Pod"
    container := input.review.object.spec.containers[_]
    security.has_dangerous_capabilities(container)
    msg := sprintf("Pod %s/%s: container %s requests dangerous capabilities", [
        input.review.namespace,
        input.review.object.metadata.name,
        container.name
    ])
}

# Enforce image trust
violation[msg] {
    input.review.kind.kind == "Pod"
    container := input.review.object.spec.containers[_]
    not security.is_trusted_registry(container.image)
    msg := sprintf("Pod %s/%s: container %s uses untrusted image %s", [
        input.review.namespace,
        input.review.object.metadata.name,
        container.name,
        container.image
    ])
}
```

### ❌ DON'T: Write Monolithic, Untestable Policies

Avoid creating single, massive policy files that combine all rules. This makes testing, debugging, and maintenance extremely difficult.

```rego
# Bad - Monolithic policy file
package kubernetes.admission

# 500+ lines of mixed concerns: security, resources, naming, etc.
deny[msg] {
    # Security rule mixed with resource limits mixed with naming conventions
    # Impossible to test individual rules
    # No reusability
}
```

### ✅ DO: Implement Mutation Policies for Automatic Remediation

Use Gatekeeper's mutation capabilities to automatically fix common issues rather than just blocking them.

```yaml
# opa/mutations/security-context-defaults.yaml
apiVersion: mutations.gatekeeper.sh/v1beta1
kind: Assign
metadata:
  name: pod-security-context-defaults
spec:
  applyTo:
  - groups: [""]
    kinds: ["Pod"]
    versions: ["v1"]
  match:
    scope: Namespaced
    kinds:
    - apiGroups: [""]
      kinds: ["Pod"]
    excludedNamespaces: ["kube-system", "gatekeeper-system"]
  location: "spec.securityContext"
  parameters:
    assign:
      value:
        runAsNonRoot: true
        runAsUser: 10000
        fsGroup: 10000
        seccompProfile:
          type: RuntimeDefault
```

### Advanced OPA Patterns

#### External Data Providers for Dynamic Policies

Integrate OPA with external systems for real-time policy decisions:

```yaml
# opa/templates/external-data-provider.yaml
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: k8srequiredlabels
spec:
  crd:
    spec:
      names:
        kind: K8sRequiredLabels
      validation:
        openAPIV3Schema:
          type: object
  targets:
  - target: admission.k8s.gatekeeper.sh
    rego: |
      package k8srequiredlabels
      
      violation[{"msg": msg}] {
        # Call external data provider (e.g., Vault for dynamic label requirements)
        response := external_data({"provider": "vault-provider", "keys": ["required-labels", input.review.namespace]})
        required_labels := response.vault_provider["required-labels"][input.review.namespace]
        
        provided := input.review.object.metadata.labels
        missing := required_labels[_]
        not provided[missing]
        
        msg := sprintf("Missing required label: %v", [missing])
      }
```

---

## 3. Vault 1.16: Enterprise Secrets Management

Vault 1.16 introduces significant improvements in performance, workflow automation, and event streaming. Design your secrets architecture to leverage these capabilities.

### ✅ DO: Implement Least-Privilege Access with Dynamic Secrets

Static credentials are a security anti-pattern. Use Vault's dynamic secrets engines wherever possible.

```hcl
# vault/secrets/database.hcl
# PostgreSQL dynamic secrets configuration
resource "vault_mount" "postgres" {
  path = "database"
  type = "database"
}

resource "vault_database_secret_backend_connection" "postgres" {
  backend       = vault_mount.postgres.path
  name          = "postgres"
  allowed_roles = ["app-*"]

  postgresql {
    connection_url = "postgresql://{{username}}:{{password}}@postgres.internal:5432/postgres?sslmode=require"
    username       = var.postgres_admin_user
    password       = var.postgres_admin_password
    
    # Vault 1.16: Enhanced connection pooling
    max_open_connections     = 10
    max_idle_connections     = 5
    max_connection_lifetime  = "5m"
  }
}

resource "vault_database_secret_backend_role" "app_read" {
  backend = vault_mount.postgres.path
  name    = "app-readonly"
  db_name = vault_database_secret_backend_connection.postgres.name
  
  creation_statements = [
    "CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}';",
    "GRANT SELECT ON ALL TABLES IN SCHEMA public TO \"{{name}}\";"
  ]
  
  revocation_statements = [
    "REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM \"{{name}}\";",
    "DROP ROLE IF EXISTS \"{{name}}\";"
  ]
  
  default_ttl = 3600  # 1 hour
  max_ttl     = 7200  # 2 hours
}
```

### ✅ DO: Use Vault's Policy Templates for Scalable RBAC

Leverage templated policies to manage access at scale:

```hcl
# vault/policies/k8s-namespace-policy.hcl
# Template policy for Kubernetes namespace isolation
path "secret/data/k8s/{{identity.entity.aliases.auth_kubernetes_cluster.metadata.namespace}}/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "secret/metadata/k8s/{{identity.entity.aliases.auth_kubernetes_cluster.metadata.namespace}}/*" {
  capabilities = ["list", "read", "delete"]
}

# Allow access to dynamic database credentials for the namespace
path "database/creds/app-{{identity.entity.aliases.auth_kubernetes_cluster.metadata.namespace}}" {
  capabilities = ["read"]
}

# Transit encryption for the namespace
path "transit/encrypt/k8s-{{identity.entity.aliases.auth_kubernetes_cluster.metadata.namespace}}" {
  capabilities = ["update"]
}

path "transit/decrypt/k8s-{{identity.entity.aliases.auth_kubernetes_cluster.metadata.namespace}}" {
  capabilities = ["update"]
}
```

### ✅ DO: Implement Secret Rotation with Vault 1.16 Events

Use Vault's new event system to trigger automatic rotation workflows:

```hcl
# vault/scripts/setup-rotation-events.sh
#!/bin/bash

# Enable event notifications for secret expiration
vault write sys/events/subscribe/secret-expiry \
  event_types="kv.secret.expiry" \
  namespace="*"

# Configure webhook for rotation automation
vault write sys/events/notifications/webhook/rotate-secrets \
  endpoint="https://rotation-service.internal/webhook" \
  events="kv.secret.expiry" \
  hmac_key="$(vault kv get -field=hmac_key secret/rotation/webhook)"
```

```go
// rotation-service/handler.go
package main

import (
    "encoding/json"
    "net/http"
    "github.com/hashicorp/vault/api"
)

type VaultEvent struct {
    EventType string          `json:"event_type"`
    Timestamp string          `json:"timestamp"`
    Namespace string          `json:"namespace"`
    Path      string          `json:"path"`
    Metadata  json.RawMessage `json:"metadata"`
}

func handleRotationWebhook(w http.ResponseWriter, r *http.Request) {
    var event VaultEvent
    if err := json.NewDecoder(r.Body).Decode(&event); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    
    // Verify HMAC signature
    if !verifyHMAC(r.Header.Get("X-Vault-Signature"), r.Body) {
        http.Error(w, "Invalid signature", http.StatusUnauthorized)
        return
    }
    
    // Trigger rotation based on path
    switch {
    case strings.HasPrefix(event.Path, "database/"):
        rotateDatabasePassword(event.Path)
    case strings.HasPrefix(event.Path, "pki/"):
        rotateCertificate(event.Path)
    default:
        rotateGenericSecret(event.Path)
    }
    
    w.WriteHeader(http.StatusOK)
}
```

### Advanced Vault Patterns

#### Multi-Region Replication with Performance Standbys

Configure Vault for global scale with local read replicas:

```hcl
# Primary cluster configuration
resource "vault_raft_secondary" "eu_west" {
  cluster_addr = "https://vault-eu-west.internal:8201"
  
  # Performance replication for local reads
  performance_mode = true
  
  # Filter paths for GDPR compliance
  paths_filter = [
    "secret/data/pii/+",
    "identity/+"
  ]
}

# Configure local mounts in secondary
resource "vault_mount" "eu_local" {
  path = "eu-secrets"
  type = "kv-v2"
  
  options = {
    version = "2"
  }
  
  # Only exists in EU region
  local = true
}
```

---

## 4. Boundary: Zero Trust Session Brokering

Boundary eliminates the need for VPNs and bastion hosts by providing identity-based access to infrastructure. Combined with Vault, it creates a powerful zero-trust access platform.

### ✅ DO: Design Target Topology for Least Privilege

Structure Boundary resources to enforce principle of least privilege:

```hcl
# boundary/terraform/organization.tf
resource "boundary_scope" "global" {
  global_scope = true
  name         = "global"
  description  = "Global scope"
}

resource "boundary_scope" "org" {
  name                     = "platform-engineering"
  description              = "Platform engineering organization"
  scope_id                 = boundary_scope.global.id
  auto_create_admin_role   = false
  auto_create_default_role = false
}

# Separate projects by environment and purpose
resource "boundary_scope" "production" {
  name        = "production"
  description = "Production infrastructure"
  scope_id    = boundary_scope.org.id
}

resource "boundary_scope" "staging" {
  name        = "staging"
  description = "Staging infrastructure"
  scope_id    = boundary_scope.org.id
}

# Host catalogs with dynamic host discovery
resource "boundary_host_catalog_plugin" "aws_production" {
  name        = "aws-production-catalog"
  description = "AWS EC2 instances in production"
  scope_id    = boundary_scope.production.id
  plugin_name = "aws"
  
  attributes_json = jsonencode({
    region                    = "us-east-1"
    disable_credential_rotation = false
  })
  
  # Dynamic host discovery via tags
  secrets_json = jsonencode({
    access_key_id     = var.aws_access_key
    secret_access_key = var.aws_secret_key
  })
}

resource "boundary_host_set_plugin" "k8s_nodes" {
  name            = "k8s-production-nodes"
  host_catalog_id = boundary_host_catalog_plugin.aws_production.id
  
  # Automatically discover K8s nodes
  attributes_json = jsonencode({
    filters = [
      "tag:kubernetes.io/cluster/production=owned",
      "tag:kubernetes.io/role/node=1"
    ]
  })
}
```

### ✅ DO: Integrate Vault for Dynamic Credential Injection

Boundary's credential brokering eliminates credential exposure:

```hcl
# boundary/terraform/credentials.tf
resource "boundary_credential_store_vault" "production" {
  name        = "vault-production"
  description = "Production Vault credential store"
  scope_id    = boundary_scope.production.id
  address     = "https://vault.internal:8200"
  
  # Boundary authenticates to Vault using K8s service account
  token = vault_token.boundary_token.client_token
  
  # TLS verification
  tls_server_name = "vault.internal"
  tls_ca_cert     = file("${path.module}/ca.crt")
}

# SSH credential library for dynamic SSH keys
resource "boundary_credential_library_vault_ssh_certificate" "k8s_nodes" {
  name                = "k8s-nodes-ssh"
  description         = "Dynamic SSH certificates for K8s nodes"
  credential_store_id = boundary_credential_store_vault.production.id
  path                = "ssh/sign/k8s-nodes"
  username            = "boundary-session"
  
  # Key signing parameters
  additional_valid_principals = ["ubuntu", "ec2-user"]
  extensions = {
    "permit-pty" = ""
    "permit-user-rc" = ""
  }
  
  critical_options = {
    "force-command" = "/usr/bin/boundary-session-recorder"
  }
}

# Database credential library
resource "boundary_credential_library_vault" "postgres" {
  name                = "postgres-dynamic"
  description         = "Dynamic PostgreSQL credentials"
  credential_store_id = boundary_credential_store_vault.production.id
  path                = "database/creds/app-production"
  
  # Ensure fresh credentials for each session
  http_method = "POST"
}
```

### ✅ DO: Implement Session Recording and Audit

Enable comprehensive session recording for compliance:

```hcl
# boundary/terraform/session-recording.tf
resource "boundary_storage_bucket" "session_recordings" {
  name        = "session-recordings"
  description = "S3 bucket for session recordings"
  scope_id    = boundary_scope.global.id
  bucket_name = "boundary-session-recordings-${var.environment}"
  region      = "us-east-1"
  
  # Encryption at rest
  attributes_json = jsonencode({
    kms_key_id = aws_kms_key.boundary.id
  })
  
  # Worker filter for recording
  worker_filter = "\"recording\" in \"/tags\""
}

resource "boundary_target" "k8s_nodes_recorded" {
  name         = "k8s-nodes-recorded"
  description  = "Kubernetes nodes with session recording"
  type         = "ssh"
  scope_id     = boundary_scope.production.id
  
  session_max_seconds     = 28800  # 8 hours
  session_connection_limit = 1      # One connection per session
  
  # Enable recording
  enable_session_recording = true
  storage_bucket_id        = boundary_storage_bucket.session_recordings.id
  
  # Inject SSH credentials
  brokered_credential_source_ids = [
    boundary_credential_library_vault_ssh_certificate.k8s_nodes.id
  ]
  
  host_source_ids = [
    boundary_host_set_plugin.k8s_nodes.id
  ]
}
```

---

## 5. Integration Patterns: OPA + Vault + Boundary

The real power comes from integrating these tools to create a comprehensive security platform.

### ✅ DO: Use OPA to Enforce Vault Policy Compliance

Validate Vault policies before deployment using OPA:

```rego
# opa/policies/vault/policy_validation.rego
package vault.policy.validation

import future.keywords.contains
import future.keywords.if
import future.keywords.in

# Ensure no wildcard permissions in production
violation[msg] {
    input.path[_].capabilities[_] == "*"
    contains(input.metadata.name, "production")
    msg := sprintf("Policy %s uses wildcard capabilities in production", [input.metadata.name])
}

# Enforce MFA for sensitive paths
violation[msg] {
    path := input.path[_]
    is_sensitive_path(path.path)
    not path.required_parameters
    msg := sprintf("Sensitive path %s missing MFA requirement", [path.path])
}

is_sensitive_path(path) if {
    prefixes := ["auth/", "sys/", "pki/issue/"]
    some prefix in prefixes
    startswith(path, prefix)
}
```

### ✅ DO: Create Boundary Targets Dynamically from Kubernetes

Automatically create Boundary targets for Kubernetes services:

```go
// integration/boundary-k8s-operator/controller.go
package main

import (
    "context"
    "fmt"
    boundaryapi "github.com/hashicorp/boundary/api"
    corev1 "k8s.io/api/core/v1"
    "sigs.k8s.io/controller-runtime/pkg/client"
    "sigs.k8s.io/controller-runtime/pkg/reconcile"
)

type ServiceReconciler struct {
    client.Client
    BoundaryClient *boundaryapi.Client
    ProjectID      string
}

func (r *ServiceReconciler) Reconcile(ctx context.Context, req reconcile.Request) (reconcile.Result, error) {
    var service corev1.Service
    if err := r.Get(ctx, req.NamespacedName, &service); err != nil {
        return reconcile.Result{}, client.IgnoreNotFound(err)
    }
    
    // Check if service should be exposed via Boundary
    if _, ok := service.Annotations["boundary.io/expose"]; !ok {
        return reconcile.Result{}, nil
    }
    
    // Create host catalog entry
    host := &boundaryapi.Host{
        Name:        fmt.Sprintf("%s-%s", service.Namespace, service.Name),
        Description: fmt.Sprintf("Kubernetes service %s/%s", service.Namespace, service.Name),
        Address:     fmt.Sprintf("%s.%s.svc.cluster.local", service.Name, service.Namespace),
    }
    
    // Create target with appropriate credential injection
    target := &boundaryapi.Target{
        Name:        host.Name,
        Description: host.Description,
        Type:        service.Annotations["boundary.io/type"], // tcp, ssh, etc.
        ScopeId:     r.ProjectID,
        HostSources: []string{host.Id},
    }
    
    // Add credential sources based on annotations
    if credLib := service.Annotations["boundary.io/credential-library"]; credLib != "" {
        target.BrokeredCredentialSourceIds = append(
            target.BrokeredCredentialSourceIds, 
            credLib,
        )
    }
    
    _, err := r.BoundaryClient.Create(ctx, target)
    return reconcile.Result{}, err
}
```

### ✅ DO: Implement Policy Decision Logging

Create a unified audit trail across all policy decisions:

```yaml
# integration/opa-vault/policy-decision-logger.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: opa-vault-integration
data:
  decision-log.rego: |
    package system.log
    
    # Log all policy decisions to Vault audit
    mask["/input/request/object/metadata/annotations"] {
      # Mask sensitive annotations
      true
    }
    
    # Enrich decisions with Vault identity
    patch[{"op": "add", "path": "/vault_identity", "value": vault_id}] {
      vault_id := {
        "entity_id": input.vault_token.entity_id,
        "policies": input.vault_token.policies,
        "metadata": input.vault_token.metadata
      }
    }
```

---

## 6. Security Patterns and Anti-Patterns

### ❌ DON'T: Store Secrets in ConfigMaps or Environment Variables

This remains one of the most common and dangerous anti-patterns.

```yaml
# Bad - Secrets in plain text
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  database_password: "super-secret-password"  # NEVER DO THIS
```

### ✅ DO: Use Vault Agent for Secret Injection

Implement the Vault Agent Injector pattern for zero-trust secret delivery:

```yaml
# deployment/app-with-vault.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  template:
    metadata:
      annotations:
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/role: "app"
        vault.hashicorp.com/agent-inject-secret-db: "database/creds/app"
        vault.hashicorp.com/agent-inject-template-db: |
          {{ with secret "database/creds/app" -}}
          export DB_USERNAME="{{ .Data.username }}"
          export DB_PASSWORD="{{ .Data.password }}"
          {{- end }}
    spec:
      serviceAccountName: app
      containers:
      - name: app
        image: app:latest
        command: ["/bin/sh"]
        args: ["-c", "source /vault/secrets/db && exec app"]
```

### ✅ DO: Implement Defense in Depth

Layer multiple policy enforcement points:

```mermaid
graph TD
    A[Developer] -->|Git Push| B[CI/CD Pipeline]
    B -->|Policy Tests| C{OPA Validation}
    C -->|Pass| D[Container Registry]
    C -->|Fail| E[Rejected]
    D -->|Deploy| F[K8s API]
    F -->|Admission| G{OPA Gatekeeper}
    G -->|Allow| H[Pod Created]
    G -->|Deny| I[Rejected]
    H -->|Runtime| J{Falco Rules}
    J -->|Violation| K[Alert/Block]
    H -->|Secret Access| L{Vault Policy}
    L -->|Authorized| M[Secret Retrieved]
    L -->|Denied| N[Access Denied]
```

---

## 7. Testing and Validation Strategies

### ✅ DO: Implement Comprehensive Policy Testing

Create a robust testing framework for all policies:

```go
// opa/tests/admission_test.go
package admission_test

import (
    "testing"
    "github.com/open-policy-agent/opa/rego"
)

func TestPodSecurityPolicy(t *testing.T) {
    tests := []struct {
        name      string
        input     string
        expected  bool
        violations int
    }{
        {
            name: "privileged_container_rejected",
            input: `{
                "review": {
                    "kind": {"kind": "Pod"},
                    "namespace": "default",
                    "object": {
                        "metadata": {"name": "test-pod"},
                        "spec": {
                            "containers": [{
                                "name": "test",
                                "image": "nginx",
                                "securityContext": {"privileged": true}
                            }]
                        }
                    }
                }
            }`,
            expected: false,
            violations: 1,
        },
        {
            name: "compliant_pod_accepted",
            input: `{
                "review": {
                    "kind": {"kind": "Pod"},
                    "namespace": "default",
                    "object": {
                        "metadata": {"name": "test-pod"},
                        "spec": {
                            "containers": [{
                                "name": "test",
                                "image": "registry.internal/nginx:1.21",
                                "securityContext": {
                                    "runAsNonRoot": true,
                                    "runAsUser": 1000,
                                    "allowPrivilegeEscalation": false,
                                    "readOnlyRootFilesystem": true,
                                    "capabilities": {"drop": ["ALL"]}
                                }
                            }]
                        }
                    }
                }
            }`,
            expected: true,
            violations: 0,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            r := rego.New(
                rego.Query("data.kubernetes.admission.violation[_]"),
                rego.Load([]string{"../policies/admission/"}, nil),
                rego.Input(tt.input),
            )
            
            rs, err := r.Eval(context.Background())
            if err != nil {
                t.Fatalf("policy evaluation failed: %v", err)
            }
            
            violations := len(rs[0].Expressions[0].Value.([]interface{}))
            if violations != tt.violations {
                t.Errorf("expected %d violations, got %d", tt.violations, violations)
            }
        })
    }
}
```

### ✅ DO: Chaos Test Your Policy Infrastructure

Ensure policies remain effective under failure conditions:

```yaml
# tests/chaos/policy-resilience.yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: opa-failure-simulation
spec:
  action: pod-kill
  mode: fixed
  selector:
    namespaces:
      - gatekeeper-system
    labelSelectors:
      app: gatekeeper
  fixed:
    - gatekeeper-controller-0
  duration: "60s"
  scheduler:
    cron: "@every 2h"
---
# Verify policies still enforce during OPA outages
apiVersion: v1
kind: Pod
metadata:
  name: chaos-test-pod
  annotations:
    test.io/expected-result: "should-fail-even-during-chaos"
spec:
  containers:
  - name: test
    image: nginx
    securityContext:
      privileged: true  # Should always be rejected
```

---

## 8. Production Operations and Monitoring

### ✅ DO: Implement Comprehensive Observability

Monitor policy decisions, performance, and security events:

```yaml
# monitoring/grafana-dashboard.json
{
  "dashboard": {
    "title": "Policy-as-Code Operations",
    "panels": [
      {
        "title": "OPA Decision Rate",
        "targets": [{
          "expr": "rate(opa_decisions_total[5m])",
          "legendFormat": "{{decision}}"
        }]
      },
      {
        "title": "Vault Token Creation Rate",
        "targets": [{
          "expr": "rate(vault_token_creation_total[5m])",
          "legendFormat": "{{auth_method}}"
        }]
      },
      {
        "title": "Boundary Session Duration",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(boundary_session_duration_seconds_bucket[5m]))",
          "legendFormat": "p95 session duration"
        }]
      },
      {
        "title": "Policy Violations by Namespace",
        "targets": [{
          "expr": "sum by (namespace, policy) (rate(gatekeeper_violation_total[5m]))",
          "legendFormat": "{{namespace}}/{{policy}}"
        }]
      }
    ]
  }
}
```

### ✅ DO: Automate Policy Drift Detection

Continuously validate that deployed policies match intended state:

```go
// monitoring/drift-detector/main.go
package main

import (
    "context"
    "crypto/sha256"
    "fmt"
)

func detectPolicyDrift(ctx context.Context) error {
    // Compare deployed Gatekeeper constraints with Git
    deployedConstraints, err := k8sClient.List(ctx, &v1beta1.ConstraintList{})
    if err != nil {
        return err
    }
    
    for _, constraint := range deployedConstraints.Items {
        gitHash := getGitHash(constraint.Name)
        deployedHash := hashConstraint(constraint)
        
        if gitHash != deployedHash {
            alert := Alert{
                Severity: "warning",
                Message:  fmt.Sprintf("Policy drift detected for %s", constraint.Name),
                Details: map[string]string{
                    "git_hash":      gitHash,
                    "deployed_hash": deployedHash,
                    "namespace":     constraint.Namespace,
                },
            }
            alertManager.Send(alert)
        }
    }
    
    return nil
}
```

---

## 9. Advanced Production Patterns

### Multi-Cluster Policy Federation

Manage policies across multiple Kubernetes clusters:

```yaml
# federation/fleet-policy.yaml
apiVersion: fleet.cattle.io/v1alpha1
kind: GitRepo
metadata:
  name: policy-federation
  namespace: fleet-default
spec:
  repo: https://git.internal/platform-policies
  branch: main
  paths:
  - opa/gatekeeper/base
  targets:
  - name: production
    clusterSelector:
      matchLabels:
        env: production
    helm:
      valuesFiles:
      - opa/gatekeeper/overlays/production/values.yaml
  - name: staging
    clusterSelector:
      matchLabels:
        env: staging
    helm:
      valuesFiles:
      - opa/gatekeeper/overlays/staging/values.yaml
```

### Policy as Code for Multi-Cloud

Extend policies across cloud providers:

```rego
# opa/policies/multi-cloud/resource-tags.rego
package cloud.resources.tags

required_tags := {
    "Environment",
    "Owner",
    "CostCenter",
    "DataClassification",
    "BackupRequired"
}

# AWS EC2 Instances
violation[msg] {
    input.provider == "aws"
    input.resource_type == "aws_instance"
    missing_tags := required_tags - {tag | tag := input.resource.tags[_]}
    count(missing_tags) > 0
    msg := sprintf("AWS instance %s missing required tags: %v", 
        [input.resource.id, missing_tags])
}

# Azure VMs
violation[msg] {
    input.provider == "azure"
    input.resource_type == "azurerm_virtual_machine"
    missing_tags := required_tags - {tag | tag := input.resource.tags[_]}
    count(missing_tags) > 0
    msg := sprintf("Azure VM %s missing required tags: %v", 
        [input.resource.name, missing_tags])
}

# GCP Instances
violation[msg] {
    input.provider == "gcp"
    input.resource_type == "google_compute_instance"
    missing_labels := required_tags - {tag | tag := input.resource.labels[_]}
    count(missing_labels) > 0
    msg := sprintf("GCP instance %s missing required labels: %v", 
        [input.resource.name, missing_labels])
}
```

### Zero-Downtime Policy Updates

Implement canary deployments for policy changes:

```yaml
# deployment/policy-canary.yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: gatekeeper-policies
spec:
  targetRef:
    apiVersion: v1
    kind: ConfigMap
    name: gatekeeper-policies
  progressDeadlineSeconds: 120
  analysis:
    interval: 30s
    threshold: 5
    metrics:
    - name: policy-rejection-rate
      templateRef:
        name: policy-rejection-rate
      thresholdRange:
        max: 5  # Max 5% increase in rejections
    - name: opa-latency
      templateRef:
        name: opa-latency
      thresholdRange:
        max: 100  # Max 100ms p99 latency
  webhooks:
    - name: policy-validation
      url: https://policy-validator.internal/validate
      timeout: 30s
```

---

## 10. Conclusion

This guide has presented production-grade patterns for implementing policy-as-code with OPA, Vault, and Boundary. Key takeaways:

1. **Policy is Code**: Version control, test, and deploy policies like any other critical infrastructure component
2. **Defense in Depth**: Layer multiple policy enforcement points throughout your stack
3. **Zero Trust by Default**: Never trust, always verify—whether it's container admission, secret access, or session brokering
4. **Automate Everything**: From policy testing to drift detection to secret rotation
5. **Prepare for Failure**: Design policies to fail closed and test their resilience under adverse conditions

Remember: security is not a feature, it's a fundamental architectural concern that must be woven throughout your infrastructure from day one.