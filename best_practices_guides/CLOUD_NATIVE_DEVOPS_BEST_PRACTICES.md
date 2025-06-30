# The Definitive Guide to Cloud-Native DevOps (Mid-2025 Edition)

This guide synthesizes production-grade patterns for building resilient, observable, and scalable cloud-native infrastructure. It reflects the convergence of GitOps, platform engineering, and AI-assisted operations that define modern DevOps in 2025.

## Prerequisites & Core Technologies

- **Kubernetes**: 1.31+ (Gateway API v1.2 GA, SidecarContainers stable)
- **K3s**: 1.31+ for edge deployments
- **Argo CD**: 2.11+ with ApplicationSet generators
- **Terraform**: 1.10+ or OpenTofu 1.9+
- **Crossplane**: 1.17+ for control plane infrastructure
- **OpenTelemetry**: Collector 0.110+, Operator 0.115+
- **Container Runtime**: containerd 2.0+ with nydus snapshotter

## Table of Contents

1. [Container Orchestration: Kubernetes in Production](#1-container-orchestration-kubernetes-in-production)
2. [GitOps: The Single Source of Truth](#2-gitops-the-single-source-of-truth)
3. [Infrastructure as Code: Beyond Basic Terraform](#3-infrastructure-as-code-beyond-basic-terraform)
4. [Observability: OpenTelemetry-First Architecture](#4-observability-opentelemetry-first-architecture)
5. [CI/CD: Cloud-Native Pipelines](#5-cicd-cloud-native-pipelines)
6. [Security: Supply Chain to Runtime](#6-security-supply-chain-to-runtime)
7. [Cost Optimization & FinOps](#7-cost-optimization--finops)
8. [Platform Engineering Patterns](#8-platform-engineering-patterns)

---

## 1. Container Orchestration: Kubernetes in Production

### Core Cluster Architecture

**✅ DO: Use Purpose-Built Distributions**

```yaml
# cluster-config.yaml for K3s edge deployment
apiVersion: k3s.io/v1
kind: K3sConfig
metadata:
  name: edge-cluster-01
spec:
  version: v1.31.0+k3s1
  serverConfig:
    # Embedded etcd for HA without external dependency
    cluster-init: true
    disable:
      - traefik  # Replace with Gateway API
      - servicelb
    kube-proxy-arg:
      - "proxy-mode=ipvs"  # Better performance than iptables
      - "ipvs-strict-arp=true"
    kubelet-arg:
      - "max-pods=250"  # K3s default is 110
      - "eviction-hard=memory.available<100Mi,nodefs.available<10%"
    tls-san:
      - "edge-01.company.internal"
  agentConfig:
    node-label:
      - "node.company.io/type=edge"
      - "node.company.io/region=us-east"
```

**✅ DO: Implement Resource Quotas and Limits from Day One**

```yaml
# namespace-template.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: team-backend
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: compute-quota
  namespace: team-backend
spec:
  hard:
    requests.cpu: "100"
    requests.memory: 200Gi
    limits.cpu: "200"
    limits.memory: 400Gi
    persistentvolumeclaims: "10"
    services.loadbalancers: "2"
---
apiVersion: v1
kind: LimitRange
metadata:
  name: pod-limit-range
  namespace: team-backend
spec:
  limits:
  - max:
      cpu: "4"
      memory: 8Gi
    min:
      cpu: 10m
      memory: 64Mi
    default:
      cpu: 100m
      memory: 128Mi
    defaultRequest:
      cpu: 50m
      memory: 64Mi
    type: Pod
  - default:
      cpu: 100m
      memory: 128Mi
    defaultRequest:
      cpu: 50m
      memory: 64Mi
    type: Container
```

### Advanced Scheduling with Descheduler

**✅ DO: Enable Continuous Optimization**

```yaml
# descheduler-policy.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: descheduler-policy
  namespace: kube-system
data:
  policy.yaml: |
    apiVersion: "descheduler/v1alpha2"
    kind: "DeschedulerPolicy"
    profiles:
    - name: default
      pluginConfigs:
      - name: DefaultEvictor
        args:
          evictFailedBarePods: true
          evictLocalStoragePods: true
          nodeFit: true
      - name: RemoveDuplicates
        args:
          namespaces:
            exclude: ["kube-system", "kube-public"]
      - name: LowNodeUtilization
        args:
          thresholds:
            cpu: 20
            memory: 20
            pods: 20
          targetThresholds:
            cpu: 50
            memory: 50
            pods: 50
      - name: RemovePodsHavingTooManyRestarts
        args:
          podRestartThreshold: 10
          includingInitContainers: true
      plugins:
        balance:
          enabled:
          - RemoveDuplicates
          - LowNodeUtilization
        deschedule:
          enabled:
          - RemovePodsHavingTooManyRestarts
```

### Gateway API for Modern Ingress

**✅ DO: Migrate from Ingress to Gateway API**

```yaml
# gateway-class.yaml
apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: cilium
spec:
  controllerName: io.cilium/gateway-controller
  parametersRef:
    group: cilium.io
    kind: CiliumGatewayConfiguration
    name: gateway-config
---
# gateway.yaml
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: api-gateway
  namespace: gateway-infra
spec:
  gatewayClassName: cilium
  listeners:
  - name: https
    protocol: HTTPS
    port: 443
    tls:
      mode: Terminate
      certificateRefs:
      - name: api-tls-cert
        namespace: gateway-infra
    allowedRoutes:
      namespaces:
        from: Selector
        selector:
          matchLabels:
            gateway-access: "true"
  - name: grpc
    protocol: GRPC
    port: 8443
    tls:
      mode: Terminate
      certificateRefs:
      - name: grpc-tls-cert
---
# httproute.yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: backend-api
  namespace: team-backend
spec:
  parentRefs:
  - name: api-gateway
    namespace: gateway-infra
  hostnames:
  - "api.company.com"
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /v1/users
    backendRefs:
    - name: user-service
      port: 8080
      weight: 100
    filters:
    - type: RequestHeaderModifier
      requestHeaderModifier:
        add:
        - name: X-Gateway-Route
          value: users-v1
    timeouts:
      request: 30s
      backendRequest: 25s
```

### Kubernetes 1.31 SidecarContainers

**✅ DO: Use Native Sidecar Support for Service Mesh**

```yaml
# pod-with-sidecar.yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-telemetry
spec:
  initContainers:
  - name: otel-agent
    image: otel/opentelemetry-collector:0.110.0
    restartPolicy: Always  # This makes it a sidecar in 1.31+
    resources:
      limits:
        memory: 512Mi
        cpu: 200m
    volumeMounts:
    - name: otel-config
      mountPath: /etc/otel
  containers:
  - name: main-app
    image: company/app:v1.2.3
    env:
    - name: OTEL_EXPORTER_OTLP_ENDPOINT
      value: "http://localhost:4317"
    resources:
      requests:
        memory: 1Gi
        cpu: 500m
      limits:
        memory: 2Gi
        cpu: 1000m
```

---

## 2. GitOps: The Single Source of Truth

### Argo CD 2.11 Advanced Patterns

**✅ DO: Structure Repositories for Scale**

```
# Recommended GitOps repository structure
gitops-monorepo/
├── clusters/
│   ├── production/
│   │   ├── us-east-1/
│   │   │   ├── platform/      # Cluster-wide resources
│   │   │   ├── tenants/       # Namespace provisioning
│   │   │   └── policies/      # OPA/Kyverno policies
│   │   └── eu-west-1/
│   └── staging/
├── apps/
│   ├── base/                  # Kustomize bases
│   └── overlays/              # Environment-specific
├── infrastructure/
│   ├── terraform/             # Cloud resources
│   └── crossplane/            # Crossplane compositions
└── applicationsets/           # Argo CD ApplicationSets
```

**✅ DO: Use ApplicationSets for Multi-Tenancy**

```yaml
# applicationset-multi-tenant.yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: tenant-apps
  namespace: argocd
spec:
  goTemplate: true
  goTemplateOptions: ["missingkey=error"]
  generators:
  - matrix:
      generators:
      - git:
          repoURL: https://github.com/company/gitops
          revision: main
          directories:
          - path: "apps/tenants/*"
      - list:
          elements:
          - cluster: prod-us-east-1
            url: https://k8s-prod-use1.internal
          - cluster: prod-eu-west-1
            url: https://k8s-prod-euw1.internal
  template:
    metadata:
      name: '{{.path.basename}}-{{.cluster}}'
      labels:
        tenant: '{{.path.basename}}'
        cluster: '{{.cluster}}'
    spec:
      project: tenant-{{.path.basename}}
      source:
        repoURL: https://github.com/company/gitops
        targetRevision: main
        path: '{{.path.path}}'
        helm:
          valueFiles:
          - values.yaml
          - values-{{.cluster}}.yaml
      destination:
        server: '{{.url}}'
        namespace: '{{.path.basename}}'
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
        retry:
          limit: 5
          backoff:
            duration: 5s
            factor: 2
            maxDuration: 3m
        syncOptions:
        - CreateNamespace=true
        - PrunePropagationPolicy=foreground
        - ServerSideApply=true  # For large objects
      revisionHistoryLimit: 10
```

### Progressive Delivery with Argo Rollouts

**✅ DO: Implement Canary Deployments**

```yaml
# rollout-canary.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: backend-api
spec:
  replicas: 10
  strategy:
    canary:
      canaryService: backend-api-canary
      stableService: backend-api-stable
      trafficRouting:
        plugins:
          argoproj-labs/gatewayAPI:
            httpRoutes:
            - name: backend-api-route
              namespace: production
      steps:
      - setWeight: 20
      - pause: {duration: 2m}
      - analysis:
          templates:
          - templateName: error-rate
          args:
          - name: service-name
            value: backend-api
      - setWeight: 40
      - pause: {duration: 2m}
      - analysis:
          templates:
          - templateName: latency-p99
          - templateName: error-rate
      - setWeight: 60
      - pause: {duration: 3m}
      - setWeight: 80
      - pause: {duration: 5m}
      - setWeight: 100
      analysis:
        successfulRunHistoryLimit: 3
        unsuccessfulRunHistoryLimit: 3
  selector:
    matchLabels:
      app: backend-api
  template:
    metadata:
      labels:
        app: backend-api
    spec:
      containers:
      - name: backend
        image: company/backend-api:v2.0.0
        ports:
        - containerPort: 8080
        env:
        - name: OTEL_SERVICE_NAME
          value: backend-api
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
---
# analysis-template.yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: error-rate
spec:
  args:
  - name: service-name
  metrics:
  - name: error-rate
    interval: 1m
    successCondition: result[0] <= 0.01
    failureLimit: 3
    provider:
      prometheus:
        address: http://prometheus.monitoring:9090
        query: |
          sum(rate(http_requests_total{job="{{args.service-name}}",status=~"5.."}[5m]))
          /
          sum(rate(http_requests_total{job="{{args.service-name}}"}[5m]))
```

### Flux CD for Decentralized GitOps

**✅ DO: Use Flux for Platform Teams Preferring Pull-Based GitOps**

```yaml
# flux-system/gotk-components.yaml
apiVersion: source.toolkit.fluxcd.io/v1
kind: GitRepository
metadata:
  name: flux-system
  namespace: flux-system
spec:
  interval: 1m
  ref:
    branch: main
  secretRef:
    name: flux-system
  url: ssh://git@github.com/company/fleet-infra
---
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: infrastructure
  namespace: flux-system
spec:
  interval: 10m
  path: ./infrastructure
  prune: true
  sourceRef:
    kind: GitRepository
    name: flux-system
  healthChecks:
  - apiVersion: apps/v1
    kind: Deployment
    name: cert-manager
    namespace: cert-manager
  timeout: 5m
  wait: true
  postBuild:
    substituteFrom:
    - kind: ConfigMap
      name: cluster-vars
    - kind: Secret
      name: cluster-secrets
---
# Multi-cluster with Flux
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: clusters
  namespace: flux-system
spec:
  interval: 10m
  path: ./clusters/production
  prune: true
  sourceRef:
    kind: GitRepository
    name: flux-system
  decryption:
    provider: sops
    secretRef:
      name: sops-age
  patches:
  - patch: |
      - op: replace
        path: /spec/values/cluster/name
        value: prod-us-east-1
    target:
      kind: HelmRelease
      name: "*"
```

---

## 3. Infrastructure as Code: Beyond Basic Terraform

### Terraform 1.10 / OpenTofu Advanced Patterns

**✅ DO: Implement Proper Module Structure**

```hcl
# modules/kubernetes-cluster/main.tf
terraform {
  required_version = ">= 1.10"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.35"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.16"
    }
  }
}

# Dynamic provider configuration
provider "kubernetes" {
  host                   = var.cluster_endpoint
  cluster_ca_certificate = base64decode(var.cluster_ca_cert)
  
  # Use exec for cloud provider auth
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", var.cluster_name]
  }
}

# Import existing resources with for_each
import {
  for_each = var.import_namespaces
  to       = kubernetes_namespace.imported[each.key]
  id       = each.value
}

resource "kubernetes_namespace" "imported" {
  for_each = var.import_namespaces
  
  metadata {
    name = each.value
    labels = {
      "managed-by"     = "terraform"
      "imported"       = "true"
      "import-date"    = formatdate("YYYY-MM-DD", timestamp())
    }
  }
}

# Terraform 1.10 Stacks configuration
variable "cluster_configs" {
  type = map(object({
    region       = string
    node_count   = number
    node_type    = string
    k8s_version  = string
  }))
}

# Enhanced state management with S3
terraform {
  backend "s3" {
    bucket         = "company-terraform-state"
    key            = "kubernetes/clusters/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    kms_key_id     = "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
    dynamodb_table = "terraform-state-lock"
    
    # Terraform 1.10 feature: workspace key prefix
    workspace_key_prefix = "clusters"
  }
}
```

### Pulumi for Type-Safe Infrastructure

**✅ DO: Use Pulumi for Complex Logic**

```typescript
// infrastructure/pulumi/index.ts
import * as pulumi from "@pulumi/pulumi";
import * as kubernetes from "@pulumi/kubernetes";
import * as aws from "@pulumi/aws";
import * as docker from "@pulumi/docker";

// Configuration with type safety
const config = new pulumi.Config();
const environment = config.require("environment");
const isProduction = environment === "production";

// Component resource for a complete microservice
export class MicroserviceStack extends pulumi.ComponentResource {
  public readonly endpoint: pulumi.Output<string>;
  
  constructor(name: string, args: MicroserviceArgs, opts?: pulumi.ComponentResourceOptions) {
    super("company:infrastructure:MicroserviceStack", name, {}, opts);
    
    // Build and push container image
    const image = new docker.Image(`${name}-image`, {
      imageName: pulumi.interpolate`${args.registry}/${name}:${args.version}`,
      build: {
        context: args.buildContext,
        dockerfile: `${args.buildContext}/Dockerfile`,
        platform: "linux/amd64",
        args: {
          BUILD_VERSION: args.version,
        },
        cacheFrom: {
          images: [pulumi.interpolate`${args.registry}/${name}:cache`],
        },
      },
      registry: {
        server: args.registry,
        username: args.registryUsername,
        password: args.registryPassword,
      },
    }, { parent: this });
    
    // Create Kubernetes deployment with automatic rollback
    const deployment = new kubernetes.apps.v1.Deployment(`${name}-deployment`, {
      metadata: {
        namespace: args.namespace,
        labels: {
          app: name,
          version: args.version,
          "app.kubernetes.io/managed-by": "pulumi",
        },
      },
      spec: {
        replicas: isProduction ? 3 : 1,
        revisionHistoryLimit: 5,
        progressDeadlineSeconds: 600,
        strategy: {
          type: "RollingUpdate",
          rollingUpdate: {
            maxSurge: 1,
            maxUnavailable: 0,
          },
        },
        selector: {
          matchLabels: { app: name },
        },
        template: {
          metadata: {
            labels: { 
              app: name,
              version: args.version,
            },
          },
          spec: {
            topologySpreadConstraints: isProduction ? [{
              maxSkew: 1,
              topologyKey: "topology.kubernetes.io/zone",
              whenUnsatisfiable: "DoNotSchedule",
              labelSelector: {
                matchLabels: { app: name },
              },
            }] : undefined,
            containers: [{
              name: "main",
              image: image.imageName,
              ports: [{ containerPort: args.port }],
              resources: {
                requests: {
                  memory: "128Mi",
                  cpu: "100m",
                },
                limits: {
                  memory: isProduction ? "512Mi" : "256Mi",
                  cpu: isProduction ? "500m" : "200m",
                },
              },
              livenessProbe: {
                httpGet: {
                  path: "/health/live",
                  port: args.port,
                },
                initialDelaySeconds: 30,
                periodSeconds: 10,
              },
              readinessProbe: {
                httpGet: {
                  path: "/health/ready",
                  port: args.port,
                },
                initialDelaySeconds: 10,
                periodSeconds: 5,
              },
              env: [
                {
                  name: "ENVIRONMENT",
                  value: environment,
                },
                {
                  name: "OTEL_SERVICE_NAME",
                  value: name,
                },
                {
                  name: "OTEL_EXPORTER_OTLP_ENDPOINT",
                  value: "http://otel-collector.observability:4317",
                },
              ],
            }],
          },
        },
      },
    }, { parent: this });
    
    // Create service
    const service = new kubernetes.core.v1.Service(`${name}-service`, {
      metadata: {
        namespace: args.namespace,
        labels: { app: name },
      },
      spec: {
        selector: { app: name },
        ports: [{
          port: 80,
          targetPort: args.port,
        }],
      },
    }, { parent: this });
    
    // Create HTTPRoute for Gateway API
    const route = new kubernetes.apiextensions.CustomResource(`${name}-route`, {
      apiVersion: "gateway.networking.k8s.io/v1",
      kind: "HTTPRoute",
      metadata: {
        namespace: args.namespace,
        labels: { app: name },
      },
      spec: {
        parentRefs: [{
          name: "api-gateway",
          namespace: "gateway-infra",
        }],
        hostnames: [args.hostname],
        rules: [{
          matches: [{
            path: {
              type: "PathPrefix",
              value: args.pathPrefix || "/",
            },
          }],
          backendRefs: [{
            name: service.metadata.name,
            port: 80,
          }],
        }],
      },
    }, { parent: this });
    
    this.endpoint = pulumi.interpolate`https://${args.hostname}${args.pathPrefix || "/"}`;
    
    this.registerOutputs({
      endpoint: this.endpoint,
    });
  }
}

interface MicroserviceArgs {
  registry: string;
  registryUsername: string;
  registryPassword: pulumi.Output<string>;
  buildContext: string;
  version: string;
  namespace: string;
  port: number;
  hostname: string;
  pathPrefix?: string;
}

// Usage
const userService = new MicroserviceStack("user-service", {
  registry: "ghcr.io/company",
  registryUsername: "github-actions",
  registryPassword: pulumi.secret(process.env.GITHUB_TOKEN!),
  buildContext: "../services/user",
  version: process.env.GIT_SHA || "latest",
  namespace: "production",
  port: 8080,
  hostname: "api.company.com",
  pathPrefix: "/v1/users",
});

export const userServiceEndpoint = userService.endpoint;
```

### Crossplane for Kubernetes-Native Infrastructure

**✅ DO: Use Crossplane for Control Plane Infrastructure**

```yaml
# crossplane/composition-rds.yaml
apiVersion: apiextensions.crossplane.io/v1
kind: Composition
metadata:
  name: xpostgresql.company.io
spec:
  compositeTypeRef:
    apiVersion: company.io/v1alpha1
    kind: XPostgreSQL
  mode: Pipeline
  pipeline:
  - step: patch-and-transform
    functionRef:
      name: function-patch-and-transform
    input:
      apiVersion: pt.fn.crossplane.io/v1beta1
      kind: Resources
      resources:
      - name: rds-instance
        base:
          apiVersion: rds.aws.upbound.io/v1beta2
          kind: Instance
          spec:
            forProvider:
              region: us-east-1
              instanceClass: db.t3.micro
              allocatedStorage: 20
              engine: postgres
              engineVersion: "16.3"
              masterUsername: postgres
              autoMinorVersionUpgrade: true
              backupRetentionPeriod: 7
              backupWindow: "03:00-04:00"
              maintenanceWindow: "sun:04:00-sun:05:00"
              storageEncrypted: true
              deletionProtection: true
        patches:
        - type: FromCompositeFieldPath
          fromFieldPath: spec.parameters.region
          toFieldPath: spec.forProvider.region
        - type: FromCompositeFieldPath
          fromFieldPath: spec.parameters.instanceClass
          toFieldPath: spec.forProvider.instanceClass
        - type: FromCompositeFieldPath
          fromFieldPath: spec.parameters.storageGB
          toFieldPath: spec.forProvider.allocatedStorage
        - type: FromCompositeFieldPath
          fromFieldPath: spec.parameters.version
          toFieldPath: spec.forProvider.engineVersion
          transforms:
          - type: map
            map:
              "16": "16.3"
              "15": "15.7"
              "14": "14.12"
        - type: ToCompositeFieldPath
          fromFieldPath: status.atProvider.endpoint
          toFieldPath: status.endpoint
      - name: security-group
        base:
          apiVersion: ec2.aws.upbound.io/v1beta1
          kind: SecurityGroup
          spec:
            forProvider:
              region: us-east-1
              description: Security group for PostgreSQL
              ingress:
              - fromPort: 5432
                toPort: 5432
                protocol: tcp
                cidrBlocks:
                - 10.0.0.0/8
---
# Usage - much simpler!
apiVersion: company.io/v1alpha1
kind: PostgreSQL
metadata:
  name: app-database
  namespace: production
spec:
  parameters:
    region: us-east-1
    instanceClass: db.r6g.large
    storageGB: 100
    version: "16"
  compositionSelector:
    matchLabels:
      provider: aws
      complexity: standard
  writeConnectionSecretToRef:
    name: app-database-connection
    namespace: production
```

---

## 4. Observability: OpenTelemetry-First Architecture

### Unified Telemetry Pipeline

**✅ DO: Deploy OpenTelemetry Collector as DaemonSet + Deployment**

```yaml
# otel-collector-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
  namespace: observability
data:
  config.yaml: |
    receivers:
      # Collect metrics from Kubernetes
      kubeletstats:
        collection_interval: 30s
        auth_type: serviceAccount
        endpoint: "${env:K8S_NODE_IP}:10250"
        insecure_skip_verify: true
      k8s_cluster:
        collection_interval: 30s
      
      # OTLP for applications
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
            max_recv_msg_size_mib: 32
          http:
            endpoint: 0.0.0.0:4318
      
      # Prometheus scraping
      prometheus:
        config:
          scrape_configs:
          - job_name: 'kubernetes-pods'
            kubernetes_sd_configs:
            - role: pod
            relabel_configs:
            - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
              action: keep
              regex: true
            - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
              action: replace
              target_label: __metrics_path__
              regex: (.+)
      
      # Collect traces from AWS X-Ray SDK
      awsxray:
        endpoint: 0.0.0.0:2000
        transport: udp
      
      # Jaeger backwards compatibility
      jaeger:
        protocols:
          thrift_compact:
            endpoint: 0.0.0.0:6831
          grpc:
            endpoint: 0.0.0.0:14250
    
    processors:
      # Add Kubernetes metadata
      k8sattributes:
        extract:
          metadata:
          - k8s.namespace.name
          - k8s.deployment.name
          - k8s.statefulset.name
          - k8s.daemonset.name
          - k8s.cronjob.name
          - k8s.job.name
          - k8s.node.name
          - k8s.pod.name
          - k8s.pod.uid
          - k8s.pod.start_time
        filter:
          node_from_env_var: K8S_NODE_NAME
      
      # Resource detection for cloud metadata
      resourcedetection:
        detectors: [env, system, gcp, aws, azure]
        timeout: 5s
        override: false
      
      # Batch processing for efficiency
      batch:
        send_batch_size: 10000
        timeout: 5s
        send_batch_max_size: 20000
      
      # Memory limiter to prevent OOM
      memory_limiter:
        check_interval: 1s
        limit_percentage: 80
        spike_limit_percentage: 95
      
      # Tail sampling for traces
      tail_sampling:
        decision_wait: 10s
        num_traces: 100000
        expected_new_traces_per_sec: 10000
        policies:
        - name: errors-policy
          type: status_code
          status_code:
            status_codes: [ERROR]
        - name: slow-traces-policy
          type: latency
          latency:
            threshold_ms: 1000
        - name: probabilistic-policy
          type: probabilistic
          probabilistic:
            sampling_percentage: 10
      
      # Transform metrics
      metricstransform:
        transforms:
        - include: system.cpu.usage
          action: update
          new_name: cpu_usage_ratio
        - include: system.memory.usage
          action: update
          operations:
          - action: divide
            value: 1073741824  # Convert to GB
    
    exporters:
      # Prometheus Remote Write for metrics
      prometheusremotewrite:
        endpoint: "http://mimir.observability:9009/api/v1/push"
        headers:
          X-Scope-OrgID: "company"
        tls:
          insecure_skip_verify: false
        retry_on_failure:
          enabled: true
          initial_interval: 5s
          max_interval: 30s
          max_elapsed_time: 300s
        resource_to_telemetry_conversion:
          enabled: true
      
      # Loki for logs
      loki:
        endpoint: "http://loki-gateway.observability:80/loki/api/v1/push"
        tenant_id: "company"
        labels:
          attributes:
            k8s.namespace.name: "namespace"
            k8s.pod.name: "pod"
            k8s.container.name: "container"
          resource:
            service.name: "service"
            service.namespace: "service_namespace"
      
      # Tempo for traces
      otlp/tempo:
        endpoint: "tempo-distributor.observability:4317"
        tls:
          insecure: true
        headers:
          X-Scope-OrgID: "company"
      
      # Debug exporter for troubleshooting
      debug:
        verbosity: detailed
        sampling_initial: 5
        sampling_thereafter: 200
    
    connectors:
      # Generate metrics from spans
      spanmetrics:
        namespace: traces.spanmetrics
        histogram:
          explicit:
            buckets: [2ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2s, 5s]
        dimensions:
        - name: service.name
        - name: service.namespace
        - name: k8s.cluster.name
        - name: http.method
        - name: http.status_code
    
    service:
      pipelines:
        traces:
          receivers: [otlp, jaeger, awsxray]
          processors: [memory_limiter, k8sattributes, resourcedetection, batch, tail_sampling]
          exporters: [otlp/tempo, spanmetrics]
        
        metrics:
          receivers: [otlp, prometheus, kubeletstats, k8s_cluster, spanmetrics]
          processors: [memory_limiter, k8sattributes, resourcedetection, metricstransform, batch]
          exporters: [prometheusremotewrite]
        
        logs:
          receivers: [otlp]
          processors: [memory_limiter, k8sattributes, resourcedetection, batch]
          exporters: [loki]
      
      extensions: [health_check, pprof, zpages]
      telemetry:
        logs:
          level: info
          initial_fields:
            service: otel-collector
        metrics:
          level: detailed
          address: 0.0.0.0:8888
---
# Deploy as DaemonSet for node metrics
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: otel-collector-daemonset
  namespace: observability
spec:
  selector:
    matchLabels:
      app: otel-collector
      mode: daemonset
  template:
    metadata:
      labels:
        app: otel-collector
        mode: daemonset
    spec:
      serviceAccountName: otel-collector
      containers:
      - name: otel-collector
        image: otel/opentelemetry-collector-k8s:0.110.0
        command: ["otelcol-k8s"]
        args: ["--config=/conf/config.yaml"]
        resources:
          limits:
            memory: 512Mi
            cpu: 200m
        env:
        - name: K8S_NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        - name: K8S_NODE_IP
          valueFrom:
            fieldRef:
              fieldPath: status.hostIP
        ports:
        - containerPort: 4317  # OTLP gRPC
        - containerPort: 4318  # OTLP HTTP
        - containerPort: 8888  # Metrics
        volumeMounts:
        - name: config
          mountPath: /conf
      volumes:
      - name: config
        configMap:
          name: otel-collector-config
      hostNetwork: true
      dnsPolicy: ClusterFirstWithHostNet
---
# Deploy as Deployment for application traces/metrics
apiVersion: apps/v1
kind: Deployment
metadata:
  name: otel-collector-deployment
  namespace: observability
spec:
  replicas: 3
  selector:
    matchLabels:
      app: otel-collector
      mode: deployment
  template:
    metadata:
      labels:
        app: otel-collector
        mode: deployment
    spec:
      serviceAccountName: otel-collector
      containers:
      - name: otel-collector
        image: otel/opentelemetry-collector-k8s:0.110.0
        command: ["otelcol-k8s"]
        args: ["--config=/conf/config.yaml"]
        resources:
          limits:
            memory: 2Gi
            cpu: 1000m
          requests:
            memory: 1Gi
            cpu: 500m
        ports:
        - containerPort: 4317
        - containerPort: 4318
        - containerPort: 14250  # Jaeger gRPC
        - containerPort: 6831   # Jaeger Thrift
          protocol: UDP
        volumeMounts:
        - name: config
          mountPath: /conf
      volumes:
      - name: config
        configMap:
          name: otel-collector-config
```

### Grafana LGTM Stack Configuration

**✅ DO: Deploy Production-Grade LGTM Stack**

```yaml
# grafana-datasources.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-datasources
  namespace: observability
data:
  datasources.yaml: |
    apiVersion: 1
    datasources:
    - name: Prometheus
      uid: prometheus
      type: prometheus
      access: proxy
      url: http://mimir-query-frontend.observability:8080/prometheus
      jsonData:
        httpHeaderName1: 'X-Scope-OrgID'
      secureJsonData:
        httpHeaderValue1: 'company'
      isDefault: true
      editable: false
    
    - name: Tempo
      uid: tempo
      type: tempo
      access: proxy
      url: http://tempo-query-frontend.observability:3100
      jsonData:
        httpHeaderName1: 'X-Scope-OrgID'
        tracesToLogsV2:
          datasourceUid: loki
          spanStartTimeShift: '-1h'
          spanEndTimeShift: '1h'
          filterByTraceID: true
          filterBySpanID: true
        serviceMap:
          datasourceUid: prometheus
        search:
          hide: false
        nodeGraph:
          enabled: true
      secureJsonData:
        httpHeaderValue1: 'company'
    
    - name: Loki
      uid: loki
      type: loki
      access: proxy
      url: http://loki-query-frontend.observability:3100
      jsonData:
        httpHeaderName1: 'X-Scope-OrgID'
        derivedFields:
        - datasourceUid: tempo
          matcherRegex: '"trace_id":"([^"]+)"'
          name: TraceID
          url: '${__value.raw}'
      secureJsonData:
        httpHeaderValue1: 'company'
    
    - name: AlertManager
      uid: alertmanager
      type: alertmanager
      access: proxy
      url: http://alertmanager.observability:9093
      jsonData:
        implementation: prometheus
---
# mimir-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mimir-config
  namespace: observability
data:
  mimir.yaml: |
    multitenancy_enabled: true
    
    blocks_storage:
      backend: s3
      s3:
        bucket_name: company-metrics-blocks
        endpoint: s3.us-east-1.amazonaws.com
        region: us-east-1
      bucket_store:
        sync_dir: /data/tsdb-sync
      tsdb:
        dir: /data/tsdb
        retention_period: 30d
        block_ranges_period: [2h, 12h, 24h]
    
    compactor:
      data_dir: /data/compactor
      sharding_ring:
        kvstore:
          store: memberlist
    
    distributor:
      ring:
        instance_interface_names: [eth0]
        kvstore:
          store: memberlist
      ha_tracker:
        enable_ha_tracker: true
        kvstore:
          store: memberlist
        ha_replica_label: __replica__
    
    ingester:
      ring:
        kvstore:
          store: memberlist
        replication_factor: 3
        num_tokens: 512
      max_series: 1000000
      max_series_per_user: 100000
      max_series_per_metric: 50000
    
    limits:
      max_label_name_length: 1024
      max_label_value_length: 2048
      max_label_names_per_series: 40
      ingestion_rate: 100000
      ingestion_burst_size: 200000
      max_global_series_per_user: 1000000
      max_global_series_per_metric: 100000
      max_series_per_query: 10000
    
    memberlist:
      abort_if_cluster_join_fails: false
      bind_port: 7946
      join_members:
      - dns+mimir-gossip-ring.observability.svc.cluster.local:7946
    
    querier:
      max_concurrent: 16
      query_store_after: 12h
      query_ingesters_within: 13h
    
    query_frontend:
      parallelize_shardable_queries: true
      cache_results: true
      results_cache:
        backend: memcached
        memcached:
          addresses: dns+memcached.observability:11211
          max_item_size: 1MB
          timeout: 500ms
    
    ruler:
      rule_path: /data/rules
      alertmanager_url: http://alertmanager.observability:9093
      enable_api: true
      ring:
        kvstore:
          store: memberlist
    
    ruler_storage:
      backend: s3
      s3:
        bucket_name: company-metrics-rules
        endpoint: s3.us-east-1.amazonaws.com
        region: us-east-1
    
    runtime_config:
      file: /var/mimir/runtime.yaml
```

### Auto-Instrumentation with OpenTelemetry Operator

**✅ DO: Enable Zero-Code Instrumentation**

```yaml
# otel-instrumentation.yaml
apiVersion: opentelemetry.io/v1alpha1
kind: Instrumentation
metadata:
  name: auto-instrumentation
  namespace: observability
spec:
  # Propagators for distributed tracing
  propagators:
  - tracecontext
  - baggage
  - b3
  - xray
  
  # Resource attributes for all telemetry
  resource:
    attributes:
      service.namespace: my-namespace
      deployment.environment: production
  
  # Auto-instrumentation configs
  java:
    image: ghcr.io/open-telemetry/opentelemetry-operator/autoinstrumentation-java:2.10.0
    env:
    - name: OTEL_JAVAAGENT_LOGGING
      value: "application"
    - name: OTEL_INSTRUMENTATION_JDBC_ENABLED
      value: "true"
    - name: OTEL_INSTRUMENTATION_KAFKA_ENABLED
      value: "true"
  
  nodejs:
    image: ghcr.io/open-telemetry/opentelemetry-operator/autoinstrumentation-nodejs:0.57.0
    env:
    - name: OTEL_NODEJS_DEBUG
      value: "true"
  
  python:
    image: ghcr.io/open-telemetry/opentelemetry-operator/autoinstrumentation-python:0.51b0
    env:
    - name: OTEL_PYTHON_LOG_CORRELATION
      value: "true"
    - name: OTEL_PYTHON_EXCLUDED_URLS
      value: "/health/.*,/metrics"
  
  go:
    image: ghcr.io/open-telemetry/opentelemetry-operator/autoinstrumentation-go:0.19.0-alpha
  
  dotnet:
    image: ghcr.io/open-telemetry/opentelemetry-operator/autoinstrumentation-dotnet:1.8.0
  
  # Default endpoint configuration
  exporter:
    endpoint: http://otel-collector.observability:4317
---
# Apply to namespace for automatic injection
apiVersion: v1
kind: Namespace
metadata:
  name: production
  annotations:
    instrumentation.opentelemetry.io/inject-java: "true"
    instrumentation.opentelemetry.io/inject-nodejs: "true"
    instrumentation.opentelemetry.io/inject-python: "true"
    instrumentation.opentelemetry.io/inject-dotnet: "true"
```

---

## 5. CI/CD: Cloud-Native Pipelines

### GitHub Actions with Dagger

**✅ DO: Use Dagger for Portable CI/CD**

```yaml
# .github/workflows/deploy.yaml
name: Deploy with Dagger
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  DAGGER_VERSION: 0.10.0

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      id-token: write  # For OIDC auth
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        role-to-assume: arn:aws:iam::123456789012:role/github-actions
        aws-region: us-east-1
    
    - name: Install Dagger
      run: |
        cd /usr/local
        curl -L https://dl.dagger.io/dagger/install.sh | sh
        
    - name: Run Dagger Pipeline
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      run: |
        dagger run go run ci/main.go
```

```go
// ci/main.go - Dagger pipeline in Go
package main

import (
	"context"
	"fmt"
	"os"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	// Initialize Dagger client
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stderr))
	if err != nil {
		panic(err)
	}
	defer client.Close()

	// Get source code
	src := client.Host().Directory(".", dagger.HostDirectoryOpts{
		Exclude: []string{
			"ci/",
			".git/",
			"*.md",
		},
	})

	// Run tests in parallel
	golang := client.Container().
		From("golang:1.23-alpine").
		WithMountedDirectory("/src", src).
		WithWorkdir("/src").
		WithExec([]string{"go", "mod", "download"})

	test := golang.
		WithExec([]string{"go", "test", "-race", "-cover", "./..."})

	lint := golang.
		WithExec([]string{"go", "install", "github.com/golangci/golangci-lint/cmd/golangci-lint@latest"}).
		WithExec([]string{"golangci-lint", "run", "--timeout", "5m"})

	// Wait for tests and linting
	_, err = test.Sync(ctx)
	if err != nil {
		panic(fmt.Sprintf("tests failed: %v", err))
	}

	_, err = lint.Sync(ctx)
	if err != nil {
		panic(fmt.Sprintf("linting failed: %v", err))
	}

	// Build container image
	dockerfile := `
FROM golang:1.23-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o server ./cmd/server

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=builder /app/server /server
ENTRYPOINT ["/server"]
`

	container := client.Container().
		Build(src, dagger.ContainerBuildOpts{
			Dockerfile: dockerfile,
		})

	// Run security scanning
	trivy := client.Container().
		From("aquasec/trivy:latest").
		WithMountedDirectory("/src", src).
		WithExec([]string{
			"trivy", "fs", "--severity", "HIGH,CRITICAL",
			"--exit-code", "1", "/src",
		})

	_, err = trivy.Sync(ctx)
	if err != nil {
		panic(fmt.Sprintf("security scan failed: %v", err))
	}

	// Push to registry
	gitSha := os.Getenv("GITHUB_SHA")
	if gitSha == "" {
		gitSha = "latest"
	}

	ref := fmt.Sprintf("ghcr.io/company/app:%s", gitSha)
	
	_, err = container.
		WithRegistryAuth("ghcr.io", "x-access-token", os.Getenv("GITHUB_TOKEN")).
		Publish(ctx, ref)
	
	if err != nil {
		panic(fmt.Sprintf("failed to publish image: %v", err))
	}

	fmt.Printf("Successfully built and pushed %s\n", ref)

	// Update GitOps repo
	gitopsRepo := client.Git("https://github.com/company/gitops").
		Branch("main").
		Tree()

	updatedRepo := client.Container().
		From("alpine/git:latest").
		WithMountedDirectory("/repo", gitopsRepo).
		WithWorkdir("/repo").
		WithExec([]string{
			"sed", "-i",
			fmt.Sprintf("s|image: ghcr.io/company/app:.*|image: %s|g", ref),
			"apps/production/deployment.yaml",
		}).
		Directory("/repo")

	// Export the updated repo (in real scenario, you'd commit and push)
	_, err = updatedRepo.Export(ctx, "./updated-gitops")
	if err != nil {
		panic(fmt.Sprintf("failed to update gitops: %v", err))
	}
}
```

### Tekton for Kubernetes-Native CI/CD

**✅ DO: Use Tekton for Complex Pipelines**

```yaml
# tekton-pipeline.yaml
apiVersion: tekton.dev/v1
kind: Pipeline
metadata:
  name: build-and-deploy
  namespace: tekton-pipelines
spec:
  params:
  - name: git-url
    type: string
  - name: git-revision
    type: string
    default: main
  - name: image-registry
    type: string
    default: ghcr.io/company
  
  workspaces:
  - name: shared-data
  - name: docker-credentials
  
  tasks:
  # Clone repository
  - name: fetch-source
    taskRef:
      resolver: hub
      params:
      - name: catalog
        value: tekton-catalog-tasks
      - name: type
        value: artifact
      - name: kind
        value: task
      - name: name
        value: git-clone
      - name: version
        value: "0.7"
    workspaces:
    - name: output
      workspace: shared-data
    params:
    - name: url
      value: $(params.git-url)
    - name: revision
      value: $(params.git-revision)
  
  # Run tests
  - name: run-tests
    runAfter: ["fetch-source"]
    taskRef:
      resolver: cluster
      params:
      - name: kind
        value: task
      - name: name
        value: golang-test
      - name: namespace
        value: tekton-pipelines
    workspaces:
    - name: source
      workspace: shared-data
  
  # Build with Buildah
  - name: build-image
    runAfter: ["run-tests"]
    taskRef:
      resolver: hub
      params:
      - name: catalog
        value: tekton-catalog-tasks
      - name: type
        value: artifact
      - name: kind
        value: task
      - name: name
        value: buildah
      - name: version
        value: "0.8"
    workspaces:
    - name: source
      workspace: shared-data
    - name: dockerconfig
      workspace: docker-credentials
    params:
    - name: IMAGE
      value: $(params.image-registry)/$(context.pipelineRun.name):$(tasks.fetch-source.results.commit)
    - name: DOCKERFILE
      value: ./Dockerfile
    - name: BUILD_EXTRA_ARGS
      value: "--cache-from=$(params.image-registry)/$(context.pipelineRun.name):cache"
  
  # Security scan
  - name: security-scan
    runAfter: ["build-image"]
    taskSpec:
      params:
      - name: image
        type: string
      steps:
      - name: scan
        image: aquasec/trivy:latest
        command:
        - trivy
        - image
        - --severity
        - HIGH,CRITICAL
        - --exit-code
        - "0"  # Don't fail pipeline, just report
        - --format
        - json
        - --output
        - /workspace/scan-results.json
        - $(params.image)
      - name: upload-results
        image: amazon/aws-cli:latest
        command:
        - aws
        - s3
        - cp
        - /workspace/scan-results.json
        - s3://security-scans/$(context.pipelineRun.name).json
    params:
    - name: image
      value: $(tasks.build-image.results.IMAGE_URL)
  
  # Deploy with ArgoCD
  - name: update-gitops
    runAfter: ["security-scan"]
    taskSpec:
      params:
      - name: image
        type: string
      - name: deployment-repo
        type: string
        default: https://github.com/company/gitops
      steps:
      - name: update-manifest
        image: alpine/git:latest
        script: |
          #!/bin/sh
          git clone $(params.deployment-repo) /workspace/gitops
          cd /workspace/gitops
          
          # Update image tag
          sed -i "s|image: .*|image: $(params.image)|g" apps/production/deployment.yaml
          
          # Commit and push
          git config user.email "tekton@company.com"
          git config user.name "Tekton Pipeline"
          git add .
          git commit -m "Update image to $(params.image)"
          git push
    params:
    - name: image
      value: $(tasks.build-image.results.IMAGE_URL)
  
  finally:
  # Cleanup
  - name: cleanup
    taskSpec:
      steps:
      - name: cleanup-workspace
        image: alpine
        command: ["rm", "-rf", "/workspace/*"]
    workspaces:
    - name: source
      workspace: shared-data
---
# PipelineRun to execute
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: build-and-deploy-run-
  namespace: tekton-pipelines
spec:
  pipelineRef:
    name: build-and-deploy
  params:
  - name: git-url
    value: https://github.com/company/app
  - name: git-revision
    value: main
  workspaces:
  - name: shared-data
    volumeClaimTemplate:
      spec:
        accessModes:
        - ReadWriteOnce
        resources:
          requests:
            storage: 1Gi
  - name: docker-credentials
    secret:
      secretName: docker-credentials
  podTemplate:
    securityContext:
      fsGroup: 65532
```

---

## 6. Security: Supply Chain to Runtime

### SLSA Level 3 Compliance

**✅ DO: Implement Supply Chain Security**

```yaml
# .github/workflows/slsa-build.yaml
name: SLSA Level 3 Build
on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    permissions:
      id-token: write
      contents: read
      actions: read
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@v2.0.0
    with:
      image: ghcr.io/${{ github.repository }}
      digest: ${{ github.sha }}
    secrets:
      registry-username: ${{ github.actor }}
      registry-password: ${{ secrets.GITHUB_TOKEN }}

  verify-and-sign:
    needs: [build]
    runs-on: ubuntu-latest
    permissions:
      packages: write
      id-token: write
    steps:
    - name: Install Cosign
      uses: sigstore/cosign-installer@v3.7.0
      
    - name: Sign container image
      env:
        DIGEST: ${{ needs.build.outputs.image-digest }}
      run: |
        cosign sign --yes \
          -a "repo=${{ github.repository }}" \
          -a "workflow=${{ github.workflow }}" \
          -a "ref=${{ github.sha }}" \
          ghcr.io/${{ github.repository }}@${DIGEST}
    
    - name: Verify SLSA Provenance
      env:
        IMAGE: ghcr.io/${{ github.repository }}@${{ needs.build.outputs.image-digest }}
      run: |
        cosign verify-attestation \
          --type slsaprovenance \
          --certificate-identity-regexp "^https://github.com/slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@refs/tags/v[0-9]+.[0-9]+.[0-9]+$" \
          --certificate-oidc-issuer https://token.actions.githubusercontent.com \
          "${IMAGE}"
```

### Runtime Security with Falco

**✅ DO: Deploy Falco for Runtime Threat Detection**

```yaml
# falco-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: falco-config
  namespace: falco
data:
  falco.yaml: |
    rules_file:
      - /etc/falco/falco_rules.yaml
      - /etc/falco/falco_rules.local.yaml
      - /etc/falco/k8s_audit_rules.yaml
      - /etc/falco/rules.d
    
    json_output: true
    json_include_output_property: true
    json_include_tags_property: true
    
    log_stderr: true
    log_syslog: false
    log_level: info
    
    libs_logger:
      enabled: false
      severity: debug
    
    outputs:
      rate: 1
      max_burst: 1000
    
    stdout_output:
      enabled: false
    
    syslog_output:
      enabled: false
    
    file_output:
      enabled: false
    
    http_output:
      enabled: true
      url: "http://falco-exporter.falco:2801/"
      user_agent: "falcosecurity/falco"
    
    grpc:
      enabled: true
      bind_address: "unix:///var/run/falco/falco.sock"
      threadiness: 8
    
    grpc_output:
      enabled: true
    
    webserver:
      enabled: true
      listen_port: 8765
      k8s_healthz_endpoint: /healthz
      ssl_enabled: false
    
    metadata_download:
      max_mb: 100
      chunk_wait_us: 1000
      watch_freq_sec: 1
    
    syscall_event_drops:
      actions:
        - log
        - alert
      rate: 0.03
      max_consecutive_alerts: 5
  
  custom_rules.yaml: |
    - rule: Unauthorized Container Registry
      desc: Detect containers from unauthorized registries
      condition: >
        container and container.image.repository not in (
          ghcr.io/company,
          docker.io/library,
          gcr.io/distroless
        )
      output: >
        Unauthorized container registry used
        (image=%container.image.repository
         container=%container.name
         pod=%k8s.pod.name
         ns=%k8s.ns.name)
      priority: WARNING
      tags: [container, cis, mitre_execution]
    
    - rule: Sensitive Mount by Container
      desc: Detect containers mounting sensitive host paths
      condition: >
        container and container.mount.dest in (
          /etc/shadow, /etc/passwd, /etc/sudoers,
          /root/.ssh, /var/run/docker.sock
        )
      output: >
        Container mounted sensitive path
        (path=%container.mount.dest
         container=%container.name
         image=%container.image.repository)
      priority: ERROR
      tags: [container, cis, mitre_privilege_escalation]
    
    - rule: Crypto Mining Detection
      desc: Detect crypto mining using the stratum protocol
      condition: >
        spawned_process and (
          (proc.name in (minerd, xmrig, ccminer)) or
          (proc.cmdline contains "stratum+tcp" or
           proc.cmdline contains "stratum2+tcp")
        )
      output: >
        Crypto miner detected
        (proc=%proc.name cmdline=%proc.cmdline
         container=%container.name)
      priority: CRITICAL
      tags: [process, mitre_impact]
```

### Policy as Code with Kyverno

**✅ DO: Enforce Security Policies**

```yaml
# kyverno-policies.yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-pod-security-standards
  annotations:
    policies.kyverno.io/title: Enforce Pod Security Standards
    policies.kyverno.io/category: Pod Security Standards
    policies.kyverno.io/severity: high
spec:
  validationFailureAction: Enforce
  background: true
  rules:
  - name: check-security-context
    match:
      any:
      - resources:
          kinds:
          - Pod
    validate:
      message: >-
        Pod must run as non-root with read-only root filesystem
      pattern:
        spec:
          securityContext:
            runAsNonRoot: true
            runAsUser: ">0"
            fsGroup: ">0"
          =(ephemeralContainers):
          - securityContext:
              allowPrivilegeEscalation: false
              readOnlyRootFilesystem: true
              capabilities:
                drop:
                - ALL
          =(initContainers):
          - securityContext:
              allowPrivilegeEscalation: false
              readOnlyRootFilesystem: true
              capabilities:
                drop:
                - ALL
          containers:
          - securityContext:
              allowPrivilegeEscalation: false
              readOnlyRootFilesystem: true
              capabilities:
                drop:
                - ALL
---
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-image-signature
  annotations:
    policies.kyverno.io/title: Verify Image Signatures
    policies.kyverno.io/category: Supply Chain Security
spec:
  validationFailureAction: Enforce
  webhookTimeoutSeconds: 30
  rules:
  - name: check-signature
    match:
      any:
      - resources:
          kinds:
          - Pod
    verifyImages:
    - imageReferences:
      - "ghcr.io/company/*"
      attestors:
      - entries:
        - keys:
            publicKeys: |
              -----BEGIN PUBLIC KEY-----
              MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEBqVJ...
              -----END PUBLIC KEY-----
            signatureAlgorithm: sha256
      - entries:
        - keyless:
            subject: "https://github.com/company/*"
            issuer: "https://token.actions.githubusercontent.com"
            rekor:
              url: https://rekor.sigstore.dev
---
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: disallow-latest-tag
  annotations:
    policies.kyverno.io/title: Disallow Latest Tag
    policies.kyverno.io/category: Best Practices
spec:
  validationFailureAction: Enforce
  background: true
  rules:
  - name: require-image-tag
    match:
      any:
      - resources:
          kinds:
          - Pod
    validate:
      message: "The ':latest' tag is not allowed"
      pattern:
        spec:
          =(initContainers):
          - image: "!*:latest"
          containers:
          - image: "!*:latest"
```

---

## 7. Cost Optimization & FinOps

### Karpenter for Intelligent Node Management

**✅ DO: Use Karpenter for Cost-Optimized Scaling**

```yaml
# karpenter-provisioner.yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: general-purpose
spec:
  # Template for nodes
  template:
    metadata:
      labels:
        karpenter.sh/nodepool: general-purpose
        node.company.io/lifecycle: spot
      annotations:
        node.company.io/cost-center: engineering
    spec:
      instanceTypes:
      # Prioritize Graviton (ARM) instances for cost savings
      - t4g.medium
      - t4g.large
      - t4g.xlarge
      - t4g.2xlarge
      # Fallback to Intel
      - t3a.medium
      - t3a.large
      - t3a.xlarge
      - t3a.2xlarge
      
      requirements:
      - key: karpenter.sh/capacity-type
        operator: In
        values: ["spot", "on-demand"]
      - key: kubernetes.io/arch
        operator: In
        values: ["amd64", "arm64"]
      - key: node.kubernetes.io/instance-type
        operator: NotIn
        values: ["t2.nano", "t2.micro"]  # Too small
      
      userData: |
        #!/bin/bash
        # Configure containerd for nydus snapshotter
        cat <<EOF >> /etc/containerd/config.toml
        [proxy_plugins]
          [proxy_plugins.nydus]
            type = "snapshot"
            address = "/run/containerd-nydus/containerd-nydus-grpc.sock"
        
        [plugins."io.containerd.grpc.v1.cri".containerd]
          snapshotter = "nydus"
        EOF
        
        systemctl restart containerd
      
      taints:
      - key: node.company.io/initialization
        value: "true"
        effect: NoSchedule
      
      startupTaints:
      - key: node.company.io/startup
        value: "true"
        effect: NoSchedule
  
  # Disruption settings
  disruption:
    # Automatically replace nodes after 30 days
    expireAfter: 30d
    
    # Allow disruption for cost optimization
    consolidationPolicy: WhenEmptyOrUnderutilized
    consolidateAfter: 1m
    
    # Don't disrupt during business hours
    budgets:
    - nodes: "33%"
      schedule: "* * * * *"
      duration: 10m
    - nodes: "0"
      schedule: "0 9-17 * * MON-FRI"  # Business hours
      duration: 8h
  
  # Resource limits for this provisioner
  limits:
    cpu: "10000"
    memory: "40Ti"
  
  # Node properties
  providerRef:
    apiVersion: karpenter.k8s.aws/v1
    kind: EC2NodeClass
    name: default
---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: default
spec:
  role: "KarpenterNodeInstanceProfile"
  
  # Cost optimization: Use gp3 instead of gp2
  blockDeviceMappings:
  - deviceName: /dev/xvda
    ebs:
      volumeSize: "100Gi"
      volumeType: "gp3"
      iops: 3000
      throughput: 125
      deleteOnTermination: true
      encrypted: true
  
  # Use latest Amazon Linux 2023 AMI
  amiSelectorTerms:
  - alias: al2023@latest
  
  # Enable detailed monitoring for better autoscaling
  detailedMonitoring: true
  
  # Metadata options
  metadataOptions:
    httpEndpoint: enabled
    httpProtocolIPv6: disabled
    httpPutResponseHopLimit: 1
    httpTokens: required
```

### OpenCost for Visibility

**✅ DO: Deploy OpenCost for Cost Attribution**

```yaml
# opencost-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: opencost-config
  namespace: opencost
data:
  opencost.yaml: |
    # Prometheus for metrics
    prometheus:
      internal:
        enabled: true
        port: 9003
      external:
        enabled: true
        url: "http://prometheus.observability:9090"
    
    # Cloud provider integration
    cloudProviderAPI:
      aws:
        athena:
          bucket: "s3://company-aws-cost-reports"
          region: "us-east-1"
          database: "athenacurcfn_cost_and_usage_report"
          table: "cost_and_usage_report"
          workgroup: "primary"
      
    # Pricing sources
    pricing:
      enabled: true
      configPath: "/tmp/pricing-configs"
      
    # Allocation
    allocation:
      idle:
        enabled: true
      
    # UI configuration
    ui:
      enabled: true
      ingress:
        enabled: true
        annotations:
          cert-manager.io/cluster-issuer: letsencrypt-prod
        hosts:
        - host: cost.company.io
          paths:
          - path: /
            pathType: Prefix
        tls:
        - secretName: opencost-tls
          hosts:
          - cost.company.io
    
    # Metrics
    metrics:
      kubeStateMetrics:
        enabled: true
      serviceMonitor:
        enabled: true
        namespace: opencost
        additionalLabels:
          prometheus: kube-prometheus
      
    # Export cost data
    export:
      csv:
        enabled: true
        path: "/tmp/cost-exports"
      s3:
        enabled: true
        bucket: "company-cost-exports"
        region: "us-east-1"
        
    # Alerts
    alerts:
      enabled: true
      dailyBudget: 1000
      weeklyGrowthThreshold: 0.25
```

---

## 8. Platform Engineering Patterns

### Developer Portal with Backstage

**✅ DO: Create a Unified Developer Experience**

```yaml
# backstage/app-config.yaml
app:
  title: Company Developer Portal
  baseUrl: https://backstage.company.io

organization:
  name: Company

backend:
  baseUrl: https://backstage.company.io
  listen:
    port: 7007
  database:
    client: pg
    connection:
      host: ${POSTGRES_HOST}
      port: ${POSTGRES_PORT}
      user: ${POSTGRES_USER}
      password: ${POSTGRES_PASSWORD}
  cors:
    origin: https://backstage.company.io
  csp:
    connect-src: ["'self'", 'http:', 'https:']

integrations:
  github:
    - host: github.com
      token: ${GITHUB_TOKEN}
  gitlab:
    - host: gitlab.company.io
      token: ${GITLAB_TOKEN}
  aws:
    mainAccount:
      accessKeyId: ${AWS_ACCESS_KEY_ID}
      secretAccessKey: ${AWS_SECRET_ACCESS_KEY}

proxy:
  endpoints:
    # Kubernetes API
    /kubernetes:
      target: https://kubernetes.default.svc
      headers:
        Authorization: Bearer ${KUBERNETES_TOKEN}
    
    # ArgoCD
    /argocd/api:
      target: https://argocd.company.io
      headers:
        Authorization: Bearer ${ARGOCD_TOKEN}
    
    # Prometheus
    /prometheus/api:
      target: http://prometheus.observability:9090
    
    # SonarQube
    /sonarqube:
      target: https://sonarqube.company.io
      headers:
        Authorization: Basic ${SONARQUBE_TOKEN}

techdocs:
  builder: 'external'
  generator:
    runIn: 'docker'
  publisher:
    type: 'awsS3'
    awsS3:
      bucketName: 'company-techdocs'
      region: 'us-east-1'

auth:
  environment: production
  providers:
    github:
      production:
        clientId: ${GITHUB_CLIENT_ID}
        clientSecret: ${GITHUB_CLIENT_SECRET}
    google:
      production:
        clientId: ${GOOGLE_CLIENT_ID}
        clientSecret: ${GOOGLE_CLIENT_SECRET}

scaffolder:
  defaultAuthor:
    name: Backstage
    email: backstage@company.io

catalog:
  rules:
    - allow: [Component, API, Location, Template]
  locations:
    # Core entities
    - type: url
      target: https://github.com/company/backstage-catalog/blob/main/catalog-info.yaml
    
    # Service templates
    - type: url
      target: https://github.com/company/service-templates/blob/main/templates.yaml
    
    # Auto-discovery
    - type: github-discovery
      target: https://github.com/company/*/blob/main/catalog-info.yaml

kubernetes:
  serviceLocatorMethod:
    type: 'multiTenant'
  clusterLocatorMethods:
    - type: 'config'
      clusters:
        - name: production-us-east-1
          url: ${K8S_PROD_US_URL}
          authProvider: 'serviceAccount'
          serviceAccountToken: ${K8S_PROD_US_TOKEN}
        - name: production-eu-west-1
          url: ${K8S_PROD_EU_URL}
          authProvider: 'serviceAccount'
          serviceAccountToken: ${K8S_PROD_EU_TOKEN}
        - name: staging
          url: ${K8S_STAGING_URL}
          authProvider: 'serviceAccount'
          serviceAccountToken: ${K8S_STAGING_TOKEN}
```

### Service Catalog Template

**✅ DO: Provide Self-Service Templates**

```yaml
# templates/microservice-template.yaml
apiVersion: scaffolder.backstage.io/v1beta3
kind: Template
metadata:
  name: microservice-template
  title: Create a new Microservice
  description: Creates a new microservice with CI/CD, monitoring, and documentation
  tags:
    - recommended
    - microservice
spec:
  owner: platform-team
  type: service
  
  parameters:
    - title: Service Information
      required:
        - name
        - description
        - owner
      properties:
        name:
          title: Name
          type: string
          description: Unique name for the service
          pattern: '^[a-z0-9-]+$'
          ui:autofocus: true
        description:
          title: Description
          type: string
          description: What does this service do?
        owner:
          title: Owner
          type: string
          description: Owner team from the catalog
          ui:field: OwnerPicker
          ui:options:
            catalogFilter:
              kind: Group
    
    - title: Technology Stack
      required:
        - language
        - database
      properties:
        language:
          title: Programming Language
          type: string
          enum:
            - go
            - java
            - nodejs
            - python
          enumNames:
            - Go
            - Java (Spring Boot)
            - Node.js
            - Python (FastAPI)
        database:
          title: Database
          type: string
          enum:
            - postgresql
            - mysql
            - mongodb
            - none
          default: postgresql
        cache:
          title: Enable Redis Cache
          type: boolean
          default: true
    
    - title: Infrastructure
      required:
        - region
      properties:
        region:
          title: Primary Region
          type: string
          enum:
            - us-east-1
            - eu-west-1
            - ap-southeast-1
          default: us-east-1
        multiRegion:
          title: Multi-region deployment
          type: boolean
          default: false
        minReplicas:
          title: Minimum Replicas
          type: integer
          default: 2
          minimum: 1
        maxReplicas:
          title: Maximum Replicas
          type: integer
          default: 10
          minimum: 2
  
  steps:
    # Fetch skeleton
    - id: fetch
      name: Fetch Skeleton + Template
      action: fetch:template
      input:
        url: ./skeleton/${{ parameters.language }}
        values:
          name: ${{ parameters.name }}
          description: ${{ parameters.description }}
          owner: ${{ parameters.owner }}
          database: ${{ parameters.database }}
          cache: ${{ parameters.cache }}
          region: ${{ parameters.region }}
    
    # Create GitHub repository
    - id: publish
      name: Publish to GitHub
      action: publish:github
      input:
        description: This is ${{ parameters.name }}
        repoUrl: github.com?owner=company&repo=${{ parameters.name }}
        defaultBranch: main
        gitCommitMessage: 'Initial commit'
        topics:
          - microservice
          - ${{ parameters.language }}
        repoVariables:
          LANGUAGE: ${{ parameters.language }}
          HAS_DATABASE: ${{ parameters.database !== 'none' }}
    
    # Create AWS resources
    - id: aws-resources
      name: Create AWS Resources
      action: aws:cloudformation:deploy
      input:
        stackName: ${{ parameters.name }}-infrastructure
        templatePath: ./infrastructure/cloudformation.yaml
        parameters:
          ServiceName: ${{ parameters.name }}
          DatabaseType: ${{ parameters.database }}
          EnableCache: ${{ parameters.cache }}
          Region: ${{ parameters.region }}
    
    # Set up CI/CD
    - id: create-pipeline
      name: Create CI/CD Pipeline
      action: github:actions:create-workflow
      input:
        repoUrl: github.com?owner=company&repo=${{ parameters.name }}
        workflowPath: .github/workflows/deploy.yaml
        workflowContent:
          name: Deploy
          on:
            push:
              branches: [main]
          jobs:
            deploy:
              uses: company/shared-workflows/.github/workflows/deploy-${{ parameters.language }}.yaml@main
              secrets: inherit
    
    # Create ArgoCD Application
    - id: create-argocd-app
      name: Register in ArgoCD
      action: argocd:create-application
      input:
        appName: ${{ parameters.name }}
        appNamespace: argocd
        repoUrl: https://github.com/company/gitops
        path: apps/${{ parameters.name }}
        project: default
        destination:
          server: https://kubernetes.default.svc
          namespace: ${{ parameters.name }}
        syncPolicy:
          automated:
            prune: true
            selfHeal: true
    
    # Register in catalog
    - id: register
      name: Register in Software Catalog
      action: catalog:register
      input:
        repoContentsUrl: ${{ steps['publish'].output.repoContentsUrl }}
        catalogInfoPath: '/catalog-info.yaml'
  
  output:
    links:
      - title: Repository
        url: ${{ steps['publish'].output.remoteUrl }}
      - title: ArgoCD Application
        url: https://argocd.company.io/applications/${{ parameters.name }}
      - title: Open in catalog
        icon: catalog
        entityRef: ${{ steps['register'].output.entityRef }}
```

### Platform API with Crossplane

**✅ DO: Build a Kubernetes-Native Platform API**

```yaml
# platform-api/xrd-application.yaml
apiVersion: apiextensions.crossplane.io/v1
kind: CompositeResourceDefinition
metadata:
  name: xapplications.platform.company.io
spec:
  group: platform.company.io
  names:
    kind: XApplication
    plural: xapplications
  claimNames:
    kind: Application
    plural: applications
  versions:
  - name: v1alpha1
    served: true
    referenceable: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              parameters:
                type: object
                properties:
                  image:
                    type: string
                    description: Container image
                  replicas:
                    type: integer
                    description: Number of replicas
                    default: 2
                  port:
                    type: integer
                    description: Service port
                    default: 8080
                  resources:
                    type: object
                    properties:
                      cpu:
                        type: string
                        default: "100m"
                      memory:
                        type: string
                        default: "128Mi"
                  autoscaling:
                    type: object
                    properties:
                      enabled:
                        type: boolean
                        default: true
                      minReplicas:
                        type: integer
                        default: 2
                      maxReplicas:
                        type: integer
                        default: 10
                      targetCPU:
                        type: integer
                        default: 80
                  ingress:
                    type: object
                    properties:
                      enabled:
                        type: boolean
                        default: false
                      host:
                        type: string
                      path:
                        type: string
                        default: "/"
                  database:
                    type: object
                    properties:
                      enabled:
                        type: boolean
                        default: false
                      type:
                        type: string
                        enum: ["postgresql", "mysql", "redis"]
                        default: "postgresql"
                      size:
                        type: string
                        enum: ["small", "medium", "large"]
                        default: "small"
                  monitoring:
                    type: object
                    properties:
                      enabled:
                        type: boolean
                        default: true
                      alerts:
                        type: array
                        items:
                          type: object
                          properties:
                            name:
                              type: string
                            query:
                              type: string
                            threshold:
                              type: number
                required:
                - image
          status:
            type: object
            properties:
              phase:
                type: string
              endpoints:
                type: object
                properties:
                  application:
                    type: string
                  database:
                    type: string
                  monitoring:
                    type: string
---
# platform-api/composition-application.yaml
apiVersion: apiextensions.crossplane.io/v1
kind: Composition
metadata:
  name: xapplications.platform.company.io
spec:
  compositeTypeRef:
    apiVersion: platform.company.io/v1alpha1
    kind: XApplication
  
  mode: Pipeline
  pipeline:
  - step: create-namespace
    functionRef:
      name: function-auto-ready
    input:
      apiVersion: kubernetes.crossplane.io/v1alpha1
      kind: Object
      metadata:
        name: namespace
      spec:
        forProvider:
          manifest:
            apiVersion: v1
            kind: Namespace
            metadata:
              name: "" # patched
              labels:
                app.kubernetes.io/managed-by: crossplane
  
  - step: create-deployment
    functionRef:
      name: function-go-templating
    input:
      apiVersion: gotemplating.fn.crossplane.io/v1beta1
      kind: GoTemplate
      source: Inline
      inline:
        template: |
          apiVersion: apps/v1
          kind: Deployment
          metadata:
            name: {{ .observed.composite.resource.metadata.name }}
            namespace: {{ .observed.composite.resource.metadata.name }}
          spec:
            replicas: {{ .observed.composite.resource.spec.parameters.replicas }}
            selector:
              matchLabels:
                app: {{ .observed.composite.resource.metadata.name }}
            template:
              metadata:
                labels:
                  app: {{ .observed.composite.resource.metadata.name }}
              spec:
                containers:
                - name: app
                  image: {{ .observed.composite.resource.spec.parameters.image }}
                  ports:
                  - containerPort: {{ .observed.composite.resource.spec.parameters.port }}
                  resources:
                    requests:
                      cpu: {{ .observed.composite.resource.spec.parameters.resources.cpu }}
                      memory: {{ .observed.composite.resource.spec.parameters.resources.memory }}
                  env:
                  - name: OTEL_SERVICE_NAME
                    value: {{ .observed.composite.resource.metadata.name }}
                  {{ if .observed.composite.resource.spec.parameters.database.enabled }}
                  - name: DATABASE_URL
                    valueFrom:
                      secretKeyRef:
                        name: {{ .observed.composite.resource.metadata.name }}-db
                        key: connection-string
                  {{ end }}
  
  - step: create-service
    functionRef:
      name: function-auto-ready
    input:
      apiVersion: kubernetes.crossplane.io/v1alpha1
      kind: Object
      spec:
        forProvider:
          manifest:
            apiVersion: v1
            kind: Service
            spec:
              selector:
                app: "" # patched
              ports:
              - port: 80
                targetPort: 8080
  
  - step: create-monitoring
    functionRef:
      name: function-go-templating
    input:
      apiVersion: gotemplating.fn.crossplane.io/v1beta1
      kind: GoTemplate
      source: Inline
      inline:
        template: |
          {{ if .observed.composite.resource.spec.parameters.monitoring.enabled }}
          apiVersion: v1
          kind: ServiceMonitor
          metadata:
            name: {{ .observed.composite.resource.metadata.name }}
            namespace: {{ .observed.composite.resource.metadata.name }}
          spec:
            selector:
              matchLabels:
                app: {{ .observed.composite.resource.metadata.name }}
            endpoints:
            - port: metrics
              interval: 30s
          ---
          apiVersion: monitoring.coreos.com/v1
          kind: PrometheusRule
          metadata:
            name: {{ .observed.composite.resource.metadata.name }}
            namespace: {{ .observed.composite.resource.metadata.name }}
          spec:
            groups:
            - name: {{ .observed.composite.resource.metadata.name }}
              rules:
              {{ range .observed.composite.resource.spec.parameters.monitoring.alerts }}
              - alert: {{ .name }}
                expr: {{ .query }}
                for: 5m
                annotations:
                  summary: "{{ .name }} triggered"
                  description: "{{ .query }} exceeded threshold {{ .threshold }}"
              {{ end }}
          {{ end }}
```

### Developer Self-Service UI

**✅ DO: Create a Self-Service Portal**

```typescript
// platform-ui/src/components/CreateApplication.tsx
import React, { useState } from 'react';
import { 
  Box, 
  Button, 
  TextField, 
  Select, 
  MenuItem, 
  FormControl,
  InputLabel,
  Stepper,
  Step,
  StepLabel,
  Card,
  CardContent,
  Typography,
  Alert,
  Chip,
  Grid
} from '@mui/material';
import { useKubernetesClient } from '../hooks/useKubernetesClient';
import { ApplicationSpec, DatabaseSize } from '../types';

const CreateApplication: React.FC = () => {
  const k8sClient = useKubernetesClient();
  const [activeStep, setActiveStep] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  
  const [spec, setSpec] = useState<ApplicationSpec>({
    name: '',
    image: '',
    replicas: 2,
    resources: {
      cpu: '100m',
      memory: '128Mi'
    },
    autoscaling: {
      enabled: true,
      minReplicas: 2,
      maxReplicas: 10,
      targetCPU: 80
    },
    database: {
      enabled: false,
      type: 'postgresql',
      size: 'small'
    },
    monitoring: {
      enabled: true,
      alerts: []
    }
  });

  const steps = ['Basic Info', 'Resources', 'Database', 'Monitoring', 'Review'];

  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const createApplication = async () => {
    setCreating(true);
    setError(null);
    
    try {
      // Create the Crossplane Application claim
      const application = {
        apiVersion: 'platform.company.io/v1alpha1',
        kind: 'Application',
        metadata: {
          name: spec.name,
          namespace: 'default'
        },
        spec: {
          parameters: spec
        }
      };
      
      await k8sClient.create(application);
      
      // Redirect to application details
      window.location.href = `/applications/${spec.name}`;
    } catch (err) {
      setError(`Failed to create application: ${err.message}`);
      setCreating(false);
    }
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <Box>
            <TextField
              fullWidth
              label="Application Name"
              value={spec.name}
              onChange={(e) => setSpec({...spec, name: e.target.value})}
              helperText="Lowercase letters, numbers, and hyphens only"
              margin="normal"
            />
            <TextField
              fullWidth
              label="Container Image"
              value={spec.image}
              onChange={(e) => setSpec({...spec, image: e.target.value})}
              helperText="e.g., ghcr.io/company/my-app:v1.0.0"
              margin="normal"
            />
          </Box>
        );
      
      case 1:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Resource Configuration
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="CPU Request"
                  value={spec.resources.cpu}
                  onChange={(e) => setSpec({
                    ...spec, 
                    resources: {...spec.resources, cpu: e.target.value}
                  })}
                  helperText="e.g., 100m, 1, 2"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="Memory Request"
                  value={spec.resources.memory}
                  onChange={(e) => setSpec({
                    ...spec,
                    resources: {...spec.resources, memory: e.target.value}
                  })}
                  helperText="e.g., 128Mi, 1Gi"
                />
              </Grid>
            </Grid>
            
            <Box mt={3}>
              <Typography variant="subtitle1" gutterBottom>
                Autoscaling
              </Typography>
              <FormControl fullWidth margin="normal">
                <InputLabel>Enable Autoscaling</InputLabel>
                <Select
                  value={spec.autoscaling.enabled}
                  onChange={(e) => setSpec({
                    ...spec,
                    autoscaling: {...spec.autoscaling, enabled: e.target.value as boolean}
                  })}
                >
                  <MenuItem value={true}>Yes</MenuItem>
                  <MenuItem value={false}>No</MenuItem>
                </Select>
              </FormControl>
              
              {spec.autoscaling.enabled && (
                <Grid container spacing={2} mt={1}>
                  <Grid item xs={4}>
                    <TextField
                      fullWidth
                      type="number"
                      label="Min Replicas"
                      value={spec.autoscaling.minReplicas}
                      onChange={(e) => setSpec({
                        ...spec,
                        autoscaling: {
                          ...spec.autoscaling,
                          minReplicas: parseInt(e.target.value)
                        }
                      })}
                    />
                  </Grid>
                  <Grid item xs={4}>
                    <TextField
                      fullWidth
                      type="number"
                      label="Max Replicas"
                      value={spec.autoscaling.maxReplicas}
                      onChange={(e) => setSpec({
                        ...spec,
                        autoscaling: {
                          ...spec.autoscaling,
                          maxReplicas: parseInt(e.target.value)
                        }
                      })}
                    />
                  </Grid>
                  <Grid item xs={4}>
                    <TextField
                      fullWidth
                      type="number"
                      label="Target CPU %"
                      value={spec.autoscaling.targetCPU}
                      onChange={(e) => setSpec({
                        ...spec,
                        autoscaling: {
                          ...spec.autoscaling,
                          targetCPU: parseInt(e.target.value)
                        }
                      })}
                    />
                  </Grid>
                </Grid>
              )}
            </Box>
          </Box>
        );
      
      case 2:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Database Configuration
            </Typography>
            <FormControl fullWidth margin="normal">
              <InputLabel>Enable Database</InputLabel>
              <Select
                value={spec.database.enabled}
                onChange={(e) => setSpec({
                  ...spec,
                  database: {...spec.database, enabled: e.target.value as boolean}
                })}
              >
                <MenuItem value={true}>Yes</MenuItem>
                <MenuItem value={false}>No</MenuItem>
              </Select>
            </FormControl>
            
            {spec.database.enabled && (
              <>
                <FormControl fullWidth margin="normal">
                  <InputLabel>Database Type</InputLabel>
                  <Select
                    value={spec.database.type}
                    onChange={(e) => setSpec({
                      ...spec,
                      database: {...spec.database, type: e.target.value as string}
                    })}
                  >
                    <MenuItem value="postgresql">PostgreSQL</MenuItem>
                    <MenuItem value="mysql">MySQL</MenuItem>
                    <MenuItem value="redis">Redis</MenuItem>
                  </Select>
                </FormControl>
                
                <FormControl fullWidth margin="normal">
                  <InputLabel>Database Size</InputLabel>
                  <Select
                    value={spec.database.size}
                    onChange={(e) => setSpec({
                      ...spec,
                      database: {...spec.database, size: e.target.value as DatabaseSize}
                    })}
                  >
                    <MenuItem value="small">Small (2 vCPU, 4GB RAM)</MenuItem>
                    <MenuItem value="medium">Medium (4 vCPU, 16GB RAM)</MenuItem>
                    <MenuItem value="large">Large (8 vCPU, 32GB RAM)</MenuItem>
                  </Select>
                </FormControl>
              </>
            )}
          </Box>
        );
      
      case 3:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Monitoring & Alerting
            </Typography>
            <FormControl fullWidth margin="normal">
              <InputLabel>Enable Monitoring</InputLabel>
              <Select
                value={spec.monitoring.enabled}
                onChange={(e) => setSpec({
                  ...spec,
                  monitoring: {...spec.monitoring, enabled: e.target.value as boolean}
                })}
              >
                <MenuItem value={true}>Yes</MenuItem>
                <MenuItem value={false}>No</MenuItem>
              </Select>
            </FormControl>
            
            {spec.monitoring.enabled && (
              <Alert severity="info" sx={{ mt: 2 }}>
                Default alerts will be configured for:
                <ul>
                  <li>High error rate (5xx responses &gt; 1%)</li>
                  <li>High latency (p95 &gt; 1s)</li>
                  <li>Pod restarts (&gt; 5 in 15 minutes)</li>
                  <li>High CPU usage (&gt; 80% for 5 minutes)</li>
                  <li>High memory usage (&gt; 90% for 5 minutes)</li>
                </ul>
              </Alert>
            )}
          </Box>
        );
      
      case 4:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Review Configuration
            </Typography>
            <Card variant="outlined">
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Application
                </Typography>
                <Typography variant="h5" component="h2">
                  {spec.name}
                </Typography>
                <Typography color="textSecondary">
                  Image: {spec.image}
                </Typography>
                
                <Box mt={2}>
                  <Chip 
                    label={`CPU: ${spec.resources.cpu}`} 
                    size="small" 
                    sx={{ mr: 1 }}
                  />
                  <Chip 
                    label={`Memory: ${spec.resources.memory}`} 
                    size="small" 
                    sx={{ mr: 1 }}
                  />
                  {spec.autoscaling.enabled && (
                    <Chip 
                      label={`Autoscaling: ${spec.autoscaling.minReplicas}-${spec.autoscaling.maxReplicas}`} 
                      size="small" 
                      color="primary"
                    />
                  )}
                </Box>
                
                {spec.database.enabled && (
                  <Box mt={2}>
                    <Typography variant="subtitle2">
                      Database: {spec.database.type} ({spec.database.size})
                    </Typography>
                  </Box>
                )}
                
                {spec.monitoring.enabled && (
                  <Box mt={2}>
                    <Typography variant="subtitle2">
                      ✓ Monitoring & Alerting Enabled
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
            
            <Alert severity="warning" sx={{ mt: 2 }}>
              This will create real cloud resources that will incur costs.
              Estimated monthly cost: ${estimateMonthlyCost(spec)}
            </Alert>
          </Box>
        );
      
      default:
        return null;
    }
  };

  return (
    <Box sx={{ width: '100%', maxWidth: 800, mx: 'auto', p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Create New Application
      </Typography>
      
      <Stepper activeStep={activeStep} sx={{ mt: 3, mb: 3 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      
      <Box sx={{ mt: 2, mb: 2, minHeight: 400 }}>
        {renderStepContent(activeStep)}
      </Box>
      
      <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
        {activeStep !== 0 && (
          <Button onClick={handleBack} sx={{ mr: 1 }}>
            Back
          </Button>
        )}
        
        {activeStep === steps.length - 1 ? (
          <Button
            variant="contained"
            onClick={createApplication}
            disabled={creating}
          >
            {creating ? 'Creating...' : 'Create Application'}
          </Button>
        ) : (
          <Button variant="contained" onClick={handleNext}>
            Next
          </Button>
        )}
      </Box>
    </Box>
  );
};

// Helper function to estimate costs
const estimateMonthlyCost = (spec: ApplicationSpec): number => {
  let cost = 0;
  
  // Compute costs (simplified)
  const cpuCores = parseFloat(spec.resources.cpu.replace('m', '')) / 1000;
  const memoryGB = parseFloat(spec.resources.memory.replace('Mi', '')) / 1024;
  
  cost += cpuCores * 30 * spec.autoscaling.minReplicas; // $30/core/month
  cost += memoryGB * 10 * spec.autoscaling.minReplicas; // $10/GB/month
  
  // Database costs
  if (spec.database.enabled) {
    const dbCosts = {
      small: 50,
      medium: 200,
      large: 500
    };
    cost += dbCosts[spec.database.size];
  }
  
  return Math.round(cost);
};

export default CreateApplication;
```

---

## Conclusion

This guide represents the state of cloud-native DevOps in mid-2025. The key themes are:

1. **GitOps Everything**: Infrastructure, applications, and policies all managed through Git
2. **Platform Engineering**: Self-service capabilities that empower developers while maintaining governance
3. **Observability-First**: OpenTelemetry as the universal standard for all telemetry data
4. **Security by Default**: Supply chain security, runtime protection, and policy enforcement built-in
5. **Cost Awareness**: FinOps practices integrated into the platform from day one

The future of DevOps is about creating platforms that developers love to use while maintaining the security, reliability, and efficiency that operations teams require. By following these patterns, you'll build infrastructure that scales with your organization's needs.

Remember: these are patterns, not prescriptions. Adapt them to your specific context, start small, and iterate based on feedback from your users.