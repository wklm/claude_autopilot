# The Definitive Guide to DevOps Fleet Management with Ansible, Grafana, and Modern Observability (Mid-2025)

This guide provides production-grade patterns for managing large-scale infrastructure across multiple cloud providers using Ansible for automation, Grafana for observability, and modern DevOps practices. It emphasizes security, scalability, and cost efficiency in multi-cloud environments.

### Prerequisites & Core Stack
- **Ansible 10.0+** with ansible-core 2.17+
- **Grafana 11.3+** with unified alerting
- **Prometheus 3.0+** with native OpenTelemetry support
- **Python 3.12+** for custom modules and filters
- **Terraform 1.9+** / **OpenTofu 2.0+** for infrastructure provisioning

### Essential Tooling (2025 Standards)
```yaml
# requirements.yml - Ansible Collections
collections:
  - name: ansible.posix
    version: ">=1.6.0"
  - name: community.general
    version: ">=9.0.0"
  - name: amazon.aws
    version: ">=8.0.0"
  - name: google.cloud
    version: ">=2.0.0"
  - name: azure.azcollection
    version: ">=3.0.0"
  - name: kubernetes.core
    version: ">=4.0.0"
  - name: community.grafana
    version: ">=2.0.0"
  - name: prometheus.prometheus
    version: ">=1.0.0"
```

---

## 1. Infrastructure Repository Structure

A well-organized repository is crucial for managing complex multi-cloud deployments at scale.

### ✅ DO: Use a Standardized Directory Layout

This structure separates concerns and enables team collaboration while supporting multiple environments and cloud providers.

```
/infrastructure
├── ansible/
│   ├── inventories/
│   │   ├── production/
│   │   │   ├── group_vars/
│   │   │   │   ├── all/
│   │   │   │   │   ├── vault.yml      # Encrypted secrets
│   │   │   │   │   └── vars.yml       # Global variables
│   │   │   │   ├── aws/
│   │   │   │   ├── gcp/
│   │   │   │   └── azure/
│   │   │   ├── host_vars/
│   │   │   └── inventory.yml          # Dynamic inventory config
│   │   ├── staging/
│   │   └── development/
│   ├── playbooks/
│   │   ├── site.yml                   # Master playbook
│   │   ├── deploy.yml                 # Application deployment
│   │   ├── security-hardening.yml    # CIS benchmarks
│   │   └── disaster-recovery.yml     # DR procedures
│   ├── roles/
│   │   ├── base-server/              # Common server setup
│   │   ├── monitoring-agent/         # Prometheus/OTel agents
│   │   └── security-baseline/        # Security hardening
│   ├── filter_plugins/               # Custom Jinja2 filters
│   ├── module_utils/                 # Shared module code
│   └── ansible.cfg                   # Ansible configuration
├── terraform/
│   ├── modules/
│   │   ├── vpc/
│   │   ├── kubernetes/
│   │   └── monitoring/
│   ├── environments/
│   │   ├── prod/
│   │   └── staging/
│   └── providers.tf
├── monitoring/
│   ├── grafana/
│   │   ├── dashboards/              # JSON dashboard definitions
│   │   ├── provisioning/            # Datasources and dashboards
│   │   └── alerts/                  # Alert rules as code
│   ├── prometheus/
│   │   ├── rules/                   # Recording and alerting rules
│   │   └── targets/                 # Service discovery configs
│   └── loki/
│       └── config/                  # Log aggregation config
├── scripts/
│   ├── backup-automation.py         # Automated backup orchestration
│   └── cost-optimizer.py            # Cloud cost analysis
└── .gitlab-ci.yml                   # CI/CD pipeline
```

### ✅ DO: Use Ansible Vault for All Secrets

Never commit plaintext secrets. Use Ansible Vault with a strong password or integrate with HashiCorp Vault.

```bash
# Create encrypted variable file
ansible-vault create inventories/production/group_vars/all/vault.yml

# Edit existing encrypted file
ansible-vault edit inventories/production/group_vars/all/vault.yml

# Encrypt inline strings (Ansible 2.17+)
ansible-vault encrypt_string 'super-secret-password' --name 'db_password'
```

---

## 2. Dynamic Inventory Management Across Clouds

Static inventory files don't scale. Use dynamic inventory to automatically discover and manage resources.

### ✅ DO: Implement Multi-Cloud Dynamic Inventory

Create a unified inventory that aggregates resources from all cloud providers with consistent tagging.

```yaml
# inventories/production/inventory.yml
plugin: constructed
strict: true
compose:
  # Standardize ansible_host across providers
  ansible_host: public_ip_address | default(private_ip_address)
  
  # Create consistent groups across clouds
  cloud_provider: |
    tags.get('provider', 'unknown')
  
  # Environment from tags
  environment: tags.get('Environment', 'unknown')
  
  # Service type grouping
  service: tags.get('Service', 'unknown')

groups:
  # Dynamic groups based on tags
  web_servers: service == 'web'
  db_servers: service == 'database'
  monitoring: service == 'monitoring'
  
  # Cloud-specific groups
  aws_instances: cloud_provider == 'aws'
  gcp_instances: cloud_provider == 'gcp'
  azure_instances: cloud_provider == 'azure'

keyed_groups:
  # Create groups for each service automatically
  - prefix: service
    key: service
  
  # Groups by availability zone
  - prefix: az
    key: availability_zone
    
  # Groups by instance type/size
  - prefix: size
    key: instance_type | default('unknown')
```

### ✅ DO: Use Plugin Composition for Complex Inventories

```yaml
# inventories/production/aws.yml
plugin: amazon.aws.aws_ec2
regions:
  - us-east-1
  - eu-west-1
filters:
  tag:Environment: production
  instance-state-name: running
hostnames:
  - dns-name
  - private-ip-address
compose:
  ansible_host: public_dns_name | default(private_ip_address)
  provider: "'aws'"

# inventories/production/gcp.yml
plugin: google.cloud.gcp_compute
projects:
  - my-project-id
auth_kind: serviceaccount
service_account_file: /path/to/service-account.json
filters:
  - labels.environment = production
compose:
  ansible_host: networkInterfaces[0].accessConfigs[0].natIP | default(networkInterfaces[0].networkIP)
  provider: "'gcp'"

# inventories/production/inventory.yml - Aggregates all sources
plugin: ansible.builtin.constructed
use_vars_plugins: true
sources:
  - aws.yml
  - gcp.yml
  - azure.yml
```

---

## 3. Modern Ansible Best Practices

Ansible has evolved significantly. Modern patterns emphasize idempotency, performance, and maintainability.

### ✅ DO: Use Execution Environments (ansible-navigator)

Execution Environments (EE) are container images that include Ansible, collections, and dependencies. They ensure consistency across teams and CI/CD.

```yaml
# execution-environment.yml
version: 3
images:
  base_image:
    name: quay.io/ansible/creator-ee:v0.24.0

dependencies:
  galaxy: requirements.yml
  python:
    - boto3>=1.34.0
    - google-cloud-compute>=1.14.0
    - azure-mgmt-compute>=30.0.0
    - kubernetes>=29.0.0
    - jmespath>=1.0.0
    - netaddr>=1.0.0
  system:
    - openssh-clients
    - rsync

build_args:
  ANSIBLE_GALAXY_SERVER_LIST: "https://galaxy.ansible.com,https://hub.example.com"

# Build and use
# ansible-builder build -t my-org/ansible-ee:latest
# ansible-navigator run playbooks/site.yml -i inventories/production --eei my-org/ansible-ee:latest
```

### ✅ DO: Implement Async Operations for Scale

When managing hundreds of hosts, synchronous execution is too slow. Use async for parallel operations.

```yaml
---
# playbooks/rolling-update.yml
- name: Perform rolling update across fleet
  hosts: web_servers
  serial: "20%"  # Update 20% of hosts at a time
  max_fail_percentage: 10  # Tolerate 10% failure rate
  
  tasks:
    - name: Drain connections from load balancer
      uri:
        url: "https://lb.example.com/api/drain/{{ inventory_hostname }}"
        method: POST
      delegate_to: localhost
      
    - name: Wait for connections to drain
      wait_for:
        port: 80
        state: drained
        timeout: 60
    
    - name: Update application (async)
      ansible.builtin.package:
        name: myapp
        state: latest
      async: 300  # 5 minute timeout
      poll: 0     # Fire and forget
      register: update_job
    
    - name: Check update status
      async_status:
        jid: "{{ update_job.ansible_job_id }}"
      register: job_result
      until: job_result.finished
      retries: 30
      delay: 10
    
    - name: Verify application health
      uri:
        url: "http://{{ ansible_host }}:8080/health"
        status_code: 200
      retries: 5
      delay: 10
      
    - name: Re-enable in load balancer
      uri:
        url: "https://lb.example.com/api/enable/{{ inventory_hostname }}"
        method: POST
      delegate_to: localhost
```

### ✅ DO: Use Ansible Facts Caching

Facts gathering is expensive. Cache facts to dramatically improve performance.

```ini
# ansible.cfg
[defaults]
gathering = smart
fact_caching = redis
fact_caching_connection = redis-server:6379:0
fact_caching_timeout = 3600  # 1 hour
fact_caching_prefix = ansible_facts_

# Alternative: JSON file caching
# fact_caching = jsonfile
# fact_caching_connection = /tmp/ansible_fact_cache

[inventory]
cache = yes
cache_connection = /tmp/ansible_inventory_cache
cache_timeout = 3600
```

### ❌ DON'T: Use Loops for Package Installation

```yaml
# Bad - Makes N separate transactions
- name: Install packages
  package:
    name: "{{ item }}"
    state: present
  loop:
    - nginx
    - postgresql
    - redis
```

### ✅ DO: Pass Lists Directly

```yaml
# Good - Single transaction, much faster
- name: Install packages
  package:
    name:
      - nginx
      - postgresql-16  # Be specific about versions
      - redis-server
    state: present
```

---

## 4. Comprehensive Monitoring Architecture

Modern observability requires multiple data types: metrics, logs, traces, and profiles. The Grafana stack provides a unified solution.

### ✅ DO: Deploy the Complete LGTM Stack

**LGTM** = Loki (logs) + Grafana (visualization) + Tempo (traces) + Mimir (metrics)

```yaml
# roles/monitoring-stack/tasks/main.yml
---
- name: Deploy Grafana with Unified Alerting
  kubernetes.core.helm:
    name: grafana
    chart_ref: grafana/grafana
    release_namespace: monitoring
    values:
      persistence:
        enabled: true
        size: 10Gi
      adminPassword: "{{ vault_grafana_admin_password }}"
      
      # Enable new unified alerting (Grafana 11+)
      alerting:
        enabled: true
        unified_alerting:
          enabled: true
          
      # Pre-configure datasources
      datasources:
        datasources.yaml:
          apiVersion: 1
          datasources:
            - name: Prometheus
              type: prometheus
              url: http://prometheus:9090
              isDefault: true
              
            - name: Loki
              type: loki
              url: http://loki:3100
              
            - name: Tempo
              type: tempo
              url: http://tempo:3200
              
      # Provision dashboards from ConfigMaps
      dashboardProviders:
        dashboardproviders.yaml:
          apiVersion: 1
          providers:
            - name: 'default'
              folder: 'Provisioned'
              type: file
              options:
                path: /var/lib/grafana/dashboards

- name: Deploy Prometheus with native OTLP support
  kubernetes.core.helm:
    name: prometheus
    chart_ref: prometheus-community/kube-prometheus-stack
    values:
      prometheus:
        prometheusSpec:
          # Enable OTLP receiver (Prometheus 3.0+)
          enableFeatures:
            - otlp-receiver
            - native-histograms
            
          # Remote write for long-term storage
          remoteWrite:
            - url: https://mimir.example.com/api/v1/push
              headers:
                X-Scope-OrgID: "{{ org_id }}"
              
          # Retention for local storage
          retention: 24h
          retentionSize: 50GB
          
          # Service discovery for multi-cloud
          additionalScrapeConfigs:
            - job_name: 'aws-ec2'
              ec2_sd_configs:
                - region: us-east-1
                  access_key: "{{ vault_aws_access_key }}"
                  secret_key: "{{ vault_aws_secret_key }}"
                  filters:
                    - name: tag:monitoring
                      values: ['enabled']
                      
            - job_name: 'gcp-gce'
              gce_sd_configs:
                - project: my-project
                  zone: us-central1-a
                  filter: 'labels.monitoring=enabled'
```

### ✅ DO: Implement SLO-Based Monitoring

Define Service Level Objectives (SLOs) as code and generate alerts automatically.

```yaml
# monitoring/prometheus/rules/slos.yml
groups:
  - name: slo_rules
    interval: 30s
    rules:
      # Error rate SLO
      - record: slo:api_error_rate
        expr: |
          (
            sum(rate(http_requests_total{job="api",code=~"5.."}[5m]))
            /
            sum(rate(http_requests_total{job="api"}[5m]))
          )
          
      # Multi-window multi-burn-rate alerts (Google SRE workbook)
      - alert: APIErrorBudgetBurn
        expr: |
          (
            slo:api_error_rate > (1 - 0.99) * 14.4
            and
            slo:api_error_rate offset 1h > (1 - 0.99) * 14.4
          )
          or
          (
            slo:api_error_rate > (1 - 0.99) * 6
            and
            slo:api_error_rate offset 6h > (1 - 0.99) * 6
          )
        labels:
          severity: page
          slo: api_availability
        annotations:
          summary: "API error budget burn rate too high"
          description: "API is burning through error budget. Current error rate: {{ $value | humanizePercentage }}"
```

### ✅ DO: Create Actionable Dashboards

```json
// monitoring/grafana/dashboards/fleet-overview.json
{
  "dashboard": {
    "title": "Multi-Cloud Fleet Overview",
    "panels": [
      {
        "title": "Instances by Cloud Provider",
        "targets": [
          {
            "expr": "count by (provider) (up{job=~\"node.*\"})",
            "legendFormat": "{{ provider }}"
          }
        ],
        "type": "piechart"
      },
      {
        "title": "Cost per Hour by Provider",
        "targets": [
          {
            "expr": "sum by (provider) (instance_cost_per_hour)",
            "legendFormat": "{{ provider }}"
          }
        ],
        "type": "timeseries",
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD",
            "custom": {
              "drawStyle": "bars"
            }
          }
        }
      },
      {
        "title": "Security Compliance Score",
        "targets": [
          {
            "expr": "avg by (environment) (security_compliance_score)",
            "legendFormat": "{{ environment }}"
          }
        ],
        "type": "gauge",
        "options": {
          "reduceOptions": {
            "calcs": ["lastNotNull"]
          }
        },
        "fieldConfig": {
          "defaults": {
            "max": 100,
            "min": 0,
            "thresholds": {
              "steps": [
                { "color": "red", "value": 0 },
                { "color": "yellow", "value": 70 },
                { "color": "green", "value": 90 }
              ]
            }
          }
        }
      }
    ]
  }
}
```

---

## 5. Security Automation and Compliance

Security must be automated and continuously validated across your fleet.

### ✅ DO: Implement CIS Benchmarks Automatically

```yaml
# roles/security-baseline/tasks/main.yml
---
- name: Apply CIS Ubuntu 22.04 Level 1 Benchmark
  block:
    - name: Ensure permissions on /etc/passwd are configured
      file:
        path: /etc/passwd
        owner: root
        group: root
        mode: '0644'
      tags: [cis_1_1_1]
    
    - name: Ensure no legacy + entries in /etc/passwd
      lineinfile:
        path: /etc/passwd
        regexp: '^\+'
        state: absent
      tags: [cis_1_1_2]
    
    - name: Configure kernel parameters
      sysctl:
        name: "{{ item.name }}"
        value: "{{ item.value }}"
        sysctl_set: yes
        state: present
        reload: yes
      loop:
        - { name: 'net.ipv4.conf.all.send_redirects', value: '0' }
        - { name: 'net.ipv4.conf.default.send_redirects', value: '0' }
        - { name: 'net.ipv4.ip_forward', value: '0' }
        - { name: 'net.ipv4.conf.all.accept_source_route', value: '0' }
      tags: [cis_3_3]
    
    - name: Install and configure auditd
      package:
        name: auditd
        state: present
    
    - name: Configure audit rules
      template:
        src: audit.rules.j2
        dest: /etc/audit/rules.d/cis.rules
      notify: restart auditd
      tags: [cis_4_1]

- name: Run OSCAP compliance scan
  command: |
    oscap xccdf eval \
      --profile xccdf_org.ssgproject.content_profile_cis_level1_server \
      --results /tmp/scan-results-{{ ansible_hostname }}.xml \
      --report /tmp/scan-report-{{ ansible_hostname }}.html \
      /usr/share/xml/scap/ssg/content/ssg-ubuntu2204-ds.xml
  register: compliance_scan
  changed_when: false
  
- name: Upload compliance results to S3
  amazon.aws.s3_object:
    bucket: compliance-reports
    object: "{{ ansible_date_time.date }}/{{ ansible_hostname }}-compliance.xml"
    src: "/tmp/scan-results-{{ ansible_hostname }}.xml"
    mode: put
```

### ✅ DO: Automate Secret Rotation

```yaml
# playbooks/rotate-secrets.yml
---
- name: Rotate database passwords across fleet
  hosts: db_servers
  gather_facts: no
  serial: 1  # One at a time to prevent outages
  
  tasks:
    - name: Generate new password
      set_fact:
        new_db_password: "{{ lookup('community.general.random_string', length=32, special=true) }}"
      no_log: true
    
    - name: Update password in database
      postgresql_query:
        query: "ALTER USER {{ db_user }} WITH PASSWORD '{{ new_db_password }}'"
      no_log: true
      
    - name: Update password in HashiCorp Vault
      community.hashi_vault.vault_kv2_write:
        url: "{{ vault_url }}"
        path: "database/{{ inventory_hostname }}"
        data:
          password: "{{ new_db_password }}"
          rotated_at: "{{ ansible_date_time.iso8601 }}"
      delegate_to: localhost
      no_log: true
      
    - name: Update application configurations
      lineinfile:
        path: /etc/myapp/database.conf
        regexp: '^password='
        line: "password={{ new_db_password }}"
      no_log: true
      delegate_to: "{{ item }}"
      loop: "{{ groups['app_servers'] }}"
      
    - name: Restart applications gracefully
      systemd:
        name: myapp
        state: restarted
      delegate_to: "{{ item }}"
      loop: "{{ groups['app_servers'] }}"
      throttle: 1
```

### ✅ DO: Implement Zero-Trust Network Policies

```yaml
# roles/zero-trust/tasks/main.yml
---
- name: Install and configure Cilium for network policies
  kubernetes.core.helm:
    name: cilium
    chart_ref: cilium/cilium
    release_namespace: kube-system
    values:
      hubble:
        enabled: true
        relay:
          enabled: true
        ui:
          enabled: true
      
      # Enable network policy enforcement
      policyEnforcementMode: "always"
      
      # Enable transparent encryption
      encryption:
        enabled: true
        type: wireguard
        
- name: Apply zero-trust network policies
  kubernetes.core.k8s:
    definition:
      apiVersion: cilium.io/v2
      kind: CiliumNetworkPolicy
      metadata:
        name: api-server-policy
        namespace: production
      spec:
        endpointSelector:
          matchLabels:
            app: api-server
        ingress:
          - fromEndpoints:
              - matchLabels:
                  app: frontend
            toPorts:
              - ports:
                  - port: "8080"
                    protocol: TCP
          - fromEndpoints:
              - matchLabels:
                  app: monitoring
            toPorts:
              - ports:
                  - port: "9090"  # Metrics only
                    protocol: TCP
        egress:
          - toEndpoints:
              - matchLabels:
                  app: database
            toPorts:
              - ports:
                  - port: "5432"
                    protocol: TCP
          - toServices:
              - serviceName: coredns
                namespace: kube-system
          - toCIDR:
              - 0.0.0.0/0
            toPorts:
              - ports:
                  - port: "443"
                    protocol: TCP
```

---

## 6. Cost Optimization Automation

Cloud costs can spiral out of control without automated governance.

### ✅ DO: Implement Automated Cost Controls

```python
#!/usr/bin/env python3
# scripts/cost-optimizer.py

import boto3
import google.cloud.compute_v1 as compute_v1
from azure.mgmt.compute import ComputeManagementClient
from datetime import datetime, timedelta
import asyncio
from prometheus_client import Gauge, push_to_gateway

# Prometheus metrics
instance_cost_metric = Gauge('instance_cost_per_hour', 'Instance cost per hour', 
                           ['provider', 'region', 'instance_type', 'instance_id'])

class MultiCloudCostOptimizer:
    def __init__(self):
        self.aws_ec2 = boto3.client('ec2')
        self.aws_ce = boto3.client('ce')  # Cost Explorer
        self.gcp_client = compute_v1.InstancesClient()
        self.azure_client = ComputeManagementClient(...)
        
    async def analyze_all_instances(self):
        """Analyze instances across all clouds for optimization opportunities"""
        tasks = [
            self.analyze_aws_instances(),
            self.analyze_gcp_instances(),
            self.analyze_azure_instances()
        ]
        results = await asyncio.gather(*tasks)
        return self.consolidate_recommendations(results)
    
    async def analyze_aws_instances(self):
        recommendations = []
        
        # Get all running instances
        response = self.aws_ec2.describe_instances(
            Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
        )
        
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instance_id = instance['InstanceId']
                instance_type = instance['InstanceType']
                
                # Get CPU utilization
                cloudwatch = boto3.client('cloudwatch')
                cpu_stats = cloudwatch.get_metric_statistics(
                    Namespace='AWS/EC2',
                    MetricName='CPUUtilization',
                    Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                    StartTime=datetime.utcnow() - timedelta(days=7),
                    EndTime=datetime.utcnow(),
                    Period=3600,
                    Statistics=['Average']
                )
                
                avg_cpu = sum(p['Average'] for p in cpu_stats['Datapoints']) / len(cpu_stats['Datapoints'])
                
                # Check if instance is underutilized
                if avg_cpu < 20:
                    # Get pricing for current and recommended instance
                    current_price = self.get_aws_instance_price(instance_type, instance['Placement']['AvailabilityZone'])
                    recommended_type = self.get_recommended_instance_type(instance_type, avg_cpu)
                    recommended_price = self.get_aws_instance_price(recommended_type, instance['Placement']['AvailabilityZone'])
                    
                    if recommended_price < current_price:
                        recommendations.append({
                            'provider': 'aws',
                            'instance_id': instance_id,
                            'current_type': instance_type,
                            'recommended_type': recommended_type,
                            'current_cost_hourly': current_price,
                            'recommended_cost_hourly': recommended_price,
                            'potential_savings_monthly': (current_price - recommended_price) * 24 * 30,
                            'reason': f'Low CPU utilization: {avg_cpu:.1f}%'
                        })
                
                # Check for Spot instance opportunities
                if not instance.get('SpotInstanceRequestId'):
                    spot_price = self.get_spot_price(instance_type, instance['Placement']['AvailabilityZone'])
                    if spot_price < current_price * 0.7:  # 30% savings threshold
                        recommendations.append({
                            'provider': 'aws',
                            'instance_id': instance_id,
                            'action': 'convert_to_spot',
                            'potential_savings_monthly': (current_price - spot_price) * 24 * 30,
                            'risk': 'medium'  # Spot instances can be terminated
                        })
                
                # Update Prometheus metrics
                instance_cost_metric.labels(
                    provider='aws',
                    region=instance['Placement']['AvailabilityZone'][:-1],
                    instance_type=instance_type,
                    instance_id=instance_id
                ).set(current_price)
                
        return recommendations
    
    def get_recommended_instance_type(self, current_type: str, cpu_usage: float) -> str:
        """Map to smaller instance type based on usage"""
        # Simplified example - real implementation would use detailed mapping
        size_map = {
            'm5.xlarge': 'm5.large',
            'm5.large': 'm5.medium',
            'm5.2xlarge': 'm5.xlarge',
            # ... more mappings
        }
        
        if cpu_usage < 20 and current_type in size_map:
            return size_map[current_type]
        return current_type
    
    async def apply_recommendations(self, recommendations: list, dry_run: bool = True):
        """Apply cost optimization recommendations"""
        for rec in recommendations:
            if rec.get('action') == 'resize':
                if dry_run:
                    print(f"Would resize {rec['instance_id']} from {rec['current_type']} to {rec['recommended_type']}")
                else:
                    await self.resize_instance(rec)
            elif rec.get('action') == 'convert_to_spot':
                if dry_run:
                    print(f"Would convert {rec['instance_id']} to Spot instance")
                else:
                    await self.convert_to_spot(rec)

# Ansible integration
if __name__ == '__main__':
    import json
    import sys
    
    optimizer = MultiCloudCostOptimizer()
    recommendations = asyncio.run(optimizer.analyze_all_instances())
    
    # Output for Ansible
    print(json.dumps({
        'changed': False,
        'recommendations': recommendations,
        'total_potential_savings': sum(r.get('potential_savings_monthly', 0) for r in recommendations)
    }))
    
    # Push metrics to Prometheus
    push_to_gateway('prometheus-pushgateway:9091', job='cost_optimizer')
```

### ✅ DO: Implement Automated Rightsizing

```yaml
# playbooks/rightsize-instances.yml
---
- name: Rightsize underutilized instances
  hosts: localhost
  gather_facts: no
  
  tasks:
    - name: Get cost optimization recommendations
      script: scripts/cost-optimizer.py
      register: cost_analysis
      
    - name: Parse recommendations
      set_fact:
        recommendations: "{{ (cost_analysis.stdout | from_json).recommendations }}"
        total_savings: "{{ (cost_analysis.stdout | from_json).total_potential_savings }}"
    
    - name: Send approval request for high-value optimizations
      uri:
        url: "{{ slack_webhook_url }}"
        method: POST
        body_format: json
        body:
          text: |
            Cost Optimization Opportunities Found:
            Total potential monthly savings: ${{ total_savings }}
            
            Top recommendations:
            {% for rec in recommendations[:5] %}
            • {{ rec.instance_id }} ({{ rec.provider }}): Save ${{ rec.potential_savings_monthly }}/month
              Current: {{ rec.current_type }} → Recommended: {{ rec.recommended_type }}
              Reason: {{ rec.reason }}
            {% endfor %}
            
            Reply with `/approve-resize` to apply recommendations
      when: total_savings | float > 1000  # Only notify for significant savings
    
    - name: Create resize playbook for approved changes
      template:
        src: resize-playbook.j2
        dest: /tmp/resize-{{ ansible_date_time.epoch }}.yml
      vars:
        resize_recommendations: "{{ recommendations | selectattr('action', 'equalto', 'resize') | list }}"
```

---

## 7. Disaster Recovery and Backup Automation

DR must be automated, tested, and measurable.

### ✅ DO: Implement Cross-Region Backup Automation

```yaml
# roles/backup-automation/tasks/main.yml
---
- name: Configure automated cross-region backups
  block:
    - name: Create backup schedule for databases
      amazon.aws.rds_snapshot:
        state: present
        db_instance_identifier: "{{ item }}"
        db_snapshot_identifier: "{{ item }}-{{ ansible_date_time.date }}-auto"
      loop: "{{ rds_instances }}"
      register: snapshots
      
    - name: Copy snapshots to DR region
      amazon.aws.rds_snapshot_copy:
        source_db_snapshot_identifier: "{{ item.db_snapshot_identifier }}"
        target_db_snapshot_identifier: "{{ item.db_snapshot_identifier }}-dr"
        source_region: "{{ primary_region }}"
        region: "{{ dr_region }}"
        kms_key_id: "{{ dr_kms_key }}"
      loop: "{{ snapshots.results }}"
      when: item.changed
      
    - name: Create EBS snapshots with tags
      amazon.aws.ec2_snapshot:
        volume_id: "{{ item.volume_id }}"
        description: "Automated backup - {{ ansible_date_time.iso8601 }}"
        tags:
          Name: "{{ item.name }}-backup"
          AutomatedBackup: "true"
          RetentionDays: "{{ item.retention_days | default(7) }}"
          Environment: "{{ environment }}"
      loop: "{{ volumes_to_backup }}"
      register: ebs_snapshots
      
    - name: Implement cross-region replication for critical data
      amazon.aws.s3_bucket:
        name: "{{ item.bucket }}"
        state: present
        versioning: yes
        lifecycle_configuration:
          rules:
            - id: transition-to-glacier
              status: enabled
              transitions:
                - storage_class: GLACIER
                  days: 90
            - id: expire-old-versions
              status: enabled
              noncurrent_version_expiration:
                days: 180
        replication_configuration:
          role: "{{ s3_replication_role_arn }}"
          rules:
            - id: replicate-to-dr
              status: enabled
              priority: 1
              destination:
                bucket: "arn:aws:s3:::{{ item.bucket }}-dr"
                storage_class: STANDARD_IA
                replication_time:
                  status: enabled
                  time:
                    minutes: 15
                metrics:
                  status: enabled
                  event_threshold:
                    minutes: 15
      loop: "{{ critical_s3_buckets }}"
```

### ✅ DO: Automate DR Testing

```yaml
# playbooks/dr-test.yml
---
- name: Automated DR environment validation
  hosts: localhost
  gather_facts: no
  
  vars:
    dr_test_id: "dr-test-{{ ansible_date_time.epoch }}"
    
  tasks:
    - name: Create temporary VPC in DR region
      amazon.aws.ec2_vpc_net:
        name: "{{ dr_test_id }}-vpc"
        cidr_block: 10.99.0.0/16
        region: "{{ dr_region }}"
      register: dr_vpc
      
    - name: Restore database from latest snapshot
      amazon.aws.rds_instance:
        id: "{{ dr_test_id }}-db"
        engine: postgres
        db_snapshot_identifier: "{{ latest_snapshot_id }}"
        region: "{{ dr_region }}"
        vpc_security_group_ids: ["{{ dr_security_group }}"]
        skip_final_snapshot: yes
        tags:
          DRTest: "{{ dr_test_id }}"
      register: dr_database
      
    - name: Deploy test application stack
      ansible.builtin.include_role:
        name: deploy-app
      vars:
        target_environment: dr_test
        database_endpoint: "{{ dr_database.endpoint.address }}"
        
    - name: Run automated smoke tests
      uri:
        url: "http://{{ dr_app_endpoint }}/api/health"
        method: GET
        status_code: 200
      retries: 10
      delay: 30
      
    - name: Run comprehensive DR validation tests
      script: scripts/dr-validation-suite.py --endpoint {{ dr_app_endpoint }}
      register: dr_test_results
      
    - name: Calculate DR metrics
      set_fact:
        rto_actual: "{{ (dr_test_end_time | int) - (dr_test_start_time | int) }}"
        rpo_validated: "{{ dr_test_results.stdout | from_json | json_query('metrics.data_lag_seconds') }}"
        
    - name: Update DR metrics in Prometheus
      uri:
        url: "http://prometheus-pushgateway:9091/metrics/job/dr_test"
        method: POST
        body: |
          # TYPE dr_rto_seconds gauge
          dr_rto_seconds{environment="{{ environment }}",region="{{ dr_region }}"} {{ rto_actual }}
          # TYPE dr_rpo_seconds gauge  
          dr_rpo_seconds{environment="{{ environment }}",region="{{ dr_region }}"} {{ rpo_validated }}
          # TYPE dr_test_success gauge
          dr_test_success{environment="{{ environment }}",region="{{ dr_region }}"} {{ dr_test_results.rc == 0 | ternary(1, 0) }}
        
    - name: Cleanup DR test resources
      block:
        - amazon.aws.rds_instance:
            id: "{{ dr_test_id }}-db"
            state: absent
            skip_final_snapshot: yes
            
        - amazon.aws.ec2_vpc_net:
            name: "{{ dr_test_id }}-vpc"
            state: absent
      tags: [cleanup]
      when: cleanup_after_test | default(true)
```

---

## 8. Advanced Patterns and Production Optimizations

### ✅ DO: Use Ansible Execution Strategies for Performance

```yaml
# ansible.cfg
[defaults]
# Use mitogen for 7-10x faster execution
strategy = mitogen_linear
# Or use free strategy for maximum parallelism
# strategy = free

# Increase forks for parallel execution
forks = 50

# Pipeline SSH for reduced latency
pipelining = True

# Batch fact gathering
gathering = smart
gather_subset = !hardware,!facter,!ohai

# Connection pooling
[ssh_connection]
ssh_args = -o ControlMaster=auto -o ControlPersist=600s -o ServerAliveInterval=60
control_path_dir = /tmp/.ansible-cp
```

### ✅ DO: Implement Blue-Green Deployments

```yaml
# playbooks/blue-green-deploy.yml
---
- name: Blue-Green Deployment
  hosts: localhost
  vars:
    app_version: "{{ lookup('env', 'APP_VERSION') }}"
    active_env: "{{ lookup('aws_ssm', '/app/active_environment', region=region) }}"
    inactive_env: "{{ 'green' if active_env == 'blue' else 'blue' }}"
    
  tasks:
    - name: Deploy to inactive environment
      ansible.builtin.include_tasks: deploy-tasks.yml
      vars:
        target_env: "{{ inactive_env }}"
        target_asg: "app-asg-{{ inactive_env }}"
        
    - name: Run smoke tests on inactive environment
      uri:
        url: "http://{{ inactive_env }}.internal.example.com/api/health"
        method: GET
        status_code: 200
      retries: 10
      delay: 30
      
    - name: Run integration test suite
      command: |
        pytest tests/integration \
          --base-url http://{{ inactive_env }}.internal.example.com \
          --junit-xml=/tmp/test-results.xml
      delegate_to: test-runner
      register: test_results
      
    - name: Gradually shift traffic (canary deployment)
      amazon.aws.route53:
        state: present
        zone: example.com
        record: api.example.com
        type: A
        alias: true
        alias_hosted_zone_id: "{{ alb_zone_id }}"
        weight: "{{ item }}"
        identifier: "{{ inactive_env }}-{{ item }}"
        alias_target:
          - "{{ inactive_env }}-alb.example.com"
      loop: [10, 25, 50, 75, 100]
      loop_control:
        pause: 300  # 5 minutes between shifts
        
    - name: Update active environment parameter
      amazon.aws.ssm_parameter:
        name: /app/active_environment
        value: "{{ inactive_env }}"
      when: test_results.rc == 0
```

### ✅ DO: Implement GitOps with Ansible

```yaml
# .gitlab-ci.yml
stages:
  - validate
  - test
  - deploy

variables:
  ANSIBLE_FORCE_COLOR: "true"
  ANSIBLE_HOST_KEY_CHECKING: "false"
  ANSIBLE_STDOUT_CALLBACK: "yaml"

validate:
  stage: validate
  image: quay.io/ansible/creator-ee:latest
  script:
    - ansible-lint playbooks/
    - ansible-playbook playbooks/site.yml --syntax-check
    - yamllint inventories/
    
security-scan:
  stage: validate
  image: aquasec/trivy:latest
  script:
    - trivy fs --security-checks vuln,config .
    - trivy config --severity HIGH,CRITICAL .

dry-run:
  stage: test
  image: quay.io/ansible/creator-ee:latest
  script:
    - ansible-playbook playbooks/site.yml -i inventories/staging --check --diff
  only:
    - merge_requests

deploy-staging:
  stage: deploy
  image: quay.io/ansible/creator-ee:latest
  script:
    - export ANSIBLE_VAULT_PASSWORD_FILE=/tmp/vault-pass
    - echo "$VAULT_PASSWORD" > /tmp/vault-pass
    - |
      ansible-playbook playbooks/site.yml \
        -i inventories/staging \
        --extra-vars "deployment_id=$CI_PIPELINE_ID"
  environment:
    name: staging
    url: https://staging.example.com
  only:
    - develop

deploy-production:
  stage: deploy
  image: quay.io/ansible/creator-ee:latest
  script:
    - export ANSIBLE_VAULT_PASSWORD_FILE=/tmp/vault-pass
    - echo "$VAULT_PASSWORD_PROD" > /tmp/vault-pass
    - |
      ansible-playbook playbooks/site.yml \
        -i inventories/production \
        --extra-vars "deployment_id=$CI_PIPELINE_ID" \
        --extra-vars "require_approval=true"
  environment:
    name: production
    url: https://api.example.com
  when: manual
  only:
    - main
```

### ✅ DO: Implement Progressive Delivery

```yaml
# roles/progressive-delivery/tasks/main.yml
---
- name: Deploy with feature flags
  block:
    - name: Update application with new version
      kubernetes.core.k8s:
        state: present
        definition:
          apiVersion: apps/v1
          kind: Deployment
          metadata:
            name: "{{ app_name }}"
            namespace: "{{ namespace }}"
            annotations:
              flagger.app/config: |
                analysis:
                  interval: 30s
                  threshold: 10
                  maxWeight: 50
                  stepWeight: 5
                  metrics:
                    - name: error-rate
                      thresholdRange:
                        max: 1
                      interval: 1m
                    - name: latency
                      thresholdRange:
                        max: 500
                      interval: 30s
          spec:
            template:
              spec:
                containers:
                  - name: app
                    image: "{{ app_image }}:{{ app_version }}"
                    env:
                      - name: FEATURE_FLAGS_ENDPOINT
                        value: "http://unleash:4242/api"
                      - name: ENABLE_CANARY_FEATURES
                        value: "{{ enable_canary }}"
                        
    - name: Configure feature flags in Unleash
      uri:
        url: "http://unleash:4242/api/admin/features"
        method: POST
        headers:
          Authorization: "{{ unleash_api_key }}"
        body_format: json
        body:
          name: "new_feature_{{ app_version | replace('.', '_') }}"
          enabled: false
          strategies:
            - name: gradualRolloutUserId
              parameters:
                percentage: "10"
                groupId: "{{ app_name }}"
            - name: userWithId
              parameters:
                userIds: "{{ beta_users | join(',') }}"
```

---

## 9. Multi-Cloud Service Mesh

### ✅ DO: Deploy Istio for Advanced Traffic Management

```yaml
# playbooks/service-mesh.yml
---
- name: Deploy Istio across multi-cloud clusters
  hosts: kubernetes_masters
  
  tasks:
    - name: Install Istio control plane
      kubernetes.core.helm:
        name: istio-control-plane
        chart_ref: istio/base
        release_namespace: istio-system
        create_namespace: yes
        
    - name: Configure multi-cluster mesh
      kubernetes.core.k8s:
        definition:
          apiVersion: install.istio.io/v1alpha1
          kind: IstioOperator
          metadata:
            name: control-plane
          spec:
            values:
              pilot:
                env:
                  PILOT_ENABLE_WORKLOAD_ENTRY_AUTOREGISTRATION: true
              global:
                meshID: mesh1
                multiCluster:
                  clusterName: "{{ cluster_name }}"
                network: "{{ cluster_network }}"
            components:
              egressGateways:
                - name: istio-egressgateway
                  enabled: true
                  k8s:
                    service:
                      type: LoadBalancer
                      
    - name: Create cross-cluster service discovery
      kubernetes.core.k8s:
        definition:
          apiVersion: networking.istio.io/v1beta1
          kind: ServiceEntry
          metadata:
            name: cross-cluster-services
            namespace: istio-system
          spec:
            hosts:
              - "*.global"
            location: MESH_EXTERNAL
            ports:
              - number: 80
                name: http
                protocol: HTTP
            resolution: DNS
            endpoints:
              - address: cluster2-gateway.example.com
                priority: 5
                weight: 50
              - address: cluster3-gateway.example.com  
                priority: 10
                weight: 50
```

---

## 10. Chaos Engineering Integration

### ✅ DO: Automate Chaos Experiments

```yaml
# roles/chaos-engineering/tasks/main.yml
---
- name: Deploy Litmus Chaos
  kubernetes.core.helm:
    name: litmus
    chart_ref: litmuschaos/litmus
    release_namespace: litmus
    
- name: Create automated chaos experiments
  kubernetes.core.k8s:
    definition:
      apiVersion: litmuschaos.io/v1alpha1
      kind: ChaosSchedule
      metadata:
        name: scheduled-pod-failure
        namespace: litmus
      spec:
        schedule:
          type: cron
          cron:
            expression: "0 10 * * 1-5"  # Weekdays at 10 AM
          minChaosInterval: "2h"
        engineTemplateSpec:
          jobCleanUpPolicy: retain
          experiments:
            - name: pod-cpu-hog
              spec:
                components:
                  env:
                    - name: TARGET_PODS
                      value: 'app=critical-service'
                    - name: CPU_CORES
                      value: '1'
                    - name: TOTAL_CHAOS_DURATION
                      value: '300'  # 5 minutes
                probe:
                  - name: check-service-availability
                    type: httpProbe
                    httpProbe/inputs:
                      url: http://service.namespace:8080/health
                      insecureSkipVerify: false
                      responseTimeout: 10
                      criteria: "=="
                      responseCode: "200"
                    runProperties:
                      probeTimeout: 30
                      interval: 10
                      attempt: 10
```

---

## 11. AIOps and Self-Healing

### ✅ DO: Implement Self-Healing Mechanisms

```python
#!/usr/bin/env python3
# scripts/self-healing-controller.py

import asyncio
import aiohttp
from prometheus_client.parser import text_string_to_metric_families
from kubernetes import client, config
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SelfHealingController:
    def __init__(self):
        config.load_incluster_config()  # When running in K8s
        self.k8s_apps_v1 = client.AppsV1Api()
        self.k8s_core_v1 = client.CoreV1Api()
        self.prometheus_url = "http://prometheus:9090"
        self.healing_actions = []
        
    async def monitor_and_heal(self):
        """Main monitoring loop"""
        while True:
            try:
                issues = await self.detect_issues()
                for issue in issues:
                    await self.attempt_healing(issue)
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in healing loop: {e}")
                await asyncio.sleep(60)
                
    async def detect_issues(self):
        """Query Prometheus for anomalies"""
        issues = []
        
        queries = {
            'high_memory_pressure': 'node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes < 0.1',
            'pod_crash_loop': 'kube_pod_container_status_restarts_total > 5',
            'high_error_rate': 'rate(http_requests_total{status=~"5.."}[5m]) > 0.1',
            'disk_pressure': 'node_filesystem_avail_bytes{fstype!~"tmpfs|fuse.lxcfs"} / node_filesystem_size_bytes < 0.1',
            'certificate_expiry': 'certmanager_certificate_expiration_timestamp_seconds - time() < 86400 * 7'
        }
        
        async with aiohttp.ClientSession() as session:
            for issue_type, query in queries.items():
                url = f"{self.prometheus_url}/api/v1/query"
                params = {'query': query}
                
                async with session.get(url, params=params) as resp:
                    data = await resp.json()
                    
                    if data['status'] == 'success' and data['data']['result']:
                        for result in data['data']['result']:
                            issues.append({
                                'type': issue_type,
                                'metric': result['metric'],
                                'value': float(result['value'][1]),
                                'timestamp': datetime.fromtimestamp(result['value'][0])
                            })
                            
        return issues
    
    async def attempt_healing(self, issue):
        """Execute healing action based on issue type"""
        logger.info(f"Attempting to heal: {issue['type']} - {issue['metric']}")
        
        healing_strategies = {
            'high_memory_pressure': self.heal_memory_pressure,
            'pod_crash_loop': self.heal_crash_loop,
            'high_error_rate': self.heal_high_error_rate,
            'disk_pressure': self.heal_disk_pressure,
            'certificate_expiry': self.heal_certificate_expiry
        }
        
        if issue['type'] in healing_strategies:
            success = await healing_strategies[issue['type']](issue)
            
            # Record healing action
            self.healing_actions.append({
                'timestamp': datetime.utcnow(),
                'issue_type': issue['type'],
                'success': success,
                'details': issue
            })
            
            # Send notification
            await self.notify_healing_action(issue, success)
            
    async def heal_memory_pressure(self, issue):
        """Handle memory pressure by evicting pods or scaling"""
        node_name = issue['metric'].get('node', 'unknown')
        
        try:
            # First, try to evict non-critical pods
            pods = self.k8s_core_v1.list_pod_for_all_namespaces(
                field_selector=f"spec.nodeName={node_name}"
            )
            
            for pod in sorted(pods.items, key=lambda p: p.metadata.creation_timestamp):
                if 'critical' not in pod.metadata.labels.get('tier', ''):
                    logger.info(f"Evicting pod {pod.metadata.name} from {node_name}")
                    
                    eviction = client.V1Eviction(
                        metadata=client.V1ObjectMeta(
                            name=pod.metadata.name,
                            namespace=pod.metadata.namespace
                        )
                    )
                    
                    self.k8s_core_v1.create_namespaced_pod_eviction(
                        name=pod.metadata.name,
                        namespace=pod.metadata.namespace,
                        body=eviction
                    )
                    
                    await asyncio.sleep(5)  # Wait between evictions
                    
                    # Check if pressure relieved
                    if await self.check_memory_pressure_relieved(node_name):
                        return True
                        
            # If still under pressure, trigger node autoscaling
            return await self.trigger_node_autoscaling()
            
        except Exception as e:
            logger.error(f"Failed to heal memory pressure: {e}")
            return False
            
    async def heal_crash_loop(self, issue):
        """Handle crash-looping pods"""
        namespace = issue['metric'].get('namespace')
        pod_name = issue['metric'].get('pod')
        
        try:
            # Get pod details
            pod = self.k8s_core_v1.read_namespaced_pod(pod_name, namespace)
            
            # Check recent events
            events = self.k8s_core_v1.list_namespaced_event(
                namespace,
                field_selector=f"involvedObject.name={pod_name}"
            )
            
            # Common fixes based on event messages
            for event in events.items:
                if 'OOMKilled' in event.message:
                    # Increase memory limits
                    return await self.increase_pod_resources(pod, 'memory')
                elif 'CrashLoopBackOff' in event.message and 'probe failed' in event.message:
                    # Adjust health check timing
                    return await self.adjust_pod_probes(pod)
                    
            # If no specific fix, try pod restart
            self.k8s_core_v1.delete_namespaced_pod(pod_name, namespace)
            return True
            
        except Exception as e:
            logger.error(f"Failed to heal crash loop: {e}")
            return False

# Ansible task to deploy self-healing
"""
- name: Deploy self-healing controller
  kubernetes.core.k8s:
    definition:
      apiVersion: v1
      kind: ConfigMap
      metadata:
        name: self-healing-script
        namespace: monitoring
      data:
        controller.py: |
{{ lookup('file', 'scripts/self-healing-controller.py') | indent(10) }}

- name: Create self-healing deployment
  kubernetes.core.k8s:
    definition:
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: self-healing-controller
        namespace: monitoring
      spec:
        replicas: 1
        selector:
          matchLabels:
            app: self-healing
        template:
          metadata:
            labels:
              app: self-healing
          spec:
            serviceAccountName: self-healing-controller
            containers:
              - name: controller
                image: python:3.12-slim
                command: ["python", "/scripts/controller.py"]
                volumeMounts:
                  - name: script
                    mountPath: /scripts
            volumes:
              - name: script
                configMap:
                  name: self-healing-script
"""
```

---

## 12. Centralized Logging with Loki

### ✅ DO: Implement Structured Logging Pipeline

```yaml
# roles/centralized-logging/tasks/main.yml
---
- name: Deploy Loki for log aggregation
  kubernetes.core.helm:
    name: loki
    chart_ref: grafana/loki-stack
    release_namespace: monitoring
    values:
      loki:
        config:
          auth_enabled: false
          
          ingester:
            chunk_idle_period: 5m
            chunk_retain_period: 30s
            max_transfer_retries: 0
            
          limits_config:
            enforce_metric_name: false
            reject_old_samples: true
            reject_old_samples_max_age: 168h
            ingestion_rate_mb: 100
            ingestion_burst_size_mb: 200
            
          storage_config:
            aws:
              s3: s3://{{ loki_s3_bucket }}
              region: "{{ aws_region }}"
            boltdb_shipper:
              active_index_directory: /loki/index
              cache_location: /loki/cache
              shared_store: s3
              
          compactor:
            working_directory: /loki/compactor
            shared_store: s3
            compaction_interval: 10m
            retention_enabled: true
            retention_delete_delay: 2h
            retention_delete_worker_count: 150
            
      promtail:
        config:
          clients:
            - url: http://loki:3100/loki/api/v1/push
              
          scrape_configs:
            - job_name: kubernetes-pods
              kubernetes_sd_configs:
                - role: pod
              relabel_configs:
                # Only scrape pods with annotation
                - source_labels: [__meta_kubernetes_pod_annotation_promtail_io_scrape]
                  action: keep
                  regex: true
                  
              pipeline_stages:
                # Parse JSON logs
                - json:
                    expressions:
                      level: level
                      timestamp: timestamp
                      message: message
                      trace_id: trace_id
                      
                # Extract additional fields
                - regex:
                    expression: '.*(?P<status_code>\d{3}).*'
                    
                # Add labels
                - labels:
                    level:
                    status_code:
                    
                # Set timestamp
                - timestamp:
                    source: timestamp
                    format: RFC3339Nano
                    
- name: Configure log shipping from legacy systems
  ansible.builtin.template:
    src: fluent-bit.conf.j2
    dest: /etc/fluent-bit/fluent-bit.conf
  notify: restart fluent-bit
  
- name: Create Grafana log dashboards
  kubernetes.core.k8s:
    definition:
      apiVersion: v1
      kind: ConfigMap
      metadata:
        name: log-dashboards
        namespace: monitoring
      data:
        error-logs.json: |
          {{ lookup('file', 'dashboards/error-logs.json') | to_json }}
```

---

## 13. Security Scanning and Compliance

### ✅ DO: Continuous Security Scanning

```yaml
# playbooks/security-scanning.yml
---
- name: Comprehensive security scanning pipeline
  hosts: all
  gather_facts: yes
  
  tasks:
    - name: Install and run Lynis security audit
      block:
        - name: Install Lynis
          package:
            name: lynis
            state: present
            
        - name: Run Lynis audit
          command: lynis audit system --quiet --no-colors
          register: lynis_output
          changed_when: false
          
        - name: Parse Lynis score
          set_fact:
            hardening_index: "{{ lynis_output.stdout | regex_search('Hardening index : \\[([0-9]+)\\]', '\\1') | first }}"
            
        - name: Push metrics to Prometheus
          uri:
            url: "http://prometheus-pushgateway:9091/metrics/job/security_scan/instance/{{ inventory_hostname }}"
            method: POST
            body: |
              # TYPE security_hardening_index gauge
              security_hardening_index {{ hardening_index }}
              # TYPE security_scan_timestamp gauge
              security_scan_timestamp {{ ansible_date_time.epoch }}
              
    - name: Container image vulnerability scanning
      when: "'kubernetes' in group_names"
      block:
        - name: Get all running images
          kubernetes.core.k8s_info:
            api_version: v1
            kind: Pod
            namespace: all
          register: all_pods
          
        - name: Extract unique images
          set_fact:
            unique_images: "{{ all_pods.resources | map(attribute='spec.containers') | flatten | map(attribute='image') | unique }}"
            
        - name: Scan images with Trivy
          command: |
            trivy image {{ item }} --format json --quiet
          loop: "{{ unique_images }}"
          register: trivy_results
          changed_when: false
          
        - name: Process vulnerability results
          set_fact:
            critical_vulns: "{{ trivy_results.results | map(attribute='stdout') | map('from_json') | selectattr('Vulnerabilities', 'defined') | map(attribute='Vulnerabilities') | flatten | selectattr('Severity', 'equalto', 'CRITICAL') | list }}"
            
        - name: Alert on critical vulnerabilities
          uri:
            url: "{{ alert_webhook }}"
            method: POST
            body_format: json
            body:
              text: "Critical vulnerabilities found in {{ inventory_hostname }}"
              vulnerabilities: "{{ critical_vulns }}"
          when: critical_vulns | length > 0
```

---

## 14. Production Troubleshooting Playbooks

### ✅ DO: Create Runbook Automation

```yaml
# playbooks/troubleshooting/high-cpu.yml
---
- name: Automated high CPU troubleshooting
  hosts: "{{ target_host | default('all') }}"
  gather_facts: no
  
  tasks:
    - name: Gather system information
      setup:
        gather_subset:
          - hardware
          - virtual
          
    - name: Identify top CPU consumers
      shell: |
        ps aux --sort=-%cpu | head -20
      register: top_processes
      
    - name: Collect detailed process information
      shell: |
        for pid in $(ps aux --sort=-%cpu | awk 'NR>1 && $3>50 {print $2}' | head -5); do
          echo "=== Process $pid ==="
          ps -p $pid -o pid,ppid,user,%cpu,%mem,vsz,rss,tty,stat,start,time,comm,args
          echo "=== Stack trace ==="
          if [ -f /proc/$pid/stack ]; then
            cat /proc/$pid/stack
          fi
          echo "=== Open files ==="
          lsof -p $pid 2>/dev/null | head -20
          echo
        done
      register: process_details
      
    - name: Check for CPU throttling
      shell: |
        if [ -f /sys/fs/cgroup/cpu/cpu.stat ]; then
          cat /sys/fs/cgroup/cpu/cpu.stat | grep throttled
        fi
      register: cpu_throttling
      
    - name: Generate diagnostics report
      template:
        src: cpu-diagnostics-report.j2
        dest: "/tmp/cpu-diagnostics-{{ inventory_hostname }}-{{ ansible_date_time.epoch }}.html"
      vars:
        system_info: "{{ ansible_facts }}"
        top_procs: "{{ top_processes.stdout }}"
        proc_details: "{{ process_details.stdout }}"
        throttling: "{{ cpu_throttling.stdout }}"
        
    - name: Upload report to S3
      amazon.aws.s3_object:
        bucket: diagnostics-reports
        object: "cpu/{{ ansible_date_time.date }}/{{ inventory_hostname }}-{{ ansible_date_time.epoch }}.html"
        src: "/tmp/cpu-diagnostics-{{ inventory_hostname }}-{{ ansible_date_time.epoch }}.html"
        mode: put
        
    - name: Suggest remediation
      debug:
        msg: |
          {% if ansible_processor_vcpus < 4 %}
          - Consider scaling up instance (current: {{ ansible_processor_vcpus }} vCPUs)
          {% endif %}
          {% if 'throttled_time' in cpu_throttling.stdout %}
          - CPU throttling detected - check cgroup limits
          {% endif %}
          {% if top_processes.stdout | regex_search('python.*runaway|java.*heap') %}
          - Possible application memory leak causing high CPU
          {% endif %}
```

---

## 15. Advanced Monitoring Queries and Dashboards

### ✅ DO: Create Business-Focused Metrics

```yaml
# monitoring/prometheus/rules/business-metrics.yml
groups:
  - name: business_metrics
    interval: 60s
    rules:
      # Calculate infrastructure cost per transaction
      - record: business:cost_per_transaction
        expr: |
          sum(instance_cost_per_hour) / sum(rate(http_requests_total{job="api"}[5m])) / 12
          
      # Revenue per compute dollar
      - record: business:revenue_per_compute_dollar
        expr: |
          sum(rate(transactions_revenue_total[1h])) / sum(instance_cost_per_hour)
          
      # Infrastructure efficiency score
      - record: infrastructure:efficiency_score
        expr: |
          (
            avg(1 - rate(node_cpu_seconds_total{mode="idle"}[5m])) * 0.3 +  # CPU utilization (30%)
            avg(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 0.3 +  # Memory utilization (30%)
            avg(rate(node_network_transmit_bytes_total[5m]) / node_network_speed_bytes) * 0.2 +  # Network utilization (20%)
            (1 - sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))) * 0.2  # Success rate (20%)
          ) * 100
          
      # Mean time to recovery (MTTR)
      - record: sre:mttr_minutes
        expr: |
          (
            time() - max(ALERTS{alertstate="firing",severity="critical"} offset 5m) unless ALERTS{alertstate="firing",severity="critical"}
          ) / 60
```

### ✅ DO: Executive Dashboard Configuration

```json
// monitoring/grafana/dashboards/executive-overview.json
{
  "dashboard": {
    "title": "Executive Infrastructure Overview",
    "panels": [
      {
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
        "title": "Monthly Cloud Spend Trajectory",
        "targets": [
          {
            "expr": "sum by (provider) (instance_cost_per_hour) * 24 * 30",
            "legendFormat": "{{ provider }} Actual"
          },
          {
            "expr": "predict_linear(sum by (provider) (instance_cost_per_hour)[7d], 30*86400) * 24 * 30",
            "legendFormat": "{{ provider }} Projected"
          }
        ],
        "options": {
          "tooltip": { "mode": "multi" },
          "legend": { "displayMode": "table", "calcs": ["last"] }
        },
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD",
            "custom": {
              "lineWidth": 2,
              "fillOpacity": 10
            }
          }
        }
      },
      {
        "gridPos": { "h": 8, "w": 6, "x": 12, "y": 0 },
        "title": "Infrastructure ROI",
        "type": "stat",
        "targets": [
          {
            "expr": "business:revenue_per_compute_dollar",
            "instant": true
          }
        ],
        "options": {
          "reduceOptions": {
            "values": false,
            "calcs": ["lastNotNull"]
          },
          "textMode": "value_and_name"
        },
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD",
            "decimals": 2,
            "thresholds": {
              "mode": "absolute",
              "steps": [
                { "color": "red", "value": 0 },
                { "color": "yellow", "value": 5 },
                { "color": "green", "value": 10 }
              ]
            }
          }
        }
      }
    ],
    "schemaVersion": 38,
    "version": 1,
    "timezone": "browser",
    "refresh": "1m"
  }
}
```

---

## Conclusion

This guide provides a comprehensive framework for modern DevOps practices in 2025. Key takeaways:

1. **Automation First**: Every manual process should be automated, from deployments to security scanning
2. **Multi-Cloud Native**: Design for portability across providers while leveraging cloud-specific features
3. **Security as Code**: Compliance and security must be automated and continuously validated
4. **Observable by Default**: Every component should emit metrics, logs, and traces
5. **Cost-Aware**: Infrastructure decisions should consider cost implications automatically
6. **Self-Healing**: Systems should detect and remediate common issues without human intervention

Remember: **The best infrastructure is invisible to your users and developers**. It should just work, scale automatically, recover from failures, and provide insights when needed.

For the latest updates and community contributions, visit:
- [Ansible Collections Galaxy](https://galaxy.ansible.com)
- [Grafana Community Dashboards](https://grafana.com/grafana/dashboards/)
- [CNCF Landscape](https://landscape.cncf.io/)

Stay curious, automate everything, and always be monitoring! 🚀