# The Definitive Guide to Kubernetes Model Inference with KServe, Seldon Core, and BentoML (mid-2025)

This guide synthesizes production-grade best practices for deploying, scaling, and managing ML model inference on Kubernetes. It moves beyond basic tutorials to provide battle-tested patterns for high-performance, cost-efficient model serving at scale.

## Prerequisites & Core Technology Stack

Ensure your environment meets these requirements:
- **Kubernetes 1.30+** with NVIDIA GPU Operator 25.x (if using GPUs)
- **KServe 1.13+** (formerly KFServing) 
- **Seldon Core 2.9+** or **BentoML 1.3+**
- **Istio 1.23+** or **KNative Serving 1.16+** for advanced traffic management
- **KEDA 2.16+** for advanced autoscaling
- **Prometheus 2.55+** & **Grafana 11.x** for observability

### Initial Cluster Configuration

```yaml
# cluster-config.yaml - Essential add-ons for ML inference
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-cluster-config
  namespace: kube-system
data:
  gpu-time-slicing: |
    version: v1
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4  # Allow 4 pods per GPU for inference
  
  node-feature-discovery: |
    sources:
      pci:
        deviceClassWhitelist:
          - "0300"  # GPUs
          - "0302"  # 3D controllers
      cpu:
        cpuid:
          attributeWhitelist:
            - "AVX512*"  # For optimized inference
```

## 1. Architectural Foundations

### The Three-Layer Inference Architecture

Modern ML inference requires separation of concerns across three distinct layers:

```
┌─────────────────────────────────────────────────────────┐
│                   Gateway Layer                          │
│  (Istio/Kong) - Rate limiting, auth, TLS termination   │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                 Inference Router Layer                   │
│  (KServe/Seldon) - Model routing, A/B testing, canary  │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                  Model Server Layer                      │
│  (Triton/TorchServe/BentoML) - Actual model execution  │
└─────────────────────────────────────────────────────────┘
```

### ✅ DO: Choose the Right Inference Platform

| Platform | Best For | Avoid When |
|----------|----------|------------|
| **KServe** | Standard model serving with autoscaling, multi-framework support | You need complex pre/post processing pipelines |
| **Seldon Core** | Complex inference graphs, custom Python transformers | Simple single-model serving (overkill) |
| **BentoML** | Python-first teams, custom serving logic, built-in model registry | You need native multi-model serving |

### Model Storage Architecture

```yaml
# model-storage-class.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: model-storage-fast
provisioner: csi.hpe.com/nimble
parameters:
  fsType: xfs
  storagePool: ssd-perf
  perfPolicy: "DockerWorker"  # Optimized for container workloads
  # Enable compression for model artifacts
  dataCompression: "true"
  # 2025: Most CSI drivers now support instant snapshots
  enableVolumeSnapshots: "true"
reclaimPolicy: Retain
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
```

## 2. KServe Advanced Patterns

### ✅ DO: Use InferenceService v1beta1 with Proper Resource Limits

The v1beta1 API is now stable and recommended for production use.

```yaml
# inference-service-advanced.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama3-70b
  namespace: production
  annotations:
    # 2025: Vertical pod autoscaling now GA
    vpa.kserve.io/mode: "Auto"
    # Enable request batching
    serving.kserve.io/enable-batcher: "true"
spec:
  predictor:
    # Model parallelism across GPUs
    parallelism: 2
    
    # Advanced autoscaling configuration
    minReplicas: 1
    maxReplicas: 10
    scaleTarget: 50  # Target concurrency per replica
    scaleMetric: "concurrency"  # or "rps", "cpu", "memory", "gpu"
    
    # Container spec with GPU
    containers:
    - name: kserve-container
      image: nvcr.io/nvidia/tritonserver:25.06-py3
      
      # Resource management is CRITICAL
      resources:
        requests:
          cpu: "4"
          memory: "64Gi"
          nvidia.com/gpu: "1"
        limits:
          cpu: "8"
          memory: "128Gi"
          nvidia.com/gpu: "1"
      
      # GPU-specific environment
      env:
      - name: NVIDIA_VISIBLE_DEVICES
        value: "all"
      - name: CUDA_MEMORY_FRACTION
        value: "0.95"  # Reserve 5% for system
      - name: TF_FORCE_GPU_ALLOW_GROWTH
        value: "false"  # Pre-allocate for consistent latency
      - name: ORT_TENSORRT_FP16_ENABLE
        value: "1"  # Enable FP16 for 2x throughput
      
      # Probes for production reliability
      readinessProbe:
        httpGet:
          path: /v2/health/ready
          port: 8080
        initialDelaySeconds: 30
        periodSeconds: 10
        timeoutSeconds: 5
        successThreshold: 1
        failureThreshold: 3
      
      livenessProbe:
        httpGet:
          path: /v2/health/live
          port: 8080
        initialDelaySeconds: 60
        periodSeconds: 30
        timeoutSeconds: 10
        failureThreshold: 3
      
      # 2025: Startup probe prevents premature kills during model loading
      startupProbe:
        httpGet:
          path: /v2/health/ready
          port: 8080
        initialDelaySeconds: 0
        periodSeconds: 10
        timeoutSeconds: 5
        failureThreshold: 30  # 5 minutes for large models
    
    # Model storage configuration
    storageUri: "s3://models/llama3-70b"
    
    # Advanced storage options
    storageSpec:
      storageClassName: "model-storage-fast"
      accessMode: ReadOnlyMany  # Share across pods
      resources:
        requests:
          storage: 200Gi
    
    # Transformer for pre/post-processing
    transformer:
      containers:
      - name: transformer
        image: myregistry/llama-transformer:latest
        resources:
          requests:
            cpu: "2"
            memory: "8Gi"
        
        # Custom transformer configuration
        env:
        - name: MAX_BATCH_SIZE
          value: "32"
        - name: TOKENIZER_PARALLELISM
          value: "true"
```

### Multi-Model Serving with Model Mesh

KServe ModelMesh enables efficient multi-model serving on shared infrastructure:

```yaml
# modelmesh-serving.yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ServingRuntime
metadata:
  name: mlserver-sklearn
spec:
  supportedModelFormats:
    - name: sklearn
      version: "1.4"
      autoSelect: true
  
  containers:
  - name: mlserver
    image: seldonio/mlserver:1.7.0
    env:
    - name: MLSERVER_MODELS_DIR
      value: "/models"
    - name: MLSERVER_GRPC_PORT
      value: "8001"
    - name: MLSERVER_HTTP_PORT
      value: "8002"
    - name: MLSERVER_METRICS_PORT
      value: "8082"
    
    resources:
      requests:
        cpu: "1"
        memory: "2Gi"
      limits:
        cpu: "2"
        memory: "4Gi"
    
    # Model mesh specific
    volumeMounts:
    - name: model-cache
      mountPath: /tmp/models
    
  volumes:
  - name: model-cache
    emptyDir:
      sizeLimit: 50Gi
      medium: Memory  # Use RAM for hot models
  
  multiModel: true
  
  grpcDataEndpoint: "port:8001"
  grpcEndpoint: "port:8085"  # Management endpoint
```

### ❌ DON'T: Use Default Autoscaling Metrics

Default CPU-based autoscaling is terrible for ML workloads. Configure custom metrics:

```yaml
# custom-autoscaling.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server
  
  minReplicas: 2
  maxReplicas: 100
  
  # Advanced scaling behavior (2025 features)
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Percent
        value: 100  # Double pods
        periodSeconds: 60
      - type: Pods
        value: 5    # Or add 5 pods
        periodSeconds: 60
      selectPolicy: Max  # Choose most aggressive
    
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 min cooldown
      policies:
      - type: Percent
        value: 50
        periodSeconds: 120
  
  metrics:
  # GPU utilization (most important for ML)
  - type: Pods
    pods:
      metric:
        name: gpu_utilization_percentage
      target:
        type: AverageValue
        averageValue: "80"
  
  # Model-specific latency
  - type: Pods
    pods:
      metric:
        name: model_inference_duration_p99_seconds
      target:
        type: AverageValue
        averageValue: "0.1"  # 100ms P99
  
  # Queue depth from Istio
  - type: External
    external:
      metric:
        name: istio_request_duration_milliseconds_bucket
        selector:
          matchLabels:
            destination_service_name: "model-service"
      target:
        type: AverageValue
        averageValue: "50"
```

## 3. Seldon Core Production Patterns

### Complex Inference Graphs

Seldon excels at multi-step inference pipelines:

```yaml
# seldon-inference-graph.yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: fraud-detection-pipeline
spec:
  name: fraud-detection
  predictors:
  - name: default
    graph:
      name: preprocessor
      type: TRANSFORMER
      endpoint:
        type: REST
      children:
      - name: fraud-ensemble
        type: COMBINER
        endpoint:
          type: GRPC  # Better performance
        children:
        - name: xgboost-model
          type: MODEL
          modelUri: s3://models/fraud/xgboost
          implementation: XGBOOST_SERVER
          envSecretRefName: aws-secret
        - name: neural-model  
          type: MODEL
          modelUri: s3://models/fraud/neural
          implementation: TENSORFLOW_SERVER
        - name: isolation-forest
          type: MODEL
          modelUri: s3://models/fraud/isolation
          implementation: SKLEARN_SERVER
      
      # Custom combiner logic
      parameters:
      - name: combiner_type
        type: STRING
        value: "weighted_average"
      - name: weights
        type: JSON
        value: "[0.4, 0.4, 0.2]"
    
    # Component-specific resources
    componentSpecs:
    - spec:
        containers:
        - name: preprocessor
          image: myregistry/fraud-preprocessor:v2
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
        
        - name: neural-model
          resources:
            requests:
              nvidia.com/gpu: "1"
              memory: "16Gi"
    
    # Advanced traffic management
    traffic: 100
    replicas: 3
    
    # Canary deployment configuration
    annotations:
      seldon.io/canary-period: "3600"  # 1 hour
      seldon.io/canary-metric: "accuracy"
      seldon.io/canary-threshold: "0.95"
```

### Custom Python Model with Async Support

```python
# fraud_preprocessor.py
import numpy as np
import asyncio
from typing import Dict, List, Union, Any
import aioredis
from seldon_core.user_model import SeldonComponent
import logging

logger = logging.getLogger(__name__)

class FraudPreprocessor(SeldonComponent):
    def __init__(self):
        super().__init__()
        self.redis_client = None
        self.feature_stats = {}
        
    async def load(self):
        """Async initialization for connection pooling"""
        self.redis_client = await aioredis.create_redis_pool(
            'redis://redis-master:6379',
            maxsize=10
        )
        
        # Load feature statistics for normalization
        self.feature_stats = await self._load_feature_stats()
        
    async def predict(
        self, 
        X: np.ndarray, 
        features_names: List[str] = None,
        meta: Dict = None
    ) -> Union[np.ndarray, Dict]:
        """Async prediction with Redis feature lookup"""
        
        # Parallel feature enrichment
        enrichment_tasks = []
        for idx, row in enumerate(X):
            user_id = str(row[0])  # Assuming first column is user_id
            enrichment_tasks.append(
                self._enrich_user_features(user_id, idx)
            )
        
        # Wait for all enrichments
        enriched_features = await asyncio.gather(*enrichment_tasks)
        
        # Combine original and enriched features
        X_enriched = np.hstack([X, np.array(enriched_features)])
        
        # Normalize using pre-computed stats
        X_normalized = self._normalize_features(X_enriched)
        
        # Add metadata for downstream models
        return {
            "data": {"ndarray": X_normalized.tolist()},
            "meta": {
                **meta,
                "feature_version": "v2",
                "enrichment_timestamp": time.time()
            }
        }
    
    async def _enrich_user_features(
        self, 
        user_id: str, 
        idx: int
    ) -> List[float]:
        """Fetch user features from Redis with caching"""
        cache_key = f"user_features:{user_id}"
        
        # Try cache first
        cached = await self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Compute features (example)
        features = [
            await self._get_transaction_velocity(user_id),
            await self._get_merchant_risk_score(user_id),
            await self._get_device_fingerprint_score(user_id)
        ]
        
        # Cache for 5 minutes
        await self.redis_client.setex(
            cache_key, 
            300, 
            json.dumps(features)
        )
        
        return features
    
    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Fast normalization using pre-computed stats"""
        return (X - self.feature_stats['mean']) / self.feature_stats['std']
```

## 4. BentoML Production Excellence

### ✅ DO: Build Optimized Bento Images

BentoML 1.3 introduces significant improvements for production deployments:

```python
# bentofile.yaml - Optimized configuration
service: "service.py:svc"
labels:
  team: ml-platform
  cost-center: inference

include:
  - "*.py"
  - "models/*"
  - "configs/*"

python:
  requirements_txt: "requirements.txt"
  wheels:
    - ./wheels/optimized_tokenizer-1.0-cp311-cp311-linux_x86_64.whl
  
  # System packages for production
  system_packages:
    - libgomp1  # OpenMP for CPU parallelism
    - libjemalloc2  # Better memory allocation
  
  # Lock dependencies for reproducibility
  lock_packages: true

# Docker configuration
docker:
  distro: ubuntu-24.04-cuda-12.6
  python_version: "3.11"
  
  # Setup script for runtime optimization
  setup_script: |
    #!/bin/bash
    # Enable TensorRT optimization
    export TF_TRT_ENABLE_FULL_PRECISION=1
    export TF_TRT_MAX_ALLOWED_ENGINES=100
    
    # Configure NUMA for multi-socket systems
    echo 'export OMP_NUM_THREADS=8' >> /etc/environment
    echo 'export MKL_NUM_THREADS=8' >> /etc/environment
    
    # Install model optimizer
    pip install --no-cache-dir onnxruntime-gpu==1.19.0
  
  # 2025: Multi-stage builds are default
  dockerfile_template: |
    {% extends bento_base_template %}
    {% block SETUP_BENTO_COMPONENTS %}
    {{ super() }}
    
    # Optimize model files
    RUN find /home/bentoml/bento/models -name "*.onnx" -exec \
        python -m onnxruntime.tools.optimizer_cli \
        --input {} --output {} \
        --opt_level 99 \;
    {% endblock %}

# Model configuration
models:
  - tag: "fraud_detector:latest"
    filter: "format:onnx"  # Only include ONNX format
  - tag: "text_encoder:latest"
    
# Resource configuration for Kubernetes deployment
resources:
  requests:
    cpu: 4
    memory: 16Gi
    nvidia.com/gpu: 1
  limits:
    cpu: 8
    memory: 32Gi
    nvidia.com/gpu: 1
```

### Advanced Service Definition

```python
# service.py - Production-grade BentoML service
import bentoml
from bentoml.io import JSON, Image, Multipart
import numpy as np
from typing import Dict, Any, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
import onnxruntime as ort

# Configure model runners with optimization
fraud_model_runner = bentoml.models.get("fraud_detector:latest").to_runner(
    name="fraud_detector",
    max_batch_size=32,
    max_latency_ms=100,
    # 2025: Native ONNX runtime integration
    runtime="onnxruntime-gpu",
    runtime_options={
        "providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
        "provider_options": [{
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "/tmp/trt_cache"
        }, {}]
    }
)

text_encoder_runner = bentoml.models.get("text_encoder:latest").to_runner(
    name="text_encoder",
    # Use thread pool for CPU-bound encoding
    strategy=bentoml.RunnerStrategy.ThreadPoolExecutor,
    resources={
        "cpu": 4,
        "memory": "8Gi"
    }
)

# Service definition with runners
svc = bentoml.Service(
    name="fraud_detection_service",
    runners=[fraud_model_runner, text_encoder_runner]
)

# Health check endpoint
@svc.api(
    route="/health",
    input=JSON(),
    output=JSON(),
)
async def health() -> Dict[str, Any]:
    """Production health check with dependency verification"""
    checks = {
        "service": "healthy",
        "models": {},
        "gpu_available": torch.cuda.is_available()
    }
    
    # Check each runner
    for runner in svc.runners:
        try:
            # Minimal inference to verify model is loaded
            if runner.name == "fraud_detector":
                test_input = np.random.randn(1, 30).astype(np.float32)
                await runner.async_run(test_input)
            checks["models"][runner.name] = "healthy"
        except Exception as e:
            checks["models"][runner.name] = f"unhealthy: {str(e)}"
            checks["service"] = "degraded"
    
    return checks

# Main inference endpoint with batching
@svc.api(
    route="/predict",
    input=JSON(),
    output=JSON(),
    batch=True,  # Enable auto-batching
    batch_config={
        "max_batch_size": 32,
        "timeout_micros": 5000  # 5ms
    }
)
async def predict(
    inputs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Batch prediction with parallel preprocessing"""
    
    # Extract features in parallel
    text_features_task = asyncio.create_task(
        extract_text_features([inp["text"] for inp in inputs])
    )
    
    numeric_features = np.array([
        inp["numeric_features"] for inp in inputs
    ])
    
    # Wait for text encoding
    text_features = await text_features_task
    
    # Combine features
    combined_features = np.hstack([numeric_features, text_features])
    
    # Run fraud detection
    predictions = await fraud_model_runner.async_run(combined_features)
    
    # Format response
    results = []
    for idx, (inp, pred) in enumerate(zip(inputs, predictions)):
        results.append({
            "transaction_id": inp["transaction_id"],
            "fraud_probability": float(pred[1]),  # Probability of fraud class
            "risk_level": "high" if pred[1] > 0.8 else "medium" if pred[1] > 0.5 else "low",
            "model_version": "v2.1.0"
        })
    
    return results

async def extract_text_features(texts: List[str]) -> np.ndarray:
    """Extract text features using encoder runner"""
    # BentoML handles the thread pool execution
    encoded = await text_encoder_runner.async_run(texts)
    return encoded

# Metrics endpoint for Prometheus
@svc.api(
    route="/metrics",
    input=JSON(),
    output=JSON(),
)
async def metrics() -> Dict[str, Any]:
    """Expose custom metrics"""
    return {
        "inference_count": svc.metrics.inference_count,
        "average_latency_ms": svc.metrics.average_latency_ms,
        "gpu_memory_used_mb": get_gpu_memory_usage(),
        "model_cache_hit_rate": get_cache_hit_rate()
    }
```

### Deploying to Kubernetes with Yatai

```yaml
# yatai-deployment.yaml
apiVersion: serving.yatai.ai/v2
kind: BentoDeployment
metadata:
  name: fraud-detection
  namespace: production
spec:
  bento: fraud_detection_service:latest
  
  # Resource allocation per replica
  resources:
    requests:
      cpu: 4
      memory: 16Gi
      nvidia.com/gpu: 1
    limits:
      cpu: 8
      memory: 32Gi
      nvidia.com/gpu: 1
  
  # Autoscaling configuration
  autoscaling:
    minReplicas: 2
    maxReplicas: 20
    
    # Custom metrics from service
    metrics:
    - type: Pods
      pods:
        metric:
          name: bentoml_service_request_duration_seconds_p99
        target:
          type: AverageValue
          averageValue: "0.1"  # 100ms P99
    
    # Scale based on GPU memory pressure
    - type: Resource
      resource:
        name: nvidia.com/gpu
        target:
          type: Utilization
          averageUtilization: 80
  
  # Runner-specific configuration
  runners:
    fraud_detector:
      # Override runner resources
      resources:
        limits:
          nvidia.com/gpu: 1
      # Enable GPU time-slicing
      annotations:
        nvidia.com/gpu-time-slicing-config: "4"
    
    text_encoder:
      # CPU-only runner
      resources:
        limits:
          cpu: 4
          memory: 8Gi
  
  # Monitoring integration
  monitoring:
    enabled: true
    namespace: monitoring
    
  # 2025: Native tracing support
  tracing:
    enabled: true
    samplingRate: 0.1
    exporter:
      type: otlp
      endpoint: tempo-distributor.monitoring:4317
```

## 5. GPU Optimization Strategies

### ✅ DO: Implement Multi-Instance GPU (MIG) for Better Utilization

Modern NVIDIA GPUs (A100, H100, H200) support MIG for hard partitioning:

```yaml
# mig-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gpu-mig-config
  namespace: gpu-operator
data:
  config.yaml: |
    version: v1
    mig-configs:
      # H100 configuration for inference
      - devices: [0,1,2,3]
        mig-enabled: true
        profiles:
          - "1g.10gb": 7  # 7 instances per GPU
      
      # A100 configuration
      - devices: [4,5,6,7]
        mig-enabled: true  
        profiles:
          - "2g.20gb": 3  # 3 instances per GPU
          - "1g.10gb": 1  # 1 instance remainder

---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: gpu-mig-manager
  namespace: gpu-operator
spec:
  selector:
    matchLabels:
      name: gpu-mig-manager
  template:
    metadata:
      labels:
        name: gpu-mig-manager
    spec:
      containers:
      - name: gpu-mig-manager
        image: nvcr.io/nvidia/cloud-native/k8s-mig-manager:v0.8.0
        securityContext:
          privileged: true
        volumeMounts:
        - name: mig-config
          mountPath: /etc/nvidia-mig-manager
        env:
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
      volumes:
      - name: mig-config
        configMap:
          name: gpu-mig-config
```

### Dynamic GPU Memory Management

```python
# gpu_memory_optimizer.py
import torch
import psutil
import gc
from typing import Dict, Any
import threading
import time

class GPUMemoryOptimizer:
    """Dynamic GPU memory management for inference"""
    
    def __init__(self, target_memory_fraction: float = 0.9):
        self.target_memory_fraction = target_memory_fraction
        self.monitoring_thread = None
        self.should_monitor = True
        
    def start_monitoring(self):
        """Start background memory monitoring"""
        self.monitoring_thread = threading.Thread(
            target=self._monitor_memory,
            daemon=True
        )
        self.monitoring_thread.start()
    
    def _monitor_memory(self):
        """Background thread to monitor and optimize memory"""
        while self.should_monitor:
            try:
                # Check GPU memory pressure
                for device_id in range(torch.cuda.device_count()):
                    mem_info = torch.cuda.mem_get_info(device_id)
                    free_memory = mem_info[0]
                    total_memory = mem_info[1]
                    used_fraction = 1 - (free_memory / total_memory)
                    
                    # Trigger cleanup if approaching limit
                    if used_fraction > self.target_memory_fraction:
                        self._cleanup_gpu_memory(device_id)
                
                # Check system memory
                system_memory = psutil.virtual_memory()
                if system_memory.percent > 90:
                    self._cleanup_system_memory()
                    
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
            
            time.sleep(5)  # Check every 5 seconds
    
    def _cleanup_gpu_memory(self, device_id: int):
        """Aggressive GPU memory cleanup"""
        logger.info(f"Cleaning GPU {device_id} memory")
        
        with torch.cuda.device(device_id):
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            # Clear gradient buffers if any
            for obj in gc.get_objects():
                if torch.is_tensor(obj) and obj.is_cuda:
                    del obj
            
            torch.cuda.empty_cache()
    
    def optimize_model_for_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for inference"""
        model.eval()
        
        # Fuse operations where possible
        if hasattr(torch.jit, "_run_pass_onnx_fuse"):
            torch.jit._run_pass_onnx_fuse(model)
        
        # Enable cuDNN autotuner for consistent input sizes
        torch.backends.cudnn.benchmark = True
        
        # Convert to half precision if supported
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
            model = model.half()
            
        # Compile with inductor (PyTorch 2.4+)
        try:
            model = torch.compile(
                model,
                mode="max-autotune",
                backend="inductor"
            )
        except:
            logger.warning("Torch compile not available")
            
        return model
```

## 6. Advanced Traffic Management

### Progressive Rollout with Flagger

```yaml
# flagger-canary.yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: fraud-model
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-model
  
  # Service mesh provider
  provider: istio
  
  # Promotion configuration
  progressDeadlineSeconds: 3600
  
  service:
    port: 8080
    targetPort: 8080
    gateways:
    - public-gateway
    hosts:
    - fraud-api.example.com
    
    # 2025: Automatic retry configuration
    retries:
      attempts: 3
      retryOn: "gateway-error,reset,connect-failure"
  
  # Canary analysis
  analysis:
    # Promotion schedule
    interval: 1m
    threshold: 10
    maxWeight: 50
    stepWeight: 5
    
    # Metrics for promotion decision
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
      interval: 1m
    
    - name: latency-p99
      templateRef:
        name: latency
        namespace: flagger-system
      thresholdRange:
        max: 100  # 100ms
    
    - name: model-accuracy
      templateRef:
        name: custom-accuracy
      thresholdRange:
        min: 0.95
    
    # Load testing during canary
    webhooks:
    - name: load-test
      url: http://flagger-loadtester.flagger-system/
      metadata:
        cmd: "hey -z 1m -q 100 -c 10 http://fraud-model-canary.production:8080/predict"
    
    # Custom validation webhook
    - name: model-validation
      url: http://model-validator.ml-platform/validate
      metadata:
        model: "fraud-detector"
        threshold: "0.95"

---
# Custom metric for model accuracy
apiVersion: flagger.app/v1beta1
kind: MetricTemplate
metadata:
  name: custom-accuracy
  namespace: flagger-system
spec:
  provider:
    type: prometheus
    address: http://prometheus:9090
  query: |
    avg(
      rate(
        model_prediction_accuracy_total{
          model="{{ args.model }}",
          version="{{ args.version }}"
        }[{{ interval }}]
      )
    )
```

### Shadow Traffic for Safe Testing

```yaml
# istio-shadow-traffic.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: fraud-model-shadow
spec:
  hosts:
  - fraud-api.example.com
  gateways:
  - public-gateway
  http:
  - match:
    - uri:
        prefix: /predict
    route:
    - destination:
        host: fraud-model-stable
        port:
          number: 8080
      weight: 100
    
    # Mirror traffic to new version
    mirror:
      host: fraud-model-canary
      port:
        number: 8080
    mirrorPercentage:
      value: 10.0  # Shadow 10% of traffic
```

## 7. Cost Optimization Strategies

### ✅ DO: Implement Spot Instance Support with Fallback

```yaml
# spot-instance-nodepool.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: spot-instance-config
  namespace: kube-system
data:
  # AWS configuration
  aws-spot-config: |
    {
      "spot_instance_pools": 10,
      "spot_allocation_strategy": "capacity-optimized-prioritized",
      "on_demand_base_capacity": 2,
      "on_demand_percentage_above_base": 25,
      "instance_types": [
        "g5.xlarge",
        "g5.2xlarge", 
        "g4dn.xlarge",
        "g4dn.2xlarge"
      ]
    }

---
# Spot instance handler for graceful shutdowns
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: spot-interrupt-handler
  namespace: kube-system
spec:
  selector:
    matchLabels:
      app: spot-interrupt-handler
  template:
    metadata:
      labels:
        app: spot-interrupt-handler
    spec:
      serviceAccountName: spot-interrupt-handler
      containers:
      - name: spot-interrupt-handler
        image: amazon/aws-node-termination-handler:v2.0.0
        env:
        - name: ENABLE_SPOT_INTERRUPTION_DRAINING
          value: "true"
        - name: ENABLE_SCHEDULED_EVENT_DRAINING
          value: "true"
        - name: DRAIN_GRACE_PERIOD
          value: "120"  # 2 minutes for model checkpoint
        - name: WEBHOOK_URL
          value: "http://ml-platform-webhook:8080/spot-interrupt"
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
```

### Intelligent Model Caching

```python
# model_cache_manager.py
import os
import shutil
import asyncio
from typing import Dict, Optional, List
import aiofiles
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
import psutil
import json

@dataclass
class ModelCacheEntry:
    model_id: str
    model_hash: str
    size_bytes: int
    last_accessed: datetime
    access_count: int
    load_time_ms: float
    
class IntelligentModelCache:
    """LRU cache with predictive prefetching for models"""
    
    def __init__(
        self, 
        cache_dir: str = "/model-cache",
        max_size_gb: int = 100,
        prefetch_threshold: float = 0.7
    ):
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.prefetch_threshold = prefetch_threshold
        self.cache_metadata: Dict[str, ModelCacheEntry] = {}
        self.access_patterns: Dict[str, List[datetime]] = {}
        
    async def get_model(
        self, 
        model_id: str, 
        source_uri: str
    ) -> str:
        """Get model with intelligent caching"""
        
        # Check if model is cached
        if model_id in self.cache_metadata:
            entry = self.cache_metadata[model_id]
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            
            # Track access pattern
            if model_id not in self.access_patterns:
                self.access_patterns[model_id] = []
            self.access_patterns[model_id].append(datetime.now())
            
            # Trigger predictive prefetch
            asyncio.create_task(self._predictive_prefetch(model_id))
            
            return os.path.join(self.cache_dir, model_id)
        
        # Download model
        return await self._download_and_cache(model_id, source_uri)
    
    async def _download_and_cache(
        self, 
        model_id: str, 
        source_uri: str
    ) -> str:
        """Download model and add to cache"""
        
        start_time = asyncio.get_event_loop().time()
        
        # Ensure cache space
        await self._ensure_cache_space(model_id)
        
        model_path = os.path.join(self.cache_dir, model_id)
        os.makedirs(model_path, exist_ok=True)
        
        # Download based on URI scheme
        if source_uri.startswith("s3://"):
            await self._download_from_s3(source_uri, model_path)
        elif source_uri.startswith("gs://"):
            await self._download_from_gcs(source_uri, model_path)
        elif source_uri.startswith("http"):
            await self._download_from_http(source_uri, model_path)
        else:
            raise ValueError(f"Unsupported URI scheme: {source_uri}")
        
        # Calculate model hash and size
        model_hash = await self._calculate_model_hash(model_path)
        model_size = self._get_directory_size(model_path)
        
        # Update cache metadata
        load_time = (asyncio.get_event_loop().time() - start_time) * 1000
        self.cache_metadata[model_id] = ModelCacheEntry(
            model_id=model_id,
            model_hash=model_hash,
            size_bytes=model_size,
            last_accessed=datetime.now(),
            access_count=1,
            load_time_ms=load_time
        )
        
        # Persist metadata
        await self._save_metadata()
        
        return model_path
    
    async def _ensure_cache_space(self, new_model_id: str):
        """Ensure enough space using intelligent eviction"""
        
        current_size = sum(
            entry.size_bytes 
            for entry in self.cache_metadata.values()
        )
        
        # Predict size needed (use historical average if unknown)
        predicted_size = await self._predict_model_size(new_model_id)
        
        while current_size + predicted_size > self.max_size_bytes:
            # Find least valuable model to evict
            eviction_candidate = self._select_eviction_candidate()
            
            if eviction_candidate:
                # Remove from cache
                model_path = os.path.join(
                    self.cache_dir, 
                    eviction_candidate
                )
                shutil.rmtree(model_path, ignore_errors=True)
                
                # Update metadata
                evicted = self.cache_metadata.pop(eviction_candidate)
                current_size -= evicted.size_bytes
                
                logger.info(
                    f"Evicted model {eviction_candidate} "
                    f"(size: {evicted.size_bytes / 1e9:.2f}GB)"
                )
            else:
                break
    
    def _select_eviction_candidate(self) -> Optional[str]:
        """Select model to evict using value-based scoring"""
        
        if not self.cache_metadata:
            return None
        
        scores = {}
        current_time = datetime.now()
        
        for model_id, entry in self.cache_metadata.items():
            # Calculate recency score (exponential decay)
            hours_since_access = (
                current_time - entry.last_accessed
            ).total_seconds() / 3600
            recency_score = np.exp(-hours_since_access / 24)  # 24hr half-life
            
            # Calculate frequency score
            frequency_score = min(entry.access_count / 100, 1.0)
            
            # Calculate size penalty (prefer evicting larger models)
            size_penalty = entry.size_bytes / self.max_size_bytes
            
            # Calculate load time penalty (prefer keeping slow-loading models)
            load_time_score = min(entry.load_time_ms / 60000, 1.0)  # 60s max
            
            # Combined score (lower is worse)
            scores[model_id] = (
                recency_score * 0.4 +
                frequency_score * 0.3 +
                load_time_score * 0.2 -
                size_penalty * 0.1
            )
        
        # Return model with lowest score
        return min(scores, key=scores.get)
    
    async def _predictive_prefetch(self, current_model: str):
        """Prefetch models likely to be needed next"""
        
        # Analyze access patterns
        if current_model not in self.access_patterns:
            return
        
        # Simple time-series prediction
        access_times = self.access_patterns[current_model]
        if len(access_times) < 3:
            return
        
        # Calculate average interval
        intervals = [
            (access_times[i+1] - access_times[i]).total_seconds()
            for i in range(len(access_times)-1)
        ]
        avg_interval = sum(intervals) / len(intervals)
        
        # Predict next access time
        predicted_next = access_times[-1] + timedelta(seconds=avg_interval)
        time_until_next = (predicted_next - datetime.now()).total_seconds()
        
        # If likely to be needed soon, check related models
        if time_until_next < 300:  # Within 5 minutes
            related_models = await self._find_related_models(current_model)
            
            for related_model, correlation_score in related_models:
                if correlation_score > self.prefetch_threshold:
                    # Prefetch in background
                    asyncio.create_task(
                        self._background_prefetch(related_model)
                    )
```

## 8. Monitoring & Observability

### ✅ DO: Implement Comprehensive Metrics

```yaml
# model-metrics-servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: model-inference-metrics
  namespace: ml-platform
spec:
  selector:
    matchLabels:
      app: model-server
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
    relabelings:
    - sourceLabels: [__meta_kubernetes_pod_label_model_name]
      targetLabel: model_name
    - sourceLabels: [__meta_kubernetes_pod_label_model_version]
      targetLabel: model_version
    
    # Add node information for GPU correlation
    - sourceLabels: [__meta_kubernetes_pod_node_name]
      targetLabel: node
    
    # Add GPU device if present
    - sourceLabels: [__meta_kubernetes_pod_annotation_nvidia_com_gpu_device]
      targetLabel: gpu_device

---
# PrometheusRule for alerting
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: model-inference-alerts
  namespace: ml-platform
spec:
  groups:
  - name: model_inference
    interval: 30s
    rules:
    # Latency alerts
    - alert: ModelInferenceHighLatency
      expr: |
        histogram_quantile(0.99, 
          sum(rate(model_inference_duration_seconds_bucket[5m])) 
          by (model_name, model_version, le)
        ) > 0.5
      for: 5m
      labels:
        severity: warning
        team: ml-platform
      annotations:
        summary: "Model {{ $labels.model_name }} has high latency"
        description: "P99 latency is {{ $value }}s (threshold: 0.5s)"
    
    # GPU memory alerts  
    - alert: GPUMemoryPressure
      expr: |
        (
          nvidia_gpu_memory_used_bytes / 
          nvidia_gpu_memory_total_bytes
        ) > 0.95
      for: 2m
      labels:
        severity: critical
        team: ml-platform
      annotations:
        summary: "GPU memory pressure on {{ $labels.node }}"
        description: "GPU {{ $labels.gpu }} is using {{ $value | humanizePercentage }} of memory"
    
    # Model error rate
    - alert: ModelInferenceErrors
      expr: |
        sum(rate(model_inference_errors_total[5m])) by (model_name) > 0.01
      for: 5m
      labels:
        severity: warning
        team: ml-platform
      annotations:
        summary: "Model {{ $labels.model_name }} has elevated errors"
        description: "Error rate: {{ $value | humanizePercentage }}"
    
    # Queue depth alert
    - alert: InferenceQueueBacklog
      expr: |
        sum(inference_queue_depth) by (model_name) > 100
      for: 2m
      labels:
        severity: warning
        team: ml-platform
      annotations:
        summary: "High queue depth for {{ $labels.model_name }}"
        description: "Queue depth: {{ $value }}"
```

### Custom Grafana Dashboard

```json
{
  "dashboard": {
    "title": "ML Model Inference - Production",
    "panels": [
      {
        "title": "Inference Requests/sec by Model",
        "targets": [{
          "expr": "sum(rate(model_inference_requests_total[5m])) by (model_name, model_version)"
        }],
        "type": "graph"
      },
      {
        "title": "P50/P95/P99 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.5, sum(rate(model_inference_duration_seconds_bucket[5m])) by (le, model_name))",
            "legendFormat": "P50 - {{ model_name }}"
          },
          {
            "expr": "histogram_quantile(0.95, sum(rate(model_inference_duration_seconds_bucket[5m])) by (le, model_name))",
            "legendFormat": "P95 - {{ model_name }}"
          },
          {
            "expr": "histogram_quantile(0.99, sum(rate(model_inference_duration_seconds_bucket[5m])) by (le, model_name))",
            "legendFormat": "P99 - {{ model_name }}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "GPU Utilization by Node",
        "targets": [{
          "expr": "nvidia_gpu_duty_cycle * 100"
        }],
        "type": "heatmap"
      },
      {
        "title": "Model Cache Hit Rate",
        "targets": [{
          "expr": "sum(rate(model_cache_hits_total[5m])) / sum(rate(model_cache_requests_total[5m]))"
        }],
        "type": "stat"
      }
    ]
  }
}
```

## 9. Security Best Practices

### ✅ DO: Implement Model Signing and Verification

```python
# model_security.py
import hashlib
import json
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import base64
from typing import Dict, Tuple, Optional

class ModelSecurityManager:
    """Handles model signing and verification"""
    
    def __init__(self, private_key_path: Optional[str] = None):
        self.private_key = None
        self.public_key = None
        
        if private_key_path:
            self._load_keys(private_key_path)
    
    def generate_keys(self) -> Tuple[bytes, bytes]:
        """Generate new RSA key pair for model signing"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=default_backend()
        )
        
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def sign_model(
        self, 
        model_path: str,
        metadata: Dict[str, any]
    ) -> Dict[str, str]:
        """Sign a model and return signature with metadata"""
        
        # Calculate model hash
        model_hash = self._calculate_model_hash(model_path)
        
        # Add hash to metadata
        metadata['model_hash'] = model_hash
        metadata['signed_at'] = datetime.utcnow().isoformat()
        
        # Serialize metadata
        metadata_json = json.dumps(metadata, sort_keys=True)
        
        # Sign the metadata + hash
        signature = self.private_key.sign(
            metadata_json.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return {
            'metadata': metadata_json,
            'signature': base64.b64encode(signature).decode(),
            'public_key': base64.b64encode(
                self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
            ).decode()
        }
    
    def verify_model(
        self, 
        model_path: str,
        signature_data: Dict[str, str]
    ) -> bool:
        """Verify model signature"""
        
        try:
            # Load public key
            public_key_bytes = base64.b64decode(signature_data['public_key'])
            public_key = serialization.load_pem_public_key(
                public_key_bytes,
                backend=default_backend()
            )
            
            # Decode signature
            signature = base64.b64decode(signature_data['signature'])
            
            # Verify signature
            public_key.verify(
                signature,
                signature_data['metadata'].encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # Verify model hash matches
            metadata = json.loads(signature_data['metadata'])
            current_hash = self._calculate_model_hash(model_path)
            
            return current_hash == metadata['model_hash']
            
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return False
    
    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate SHA256 hash of model files"""
        sha256_hash = hashlib.sha256()
        
        # Hash all files in model directory
        for root, dirs, files in os.walk(model_path):
            for file in sorted(files):  # Sort for consistency
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
```

### Network Policies for Model Isolation

```yaml
# model-network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: model-inference-isolation
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: model-server
  
  policyTypes:
  - Ingress
  - Egress
  
  ingress:
  # Allow from gateway only
  - from:
    - namespaceSelector:
        matchLabels:
          name: istio-system
      podSelector:
        matchLabels:
          app: istio-ingressgateway
    ports:
    - protocol: TCP
      port: 8080
  
  # Allow Prometheus scraping
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
      podSelector:
        matchLabels:
          app: prometheus
    ports:
    - protocol: TCP
      port: 9090
  
  egress:
  # Allow DNS
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
      podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53
  
  # Allow model storage access
  - to:
    - podSelector:
        matchLabels:
          app: model-storage
    ports:
    - protocol: TCP
      port: 443
  
  # Block everything else
  - to: []
```

## 10. Disaster Recovery & High Availability

### ✅ DO: Implement Multi-Region Model Serving

```yaml
# multi-region-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server-us-east
  namespace: production
  labels:
    region: us-east-1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-server
      region: us-east-1
  template:
    metadata:
      labels:
        app: model-server
        region: us-east-1
    spec:
      # Region-specific node selector
      nodeSelector:
        failure-domain.beta.kubernetes.io/region: us-east-1
      
      # Anti-affinity for HA
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - model-server
            topologyKey: kubernetes.io/hostname
        
        # Prefer nodes with GPU
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            preference:
              matchExpressions:
              - key: accelerator
                operator: In
                values:
                - nvidia-tesla-a100
                - nvidia-tesla-v100
      
      # Topology spread for even distribution
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: model-server
      
      containers:
      - name: model-server
        image: model-server:v2.0.0
        env:
        - name: REGION
          value: "us-east-1"
        - name: MODEL_CACHE_REGION
          value: "us-east-1"
```

### Model Checkpoint and Recovery

```python
# checkpoint_manager.py
import asyncio
import pickle
import torch
from typing import Dict, Any, Optional
import aioredis
import aioboto3
from datetime import datetime

class CheckpointManager:
    """Manages model checkpoints for disaster recovery"""
    
    def __init__(
        self,
        s3_bucket: str,
        redis_url: str,
        checkpoint_interval_seconds: int = 300
    ):
        self.s3_bucket = s3_bucket
        self.redis_url = redis_url
        self.checkpoint_interval = checkpoint_interval_seconds
        self.redis_client = None
        self.s3_session = None
        self._checkpoint_task = None
        
    async def start(self):
        """Start checkpoint manager"""
        self.redis_client = await aioredis.create_redis_pool(self.redis_url)
        self.s3_session = aioboto3.Session()
        
        # Start periodic checkpointing
        self._checkpoint_task = asyncio.create_task(
            self._periodic_checkpoint()
        )
    
    async def save_checkpoint(
        self,
        model_id: str,
        model_state: Dict[str, Any],
        metadata: Dict[str, Any]
    ):
        """Save model checkpoint"""
        
        checkpoint_id = f"{model_id}_{datetime.utcnow().isoformat()}"
        
        # Save to Redis for fast recovery
        redis_key = f"checkpoint:{model_id}:latest"
        checkpoint_data = {
            'model_state': model_state,
            'metadata': metadata,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self.redis_client.set(
            redis_key,
            pickle.dumps(checkpoint_data),
            expire=3600  # Keep for 1 hour
        )
        
        # Async upload to S3 for long-term storage
        asyncio.create_task(
            self._upload_to_s3(checkpoint_id, checkpoint_data)
        )
        
        return checkpoint_id
    
    async def restore_checkpoint(
        self,
        model_id: str,
        checkpoint_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Restore model from checkpoint"""
        
        # Try Redis first (fastest)
        if not checkpoint_id:
            redis_key = f"checkpoint:{model_id}:latest"
            cached = await self.redis_client.get(redis_key)
            if cached:
                return pickle.loads(cached)
        
        # Fallback to S3
        if checkpoint_id:
            return await self._download_from_s3(checkpoint_id)
        else:
            # Find latest checkpoint in S3
            latest = await self._find_latest_checkpoint(model_id)
            if latest:
                return await self._download_from_s3(latest)
        
        return None
    
    async def _periodic_checkpoint(self):
        """Periodic checkpoint task"""
        while True:
            try:
                await asyncio.sleep(self.checkpoint_interval)
                
                # Get active models from registry
                active_models = await self._get_active_models()
                
                for model_id in active_models:
                    # Check if model needs checkpointing
                    if await self._should_checkpoint(model_id):
                        model_state = await self._get_model_state(model_id)
                        metadata = await self._get_model_metadata(model_id)
                        
                        await self.save_checkpoint(
                            model_id,
                            model_state,
                            metadata
                        )
                        
            except Exception as e:
                logger.error(f"Checkpoint error: {e}")
    
    async def _upload_to_s3(
        self,
        checkpoint_id: str,
        data: Dict[str, Any]
    ):
        """Upload checkpoint to S3"""
        async with self.s3_session.client('s3') as s3:
            # Serialize checkpoint
            checkpoint_bytes = pickle.dumps(data)
            
            # Compress if large
            if len(checkpoint_bytes) > 10 * 1024 * 1024:  # 10MB
                import gzip
                checkpoint_bytes = gzip.compress(checkpoint_bytes)
                key = f"checkpoints/{checkpoint_id}.pkl.gz"
            else:
                key = f"checkpoints/{checkpoint_id}.pkl"
            
            # Upload with metadata
            await s3.put_object(
                Bucket=self.s3_bucket,
                Key=key,
                Body=checkpoint_bytes,
                Metadata={
                    'checkpoint_id': checkpoint_id,
                    'created_at': datetime.utcnow().isoformat()
                }
            )
```

## 11. Advanced Deployment Patterns

### Blue-Green Deployment with Instant Rollback

```yaml
# blue-green-deployment.yaml
apiVersion: v1
kind: Service
metadata:
  name: model-service
  namespace: production
spec:
  selector:
    app: model-server
    version: active  # This label switches between blue/green
  ports:
  - port: 8080
    targetPort: 8080

---
# Blue deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server-blue
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-server
      version: blue
  template:
    metadata:
      labels:
        app: model-server
        version: blue
    spec:
      containers:
      - name: model-server
        image: model-server:v1.0.0
        # ... rest of spec

---
# Green deployment  
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server-green
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-server
      version: green
  template:
    metadata:
      labels:
        app: model-server
        version: green
    spec:
      containers:
      - name: model-server
        image: model-server:v2.0.0
        # ... rest of spec

---
# Switching script (ConfigMap)
apiVersion: v1
kind: ConfigMap
metadata:
  name: blue-green-switch
  namespace: production
data:
  switch.sh: |
    #!/bin/bash
    CURRENT=$(kubectl get svc model-service -o jsonpath='{.spec.selector.version}')
    
    if [ "$CURRENT" == "blue" ]; then
      TARGET="green"
    else
      TARGET="blue"
    fi
    
    # Pre-switch validation
    kubectl wait --for=condition=ready pod -l version=$TARGET -n production --timeout=300s
    
    # Switch traffic
    kubectl patch svc model-service -p '{"spec":{"selector":{"version":"'$TARGET'"}}}'
    
    # Mark new version as active
    kubectl label deployment model-server-$TARGET version=active --overwrite
    kubectl label deployment model-server-$CURRENT version=inactive --overwrite
    
    echo "Switched from $CURRENT to $TARGET"
```

### GitOps with ArgoCD

```yaml
# argocd-application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ml-inference-platform
  namespace: argocd
spec:
  project: ml-platform
  
  source:
    repoURL: https://github.com/company/ml-platform
    targetRevision: HEAD
    path: k8s/overlays/production
    
    # Kustomize with environment-specific patches
    kustomize:
      images:
      - model-server:v2.0.0
      
      # Strategic merge patches
      patches:
      - target:
          kind: Deployment
          name: model-server
        patch: |-
          - op: replace
            path: /spec/replicas
            value: 5
  
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  
  # Sync policy
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    
    syncOptions:
    - CreateNamespace=true
    - PrunePropagationPolicy=foreground
    
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
  
  # Health assessment
  health:
    progressDeadlineSeconds: 600
```

## 12. Performance Optimization Cookbook

### ✅ DO: Implement Request Batching

```python
# batching_server.py
import asyncio
from typing import List, Dict, Any
import numpy as np
from collections import deque
import time

class BatchingInferenceServer:
    """Dynamic batching for optimal throughput"""
    
    def __init__(
        self,
        model,
        max_batch_size: int = 32,
        max_latency_ms: int = 50,
        min_batch_size: int = 1
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_latency_ms = max_latency_ms
        self.min_batch_size = min_batch_size
        
        self.request_queue = asyncio.Queue()
        self.batch_processor_task = None
        
    async def start(self):
        """Start the batching processor"""
        self.batch_processor_task = asyncio.create_task(
            self._batch_processor()
        )
    
    async def predict(self, request_id: str, data: np.ndarray) -> Any:
        """Queue a prediction request"""
        
        # Create future for this request
        future = asyncio.Future()
        
        # Add to queue
        await self.request_queue.put({
            'id': request_id,
            'data': data,
            'future': future,
            'timestamp': time.time()
        })
        
        # Wait for result
        return await future
    
    async def _batch_processor(self):
        """Process requests in batches"""
        
        while True:
            batch = []
            batch_start_time = None
            
            # Collect batch
            while len(batch) < self.max_batch_size:
                try:
                    # Calculate timeout
                    if batch_start_time is None:
                        timeout = None
                    else:
                        elapsed = (time.time() - batch_start_time) * 1000
                        remaining = self.max_latency_ms - elapsed
                        timeout = max(0, remaining / 1000)
                    
                    # Get request with timeout
                    request = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=timeout
                    )
                    
                    batch.append(request)
                    
                    # Start timer on first request
                    if batch_start_time is None:
                        batch_start_time = time.time()
                    
                    # Process if we have enough
                    if len(batch) >= self.min_batch_size:
                        # Check if we should wait for more
                        elapsed = (time.time() - batch_start_time) * 1000
                        if elapsed >= self.max_latency_ms * 0.8:
                            break
                            
                except asyncio.TimeoutError:
                    # Timeout reached, process what we have
                    if batch:
                        break
            
            # Process batch
            if batch:
                await self._process_batch(batch)
    
    async def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of requests"""
        
        try:
            # Combine inputs
            batch_data = np.stack([req['data'] for req in batch])
            
            # Run inference
            start_time = time.time()
            predictions = await asyncio.to_thread(
                self.model.predict,
                batch_data
            )
            inference_time = time.time() - start_time
            
            # Distribute results
            for i, request in enumerate(batch):
                request['future'].set_result(predictions[i])
            
            # Log metrics
            logger.info(
                f"Processed batch of {len(batch)} in {inference_time:.3f}s "
                f"({inference_time/len(batch)*1000:.1f}ms per request)"
            )
            
        except Exception as e:
            # Set exception for all requests in batch
            for request in batch:
                request['future'].set_exception(e)
```

## 13. Edge Cases & Troubleshooting

### Handling Large Language Models

```yaml
# llm-specific-config.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama-405b
  annotations:
    # LLM-specific optimizations
    serving.kserve.io/deploymentMode: RawDeployment
    serving.kserve.io/autoscalerClass: external
spec:
  predictor:
    # Model parallelism across 8 GPUs
    parallelism: 8
    
    # No autoscaling - fixed resources
    minReplicas: 1
    maxReplicas: 1
    
    containers:
    - name: llm-server
      image: vllm/vllm-openai:v0.6.0
      
      command:
      - python
      - -m
      - vllm.entrypoints.openai.api_server
      
      args:
      - --model=/models/llama-405b
      - --tensor-parallel-size=8
      - --max-model-len=4096
      - --gpu-memory-utilization=0.95
      - --enforce-eager  # Disable CUDA graphs for stability
      - --enable-prefix-caching
      - --max-num-seqs=256
      
      resources:
        limits:
          nvidia.com/gpu: 8
          memory: 640Gi  # 80GB per GPU
          ephemeral-storage: 2Ti
      
      # Custom probes for LLMs
      readinessProbe:
        httpGet:
          path: /health
          port: 8000
        initialDelaySeconds: 600  # 10 minutes for model loading
        periodSeconds: 30
        timeoutSeconds: 30
        
      volumeMounts:
      - name: shm
        mountPath: /dev/shm
      - name: model-cache
        mountPath: /models
    
    volumes:
    # Shared memory for tensor parallelism
    - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 128Gi
          
    # Fast local SSD for model
    - name: model-cache
      hostPath:
        path: /mnt/local-ssd/models
        type: Directory
```

### Debugging Production Issues

```bash
#!/bin/bash
# debug-inference.sh - Production debugging toolkit

# 1. Check pod resource usage
echo "=== Pod Resource Usage ==="
kubectl top pods -n production -l app=model-server

# 2. GPU diagnostics
echo "=== GPU Status ==="
kubectl exec -it deploy/model-server -n production -- nvidia-smi

# 3. Check model loading
echo "=== Model Loading Logs ==="
kubectl logs -n production -l app=model-server --tail=100 | grep -E "(Loading|ERROR|WARNING)"

# 4. Latency breakdown
echo "=== Latency Analysis ==="
kubectl exec -it deploy/model-server -n production -- curl -s localhost:9090/metrics | grep -E "(model_load_time|inference_time|preprocessing_time)"

# 5. Network policies
echo "=== Network Policies ==="
kubectl get networkpolicies -n production

# 6. Service mesh configuration
echo "=== Istio Configuration ==="
istioctl analyze -n production

# 7. Memory profiling
echo "=== Memory Profile ==="
kubectl exec -it deploy/model-server -n production -- python -c "
import tracemalloc
import psutil
import torch

tracemalloc.start()
process = psutil.Process()

print(f'CPU Memory: {process.memory_info().rss / 1e9:.2f} GB')
print(f'GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
print(f'GPU Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB')

current, peak = tracemalloc.get_traced_memory()
print(f'Python Current: {current / 1e9:.2f} GB')
print(f'Python Peak: {peak / 1e9:.2f} GB')
"
```

## Conclusion

This guide covered production-grade patterns for ML inference on Kubernetes. The key principles to remember:

1. **Architecture Matters**: Separate gateway, routing, and serving layers
2. **Optimization is Continuous**: Monitor, profile, and optimize based on real usage
3. **Security is Non-Negotiable**: Sign models, isolate networks, audit access
4. **Failure is Expected**: Plan for GPU failures, spot interruptions, and model corruption
5. **Observability is Essential**: You can't optimize what you can't measure

As model sizes grow and inference demands increase, these patterns will help you build resilient, scalable, and cost-effective ML serving infrastructure.