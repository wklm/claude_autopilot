# The Definitive Guide to Edge Computing and Serverless Architecture (Mid-2025 Edition)

This guide synthesizes production-grade patterns for building scalable, performant, and cost-effective applications using edge computing platforms and serverless architectures. It covers the latest developments in V8 isolates, WASM-WASI Preview2, and modern FaaS platforms.

### Prerequisites & Runtime Requirements

Ensure your development environment includes:
- **Node.js 22.5+** or **Bun 1.2+** (for Cloudflare Workers/Vercel Edge)
- **Rust 1.85+** with `wasm32-wasi` target (for WASM development)
- **Python 3.13+** with free-threaded build support (for Lambda)
- **Deno 2.1+** (for Deno Deploy)
- **Docker 27+** with containerd 2.0 (for Cloud Run/Container Apps)

### Platform Version Matrix (June 2025)

| Platform | Runtime | Cold Start | Memory Limit | CPU Time | Persistent Storage |
|----------|---------|------------|--------------|----------|-------------------|
| **Cloudflare Workers** | V8 Isolates | <5ms | 512MB | 50ms (Paid) | KV, R2, D1, Durable Objects |
| **Vercel Edge** | Edge Runtime 5.2 | <10ms | 256MB | 150ms | KV (Beta) |
| **Deno Deploy** | Deno 2.1 | <15ms | 1GB | 50ms | Deno KV |
| **AWS Lambda** | Custom/Node/Python | 100-500ms | 10GB | 15min | EFS, S3 |
| **Google Cloud Run** | Containers | 500-2000ms | 32GB | 60min | Persistent Disks |
| **Azure Container Apps** | Containers | 1-3s | 16GB | Unlimited | Managed Disks |

---

## 1. Choosing Your Architecture: Edge vs Traditional Serverless

### Decision Matrix

```typescript
// edge-or-serverless.ts
interface WorkloadCharacteristics {
  coldStartSensitive: boolean      // <50ms requirement?
  cpuIntensive: boolean            // >50ms CPU time needed?
  memoryIntensive: boolean         // >512MB RAM needed?
  needsPersistentConnections: boolean  // WebSockets, SSE?
  requiresNodeAPIs: boolean        // fs, child_process, etc?
  multiRegionLatency: boolean      // Global <50ms response?
}

function selectArchitecture(workload: WorkloadCharacteristics): Architecture {
  // Edge Runtime: V8 Isolates (Cloudflare Workers, Vercel Edge)
  if (workload.coldStartSensitive && !workload.cpuIntensive && 
      !workload.memoryIntensive && workload.multiRegionLatency) {
    return 'edge-isolate'
  }
  
  // Container-based Serverless (Cloud Run, Container Apps)
  if (workload.requiresNodeAPIs || workload.memoryIntensive) {
    return 'container-serverless'
  }
  
  // Traditional FaaS (Lambda)
  if (workload.cpuIntensive && !workload.coldStartSensitive) {
    return 'traditional-faas'
  }
  
  // Deno Deploy: Balance between edge and Node.js compatibility
  if (workload.multiRegionLatency && workload.requiresNodeAPIs) {
    return 'deno-deploy'
  }
  
  return 'edge-isolate' // Default to fastest cold starts
}
```

### ✅ DO: Use Edge Runtime for Request/Response Workloads

Edge platforms excel at low-latency, globally distributed request handling:

```typescript
// cloudflare-worker.ts
export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url)
    
    // Geolocation available at edge
    const country = request.cf?.country || 'US'
    
    // Smart caching with regional awareness
    const cacheKey = new Request(`https://cache.${country}.example.com${url.pathname}`)
    const cache = caches.default
    
    let response = await cache.match(cacheKey)
    if (response) return response
    
    // Process request
    response = await handleRequest(request, env, country)
    
    // Cache with regional keys
    ctx.waitUntil(cache.put(cacheKey, response.clone()))
    
    return response
  }
}
```

### ❌ DON'T: Use Edge for CPU-Intensive Tasks

Edge platforms have strict CPU limits. This will timeout:

```typescript
// Bad - Will exceed CPU limits on edge
export default {
  async fetch(request: Request): Promise<Response> {
    const data = await request.json()
    
    // CPU-intensive operation
    const result = calculatePrimeFactors(BigInt(data.number)) // ❌ Timeout
    
    return Response.json({ factors: result })
  }
}
```

Instead, use Lambda or Cloud Run for compute-heavy tasks.

---

## 2. WASM Integration with WASI-Preview2

WebAssembly with WASI-Preview2 (Component Model) enables portable, high-performance modules across all platforms.

### ✅ DO: Use WASM for CPU-Bound Edge Logic

Compile performance-critical code to WASM for 2-10x speedup:

```rust
// image-processor/src/lib.rs
use wasm_bindgen::prelude::*;
use image::{DynamicImage, ImageFormat};

#[wasm_bindgen]
pub struct ImageProcessor {
    image: DynamicImage,
}

#[wasm_bindgen]
impl ImageProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[u8]) -> Result<ImageProcessor, String> {
        let image = image::load_from_memory(data)
            .map_err(|e| e.to_string())?;
        Ok(ImageProcessor { image })
    }
    
    pub fn resize(&mut self, width: u32, height: u32) -> Vec<u8> {
        use image::imageops::FilterType;
        
        let resized = self.image.resize_exact(
            width, 
            height, 
            FilterType::Lanczos3
        );
        
        let mut buffer = Vec::new();
        resized.write_to(&mut buffer, ImageFormat::WebP)
            .expect("Failed to encode");
        
        buffer
    }
    
    pub fn blur(&mut self, sigma: f32) -> Vec<u8> {
        let blurred = self.image.blur(sigma);
        
        let mut buffer = Vec::new();
        blurred.write_to(&mut buffer, ImageFormat::WebP)
            .expect("Failed to encode");
            
        buffer
    }
}
```

Compile with `wasm-pack`:

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build for web target (Workers, Vercel Edge)
wasm-pack build --target web --out-dir pkg

# Build for WASI (Deno Deploy, Wasmtime)
cargo build --target wasm32-wasi --release
```

Use in Cloudflare Worker:

```typescript
// worker.ts
import { ImageProcessor } from './pkg/image_processor'

export default {
  async fetch(request: Request): Promise<Response> {
    if (request.method !== 'POST') {
      return new Response('Method not allowed', { status: 405 })
    }
    
    const formData = await request.formData()
    const file = formData.get('image') as File
    const width = parseInt(formData.get('width') as string)
    const height = parseInt(formData.get('height') as string)
    
    const arrayBuffer = await file.arrayBuffer()
    const uint8Array = new Uint8Array(arrayBuffer)
    
    try {
      // WASM processing happens here
      const processor = new ImageProcessor(uint8Array)
      const resized = processor.resize(width, height)
      
      return new Response(resized, {
        headers: {
          'Content-Type': 'image/webp',
          'Cache-Control': 'public, max-age=31536000',
        }
      })
    } catch (error) {
      return new Response(`Error: ${error}`, { status: 500 })
    }
  }
}
```

### Component Model with WASI-Preview2

The new Component Model enables language-agnostic plugins:

```wit
// calculator.wit - WebAssembly Interface Types
package example:calculator@0.1.0;

interface calculate {
  record expression {
    left: f64,
    right: f64,
    operator: string,
  }
  
  evaluate: func(expr: expression) -> result<f64, string>;
}

world calculator {
  export calculate;
}
```

Implement in any language and use everywhere:

```rust
// Rust implementation
wit_bindgen::generate!({
    world: "calculator",
    path: "./calculator.wit",
});

struct Component;

impl Calculate for Component {
    fn evaluate(&self, expr: Expression) -> Result<f64, String> {
        match expr.operator.as_str() {
            "+" => Ok(expr.left + expr.right),
            "-" => Ok(expr.left - expr.right),
            "*" => Ok(expr.left * expr.right),
            "/" => {
                if expr.right == 0.0 {
                    Err("Division by zero".to_string())
                } else {
                    Ok(expr.left / expr.right)
                }
            }
            _ => Err(format!("Unknown operator: {}", expr.operator)),
        }
    }
}
```

---

## 3. State Management at the Edge

Edge platforms require different state management patterns than traditional servers.

### Cloudflare Durable Objects for Stateful Edge

Durable Objects provide consistent, low-latency state with automatic failover:

```typescript
// durable-object.ts
export class RateLimiter implements DurableObject {
  private state: DurableObjectState
  private storage: DurableObjectStorage
  
  constructor(state: DurableObjectState, env: Env) {
    this.state = state
    this.storage = state.storage
  }
  
  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url)
    const key = url.searchParams.get('key') || 'default'
    
    // Use transactions for consistency
    const allowed = await this.storage.transaction(async (txn) => {
      const current = await txn.get<number>(key) || 0
      const resetTime = await txn.get<number>(`${key}:reset`) || 0
      
      const now = Date.now()
      const windowMs = 60000 // 1 minute
      
      if (now > resetTime) {
        // Reset window
        await txn.put(key, 1)
        await txn.put(`${key}:reset`, now + windowMs)
        return true
      }
      
      if (current >= 100) {
        return false // Rate limit exceeded
      }
      
      await txn.put(key, current + 1)
      return true
    })
    
    return Response.json({ 
      allowed,
      remaining: allowed ? 99 - (await this.storage.get<number>(key) || 0) : 0
    })
  }
}

// Worker using Durable Object
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const clientIp = request.headers.get('CF-Connecting-IP') || 'unknown'
    
    // Get or create Durable Object instance
    const id = env.RATE_LIMITER.idFromName(clientIp)
    const limiter = env.RATE_LIMITER.get(id)
    
    // Forward request to Durable Object
    const response = await limiter.fetch(request)
    const { allowed } = await response.json()
    
    if (!allowed) {
      return new Response('Rate limit exceeded', { status: 429 })
    }
    
    // Process the actual request
    return handleRequest(request, env)
  }
}
```

### Deno KV for Global State

Deno Deploy's built-in KV database with global replication:

```typescript
// deno-kv-cache.ts
const kv = await Deno.openKv()

// Atomic operations with versioning
export async function incrementPageView(path: string): Promise<number> {
  const key = ["pageviews", path]
  
  // Atomic increment with consistency
  const result = await kv.atomic()
    .sum(key, 1n)  // BigInt for large numbers
    .commit()
  
  if (!result.ok) {
    throw new Error("Failed to increment")
  }
  
  // Get current value
  const entry = await kv.get<bigint>(key)
  return Number(entry.value || 0n)
}

// Watch for changes (reactive patterns)
export async function watchKey(key: string[], callback: (value: any) => void) {
  const watcher = kv.watch([key])
  
  for await (const entries of watcher) {
    const [entry] = entries
    if (entry.value !== null) {
      callback(entry.value)
    }
  }
}

// TTL-based caching
export async function cacheWithTTL<T>(
  key: string[],
  fetcher: () => Promise<T>,
  ttlSeconds: number = 300
): Promise<T> {
  const cached = await kv.get<T>(key)
  
  if (cached.value !== null) {
    return cached.value
  }
  
  const fresh = await fetcher()
  
  await kv.set(key, fresh, {
    expireIn: ttlSeconds * 1000
  })
  
  return fresh
}
```

---

## 4. AWS Lambda with Rust for Maximum Performance

Rust on Lambda provides near-native performance with minimal cold starts.

### ✅ DO: Use Rust for Performance-Critical Lambda Functions

```rust
// Cargo.toml
[package]
name = "image-lambda"
version = "0.1.0"
edition = "2021"

[dependencies]
lambda_runtime = "0.13"
lambda_http = "0.13"
tokio = { version = "1.42", features = ["macros"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
aws-config = "1.5"
aws-sdk-s3 = "1.54"
image = "0.25"
bytes = "1.8"

[profile.release]
lto = true
codegen-units = 1
strip = true
opt-level = "z"  # Optimize for size
```

Lambda handler with S3 integration:

```rust
// src/main.rs
use lambda_http::{run, service_fn, Error, Request, Response, Body};
use aws_sdk_s3::Client as S3Client;
use aws_config::BehaviorVersion;
use serde::{Deserialize, Serialize};
use image::{DynamicImage, ImageFormat};
use bytes::Bytes;
use tracing::info;

#[derive(Deserialize)]
struct ImageRequest {
    bucket: String,
    key: String,
    width: u32,
    height: u32,
}

#[derive(Serialize)]
struct ImageResponse {
    original_size: usize,
    resized_size: usize,
    processing_time_ms: u128,
}

async fn function_handler(event: Request) -> Result<Response<Body>, Error> {
    let start = std::time::Instant::now();
    
    // Parse request
    let body = event.body();
    let request: ImageRequest = serde_json::from_slice(body)?;
    
    // Initialize S3 client (reused across invocations)
    let config = aws_config::defaults(BehaviorVersion::latest()).load().await;
    let s3_client = S3Client::new(&config);
    
    // Download image from S3
    let object = s3_client
        .get_object()
        .bucket(&request.bucket)
        .key(&request.key)
        .send()
        .await?;
    
    let bytes = object.body.collect().await?.into_bytes();
    let original_size = bytes.len();
    
    // Process image
    let img = image::load_from_memory(&bytes)?;
    let resized = img.resize_exact(
        request.width,
        request.height,
        image::imageops::FilterType::Lanczos3,
    );
    
    // Encode to WebP for smaller size
    let mut output = Vec::new();
    resized.write_to(&mut std::io::Cursor::new(&mut output), ImageFormat::WebP)?;
    
    // Upload back to S3
    let output_key = format!("{}-{}x{}.webp", 
        request.key.trim_end_matches(|c: char| c == '.' || c.is_ascii_alphabetic()),
        request.width,
        request.height
    );
    
    s3_client
        .put_object()
        .bucket(&request.bucket)
        .key(&output_key)
        .body(output.clone().into())
        .content_type("image/webp")
        .send()
        .await?;
    
    let processing_time_ms = start.elapsed().as_millis();
    
    info!(
        original_size,
        resized_size = output.len(),
        processing_time_ms,
        "Image processed successfully"
    );
    
    let response = ImageResponse {
        original_size,
        resized_size: output.len(),
        processing_time_ms,
    };
    
    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_string(&response)?.into())?)
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("image_lambda=info".parse()?)
        )
        .json()
        .init();
    
    run(service_fn(function_handler)).await
}
```

Build and deploy:

```bash
# Install cargo-lambda
cargo install cargo-lambda

# Build for Lambda
cargo lambda build --release --arm64

# Deploy with SAM or CDK
# SAM template.yaml
```

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Globals:
  Function:
    Timeout: 30
    MemorySize: 512
    Runtime: provided.al2023
    Architectures:
      - arm64

Resources:
  ImageProcessor:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: target/lambda/image-lambda/
      Handler: bootstrap
      Environment:
        Variables:
          RUST_LOG: info
      Policies:
        - S3CrudPolicy:
            BucketName: !Ref ImageBucket
      Events:
        Api:
          Type: Api
          Properties:
            Path: /process
            Method: post

  ImageBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub ${AWS::StackName}-images
```

### Python 3.13 Free-Threaded Lambda

For CPU-bound Python workloads, use the experimental free-threaded build:

```python
# lambda_function.py
import asyncio
import concurrent.futures
import json
import time
from typing import Any, Dict
import numpy as np
import boto3

# Enable free-threaded Python features
import sys
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)  # No limit

# Thread pool for CPU-bound tasks (actually parallel in free-threaded build)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def cpu_intensive_task(data: np.ndarray) -> np.ndarray:
    """Runs in true parallel thread with free-threaded Python"""
    # Simulate complex computation
    result = np.fft.fft2(data)
    result = np.abs(result)
    return result

async def process_batch(batch: list[np.ndarray]) -> list[np.ndarray]:
    """Process multiple arrays in parallel"""
    loop = asyncio.get_event_loop()
    
    # Submit all tasks to thread pool
    tasks = [
        loop.run_in_executor(executor, cpu_intensive_task, item)
        for item in batch
    ]
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks)
    return results

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    start_time = time.time()
    
    # Parse input
    batch_size = event.get('batch_size', 10)
    array_size = event.get('array_size', 1000)
    
    # Generate test data
    batch = [
        np.random.rand(array_size, array_size) 
        for _ in range(batch_size)
    ]
    
    # Process in parallel
    results = asyncio.run(process_batch(batch))
    
    processing_time = time.time() - start_time
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'processed_items': len(results),
            'processing_time_seconds': processing_time,
            'items_per_second': len(results) / processing_time,
            'thread_count': executor._max_workers,
            'python_version': sys.version,
            'gil_disabled': not hasattr(sys, '_is_gil_enabled') or not sys._is_gil_enabled()
        })
    }
```

Deploy with custom runtime:

```dockerfile
# Dockerfile for free-threaded Python Lambda
FROM public.ecr.aws/lambda/python:3.13-ft-preview as builder

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM public.ecr.aws/lambda/python:3.13-ft-preview
COPY --from=builder /var/lang/lib/python3.13/site-packages /var/lang/lib/python3.13/site-packages
COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]
```

---

## 5. Google Cloud Run: Container-Based Serverless

Cloud Run provides the flexibility of containers with serverless scaling.

### ✅ DO: Use Cloud Run for Complex Applications

```python
# main.py - FastAPI on Cloud Run
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
import asyncio
import httpx
import os
from contextlib import asynccontextmanager
from google.cloud import firestore, storage, pubsub_v1
import uvloop

# Use uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Global clients (reused across requests)
db = firestore.AsyncClient()
storage_client = storage.Client()
publisher = pubsub_v1.PublisherClient()

# HTTP client with connection pooling
http_client = httpx.AsyncClient(
    limits=httpx.Limits(max_keepalive_connections=100, max_connections=1000),
    timeout=30.0,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    await http_client.aclose()

app = FastAPI(lifespan=lifespan)

# Cloud Run automatically sets this
SERVICE_NAME = os.environ.get('K_SERVICE', 'unknown')
REVISION = os.environ.get('K_REVISION', 'unknown')

@app.get("/health")
async def health_check():
    """Cloud Run health check endpoint"""
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "revision": REVISION,
        "region": os.environ.get('REGION', 'unknown'),
    }

@app.post("/process-image")
async def process_image_stream(
    request: Request,
    background_tasks: BackgroundTasks
):
    """Stream processing with Cloud Run's 32GB memory limit"""
    
    async def generate():
        # Stream large file from Cloud Storage
        bucket = storage_client.bucket("my-images")
        blob = bucket.blob("large-image.tiff")
        
        chunk_size = 1024 * 1024  # 1MB chunks
        offset = 0
        
        while True:
            chunk = blob.download_as_bytes(
                start=offset,
                end=offset + chunk_size - 1
            )
            
            if not chunk:
                break
                
            # Process chunk
            processed = await process_chunk(chunk)
            yield processed
            
            offset += chunk_size
    
    # Queue background job via Pub/Sub
    background_tasks.add_task(
        publish_completion_event,
        request.headers.get("x-request-id")
    )
    
    return StreamingResponse(
        generate(),
        media_type="application/octet-stream"
    )

async def publish_completion_event(request_id: str):
    """Publish to Pub/Sub for downstream processing"""
    topic_path = publisher.topic_path(
        os.environ['PROJECT_ID'],
        'image-processing-complete'
    )
    
    publisher.publish(
        topic_path,
        data=request_id.encode('utf-8'),
        service=SERVICE_NAME,
        revision=REVISION,
    )

# Graceful shutdown handling
import signal
import sys

shutdown_event = asyncio.Event()

def signal_handler(sig, frame):
    print(f"Received signal {sig}, initiating graceful shutdown...")
    shutdown_event.set()

signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    import uvicorn
    
    # Cloud Run sets PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        loop="uvloop",
        # Cloud Run handles SSL termination
        proxy_headers=True,
        forwarded_allow_ips="*",
        # Access logs are handled by Cloud Logging
        access_log=False,
    )
```

Cloud Run configuration:

```yaml
# service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: image-processor
  annotations:
    run.googleapis.com/launch-stage: GA
spec:
  template:
    metadata:
      annotations:
        # Scale configuration
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "1000"
        # CPU allocation
        run.googleapis.com/cpu-throttling: "false"
        # Startup probe for slower containers
        run.googleapis.com/startup-cpu-boost: "true"
    spec:
      containerConcurrency: 1000
      timeoutSeconds: 3600  # 1 hour for long-running tasks
      serviceAccountName: image-processor@project.iam.gserviceaccount.com
      containers:
      - image: gcr.io/my-project/image-processor:latest
        resources:
          limits:
            cpu: "8"
            memory: "32Gi"
        env:
        - name: PYTHON_ASYNC_DEBUG
          value: "1"
        startupProbe:
          httpGet:
            path: /health
          initialDelaySeconds: 0
          periodSeconds: 1
          timeoutSeconds: 3
          failureThreshold: 30
```

---

## 6. Azure Container Apps with KEDA Scaling

Container Apps provides Kubernetes-based serverless with advanced scaling via KEDA.

### Event-Driven Scaling with KEDA

```csharp
// Program.cs - .NET 9 on Container Apps
using Azure.Messaging.ServiceBus;
using Azure.Storage.Blobs;
using Azure.Identity;

var builder = WebApplication.CreateBuilder(args);

// Add services
builder.Services.AddSingleton(_ => 
    new ServiceBusClient(
        builder.Configuration["ServiceBusConnectionString"],
        new DefaultAzureCredential()
    )
);

builder.Services.AddSingleton(_ =>
    new BlobServiceClient(
        new Uri($"https://{builder.Configuration["StorageAccount"]}.blob.core.windows.net"),
        new DefaultAzureCredential()
    )
);

// Health checks for Container Apps probes
builder.Services.AddHealthChecks()
    .AddAzureServiceBusQueue(
        builder.Configuration["ServiceBusConnectionString"],
        "image-processing-queue"
    )
    .AddAzureBlobStorage(
        builder.Configuration["StorageConnectionString"]
    );

var app = builder.Build();

app.MapHealthChecks("/health/startup", new HealthCheckOptions
{
    Predicate = check => check.Tags.Contains("startup")
});

app.MapHealthChecks("/health/liveness", new HealthCheckOptions
{
    Predicate = _ => false // Always healthy once started
});

app.MapHealthChecks("/health/readiness");

app.MapPost("/process", async (
    ImageRequest request,
    ServiceBusClient serviceBus,
    BlobServiceClient blobService,
    ILogger<Program> logger) =>
{
    // Download from blob
    var containerClient = blobService.GetBlobContainerClient("images");
    var blobClient = containerClient.GetBlobClient(request.BlobName);
    
    using var ms = new MemoryStream();
    await blobClient.DownloadToAsync(ms);
    
    // Process image (CPU-intensive)
    var processed = await ProcessImageAsync(ms.ToArray());
    
    // Upload result
    var resultBlob = containerClient.GetBlobClient($"processed/{request.BlobName}");
    await resultBlob.UploadAsync(new BinaryData(processed));
    
    // Send completion message
    var sender = serviceBus.CreateSender("completion-queue");
    await sender.SendMessageAsync(new ServiceBusMessage
    {
        Body = BinaryData.FromObjectAsJson(new
        {
            BlobName = request.BlobName,
            ProcessedAt = DateTimeOffset.UtcNow,
            Size = processed.Length
        }),
        ContentType = "application/json"
    });
    
    return Results.Ok(new { ProcessedSize = processed.Length });
});

app.Run();

record ImageRequest(string BlobName, int TargetWidth, int TargetHeight);
```

Bicep deployment with KEDA scaling:

```bicep
// main.bicep
param location string = resourceGroup().location
param containerAppEnvName string
param registryName string

resource containerAppEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' existing = {
  name: containerAppEnvName
}

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: registryName
}

resource serviceBusNamespace 'Microsoft.ServiceBus/namespaces@2023-01-01-preview' = {
  name: 'sb-${uniqueString(resourceGroup().id)}'
  location: location
  sku: {
    name: 'Standard'
  }
}

resource imageProcessorApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'image-processor'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: containerAppEnvironment.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8080
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
      registries: [
        {
          server: containerRegistry.properties.loginServer
          identity: 'system'
        }
      ]
      secrets: [
        {
          name: 'sb-connection'
          value: serviceBusNamespace.listKeys().primaryConnectionString
        }
      ]
    }
    template: {
      containers: [
        {
          image: '${containerRegistry.properties.loginServer}/image-processor:latest'
          name: 'image-processor'
          resources: {
            cpu: 4
            memory: '8Gi'
          }
          env: [
            {
              name: 'ServiceBusConnectionString'
              secretRef: 'sb-connection'
            }
          ]
          probes: [
            {
              type: 'Startup'
              httpGet: {
                path: '/health/startup'
                port: 8080
              }
              failureThreshold: 30
              periodSeconds: 10
            }
            {
              type: 'Liveness'
              httpGet: {
                path: '/health/liveness'
                port: 8080
              }
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 100
        rules: [
          {
            name: 'queue-scale'
            custom: {
              type: 'azure-servicebus'
              metadata: {
                queueName: 'image-processing-queue'
                messageCount: '5'  // Scale up for every 5 messages
                connectionFromEnv: 'ServiceBusConnectionString'
              }
            }
          }
          {
            name: 'http-scale'
            http: {
              metadata: {
                concurrentRequests: '50'
              }
            }
          }
        ]
      }
    }
  }
}
```

---

## 7. Multi-Region Edge Deployment Patterns

### ✅ DO: Implement Smart Request Routing

Route requests to the optimal region based on latency, compliance, and cost:

```typescript
// smart-router.ts - Cloudflare Worker
interface RegionConfig {
  endpoint: string
  priority: number
  maxLatency: number
  dataResidency?: string[]  // GDPR compliance
}

const REGIONS: Record<string, RegionConfig> = {
  'us-east': {
    endpoint: 'https://us-east.api.example.com',
    priority: 1,
    maxLatency: 50,
  },
  'eu-west': {
    endpoint: 'https://eu-west.api.example.com',
    priority: 1,
    maxLatency: 50,
    dataResidency: ['EU', 'UK'],
  },
  'ap-south': {
    endpoint: 'https://ap-south.api.example.com',
    priority: 2,
    maxLatency: 100,
  },
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const country = request.cf?.country || 'US'
    const continent = request.cf?.continent || 'NA'
    
    // Check data residency requirements
    const requiresEU = ['DE', 'FR', 'IT', 'ES', 'NL'].includes(country)
    
    // Health check all regions in parallel
    const healthChecks = await Promise.allSettled(
      Object.entries(REGIONS).map(async ([region, config]) => {
        const start = Date.now()
        
        try {
          const response = await fetch(`${config.endpoint}/health`, {
            method: 'GET',
            signal: AbortSignal.timeout(config.maxLatency),
          })
          
          if (!response.ok) throw new Error('Unhealthy')
          
          const latency = Date.now() - start
          
          return {
            region,
            config,
            latency,
            healthy: true,
            score: (100 - latency) * config.priority,
          }
        } catch {
          return {
            region,
            config,
            latency: Infinity,
            healthy: false,
            score: -1,
          }
        }
      })
    )
    
    // Filter and sort regions
    const availableRegions = healthChecks
      .filter((result): result is PromiseFulfilledResult<any> => 
        result.status === 'fulfilled' && result.value.healthy
      )
      .map(result => result.value)
      .filter(region => {
        // Respect data residency
        if (requiresEU && !region.config.dataResidency?.includes('EU')) {
          return false
        }
        return true
      })
      .sort((a, b) => b.score - a.score)
    
    if (availableRegions.length === 0) {
      return new Response('No healthy regions available', { status: 503 })
    }
    
    // Route to best region
    const selectedRegion = availableRegions[0]
    
    // Add region info to headers
    const headers = new Headers(request.headers)
    headers.set('X-Edge-Region', request.cf?.colo || 'unknown')
    headers.set('X-Origin-Region', selectedRegion.region)
    headers.set('X-Region-Latency', selectedRegion.latency.toString())
    
    // Proxy request
    const response = await fetch(selectedRegion.config.endpoint + new URL(request.url).pathname, {
      method: request.method,
      headers,
      body: request.body,
      cf: {
        cacheTtl: 300,
        cacheEverything: request.method === 'GET',
      },
    })
    
    // Add performance headers
    const modifiedResponse = new Response(response.body, response)
    modifiedResponse.headers.set('X-Region', selectedRegion.region)
    modifiedResponse.headers.set('X-Latency', selectedRegion.latency.toString())
    
    return modifiedResponse
  },
}
```

---

## 8. Edge Authentication Patterns

### ✅ DO: Validate JWTs at the Edge

Perform authentication as close to users as possible:

```typescript
// edge-auth.ts - JWT validation at edge
import { jwtVerify, importSPKI } from 'jose'

// Cache public keys in KV
const PUBLIC_KEY_CACHE = new Map<string, CryptoKey>()

interface JWTPayload {
  sub: string
  email: string
  roles: string[]
  exp: number
}

async function getPublicKey(env: Env, kid: string): Promise<CryptoKey> {
  // Check memory cache first
  if (PUBLIC_KEY_CACHE.has(kid)) {
    return PUBLIC_KEY_CACHE.get(kid)!
  }
  
  // Check KV cache
  const cached = await env.AUTH_KEYS.get(kid, 'text')
  if (cached) {
    const key = await importSPKI(cached, 'RS256')
    PUBLIC_KEY_CACHE.set(kid, key)
    return key
  }
  
  // Fetch from JWKS endpoint
  const response = await fetch(`${env.AUTH_DOMAIN}/.well-known/jwks.json`)
  const jwks = await response.json()
  
  const jwk = jwks.keys.find((k: any) => k.kid === kid)
  if (!jwk) {
    throw new Error('Unknown key ID')
  }
  
  const publicKey = await crypto.subtle.importKey(
    'jwk',
    jwk,
    { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' },
    true,
    ['verify']
  )
  
  // Export and cache the key
  const exported = await crypto.subtle.exportKey('spki', publicKey)
  const pem = toPEM(exported)
  
  await env.AUTH_KEYS.put(kid, pem, {
    expirationTtl: 86400, // 24 hours
  })
  
  PUBLIC_KEY_CACHE.set(kid, publicKey)
  return publicKey
}

export async function authenticateRequest(
  request: Request,
  env: Env
): Promise<JWTPayload | null> {
  const authorization = request.headers.get('Authorization')
  if (!authorization?.startsWith('Bearer ')) {
    return null
  }
  
  const token = authorization.slice(7)
  
  try {
    // Decode header to get kid
    const parts = token.split('.')
    const header = JSON.parse(atob(parts[0]))
    
    // Get public key
    const publicKey = await getPublicKey(env, header.kid)
    
    // Verify JWT
    const { payload } = await jwtVerify(token, publicKey, {
      issuer: env.AUTH_DOMAIN,
      audience: env.API_AUDIENCE,
    })
    
    return payload as JWTPayload
  } catch (error) {
    console.error('JWT verification failed:', error)
    return null
  }
}

// Middleware pattern
export function requireAuth(
  handler: (request: Request, env: Env, auth: JWTPayload) => Promise<Response>
) {
  return async (request: Request, env: Env): Promise<Response> => {
    const auth = await authenticateRequest(request, env)
    
    if (!auth) {
      return new Response('Unauthorized', { 
        status: 401,
        headers: {
          'WWW-Authenticate': 'Bearer',
        },
      })
    }
    
    // Check expiration
    if (auth.exp * 1000 < Date.now()) {
      return new Response('Token expired', { status: 401 })
    }
    
    return handler(request, env, auth)
  }
}

// Usage
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url)
    
    // Public endpoints
    if (url.pathname === '/health') {
      return new Response('OK')
    }
    
    // Protected endpoints
    if (url.pathname.startsWith('/api/')) {
      return requireAuth(async (req, env, auth) => {
        // Access auth.sub, auth.email, auth.roles
        return Response.json({
          message: `Hello ${auth.email}`,
          roles: auth.roles,
        })
      })(request, env)
    }
    
    return new Response('Not found', { status: 404 })
  },
}
```

---

## 9. Observability and Monitoring

### Distributed Tracing Across Platforms

Implement OpenTelemetry for unified observability:

```typescript
// telemetry.ts - Edge-compatible tracing
import { trace, context, SpanStatusCode } from '@opentelemetry/api'

// Initialize tracer (platform-specific)
const tracer = trace.getTracer('edge-api', '1.0.0')

export function instrument<T extends (...args: any[]) => any>(
  name: string,
  fn: T,
  attributes?: Record<string, any>
): T {
  return ((...args) => {
    const span = tracer.startSpan(name, {
      attributes: {
        'edge.runtime': 'cloudflare',
        'edge.region': globalThis.CF_REGION || 'unknown',
        ...attributes,
      },
    })
    
    return context.with(trace.setSpan(context.active(), span), async () => {
      try {
        const result = await fn(...args)
        span.setStatus({ code: SpanStatusCode.OK })
        return result
      } catch (error) {
        span.setStatus({ 
          code: SpanStatusCode.ERROR,
          message: error instanceof Error ? error.message : 'Unknown error',
        })
        span.recordException(error as Error)
        throw error
      } finally {
        span.end()
      }
    })
  }) as T
}

// Cloudflare Analytics Engine for metrics
interface AnalyticsEngine {
  writeDataPoint(data: {
    indexes: string[]
    doubles: number[]
    blobs: string[]
  }): void
}

export class EdgeMetrics {
  constructor(private ae: AnalyticsEngine) {}
  
  recordLatency(operation: string, latencyMs: number, tags: Record<string, string> = {}) {
    this.ae.writeDataPoint({
      indexes: [
        operation,
        tags.region || 'unknown',
        tags.status || 'success',
      ],
      doubles: [latencyMs, 1], // latency and count
      blobs: [new Date().toISOString()],
    })
  }
  
  recordError(operation: string, error: Error, tags: Record<string, string> = {}) {
    this.ae.writeDataPoint({
      indexes: [
        `error:${operation}`,
        error.name,
        tags.region || 'unknown',
      ],
      doubles: [1], // error count
      blobs: [error.message, error.stack || ''],
    })
  }
}

// Usage in Worker
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const metrics = new EdgeMetrics(env.ANALYTICS)
    
    return instrument('handle_request', async () => {
      const start = Date.now()
      
      try {
        const response = await handleRequest(request, env)
        
        metrics.recordLatency('request', Date.now() - start, {
          status: response.status.toString(),
          region: request.cf?.colo,
        })
        
        return response
      } catch (error) {
        metrics.recordError('request', error as Error, {
          region: request.cf?.colo,
        })
        throw error
      }
    })()
  },
}
```

---

## 10. Cost Optimization Strategies

### ✅ DO: Implement Request Coalescing

Reduce costs by deduplicating concurrent identical requests:

```typescript
// request-coalescer.ts
class RequestCoalescer {
  private pending = new Map<string, Promise<Response>>()
  
  async fetch(
    key: string,
    fetcher: () => Promise<Response>
  ): Promise<Response> {
    // Check if identical request is in flight
    const existing = this.pending.get(key)
    if (existing) {
      // Return cloned response to avoid body consumption issues
      const response = await existing
      return response.clone()
    }
    
    // Create new request
    const promise = fetcher().finally(() => {
      // Clean up after 100ms to allow for burst of identical requests
      setTimeout(() => this.pending.delete(key), 100)
    })
    
    this.pending.set(key, promise)
    
    const response = await promise
    return response.clone()
  }
}

// Usage in edge function
const coalescer = new RequestCoalescer()

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url)
    const cacheKey = `${url.pathname}${url.search}`
    
    // Only coalesce GET requests
    if (request.method !== 'GET') {
      return handleRequest(request, env)
    }
    
    return coalescer.fetch(cacheKey, () => handleRequest(request, env))
  },
}
```

### Intelligent Caching Strategy

```typescript
// cache-strategy.ts
interface CacheStrategy {
  ttl: number
  staleWhileRevalidate?: number
  tags?: string[]
  varyBy?: string[]
}

const CACHE_STRATEGIES: Record<string, CacheStrategy> = {
  // Static assets - cache forever
  '/assets/*': {
    ttl: 31536000, // 1 year
    varyBy: ['accept-encoding'],
  },
  
  // API responses - short TTL with background refresh
  '/api/products': {
    ttl: 300, // 5 minutes
    staleWhileRevalidate: 86400, // 24 hours
    tags: ['products'],
    varyBy: ['accept', 'authorization'],
  },
  
  // Personalized content - cache per user
  '/api/user/*': {
    ttl: 60,
    varyBy: ['authorization', 'accept'],
  },
}

export async function withCache(
  request: Request,
  env: Env,
  handler: () => Promise<Response>
): Promise<Response> {
  const url = new URL(request.url)
  const strategy = findStrategy(url.pathname)
  
  if (!strategy) {
    return handler()
  }
  
  // Build cache key with vary headers
  const varyHeaders = strategy.varyBy?.map(h => 
    `${h}:${request.headers.get(h) || 'null'}`
  ).join(',') || ''
  
  const cacheKey = new Request(
    `https://cache.internal${url.pathname}${url.search}?vary=${varyHeaders}`
  )
  
  const cache = caches.default
  
  // Check cache
  const cached = await cache.match(cacheKey)
  
  if (cached) {
    const age = Date.now() - new Date(cached.headers.get('date') || 0).getTime()
    const maxAge = strategy.ttl * 1000
    const staleTime = (strategy.staleWhileRevalidate || 0) * 1000
    
    if (age < maxAge) {
      // Fresh - return immediately
      return cached
    } else if (age < maxAge + staleTime) {
      // Stale - return cached but refresh in background
      env.waitUntil(
        handler().then(fresh => {
          if (fresh.ok) {
            cache.put(cacheKey, fresh.clone())
          }
        })
      )
      
      const staleResponse = new Response(cached.body, cached)
      staleResponse.headers.set('x-cache-status', 'stale')
      return staleResponse
    }
  }
  
  // Cache miss or expired - fetch fresh
  const fresh = await handler()
  
  if (fresh.ok && strategy.ttl > 0) {
    const cacheResponse = new Response(fresh.body, fresh)
    cacheResponse.headers.set('cache-control', `public, max-age=${strategy.ttl}`)
    
    if (strategy.tags) {
      cacheResponse.headers.set('cache-tags', strategy.tags.join(','))
    }
    
    env.waitUntil(cache.put(cacheKey, cacheResponse.clone()))
  }
  
  fresh.headers.set('x-cache-status', 'miss')
  return fresh
}
```

---

## 11. Testing Strategies

### Unit Testing WASM Components

```rust
// tests/wasm_tests.rs
#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;
    
    wasm_bindgen_test_configure!(run_in_browser);
    
    #[wasm_bindgen_test]
    fn test_image_resize() {
        // Create test image
        let test_data = include_bytes!("../fixtures/test.jpg");
        
        let processor = ImageProcessor::new(test_data)
            .expect("Failed to create processor");
        
        let resized = processor.resize(100, 100);
        
        assert!(resized.len() > 0);
        assert!(resized.len() < test_data.len()); // Should be smaller
    }
    
    #[wasm_bindgen_test]
    async fn test_async_processing() {
        use wasm_bindgen_futures::JsFuture;
        
        let promise = js_sys::Promise::resolve(&JsValue::from(42));
        let result = JsFuture::from(promise).await.unwrap();
        
        assert_eq!(result, 42);
    }
}
```

### Integration Testing Edge Functions

```typescript
// test/edge.test.ts
import { describe, it, expect, beforeAll } from 'vitest'
import { unstable_dev } from 'wrangler'
import type { UnstableDevWorker } from 'wrangler'

describe('Edge Worker Tests', () => {
  let worker: UnstableDevWorker
  
  beforeAll(async () => {
    worker = await unstable_dev('src/index.ts', {
      experimental: { disableExperimentalWarning: true },
      vars: {
        AUTH_DOMAIN: 'https://test.auth.com',
        API_AUDIENCE: 'test-api',
      },
    })
  })
  
  afterAll(async () => {
    await worker.stop()
  })
  
  it('should handle health check', async () => {
    const response = await worker.fetch('/health')
    expect(response.status).toBe(200)
    
    const text = await response.text()
    expect(text).toBe('OK')
  })
  
  it('should reject unauthenticated requests', async () => {
    const response = await worker.fetch('/api/protected')
    expect(response.status).toBe(401)
    expect(response.headers.get('WWW-Authenticate')).toBe('Bearer')
  })
  
  it('should validate JWT tokens', async () => {
    // Mock JWT token
    const mockToken = generateMockJWT({
      sub: 'test-user',
      email: 'test@example.com',
      roles: ['user'],
    })
    
    const response = await worker.fetch('/api/protected', {
      headers: {
        'Authorization': `Bearer ${mockToken}`,
      },
    })
    
    expect(response.status).toBe(200)
    const data = await response.json()
    expect(data.message).toBe('Hello test@example.com')
  })
  
  it('should coalesce duplicate requests', async () => {
    // Send 10 identical requests simultaneously
    const requests = Array(10).fill(null).map(() => 
      worker.fetch('/api/expensive-operation')
    )
    
    const responses = await Promise.all(requests)
    
    // All should succeed
    responses.forEach(r => expect(r.status).toBe(200))
    
    // Check that operation was only called once (via headers or logs)
    const callCounts = responses.map(r => 
      parseInt(r.headers.get('x-operation-calls') || '0')
    )
    
    expect(Math.max(...callCounts)).toBe(1)
  })
})
```

### Load Testing Serverless Functions

```yaml
# artillery-config.yml
config:
  target: "https://edge.example.com"
  phases:
    - duration: 60
      arrivalRate: 10
      name: "Warm up"
    - duration: 300
      arrivalRate: 100
      rampTo: 1000
      name: "Ramp up load"
    - duration: 600
      arrivalRate: 1000
      name: "Sustained load"
  processor: "./load-test-processor.js"
  variables:
    authToken: "{{ $processEnvironment.AUTH_TOKEN }}"

scenarios:
  - name: "Image Processing"
    weight: 60
    flow:
      - post:
          url: "/process"
          headers:
            Authorization: "Bearer {{ authToken }}"
          json:
            imageUrl: "https://example.com/test-{{ $randomNumber(1, 100) }}.jpg"
            width: "{{ $randomNumber(100, 1000) }}"
            height: "{{ $randomNumber(100, 1000) }}"
          capture:
            - json: "$.taskId"
              as: "taskId"
      
      - think: 2
      
      - get:
          url: "/status/{{ taskId }}"
          headers:
            Authorization: "Bearer {{ authToken }}"
          expect:
            - statusCode: 200
            - contentType: json
            - hasProperty: result
  
  - name: "Cache Hit Test"
    weight: 40
    flow:
      - get:
          url: "/api/products?category=featured"
          capture:
            - header: "x-cache-status"
              as: "cacheStatus"
          expect:
            - statusCode: 200
      
      - think: 0.5
      
      - get:
          url: "/api/products?category=featured"
          expect:
            - statusCode: 200
            - header: "x-cache-status"
              equals: "hit"
```

---

## 12. CI/CD for Multi-Platform Deployment

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy to Multiple Platforms

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  RUST_VERSION: "1.85.0"
  NODE_VERSION: "22.5.0"
  PYTHON_VERSION: "3.13"

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          toolchain: ${{ env.RUST_VERSION }}
          targets: wasm32-unknown-unknown,wasm32-wasi
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
      
      - name: Cache dependencies
        uses: Swatinem/rust-cache@v2
        with:
          workspaces: |
            rust-components
            wasm-modules
      
      - name: Install tools
        run: |
          curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
          npm install -g wrangler@latest
          cargo install cargo-component
      
      - name: Test Rust components
        run: |
          cd rust-components
          cargo test --all-features
          cargo clippy -- -D warnings
      
      - name: Build WASM modules
        run: |
          cd wasm-modules
          wasm-pack build --target web --out-dir ../edge-functions/pkg
      
      - name: Test Edge functions
        run: |
          cd edge-functions
          npm ci
          npm test

  deploy-cloudflare:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to Cloudflare Workers
        uses: cloudflare/wrangler-action@v3
        with:
          apiToken: ${{ secrets.CLOUDFLARE_API_TOKEN }}
          workingDirectory: 'edge-functions'
          command: deploy --env production
      
      - name: Verify deployment
        run: |
          curl -f https://api.example.com/health || exit 1

  deploy-lambda:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Rust for Lambda
        uses: dtolnay/rust-toolchain@stable
        with:
          toolchain: ${{ env.RUST_VERSION }}
          targets: aarch64-unknown-linux-gnu
      
      - name: Install cargo-lambda
        run: cargo install cargo-lambda
      
      - name: Build Lambda functions
        run: |
          cd lambda-functions
          cargo lambda build --release --arm64
      
      - name: Deploy with SAM
        run: |
          cd lambda-functions
          sam build
          sam deploy --no-confirm-changeset \
            --parameter-overrides \
              Environment=production \
            --stack-name image-processor-prod

  deploy-cloud-run:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup GCP
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      
      - name: Configure Docker
        run: gcloud auth configure-docker
      
      - name: Build and push image
        run: |
          cd cloud-run-service
          docker build -t gcr.io/${{ vars.GCP_PROJECT }}/image-processor:${{ github.sha }} .
          docker push gcr.io/${{ vars.GCP_PROJECT }}/image-processor:${{ github.sha }}
      
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy image-processor \
            --image gcr.io/${{ vars.GCP_PROJECT }}/image-processor:${{ github.sha }} \
            --region us-central1 \
            --platform managed \
            --memory 2Gi \
            --cpu 2 \
            --max-instances 1000 \
            --min-instances 1
```

---

## 13. Security Best Practices

### Content Security Policy at the Edge

```typescript
// security-headers.ts
export function addSecurityHeaders(response: Response): Response {
  const headers = new Headers(response.headers)
  
  // Content Security Policy
  headers.set('Content-Security-Policy', [
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.example.com",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com",
    "img-src 'self' data: https:",
    "connect-src 'self' https://api.example.com wss://ws.example.com",
    "frame-ancestors 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "upgrade-insecure-requests",
  ].join('; '))
  
  // Other security headers
  headers.set('X-Content-Type-Options', 'nosniff')
  headers.set('X-Frame-Options', 'DENY')
  headers.set('X-XSS-Protection', '1; mode=block')
  headers.set('Referrer-Policy', 'strict-origin-when-cross-origin')
  headers.set('Permissions-Policy', 'geolocation=(), microphone=(), camera=()')
  
  // HSTS (only on HTTPS)
  if (response.url.startsWith('https://')) {
    headers.set('Strict-Transport-Security', 'max-age=31536000; includeSubDomains; preload')
  }
  
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  })
}

// Rate limiting with sliding window
export class RateLimiter {
  constructor(
    private kv: KVNamespace,
    private limit: number = 100,
    private windowMs: number = 60000
  ) {}
  
  async checkLimit(identifier: string): Promise<{ allowed: boolean; remaining: number }> {
    const now = Date.now()
    const key = `ratelimit:${identifier}`
    const windowStart = now - this.windowMs
    
    // Get current window data
    const data = await this.kv.get<number[]>(key, 'json') || []
    
    // Filter out old entries
    const validEntries = data.filter(timestamp => timestamp > windowStart)
    
    if (validEntries.length >= this.limit) {
      return { allowed: false, remaining: 0 }
    }
    
    // Add current request
    validEntries.push(now)
    
    // Save back to KV with TTL
    await this.kv.put(key, JSON.stringify(validEntries), {
      expirationTtl: Math.ceil(this.windowMs / 1000),
    })
    
    return {
      allowed: true,
      remaining: this.limit - validEntries.length,
    }
  }
}
```

---

## 14. Performance Optimization Techniques

### ✅ DO: Optimize Cold Starts

```javascript
// Lambda optimization for cold starts
// handler.js

// Move heavy imports inside handler for better tree-shaking
let heavyDependency;

// Reuse connections across invocations
let dbConnection;

export async function handler(event, context) {
  // Lazy load heavy dependencies
  if (!heavyDependency) {
    heavyDependency = await import('heavy-library')
  }
  
  // Reuse database connection
  if (!dbConnection) {
    dbConnection = await createConnection({
      // Connection config
    })
  }
  
  // Keep Lambda warm by not waiting for empty event loop
  context.callbackWaitsForEmptyEventLoop = false
  
  try {
    const result = await processRequest(event, dbConnection)
    return {
      statusCode: 200,
      body: JSON.stringify(result),
      headers: {
        'Content-Type': 'application/json',
      },
    }
  } catch (error) {
    console.error('Error:', error)
    return {
      statusCode: 500,
      body: JSON.stringify({ error: 'Internal server error' }),
    }
  }
}

// Rust Lambda optimization
use once_cell::sync::Lazy;
use aws_sdk_s3::Client as S3Client;

// Initialize clients once
static S3_CLIENT: Lazy<S3Client> = Lazy::new(|| {
    let config = aws_config::load_from_env().await;
    S3Client::new(&config)
});

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Pre-warm the lazy statics
    Lazy::force(&S3_CLIENT);
    
    // Start the runtime
    lambda_runtime::run(service_fn(function_handler)).await
}
```

### WebAssembly SIMD Optimization

Enable SIMD for 4-8x performance on supported operations:

```rust
// Cargo.toml
[dependencies]
packed_simd = "0.3"

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = "0.2"

// lib.rs
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_feature = "simd128")]
use core::arch::wasm32::*;

#[wasm_bindgen]
pub fn process_image_simd(data: &[u8], width: u32, height: u32) -> Vec<u8> {
    #[cfg(target_feature = "simd128")]
    {
        // SIMD-optimized image processing
        let mut output = vec![0u8; data.len()];
        
        for chunk in data.chunks_exact(16) {
            unsafe {
                // Load 16 bytes at once
                let pixels = v128_load(chunk.as_ptr() as *const v128);
                
                // Apply transformations using SIMD
                let brightened = u8x16_add_sat(pixels, u8x16_splat(20));
                
                // Store result
                v128_store(output.as_mut_ptr() as *mut v128, brightened);
            }
        }
        
        output
    }
    
    #[cfg(not(target_feature = "simd128"))]
    {
        // Fallback to scalar implementation
        data.iter().map(|&p| p.saturating_add(20)).collect()
    }
}
```

---

## 15. Future-Proofing Your Architecture

### Prepare for Emerging Standards

```typescript
// future-ready-patterns.ts

// 1. Web Streams API for efficient data processing
export async function* processLargeFile(stream: ReadableStream<Uint8Array>) {
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      
      buffer += decoder.decode(value, { stream: true })
      
      // Process complete lines
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      
      for (const line of lines) {
        yield processLine(line)
      }
    }
    
    // Process remaining buffer
    if (buffer) {
      yield processLine(buffer)
    }
  } finally {
    reader.releaseLock()
  }
}

// 2. Native ES Modules in Workers
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // Dynamic imports for code splitting
    const { processRequest } = await import('./handlers/main.js')
    return processRequest(request, env)
  }
}

// 3. WebGPU for ML inference at the edge (experimental)
interface GPUDevice {
  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer
  createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline
}

async function runMLInference(device: GPUDevice, inputData: Float32Array) {
  // Create GPU buffers
  const inputBuffer = device.createBuffer({
    size: inputData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  
  // Run inference
  // ... GPU compute shader execution
}

// 4. HTTP/3 and QUIC support
export async function fetchWithQuic(url: string): Promise<Response> {
  return fetch(url, {
    // Future: explicit QUIC hints
    // @ts-ignore - Future API
    transport: 'quic',
    // Current: rely on automatic protocol negotiation
  })
}
```

### Migration Strategies

Plan for platform changes and API evolution:

```typescript
// platform-abstraction.ts
interface RuntimeAdapter {
  getRequestIP(request: Request): string
  getRequestCountry(request: Request): string
  waitUntil(promise: Promise<any>): void
  getKVStore(): KVStore
}

// Cloudflare adapter
class CloudflareAdapter implements RuntimeAdapter {
  constructor(private env: any, private ctx: ExecutionContext) {}
  
  getRequestIP(request: Request): string {
    return request.headers.get('CF-Connecting-IP') || 'unknown'
  }
  
  getRequestCountry(request: Request): string {
    return (request as any).cf?.country || 'unknown'
  }
  
  waitUntil(promise: Promise<any>): void {
    this.ctx.waitUntil(promise)
  }
  
  getKVStore(): KVStore {
    return new CloudflareKVStore(this.env.KV)
  }
}

// Deno adapter
class DenoAdapter implements RuntimeAdapter {
  private kv = Deno.openKv()
  
  getRequestIP(request: Request): string {
    return request.headers.get('X-Forwarded-For')?.split(',')[0] || 'unknown'
  }
  
  getRequestCountry(request: Request): string {
    // Use GeoIP service or header
    return request.headers.get('CloudFront-Viewer-Country') || 'unknown'
  }
  
  waitUntil(promise: Promise<any>): void {
    // Deno handles this automatically
  }
  
  async getKVStore(): Promise<KVStore> {
    return new DenoKVStore(await this.kv)
  }
}

// Platform-agnostic handler
export function createHandler(adapter: RuntimeAdapter) {
  return async (request: Request): Promise<Response> => {
    const ip = adapter.getRequestIP(request)
    const country = adapter.getRequestCountry(request)
    const kv = adapter.getKVStore()
    
    // Your business logic here
    const cached = await kv.get(`cache:${request.url}`)
    if (cached) {
      return new Response(cached, {
        headers: { 'X-Cache': 'HIT' }
      })
    }
    
    const response = await generateResponse(request, { ip, country })
    
    // Cache in background
    adapter.waitUntil(
      kv.put(`cache:${request.url}`, response.clone(), { ttl: 300 })
    )
    
    return response
  }
}
```

---

## Conclusion

The edge computing and serverless landscape in 2025 offers unprecedented flexibility and performance. Key takeaways:

1. **Choose the right platform**: V8 isolates for low-latency global distribution, FaaS for compute-intensive tasks, containers for complex applications
2. **Embrace WASM**: Use WebAssembly for performance-critical code that runs everywhere
3. **Think globally**: Design for multi-region deployment from day one
4. **Optimize aggressively**: Every millisecond counts at the edge
5. **Plan for scale**: Use smart caching, request coalescing, and efficient state management
6. **Monitor everything**: Distributed systems require comprehensive observability
7. **Stay secure**: Implement defense in depth with authentication at the edge
8. **Test thoroughly**: From unit tests to global load testing
9. **Automate deployment**: Multi-platform CI/CD is essential
10. **Future-proof**: Abstract platform differences and prepare for emerging standards

The future is distributed, and the edge is where innovation happens. Build fast, deploy everywhere, and delight your users with sub-50ms response times globally.