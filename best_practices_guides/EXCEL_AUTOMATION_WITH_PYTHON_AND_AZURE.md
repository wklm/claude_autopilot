# The Definitive Guide to Python Excel Automation with Azure VM Orchestration (mid-2025 Edition)

This guide presents production-grade patterns for building a scalable, secure, and cost-efficient Excel automation platform using Python 3.13+, FastAPI, Azure VMs, and modern Excel integration techniques. We'll move beyond simple COM automation to create a distributed computation engine that leverages Excel's calculation engine at scale.

## Table of Contents
1. [Architecture Overview & Core Principles](#1-architecture-overview--core-principles)
2. [High-Performance Python UDFs with xlwings Pro](#2-high-performance-python-udfs-with-xlwings-pro)
3. [FastAPI Orchestration Layer](#3-fastapi-orchestration-layer)
4. [Azure VM Pool Management](#4-azure-vm-pool-management)
5. [SharePoint Integration & File Management](#5-sharepoint-integration--file-management)
6. [Security & Network Architecture](#6-security--network-architecture)
7. [Cost Optimization Strategies](#7-cost-optimization-strategies)
8. [Production Deployment](#8-production-deployment)
9. [Advanced Patterns](#9-advanced-patterns)

## Prerequisites & Technology Stack

- **Python 3.13+** with free-threaded build support
- **FastAPI 0.120+** with async-first patterns
- **xlwings Pro 0.32+** for high-performance UDFs
- **Azure SDK 2025.1+** with native async support
- **Windows Server 2022/2025** for Excel hosts
- **Excel 2021/365** with modern calculation engine

```toml
# pyproject.toml excerpt
[project]
requires-python = ">=3.13"
dependencies = [
    "fastapi>=0.120.0",
    "uvicorn[standard]>=0.35.0",
    "xlwings>=0.32.0",
    "azure-mgmt-compute>=34.0.0",
    "azure-identity>=1.20.0",
    "azure-storage-blob>=12.25.0",
    "pywin32>=308",
    "aiocache[redis]>=0.12.3",
    "httpx>=0.25.0",
    "pydantic>=2.7.0",
    "sqlmodel>=0.0.15",
    "rich>=13.7.0",
    "python-decouple>=3.8",
    "asyncssh>=2.18.0",
    "pywinrm[kerberos]>=0.5.0",
    "msgraph-sdk>=1.12.0",
]
```

---

## 1. Architecture Overview & Core Principles

### System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client Apps   │────▶│  FastAPI Gateway │────▶│   Redis Cache   │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
              ┌─────▼──────┐           ┌─────▼──────┐
              │ VM Pool     │           │ SharePoint │
              │ Manager     │           │ Connector  │
              └─────┬──────┘           └────────────┘
                    │
         ┌──────────┴───────────┐
         │                      │
    ┌────▼─────┐          ┌────▼─────┐
    │ Excel VM │          │ Excel VM │
    │ (Warm)   │          │ (Cold)   │
    └──────────┘          └──────────┘
```

### Core Design Principles

1. **Pool-Based VM Management**: Maintain warm VM pools to eliminate cold start latency
2. **Async-First Architecture**: Every operation is non-blocking from API to Excel
3. **Smart Caching**: Cache both computation results and intermediate Excel states
4. **Security by Default**: Zero-trust networking, managed identities, encrypted channels
5. **Cost Optimization**: Spot instances, auto-scaling, aggressive cleanup policies

### Project Structure

```
/excel-automation-platform
├── src/
│   ├── api/                      # FastAPI application layer
│   │   ├── main.py               # Application entry point
│   │   ├── routers/              # API endpoints
│   │   │   ├── calculations.py   # Excel calculation endpoints
│   │   │   └── health.py         # Health checks
│   │   └── dependencies.py       # Shared dependencies
│   ├── core/                     # Core business logic
│   │   ├── vm_manager.py         # Azure VM lifecycle management
│   │   ├── excel_engine.py       # Excel COM automation
│   │   ├── sharepoint_client.py  # SharePoint file operations
│   │   └── cache_manager.py      # Redis caching layer
│   ├── excel/                    # Excel-specific code
│   │   ├── udfs/                 # Python UDF modules
│   │   │   ├── __init__.py
│   │   │   ├── calculations.py   # Business logic UDFs
│   │   │   └── data_fetchers.py  # Data access UDFs
│   │   └── xlwings_server.py     # xlwings server configuration
│   ├── models/                   # Pydantic models
│   │   ├── requests.py           # API request models
│   │   ├── responses.py          # API response models
│   │   └── azure.py              # Azure resource models
│   └── utils/                    # Utility functions
│       ├── security.py           # Security helpers
│       └── monitoring.py         # Telemetry and logging
├── deployment/                   # Deployment configurations
│   ├── terraform/                # Infrastructure as Code
│   ├── docker/                   # Container definitions
│   └── scripts/                  # Deployment scripts
├── tests/                        # Integration tests
└── pyproject.toml                # Project configuration
```

---

## 2. High-Performance Python UDFs with xlwings Pro

### ✅ DO: Use xlwings Pro REST API for Maximum Performance

The xlwings Pro REST API server enables Excel to call Python functions with minimal overhead, supporting async operations and connection pooling.

```python
# src/excel/xlwings_server.py
from decouple import Config as DecoupleConfig, RepositoryEnv
from xlwings import serve
import asyncio
from contextlib import asynccontextmanager
import httpx
from redis import asyncio as aioredis

# Load configuration
decouple_config = DecoupleConfig(RepositoryEnv(".env"))

# Global resources for connection pooling
http_client: httpx.AsyncClient = None
redis_pool: aioredis.Redis = None

@asynccontextmanager
async def lifespan():
    """Initialize and cleanup global resources"""
    global http_client, redis_pool
    
    # Startup
    http_client = httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(max_keepalive_connections=20)
    )
    redis_pool = await aioredis.from_url(
        decouple_config("REDIS_URL"),
        encoding="utf-8",
        decode_responses=True,
        max_connections=50
    )
    
    yield
    
    # Cleanup
    await http_client.aclose()
    await redis_pool.close()

# Configure xlwings server
app = serve(
    lifespan=lifespan,
    host="0.0.0.0",
    port=5000,
    ssl_keyfile=decouple_config("SSL_KEY_PATH", default=None),
    ssl_certfile=decouple_config("SSL_CERT_PATH", default=None),
)
```

### ✅ DO: Implement Async UDFs with Caching

```python
# src/excel/udfs/calculations.py
import xlwings as xw
import numpy as np
from typing import List, Dict, Any, Optional
import hashlib
import json
import asyncio
from functools import wraps

def cache_result(ttl: int = 3600):
    """Decorator for caching UDF results in Redis"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"udf:{func.__name__}:{hashlib.md5(
                json.dumps([args, kwargs], sort_keys=True).encode()
            ).hexdigest()}"
            
            # Check cache first
            cached = await redis_pool.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await redis_pool.setex(cache_key, ttl, json.dumps(result))
            
            return result
        return wrapper
    return decorator

@xw.func
@xw.arg('data', ndim=2)  # Expect 2D array
@xw.ret(expand='table')   # Return as dynamic array
async def monte_carlo_simulation(
    data: np.ndarray,
    iterations: int = 10000,
    confidence_level: float = 0.95
) -> np.ndarray:
    """
    High-performance Monte Carlo simulation using NumPy
    Leverages AVX-512 SIMD instructions on modern CPUs
    """
    # Validate inputs
    if data.shape[0] < 2:
        raise ValueError("Need at least 2 data points")
    
    # Calculate parameters from historical data
    returns = np.diff(data, axis=0) / data[:-1]
    mean_return = np.mean(returns, axis=0)
    cov_matrix = np.cov(returns.T)
    
    # Vectorized Monte Carlo simulation
    num_assets = data.shape[1]
    latest_prices = data[-1]
    
    # Generate random samples using NumPy's fast random generator
    rng = np.random.default_rng()
    random_returns = rng.multivariate_normal(
        mean_return, cov_matrix, size=iterations
    )
    
    # Calculate future values
    future_values = latest_prices * np.exp(random_returns)
    
    # Calculate statistics
    percentiles = np.percentile(future_values, [5, 50, 95], axis=0)
    
    # Return results as 2D array for Excel
    results = np.vstack([
        latest_prices,
        percentiles[1],  # Median
        percentiles[0],  # 5th percentile
        percentiles[2],  # 95th percentile
    ])
    
    return results.T

@xw.func
@cache_result(ttl=300)  # Cache for 5 minutes
async def fetch_market_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> List[List[Any]]:
    """
    Async market data fetcher with caching
    Returns data formatted for Excel tables
    """
    # Use global HTTP client for connection pooling
    url = f"{decouple_config('MARKET_DATA_API')}/historical"
    
    response = await http_client.get(url, params={
        "symbol": ticker,
        "from": start_date,
        "to": end_date
    })
    response.raise_for_status()
    
    data = response.json()
    
    # Format for Excel (headers + data rows)
    headers = ["Date", "Open", "High", "Low", "Close", "Volume"]
    rows = [[headers]]
    
    for item in data["results"]:
        rows.append([
            item["date"],
            item["open"],
            item["high"],
            item["low"],
            item["close"],
            item["volume"]
        ])
    
    return rows

@xw.func
@xw.arg('matrix1', ndim=2)
@xw.arg('matrix2', ndim=2)
@xw.ret(expand='table')
def optimized_matrix_multiply(
    matrix1: np.ndarray,
    matrix2: np.ndarray
) -> np.ndarray:
    """
    Matrix multiplication optimized for Excel ranges
    Uses NumPy's BLAS-backed operations
    """
    # NumPy automatically uses optimized BLAS libraries (MKL, OpenBLAS)
    # For Excel, we often get ranges with empty cells as NaN
    matrix1 = np.nan_to_num(matrix1, 0)
    matrix2 = np.nan_to_num(matrix2, 0)
    
    try:
        result = matrix1 @ matrix2  # Python 3.5+ matrix multiplication
        return result
    except ValueError as e:
        # Return error message in a format Excel can display
        return np.array([[f"Error: {str(e)}"]])
```

### ❌ DON'T: Use Synchronous COM Automation for Bulk Operations

```python
# Bad - Synchronous cell-by-cell operations
import win32com.client

def slow_data_transfer(data: List[List[Any]]):
    excel = win32com.client.Dispatch("Excel.Application")
    wb = excel.Workbooks.Open(r"C:\models\model.xlsx")
    ws = wb.Worksheets(1)
    
    # This is incredibly slow!
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            ws.Cells(i+1, j+1).Value = value
    
    wb.Save()
    wb.Close()
```

### ✅ DO: Use Array Formulas and Bulk Operations

```python
# src/excel/excel_engine.py
import asyncio
import win32com.client
import pythoncom
from typing import Any, List, Tuple, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import psutil

class ExcelEngine:
    """High-performance Excel automation engine"""
    
    def __init__(self, visible: bool = False):
        self.visible = visible
        self.executor = ThreadPoolExecutor(
            max_workers=psutil.cpu_count(logical=False)  # Physical cores only
        )
    
    async def _run_in_thread(self, func, *args, **kwargs):
        """Run COM operations in thread pool to avoid blocking"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)
    
    def _init_com(self):
        """Initialize COM in thread - required for each thread"""
        pythoncom.CoInitialize()
    
    def _write_array_formula(
        self,
        filepath: str,
        sheet_name: str,
        start_cell: str,
        data: np.ndarray
    ) -> None:
        """Write data as array formula - 1000x faster than cell-by-cell"""
        self._init_com()
        
        excel = None
        try:
            excel = win32com.client.Dispatch("Excel.Application")
            excel.Visible = self.visible
            excel.DisplayAlerts = False
            excel.ScreenUpdating = False  # Massive performance boost
            
            wb = excel.Workbooks.Open(filepath)
            ws = wb.Worksheets(sheet_name)
            
            # Convert numpy array to Python list for COM
            if isinstance(data, np.ndarray):
                data = data.tolist()
            
            # Get the range for bulk write
            start_row, start_col = self._cell_to_indices(start_cell)
            end_row = start_row + len(data) - 1
            end_col = start_col + len(data[0]) - 1
            
            # Create range object
            cell_range = ws.Range(
                ws.Cells(start_row, start_col),
                ws.Cells(end_row, end_col)
            )
            
            # Single bulk write operation
            cell_range.Value = data
            
            wb.Save()
            wb.Close(SaveChanges=False)
            
        finally:
            if excel:
                excel.Quit()
            pythoncom.CoUninitialize()
    
    async def write_data(
        self,
        filepath: str,
        sheet_name: str,
        start_cell: str,
        data: np.ndarray
    ) -> None:
        """Async wrapper for array formula writing"""
        await self._run_in_thread(
            self._write_array_formula,
            filepath,
            sheet_name,
            start_cell,
            data
        )
    
    def _calculate_range(
        self,
        filepath: str,
        sheet_name: str,
        range_name: str
    ) -> Any:
        """Calculate specific named range and return results"""
        self._init_com()
        
        excel = None
        try:
            excel = win32com.client.Dispatch("Excel.Application")
            excel.Visible = self.visible
            excel.DisplayAlerts = False
            
            # Open in read-only mode for better concurrency
            wb = excel.Workbooks.Open(filepath, ReadOnly=True)
            ws = wb.Worksheets(sheet_name)
            
            # Force calculation of specific range only
            target_range = ws.Range(range_name)
            target_range.Calculate()
            
            # Read results as array
            result = target_range.Value
            
            wb.Close(SaveChanges=False)
            
            return result
            
        finally:
            if excel:
                excel.Quit()
            pythoncom.CoUninitialize()
    
    async def calculate_range(
        self,
        filepath: str,
        sheet_name: str,
        range_name: str
    ) -> Any:
        """Async wrapper for range calculation"""
        return await self._run_in_thread(
            self._calculate_range,
            filepath,
            sheet_name,
            range_name
        )
    
    @staticmethod
    def _cell_to_indices(cell: str) -> Tuple[int, int]:
        """Convert Excel cell reference (e.g., 'A1') to row, col indices"""
        import re
        match = re.match(r'([A-Z]+)(\d+)', cell.upper())
        if not match:
            raise ValueError(f"Invalid cell reference: {cell}")
        
        col_str, row_str = match.groups()
        
        # Convert column letters to number (A=1, B=2, ..., AA=27, etc.)
        col = 0
        for char in col_str:
            col = col * 26 + (ord(char) - ord('A') + 1)
        
        return int(row_str), col
```

---

## 3. FastAPI Orchestration Layer

### API Structure and Dependencies

```python
# src/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from .routers import calculations, health, admin
from ..core import vm_manager, cache_manager
from ..utils.monitoring import setup_telemetry

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
)

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting Excel Automation Platform")
    
    # Initialize VM pool
    await vm_manager.initialize_pool()
    
    # Warm up Redis connections
    await cache_manager.initialize()
    
    # Start background tasks
    app.state.vm_monitor_task = asyncio.create_task(
        vm_manager.monitor_pool_health()
    )
    
    yield
    
    # Shutdown
    logger.info("Shutting down Excel Automation Platform")
    
    # Cancel background tasks
    app.state.vm_monitor_task.cancel()
    
    # Cleanup VM pool
    await vm_manager.cleanup_pool()
    
    # Close connections
    await cache_manager.close()

app = FastAPI(
    title="Excel Automation Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response

# CORS for frontend apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# OpenTelemetry instrumentation
setup_telemetry()
FastAPIInstrumentor.instrument_app(app)

# Include routers
app.include_router(calculations.router, prefix="/api/v1/calculations", tags=["calculations"])
app.include_router(health.router, prefix="/api/v1/health", tags=["health"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
```

### Calculation Endpoints

```python
# src/api/routers/calculations.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Optional, List
import uuid
from datetime import datetime

from ...models.requests import CalculationRequest, BatchCalculationRequest
from ...models.responses import CalculationResponse, CalculationStatus
from ...core.vm_manager import VMManager
from ...core.excel_engine import ExcelEngine
from ...core.cache_manager import CacheManager
from ..dependencies import get_vm_manager, get_cache_manager, verify_api_key

router = APIRouter()

@router.post("/submit", response_model=CalculationResponse)
async def submit_calculation(
    request: CalculationRequest,
    background_tasks: BackgroundTasks,
    vm_manager: VMManager = Depends(get_vm_manager),
    cache: CacheManager = Depends(get_cache_manager),
    api_key: str = Depends(verify_api_key)
):
    """Submit a calculation job"""
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Check cache first
    cache_key = request.generate_cache_key()
    cached_result = await cache.get_result(cache_key)
    
    if cached_result:
        return CalculationResponse(
            job_id=job_id,
            status=CalculationStatus.COMPLETED,
            result=cached_result,
            cached=True,
            completed_at=datetime.utcnow()
        )
    
    # Initialize job status
    await cache.set_job_status(job_id, CalculationStatus.QUEUED)
    
    # Queue the calculation
    background_tasks.add_task(
        process_calculation,
        job_id,
        request,
        vm_manager,
        cache
    )
    
    return CalculationResponse(
        job_id=job_id,
        status=CalculationStatus.QUEUED,
        cached=False
    )

async def process_calculation(
    job_id: str,
    request: CalculationRequest,
    vm_manager: VMManager,
    cache: CacheManager
):
    """Process calculation in background"""
    try:
        # Update status
        await cache.set_job_status(job_id, CalculationStatus.PROCESSING)
        
        # Acquire a VM from the pool
        async with vm_manager.acquire_vm() as vm:
            # Download file from SharePoint if needed
            if request.file_source == "sharepoint":
                local_path = await vm.download_from_sharepoint(
                    request.file_path,
                    request.sharepoint_site
                )
            else:
                local_path = request.file_path
            
            # Open Excel and perform calculation
            engine = ExcelEngine()
            
            # Write input data if provided
            if request.input_data:
                await engine.write_data(
                    local_path,
                    request.sheet_name,
                    request.input_range,
                    request.input_data
                )
            
            # Calculate specified range
            result = await engine.calculate_range(
                local_path,
                request.sheet_name,
                request.output_range
            )
            
            # Cache result
            cache_key = request.generate_cache_key()
            await cache.set_result(
                cache_key,
                result,
                ttl=request.cache_ttl or 3600
            )
            
            # Update job status
            await cache.set_job_result(job_id, result)
            await cache.set_job_status(job_id, CalculationStatus.COMPLETED)
            
    except Exception as e:
        logger.error(f"Calculation failed for job {job_id}", exc_info=e)
        await cache.set_job_status(job_id, CalculationStatus.FAILED)
        await cache.set_job_error(job_id, str(e))

@router.get("/status/{job_id}", response_model=CalculationResponse)
async def get_calculation_status(
    job_id: str,
    cache: CacheManager = Depends(get_cache_manager),
    api_key: str = Depends(verify_api_key)
):
    """Get calculation job status"""
    status = await cache.get_job_status(job_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = CalculationResponse(
        job_id=job_id,
        status=status
    )
    
    if status == CalculationStatus.COMPLETED:
        result = await cache.get_job_result(job_id)
        response.result = result
        response.completed_at = datetime.utcnow()
    
    elif status == CalculationStatus.FAILED:
        error = await cache.get_job_error(job_id)
        response.error = error
    
    return response

@router.post("/batch", response_model=List[CalculationResponse])
async def submit_batch_calculation(
    request: BatchCalculationRequest,
    background_tasks: BackgroundTasks,
    vm_manager: VMManager = Depends(get_vm_manager),
    cache: CacheManager = Depends(get_cache_manager),
    api_key: str = Depends(verify_api_key)
):
    """Submit multiple calculations as a batch"""
    responses = []
    
    # Use asyncio.gather for parallel submission
    tasks = []
    for calc_request in request.calculations:
        task = submit_calculation(
            calc_request,
            background_tasks,
            vm_manager,
            cache,
            api_key
        )
        tasks.append(task)
    
    responses = await asyncio.gather(*tasks)
    return responses
```

---

## 4. Azure VM Pool Management

### VM Pool Architecture

```python
# src/core/vm_manager.py
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager
from azure.identity.aio import DefaultAzureCredential
from azure.mgmt.compute.aio import ComputeManagementClient
from azure.mgmt.network.aio import NetworkManagementClient
from azure.mgmt.resource.aio import ResourceManagementClient
import structlog

from ..models.azure import VMConfig, VMState, VMPoolConfig
from ..utils.security import generate_vm_password

logger = structlog.get_logger()

@dataclass
class PooledVM:
    """Represents a VM in the pool"""
    instance_id: str
    private_ip: str
    state: VMState
    created_at: datetime
    last_used: datetime
    health_check_failures: int = 0
    allocated_to: Optional[str] = None

class VMManager:
    """Manages a pool of Azure VMs for Excel automation"""
    
    def __init__(self, config: VMPoolConfig):
        self.config = config
        self.credential = DefaultAzureCredential()
        self.compute_client = None
        self.network_client = None
        self.resource_client = None
        self.pool: Dict[str, PooledVM] = {}
        self.pool_lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize_pool(self):
        """Initialize the VM pool with warm instances"""
        if self._initialized:
            return
        
        # Initialize Azure clients
        self.compute_client = ComputeManagementClient(
            self.credential,
            self.config.subscription_id
        )
        self.network_client = NetworkManagementClient(
            self.credential,
            self.config.subscription_id
        )
        self.resource_client = ResourceManagementClient(
            self.credential,
            self.config.subscription_id
        )
        
        # Create initial pool
        logger.info(f"Initializing VM pool with {self.config.min_pool_size} instances")
        
        tasks = []
        for i in range(self.config.min_pool_size):
            task = self._create_vm(f"excel-vm-{i:03d}")
            tasks.append(task)
        
        # Create VMs in parallel
        created_vms = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Add successful VMs to pool
        async with self.pool_lock:
            for vm in created_vms:
                if isinstance(vm, PooledVM):
                    self.pool[vm.instance_id] = vm
                    logger.info(f"Added VM {vm.instance_id} to pool")
                else:
                    logger.error(f"Failed to create VM: {vm}")
        
        self._initialized = True
    
    async def _create_vm(self, vm_name: str) -> PooledVM:
        """Create a new VM with Excel pre-installed"""
        # VM configuration
        vm_parameters = {
            "location": self.config.location,
            "identity": {
                "type": "SystemAssigned"  # Managed identity for secure access
            },
            "properties": {
                "hardwareProfile": {
                    "vmSize": self.config.vm_size  # e.g., "Standard_D4s_v5"
                },
                "storageProfile": {
                    "imageReference": {
                        "id": self.config.custom_image_id  # Custom image with Excel
                    },
                    "osDisk": {
                        "createOption": "FromImage",
                        "managedDisk": {
                            "storageAccountType": "Premium_LRS"
                        },
                        "deleteOption": "Delete"  # Clean up disk on VM deletion
                    }
                },
                "osProfile": {
                    "computerName": vm_name,
                    "adminUsername": "exceladmin",
                    "adminPassword": generate_vm_password(),
                    "windowsConfiguration": {
                        "enableAutomaticUpdates": False,  # Control updates
                        "timeZone": "UTC"
                    }
                },
                "networkProfile": {
                    "networkInterfaces": [{
                        "id": await self._create_network_interface(vm_name),
                        "properties": {
                            "deleteOption": "Delete"  # Clean up NIC on VM deletion
                        }
                    }]
                },
                "priority": "Spot" if self.config.use_spot_instances else "Regular",
                "evictionPolicy": "Delete" if self.config.use_spot_instances else None,
                "billingProfile": {
                    "maxPrice": self.config.max_spot_price if self.config.use_spot_instances else None
                }
            }
        }
        
        # Create VM
        async with self.compute_client:
            poller = await self.compute_client.virtual_machines.begin_create_or_update(
                self.config.resource_group,
                vm_name,
                vm_parameters
            )
            vm = await poller.result()
        
        # Get private IP
        private_ip = await self._get_vm_private_ip(vm_name)
        
        # Wait for VM to be ready
        await self._wait_for_vm_ready(vm_name, private_ip)
        
        # Install xlwings server
        await self._install_xlwings_server(private_ip)
        
        return PooledVM(
            instance_id=vm_name,
            private_ip=private_ip,
            state=VMState.AVAILABLE,
            created_at=datetime.utcnow(),
            last_used=datetime.utcnow()
        )
    
    @asynccontextmanager
    async def acquire_vm(self, timeout: int = 300):
        """Acquire a VM from the pool"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            async with self.pool_lock:
                # Find available VM
                for vm_id, vm in self.pool.items():
                    if vm.state == VMState.AVAILABLE:
                        vm.state = VMState.IN_USE
                        vm.allocated_to = asyncio.current_task().get_name()
                        vm.last_used = datetime.utcnow()
                        
                        logger.info(f"Acquired VM {vm_id}")
                        
                        try:
                            yield vm
                            return
                        finally:
                            # Release VM
                            async with self.pool_lock:
                                vm.state = VMState.AVAILABLE
                                vm.allocated_to = None
                                logger.info(f"Released VM {vm_id}")
            
            # No available VMs, check if we can scale up
            async with self.pool_lock:
                current_size = len(self.pool)
                if current_size < self.config.max_pool_size:
                    logger.info("Scaling up VM pool")
                    asyncio.create_task(self._scale_up())
            
            # Wait before retrying
            await asyncio.sleep(5)
        
        raise TimeoutError("Could not acquire VM within timeout")
    
    async def _scale_up(self):
        """Add a new VM to the pool"""
        current_size = len(self.pool)
        vm_name = f"excel-vm-{current_size:03d}"
        
        try:
            vm = await self._create_vm(vm_name)
            async with self.pool_lock:
                self.pool[vm.instance_id] = vm
                logger.info(f"Scaled up pool with VM {vm.instance_id}")
        except Exception as e:
            logger.error(f"Failed to scale up pool: {e}")
    
    async def monitor_pool_health(self):
        """Background task to monitor VM health"""
        while True:
            try:
                async with self.pool_lock:
                    tasks = []
                    for vm_id, vm in self.pool.items():
                        if vm.state == VMState.AVAILABLE:
                            task = self._health_check(vm)
                            tasks.append(task)
                
                # Run health checks in parallel
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results
                    async with self.pool_lock:
                        for vm, result in zip(
                            [v for v in self.pool.values() if v.state == VMState.AVAILABLE],
                            results
                        ):
                            if isinstance(result, Exception):
                                vm.health_check_failures += 1
                                logger.warning(
                                    f"Health check failed for {vm.instance_id}: {result}"
                                )
                                
                                # Replace unhealthy VMs
                                if vm.health_check_failures >= 3:
                                    logger.error(
                                        f"VM {vm.instance_id} is unhealthy, replacing"
                                    )
                                    asyncio.create_task(self._replace_vm(vm))
                            else:
                                vm.health_check_failures = 0
                
                # Scale down if needed
                await self._check_scale_down()
                
            except Exception as e:
                logger.error(f"Pool monitoring error: {e}")
            
            # Wait before next check
            await asyncio.sleep(30)
    
    async def _check_scale_down(self):
        """Scale down pool if underutilized"""
        async with self.pool_lock:
            available_vms = [
                vm for vm in self.pool.values()
                if vm.state == VMState.AVAILABLE
            ]
            
            # Check if we have too many idle VMs
            if len(available_vms) > self.config.min_pool_size:
                # Find VMs idle for too long
                idle_threshold = datetime.utcnow() - timedelta(
                    minutes=self.config.idle_timeout_minutes
                )
                
                idle_vms = [
                    vm for vm in available_vms
                    if vm.last_used < idle_threshold
                ]
                
                # Remove excess idle VMs
                vms_to_remove = len(self.pool) - self.config.min_pool_size
                for vm in idle_vms[:vms_to_remove]:
                    logger.info(f"Scaling down: removing idle VM {vm.instance_id}")
                    asyncio.create_task(self._delete_vm(vm))
```

---

## 5. SharePoint Integration & File Management

### SharePoint Client with Microsoft Graph

```python
# src/core/sharepoint_client.py
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import aiofiles
import httpx
from msgraph import GraphServiceClient
from msgraph.generated.models.drive_item import DriveItem
from azure.identity.aio import ClientSecretCredential
import hashlib
from pathlib import Path

from ..utils.monitoring import measure_time
from ..models.sharepoint import SharePointFile, SharePointSite

class SharePointClient:
    """Async SharePoint client using Microsoft Graph API"""
    
    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        cache_dir: Path = Path("/tmp/sharepoint_cache")
    ):
        self.credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )
        
        self.client = GraphServiceClient(
            credentials=self.credential,
            scopes=['https://graph.microsoft.com/.default']
        )
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # File metadata cache
        self._metadata_cache: Dict[str, SharePointFile] = {}
        self._cache_expiry: Dict[str, datetime] = {}
    
    async def get_site(self, site_name: str) -> SharePointSite:
        """Get SharePoint site by name"""
        # Search for site
        sites = await self.client.sites.by_site_id('root').sites.get()
        
        for site in sites.value:
            if site.display_name.lower() == site_name.lower():
                return SharePointSite(
                    id=site.id,
                    name=site.display_name,
                    web_url=site.web_url
                )
        
        raise ValueError(f"Site '{site_name}' not found")
    
    @measure_time
    async def download_file(
        self,
        site_id: str,
        file_path: str,
        local_path: Optional[Path] = None,
        use_cache: bool = True
    ) -> Path:
        """Download file from SharePoint with caching"""
        # Generate cache key
        cache_key = hashlib.sha256(
            f"{site_id}:{file_path}".encode()
        ).hexdigest()
        
        # Check local cache first
        if use_cache:
            cached_file = self.cache_dir / cache_key
            if cached_file.exists():
                # Verify cache validity
                metadata = await self._get_file_metadata(site_id, file_path)
                cached_stat = cached_file.stat()
                
                # Use cached file if not modified
                if cached_stat.st_mtime >= metadata.last_modified.timestamp():
                    logger.info(f"Using cached file for {file_path}")
                    return cached_file
        
        # Download file
        drive_item = await self._get_drive_item(site_id, file_path)
        
        if not drive_item.id:
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Stream download for large files
        download_url = drive_item.microsoft_graph_download_url
        
        if not local_path:
            local_path = self.cache_dir / cache_key
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                download_url,
                follow_redirects=True,
                headers={"Authorization": f"Bearer {await self._get_token()}"}
            )
            response.raise_for_status()
            
            # Write to file asynchronously
            async with aiofiles.open(local_path, 'wb') as f:
                async for chunk in response.aiter_bytes(chunk_size=1024*1024):
                    await f.write(chunk)
        
        # Update modification time
        if metadata.last_modified:
            import os
            os.utime(
                local_path,
                (
                    metadata.last_modified.timestamp(),
                    metadata.last_modified.timestamp()
                )
            )
        
        logger.info(f"Downloaded {file_path} to {local_path}")
        return local_path
    
    async def upload_file(
        self,
        site_id: str,
        local_path: Path,
        remote_path: str,
        conflict_behavior: str = "replace"
    ) -> SharePointFile:
        """Upload file to SharePoint"""
        # For large files, use resumable upload
        file_size = local_path.stat().st_size
        
        if file_size > 4 * 1024 * 1024:  # 4MB threshold
            return await self._upload_large_file(
                site_id,
                local_path,
                remote_path,
                conflict_behavior
            )
        
        # Small file upload
        async with aiofiles.open(local_path, 'rb') as f:
            content = await f.read()
        
        # Parse path
        path_parts = remote_path.strip('/').split('/')
        filename = path_parts[-1]
        folder_path = '/'.join(path_parts[:-1])
        
        # Get parent folder
        parent = await self._get_or_create_folder(site_id, folder_path)
        
        # Upload file
        drive_item = await self.client.sites.by_site_id(site_id).drive.items.by_drive_item_id(
            parent.id
        ).children.by_drive_item_id(filename).content.put(
            content,
            conflict_behavior=conflict_behavior
        )
        
        return SharePointFile(
            id=drive_item.id,
            name=drive_item.name,
            path=remote_path,
            size=drive_item.size,
            last_modified=drive_item.last_modified_date_time,
            download_url=drive_item.microsoft_graph_download_url
        )
    
    async def _upload_large_file(
        self,
        site_id: str,
        local_path: Path,
        remote_path: str,
        conflict_behavior: str
    ) -> SharePointFile:
        """Upload large file using resumable session"""
        file_size = local_path.stat().st_size
        
        # Create upload session
        upload_session = await self.client.sites.by_site_id(
            site_id
        ).drive.root.item_with_path(
            remote_path
        ).create_upload_session.post(
            conflict_behavior=conflict_behavior
        )
        
        # Upload in chunks
        chunk_size = 10 * 1024 * 1024  # 10MB chunks
        
        async with aiofiles.open(local_path, 'rb') as f:
            uploaded = 0
            
            while uploaded < file_size:
                # Read chunk
                chunk = await f.read(chunk_size)
                chunk_length = len(chunk)
                
                # Upload chunk
                headers = {
                    'Content-Length': str(chunk_length),
                    'Content-Range': f'bytes {uploaded}-{uploaded + chunk_length - 1}/{file_size}'
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.put(
                        upload_session.upload_url,
                        content=chunk,
                        headers=headers
                    )
                    response.raise_for_status()
                
                uploaded += chunk_length
                
                # Log progress
                progress = (uploaded / file_size) * 100
                logger.info(f"Upload progress: {progress:.1f}%")
        
        # Get uploaded file metadata
        result = response.json()
        return SharePointFile(
            id=result['id'],
            name=result['name'],
            path=remote_path,
            size=result['size'],
            last_modified=datetime.fromisoformat(result['lastModifiedDateTime']),
            download_url=result['@microsoft.graph.downloadUrl']
        )
    
    async def list_files(
        self,
        site_id: str,
        folder_path: str = "/",
        recursive: bool = False
    ) -> List[SharePointFile]:
        """List files in a SharePoint folder"""
        drive = await self.client.sites.by_site_id(site_id).drive.get()
        
        # Get folder
        if folder_path == "/":
            folder = await self.client.drives.by_drive_id(drive.id).root.get()
        else:
            folder = await self.client.drives.by_drive_id(
                drive.id
            ).root.item_with_path(folder_path).get()
        
        # List children
        children = await self.client.drives.by_drive_id(
            drive.id
        ).items.by_drive_item_id(folder.id).children.get()
        
        files = []
        for item in children.value:
            if item.file:  # It's a file
                files.append(SharePointFile(
                    id=item.id,
                    name=item.name,
                    path=f"{folder_path}/{item.name}",
                    size=item.size,
                    last_modified=item.last_modified_date_time,
                    download_url=item.microsoft_graph_download_url
                ))
            elif recursive and item.folder:  # Recurse into folders
                subfolder_path = f"{folder_path}/{item.name}"
                subfiles = await self.list_files(
                    site_id,
                    subfolder_path,
                    recursive=True
                )
                files.extend(subfiles)
        
        return files
```

### File Version Management

```python
# src/core/file_versioning.py
from typing import List, Optional
from datetime import datetime
import asyncio

class FileVersionManager:
    """Manages Excel file versions and snapshots"""
    
    def __init__(self, sharepoint_client: SharePointClient):
        self.sharepoint = sharepoint_client
        self.version_folder = "/_versions"
    
    async def create_snapshot(
        self,
        site_id: str,
        file_path: str,
        description: str = ""
    ) -> str:
        """Create a versioned snapshot of a file"""
        # Generate version name
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_name = Path(file_path).name
        version_name = f"{file_name}.{timestamp}"
        
        # Download current file
        local_file = await self.sharepoint.download_file(
            site_id,
            file_path
        )
        
        # Upload as version
        version_path = f"{self.version_folder}/{version_name}"
        await self.sharepoint.upload_file(
            site_id,
            local_file,
            version_path
        )
        
        # Store metadata
        metadata = {
            "original_path": file_path,
            "version_path": version_path,
            "timestamp": timestamp,
            "description": description
        }
        
        await self._save_version_metadata(site_id, metadata)
        
        return version_path
    
    async def restore_version(
        self,
        site_id: str,
        version_path: str,
        target_path: str
    ):
        """Restore a specific version"""
        # Create backup of current
        await self.create_snapshot(
            site_id,
            target_path,
            "Auto-backup before restore"
        )
        
        # Download version
        local_file = await self.sharepoint.download_file(
            site_id,
            version_path
        )
        
        # Upload to target
        await self.sharepoint.upload_file(
            site_id,
            local_file,
            target_path,
            conflict_behavior="replace"
        )
        
        logger.info(f"Restored {version_path} to {target_path}")
```

---

## 6. Security & Network Architecture

### Zero-Trust Network Design

```python
# src/utils/security.py
import secrets
import string
from typing import Dict, Any
import jwt
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
import hashlib

from ..models.security import APIKey, VMCredentials

class SecurityManager:
    """Centralized security management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fernet = Fernet(config['encryption_key'].encode())
        self.jwt_secret = config['jwt_secret']
    
    def generate_api_key(
        self,
        client_id: str,
        permissions: List[str],
        expires_in_days: int = 365
    ) -> APIKey:
        """Generate a new API key with permissions"""
        # Generate cryptographically secure key
        key = secrets.token_urlsafe(32)
        
        # Create JWT with permissions
        payload = {
            'client_id': client_id,
            'permissions': permissions,
            'exp': datetime.utcnow() + timedelta(days=expires_in_days),
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        
        # Store key hash (never store plain keys)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        return APIKey(
            key=key,
            token=token,
            key_hash=key_hash,
            client_id=client_id,
            permissions=permissions,
            expires_at=payload['exp']
        )
    
    def verify_api_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Verify and decode API key"""
        try:
            # Verify key hash exists in database
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            # ... database lookup ...
            
            # Decode JWT
            payload = jwt.decode(
                key,
                self.jwt_secret,
                algorithms=['HS256']
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Expired API key attempted")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid API key attempted")
            return None
    
    def encrypt_credentials(self, credentials: VMCredentials) -> str:
        """Encrypt VM credentials for storage"""
        data = f"{credentials.username}:{credentials.password}"
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_credentials(self, encrypted: str) -> VMCredentials:
        """Decrypt VM credentials"""
        data = self.fernet.decrypt(encrypted.encode()).decode()
        username, password = data.split(':', 1)
        return VMCredentials(username=username, password=password)
    
    @staticmethod
    def generate_vm_password() -> str:
        """Generate secure VM password"""
        # Azure password requirements:
        # 12-123 characters, 3 of: uppercase, lowercase, digit, special
        length = 20
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        
        while True:
            password = ''.join(secrets.choice(chars) for _ in range(length))
            
            # Verify complexity
            if (any(c.isupper() for c in password) and
                any(c.islower() for c in password) and
                any(c.isdigit() for c in password) and
                any(c in "!@#$%^&*" for c in password)):
                return password

# Network Security Configuration
NETWORK_SECURITY_RULES = [
    {
        "name": "AllowHTTPS",
        "priority": 100,
        "direction": "Inbound",
        "access": "Allow",
        "protocol": "Tcp",
        "source_port_range": "*",
        "destination_port_range": "443",
        "source_address_prefix": "10.0.0.0/8",  # Internal only
        "destination_address_prefix": "*"
    },
    {
        "name": "AllowWinRM",
        "priority": 110,
        "direction": "Inbound",
        "access": "Allow",
        "protocol": "Tcp",
        "source_port_range": "*",
        "destination_port_range": "5985-5986",
        "source_address_prefix": "10.0.1.0/24",  # Management subnet
        "destination_address_prefix": "*"
    },
    {
        "name": "DenyAllInbound",
        "priority": 4096,
        "direction": "Inbound",
        "access": "Deny",
        "protocol": "*",
        "source_port_range": "*",
        "destination_port_range": "*",
        "source_address_prefix": "*",
        "destination_address_prefix": "*"
    }
]

# VM Isolation Configuration
class VMNetworkIsolation:
    """Ensures VMs are properly isolated"""
    
    @staticmethod
    async def create_isolated_subnet(
        network_client: NetworkManagementClient,
        resource_group: str,
        vnet_name: str,
        subnet_name: str,
        address_prefix: str = "10.0.2.0/24"
    ):
        """Create an isolated subnet for Excel VMs"""
        # Create Network Security Group
        nsg_params = {
            "location": "eastus",
            "security_rules": NETWORK_SECURITY_RULES
        }
        
        nsg = await network_client.network_security_groups.begin_create_or_update(
            resource_group,
            f"{subnet_name}-nsg",
            nsg_params
        ).result()
        
        # Create subnet with NSG
        subnet_params = {
            "address_prefix": address_prefix,
            "network_security_group": {"id": nsg.id},
            "service_endpoints": [
                {"service": "Microsoft.Storage"},
                {"service": "Microsoft.KeyVault"}
            ]
        }
        
        subnet = await network_client.subnets.begin_create_or_update(
            resource_group,
            vnet_name,
            subnet_name,
            subnet_params
        ).result()
        
        return subnet
```

### API Security Middleware

```python
# src/api/dependencies.py
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from ..utils.security import SecurityManager
from ..core.vm_manager import VMManager
from ..core.cache_manager import CacheManager

security = HTTPBearer()

async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, Any]:
    """Verify API key and return permissions"""
    security_manager = SecurityManager(get_config())
    
    payload = security_manager.verify_api_key(credentials.credentials)
    
    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired API key"
        )
    
    return payload

async def require_permission(permission: str):
    """Require specific permission"""
    async def permission_checker(
        auth_payload: Dict[str, Any] = Depends(verify_api_key)
    ):
        if permission not in auth_payload.get('permissions', []):
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required"
            )
        return auth_payload
    
    return permission_checker

# Dependency providers
_vm_manager: Optional[VMManager] = None
_cache_manager: Optional[CacheManager] = None

async def get_vm_manager() -> VMManager:
    """Get VM manager instance"""
    global _vm_manager
    if not _vm_manager:
        config = VMPoolConfig.from_env()
        _vm_manager = VMManager(config)
    return _vm_manager

async def get_cache_manager() -> CacheManager:
    """Get cache manager instance"""
    global _cache_manager
    if not _cache_manager:
        _cache_manager = CacheManager.from_env()
    return _cache_manager
```

---

## 7. Cost Optimization Strategies

### Intelligent VM Lifecycle Management

```python
# src/core/cost_optimizer.py
from typing import List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio

from ..models.azure import VMUsageMetrics, CostReport
from ..utils.monitoring import MetricsCollector

@dataclass
class VMCostProfile:
    """Cost profile for different VM configurations"""
    vm_size: str
    hourly_cost: float
    spot_hourly_cost: float
    performance_score: float  # Relative performance metric

class CostOptimizer:
    """Optimizes VM usage for cost efficiency"""
    
    # VM cost profiles (2025 pricing)
    VM_PROFILES = {
        "Standard_D2s_v5": VMCostProfile(
            vm_size="Standard_D2s_v5",
            hourly_cost=0.096,
            spot_hourly_cost=0.019,
            performance_score=1.0
        ),
        "Standard_D4s_v5": VMCostProfile(
            vm_size="Standard_D4s_v5",
            hourly_cost=0.192,
            spot_hourly_cost=0.038,
            performance_score=2.1
        ),
        "Standard_D8s_v5": VMCostProfile(
            vm_size="Standard_D8s_v5",
            hourly_cost=0.384,
            spot_hourly_cost=0.077,
            performance_score=4.3
        ),
    }
    
    def __init__(self, vm_manager: VMManager, metrics_collector: MetricsCollector):
        self.vm_manager = vm_manager
        self.metrics = metrics_collector
        self.usage_history: Dict[str, List[VMUsageMetrics]] = {}
    
    async def optimize_pool_configuration(self) -> Dict[str, Any]:
        """Analyze usage and recommend optimal pool configuration"""
        # Collect usage metrics for the past week
        metrics = await self.metrics.get_vm_usage_metrics(days=7)
        
        # Analyze usage patterns
        peak_concurrent_vms = max(m.concurrent_vms for m in metrics)
        avg_concurrent_vms = sum(m.concurrent_vms for m in metrics) / len(metrics)
        utilization_rate = avg_concurrent_vms / self.vm_manager.config.max_pool_size
        
        # Calculate current costs
        current_cost = self._calculate_current_cost(metrics)
        
        # Generate recommendations
        recommendations = []
        
        # Recommendation 1: Pool size optimization
        if utilization_rate < 0.3:
            new_max_size = max(
                self.vm_manager.config.min_pool_size,
                int(peak_concurrent_vms * 1.2)
            )
            recommendations.append({
                "type": "reduce_pool_size",
                "current": self.vm_manager.config.max_pool_size,
                "recommended": new_max_size,
                "estimated_savings": self._estimate_pool_savings(
                    self.vm_manager.config.max_pool_size,
                    new_max_size
                )
            })
        
        # Recommendation 2: Spot instance usage
        spot_eligible_hours = self._calculate_spot_eligible_hours(metrics)
        if spot_eligible_hours > 0.5:  # >50% of time eligible for spot
            recommendations.append({
                "type": "increase_spot_usage",
                "current_spot_ratio": self.vm_manager.config.spot_instance_ratio,
                "recommended_spot_ratio": 0.8,
                "estimated_savings": self._estimate_spot_savings(0.8)
            })
        
        # Recommendation 3: VM size optimization
        avg_calculation_time = await self._get_avg_calculation_time()
        if avg_calculation_time < 30:  # Fast calculations
            recommendations.append({
                "type": "downsize_vms",
                "current_size": self.vm_manager.config.vm_size,
                "recommended_size": "Standard_D2s_v5",
                "estimated_savings": self._estimate_size_savings(
                    self.vm_manager.config.vm_size,
                    "Standard_D2s_v5"
                )
            })
        
        return {
            "current_monthly_cost": current_cost * 30,
            "utilization_rate": utilization_rate,
            "recommendations": recommendations,
            "total_potential_savings": sum(
                r.get("estimated_savings", 0) for r in recommendations
            )
        }
    
    async def implement_auto_scaling(self):
        """Implement predictive auto-scaling based on usage patterns"""
        while True:
            try:
                # Get current time and day of week
                now = datetime.utcnow()
                hour = now.hour
                day_of_week = now.weekday()
                
                # Predict load based on historical patterns
                predicted_load = await self._predict_load(hour, day_of_week)
                
                # Adjust pool size
                current_size = len(self.vm_manager.pool)
                target_size = max(
                    self.vm_manager.config.min_pool_size,
                    int(predicted_load * 1.2)  # 20% buffer
                )
                
                if target_size > current_size:
                    # Scale up
                    logger.info(f"Scaling up from {current_size} to {target_size}")
                    for _ in range(target_size - current_size):
                        asyncio.create_task(self.vm_manager._scale_up())
                
                elif target_size < current_size and self._can_scale_down():
                    # Scale down
                    logger.info(f"Scaling down from {current_size} to {target_size}")
                    await self._scale_down_gradually(current_size - target_size)
                
                # Wait before next check
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                await asyncio.sleep(60)
    
    async def _predict_load(self, hour: int, day_of_week: int) -> int:
        """Predict VM load based on historical patterns"""
        # Get historical data for this hour and day
        history = await self.metrics.get_usage_by_hour_and_day(
            hour=hour,
            day_of_week=day_of_week,
            weeks=4
        )
        
        if not history:
            return self.vm_manager.config.min_pool_size
        
        # Use simple moving average for prediction
        # In production, use more sophisticated ML models
        avg_load = sum(h.concurrent_vms for h in history) / len(history)
        
        # Account for growth trend
        growth_factor = 1.05  # 5% growth assumption
        
        return int(avg_load * growth_factor)
    
    def _calculate_spot_savings(self, spot_ratio: float) -> float:
        """Calculate savings from using spot instances"""
        profile = self.VM_PROFILES[self.vm_manager.config.vm_size]
        
        regular_cost = profile.hourly_cost
        spot_cost = profile.spot_hourly_cost
        
        # Average hours per month
        hours_per_month = 730
        
        # Calculate blended cost
        blended_cost = (
            regular_cost * (1 - spot_ratio) +
            spot_cost * spot_ratio
        )
        
        # Savings per VM per month
        savings_per_vm = (regular_cost - blended_cost) * hours_per_month
        
        # Total savings for pool
        avg_pool_size = (
            self.vm_manager.config.min_pool_size +
            self.vm_manager.config.max_pool_size
        ) / 2
        
        return savings_per_vm * avg_pool_size

# Cost monitoring endpoint
@router.get("/cost-report", response_model=CostReport)
async def get_cost_report(
    days: int = 30,
    optimizer: CostOptimizer = Depends(get_cost_optimizer),
    auth: Dict = Depends(require_permission("admin"))
):
    """Get detailed cost report and optimization recommendations"""
    # Get current costs
    current_costs = await optimizer.calculate_current_costs(days)
    
    # Get optimization recommendations
    recommendations = await optimizer.optimize_pool_configuration()
    
    # Historical trend
    trend = await optimizer.get_cost_trend(days)
    
    return CostReport(
        period_days=days,
        total_cost=current_costs["total"],
        cost_breakdown=current_costs["breakdown"],
        recommendations=recommendations["recommendations"],
        potential_savings=recommendations["total_potential_savings"],
        cost_trend=trend
    )
```

### Automatic Spot Instance Failover

```python
# src/core/spot_manager.py
class SpotInstanceManager:
    """Manages spot instances with automatic failover"""
    
    async def handle_spot_eviction(self, vm_id: str):
        """Handle spot instance eviction notice"""
        logger.warning(f"Spot eviction notice for {vm_id}")
        
        # Get VM from pool
        vm = self.vm_manager.pool.get(vm_id)
        if not vm:
            return
        
        # Mark as evicting
        vm.state = VMState.EVICTING
        
        # If VM is in use, migrate workload
        if vm.allocated_to:
            await self._migrate_workload(vm)
        
        # Pre-provision replacement
        asyncio.create_task(self._replace_spot_instance(vm))
    
    async def _migrate_workload(self, evicting_vm: PooledVM):
        """Migrate workload from evicting VM"""
        # Try to acquire another VM
        try:
            async with self.vm_manager.acquire_vm() as new_vm:
                # Notify the task about migration
                # In practice, this would involve more complex state transfer
                logger.info(
                    f"Migrating workload from {evicting_vm.instance_id} "
                    f"to {new_vm.instance_id}"
                )
        except TimeoutError:
            logger.error("Failed to acquire replacement VM for migration")
```

---

## 8. Production Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.13-slim-bookworm as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=true

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv venv .venv && \
    . .venv/bin/activate && \
    uv sync --all-extras

# Final stage
FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser ./src ./src
COPY --chown=appuser:appuser ./alembic ./alembic
COPY --chown=appuser:appuser ./alembic.ini ./

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/liveness || exit 1

# Run with gunicorn
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", \
     "-c", "gunicorn.conf.py", "src.api.main:app"]
```

### Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: excel-automation-api
  labels:
    app: excel-automation
    component: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: excel-automation
      component: api
  template:
    metadata:
      labels:
        app: excel-automation
        component: api
    spec:
      serviceAccountName: excel-automation
      containers:
      - name: api
        image: myregistry.azurecr.io/excel-automation:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: AZURE_CLIENT_ID
          valueFrom:
            secretKeyRef:
              name: azure-credentials
              key: client-id
        - name: AZURE_TENANT_ID
          valueFrom:
            secretKeyRef:
              name: azure-credentials
              key: tenant-id
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /api/v1/health/liveness
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health/readiness
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
      nodeSelector:
        agentpool: default

---
apiVersion: v1
kind: Service
metadata:
  name: excel-automation-api
spec:
  selector:
    app: excel-automation
    component: api
  ports:
  - port: 80
    targetPort: 8000
    name: http
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: excel-automation-ingress
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.excel-automation.example.com
    secretName: excel-automation-tls
  rules:
  - host: api.excel-automation.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: excel-automation-api
            port:
              number: 80
```

### Monitoring and Observability

```python
# src/utils/monitoring.py
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

# Prometheus metrics
calculation_requests = Counter(
    'excel_calculation_requests_total',
    'Total calculation requests',
    ['status', 'cached']
)

calculation_duration = Histogram(
    'excel_calculation_duration_seconds',
    'Calculation duration',
    ['operation']
)

vm_pool_size = Gauge(
    'excel_vm_pool_size',
    'Current VM pool size',
    ['state']
)

vm_allocation_time = Histogram(
    'excel_vm_allocation_seconds',
    'Time to allocate VM from pool'
)

# OpenTelemetry setup
def setup_telemetry():
    """Configure OpenTelemetry for distributed tracing"""
    # Tracing
    otlp_exporter = OTLPSpanExporter(
        endpoint="http://otel-collector:4317",
        insecure=True
    )
    
    trace_provider = TracerProvider()
    processor = BatchSpanProcessor(otlp_exporter)
    trace_provider.add_span_processor(processor)
    trace.set_tracer_provider(trace_provider)
    
    # Metrics
    metric_exporter = OTLPMetricExporter(
        endpoint="http://otel-collector:4317",
        insecure=True
    )
    
    metric_reader = PeriodicExportingMetricReader(
        exporter=metric_exporter,
        export_interval_millis=10000
    )
    
    metrics_provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(metrics_provider)

# Decorators for instrumentation
def measure_time(metric_name: str = None):
    """Decorator to measure execution time"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                if metric_name:
                    calculation_duration.labels(
                        operation=metric_name
                    ).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                if metric_name:
                    calculation_duration.labels(
                        operation=metric_name
                    ).observe(duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Custom span attributes
def add_calculation_attributes(span, calculation_request):
    """Add calculation-specific attributes to span"""
    span.set_attribute("excel.file_path", calculation_request.file_path)
    span.set_attribute("excel.sheet_name", calculation_request.sheet_name)
    span.set_attribute("excel.calculation_type", calculation_request.calculation_type)
    if calculation_request.cache_key:
        span.set_attribute("excel.cache_key", calculation_request.cache_key)
```

---

## 9. Advanced Patterns

### Distributed Calculation with Dask

```python
# src/core/distributed_calc.py
import dask
from dask.distributed import Client, as_completed
import numpy as np
from typing import List, Dict, Any

class DistributedCalculationEngine:
    """Distributes large calculations across multiple Excel instances"""
    
    def __init__(self, vm_manager: VMManager):
        self.vm_manager = vm_manager
        self.dask_client = None
    
    async def initialize(self):
        """Initialize Dask cluster using available VMs"""
        # Get available VMs
        available_vms = [
            vm for vm in self.vm_manager.pool.values()
            if vm.state == VMState.AVAILABLE
        ]
        
        # Start Dask workers on VMs
        scheduler_vm = available_vms[0]
        worker_vms = available_vms[1:]
        
        # Start scheduler
        scheduler_address = await self._start_dask_scheduler(scheduler_vm)
        
        # Start workers
        worker_tasks = []
        for vm in worker_vms:
            task = self._start_dask_worker(vm, scheduler_address)
            worker_tasks.append(task)
        
        await asyncio.gather(*worker_tasks)
        
        # Connect client
        self.dask_client = Client(scheduler_address, asynchronous=True)
        
    async def parallel_monte_carlo(
        self,
        model_path: str,
        scenarios: int = 100000,
        parameters: Dict[str, Any] = None
    ) -> np.ndarray:
        """Run Monte Carlo simulation in parallel across VMs"""
        
        # Partition scenarios across workers
        n_workers = len(self.dask_client.nthreads())
        scenarios_per_worker = scenarios // n_workers
        
        # Create tasks
        futures = []
        for i in range(n_workers):
            start_idx = i * scenarios_per_worker
            end_idx = start_idx + scenarios_per_worker
            
            if i == n_workers - 1:  # Last worker handles remainder
                end_idx = scenarios
            
            future = self.dask_client.submit(
                self._run_excel_monte_carlo,
                model_path,
                start_idx,
                end_idx,
                parameters
            )
            futures.append(future)
        
        # Gather results as they complete
        results = []
        for future in as_completed(futures):
            result = await future
            results.append(result)
            
            # Update progress
            progress = len(results) / n_workers * 100
            logger.info(f"Monte Carlo progress: {progress:.1f}%")
        
        # Combine results
        combined = np.concatenate(results, axis=0)
        
        # Calculate statistics
        return self._calculate_statistics(combined)
    
    @staticmethod
    def _run_excel_monte_carlo(
        model_path: str,
        start_idx: int,
        end_idx: int,
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """Run Monte Carlo scenarios in Excel (runs on worker)"""
        import win32com.client
        import numpy as np
        
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False
        
        try:
            wb = excel.Workbooks.Open(model_path)
            ws = wb.Worksheets("MonteCarlo")
            
            results = []
            
            for i in range(start_idx, end_idx):
                # Set random seed
                ws.Range("RandomSeed").Value = i
                
                # Set parameters
                if parameters:
                    for param, value in parameters.items():
                        ws.Range(param).Value = value
                
                # Calculate
                excel.Calculate()
                
                # Read results
                result_range = ws.Range("Results")
                result = np.array(result_range.Value)
                results.append(result)
            
            wb.Close(SaveChanges=False)
            return np.array(results)
            
        finally:
            excel.Quit()
```

### Real-time Collaboration with WebSockets

```python
# src/api/websocket.py
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.calculation_subscriptions: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = set()
        self.active_connections[client_id].add(websocket)
    
    def disconnect(self, websocket: WebSocket, client_id: str):
        self.active_connections[client_id].remove(websocket)
        if not self.active_connections[client_id]:
            del self.active_connections[client_id]
    
    async def send_personal_message(
        self,
        message: str,
        client_id: str
    ):
        if client_id in self.active_connections:
            for connection in self.active_connections[client_id]:
                await connection.send_text(message)
    
    async def broadcast_calculation_update(
        self,
        job_id: str,
        status: str,
        progress: float = None,
        result: Any = None
    ):
        """Broadcast calculation updates to subscribed clients"""
        message = {
            "type": "calculation_update",
            "job_id": job_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if progress is not None:
            message["progress"] = progress
        
        if result is not None:
            message["result"] = result
        
        # Send to all subscribed clients
        if job_id in self.calculation_subscriptions:
            for client_id in self.calculation_subscriptions[job_id]:
                await self.send_personal_message(
                    json.dumps(message),
                    client_id
                )

manager = ConnectionManager()

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
    api_key: str = Query(...)
):
    # Verify API key
    security_manager = SecurityManager(get_config())
    if not security_manager.verify_api_key(api_key):
        await websocket.close(code=1008)  # Policy Violation
        return
    
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive messages
            data = await websocket.receive_json()
            
            if data["type"] == "subscribe_calculation":
                job_id = data["job_id"]
                if job_id not in manager.calculation_subscriptions:
                    manager.calculation_subscriptions[job_id] = set()
                manager.calculation_subscriptions[job_id].add(client_id)
                
                # Send current status
                status = await cache_manager.get_job_status(job_id)
                await manager.broadcast_calculation_update(
                    job_id,
                    status
                )
            
            elif data["type"] == "unsubscribe_calculation":
                job_id = data["job_id"]
                if job_id in manager.calculation_subscriptions:
                    manager.calculation_subscriptions[job_id].discard(client_id)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
        # Clean up subscriptions
        for job_id, clients in manager.calculation_subscriptions.items():
            clients.discard(client_id)
```

### Excel Model Versioning and Testing

```python
# src/core/model_testing.py
class ExcelModelTester:
    """Automated testing framework for Excel models"""
    
    async def run_regression_tests(
        self,
        model_path: str,
        test_suite: Dict[str, Any]
    ) -> TestResults:
        """Run regression tests on Excel model"""
        results = TestResults()
        
        async with self.vm_manager.acquire_vm() as vm:
            engine = ExcelEngine()
            
            for test_name, test_config in test_suite.items():
                try:
                    # Set inputs
                    await engine.write_data(
                        model_path,
                        test_config["sheet"],
                        test_config["input_range"],
                        test_config["inputs"]
                    )
                    
                    # Calculate
                    actual = await engine.calculate_range(
                        model_path,
                        test_config["sheet"],
                        test_config["output_range"]
                    )
                    
                    # Validate
                    expected = test_config["expected"]
                    tolerance = test_config.get("tolerance", 0.0001)
                    
                    if self._compare_results(actual, expected, tolerance):
                        results.passed.append(test_name)
                    else:
                        results.failed.append({
                            "test": test_name,
                            "expected": expected,
                            "actual": actual
                        })
                
                except Exception as e:
                    results.errors.append({
                        "test": test_name,
                        "error": str(e)
                    })
        
        return results
    
    async def performance_benchmark(
        self,
        model_path: str,
        benchmark_config: Dict[str, Any]
    ) -> BenchmarkResults:
        """Benchmark Excel model performance"""
        results = BenchmarkResults()
        
        # Test with different data sizes
        for size in benchmark_config["data_sizes"]:
            # Generate test data
            test_data = self._generate_test_data(size)
            
            # Measure calculation time
            start = time.time()
            
            async with self.vm_manager.acquire_vm() as vm:
                engine = ExcelEngine()
                
                await engine.write_data(
                    model_path,
                    benchmark_config["sheet"],
                    benchmark_config["input_range"],
                    test_data
                )
                
                await engine.calculate_range(
                    model_path,
                    benchmark_config["sheet"],
                    benchmark_config["output_range"]
                )
            
            duration = time.time() - start
            
            results.add_measurement(size, duration)
            
            # Check if performance degrades
            if duration > benchmark_config.get("max_duration", 60):
                logger.warning(
                    f"Performance degradation detected: "
                    f"{size} rows took {duration:.2f}s"
                )
        
        return results
```

## Conclusion

This guide provides a comprehensive framework for building a production-grade Excel automation platform with Python, FastAPI, and Azure. The key principles to remember:

1. **Always use async patterns** - From FastAPI endpoints to Excel operations
2. **Pool-based VM management** - Eliminate cold starts with warm VM pools
3. **Security by default** - Zero-trust networking, managed identities, encrypted communications
4. **Cost optimization** - Spot instances, auto-scaling, intelligent lifecycle management
5. **High-performance UDFs** - xlwings Pro with REST API for minimal overhead
6. **Comprehensive monitoring** - OpenTelemetry, Prometheus metrics, structured logging

The patterns shown here scale from simple single-model calculations to complex distributed Monte Carlo simulations across multiple VMs. By following these practices, you can build a robust, secure, and cost-effective Excel automation platform that leverages the full power of cloud computing while maintaining the familiarity of Excel for business users.