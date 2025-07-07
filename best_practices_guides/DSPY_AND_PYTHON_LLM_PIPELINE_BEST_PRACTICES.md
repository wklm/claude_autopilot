# The Definitive Guide to DSPy for Complex LLM Pipelines (Mid-2025 Edition)

This guide provides production-grade patterns for building scalable, reliable, and cost-effective LLM applications using DSPy. It moves beyond basic tutorials to address the challenges of deploying complex prompt pipelines in production environments.

## Prerequisites & Modern Python Setup

Ensure your project uses **Python 3.13+**, **DSPy 3.0+**, and **uv** for dependency management (NOT pip/poetry/conda!). DSPy has evolved significantly, with v3.0 introducing native async support, built-in caching mechanisms, and production-ready optimizers.

### Project Configuration

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "llm-pipeline"
version = "0.1.0"
description = "Production LLM pipeline using DSPy"
readme = "README.md"
requires-python = ">=3.13"
license = { text = "MIT" }

dependencies = [
    # Core DSPy and LLM libraries
    "dspy-ai>=3.0.0b2",
    "openai>=2.0.0",
    "anthropic>=0.35.0",
    "litellm>=1.51.0",  # Unified LLM interface with built-in cost hooks
    
    # Async and performance
    "httpx>=0.27.0",
    "aiofiles>=24.1.0",
    # aiocache removed - DSPy 3.0 has built-in caching
    
    # Observability and monitoring
    "opentelemetry-api>=1.28.0",
    "opentelemetry-instrumentation-httpx>=0.48",
    "structlog>=24.4.0",
    "prometheus-client>=0.21.0",
    "mlflow>=2.17.0",  # Native DSPy integration
    
    # Data processing
    "pydantic>=2.9.0",
    "pandas>=2.2.0",
    "polars>=1.15.0",  # For high-performance data processing
    
    # Configuration and environment
    "python-decouple>=3.8",
    "pydantic-settings>=2.5.0",
    
    # Testing and validation - REAL integration tests only!
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "deepeval>=1.4.0",  # LLM output evaluation
    
    # CLI and visualization  
    "typer>=0.15.0",
    "rich>=13.9.0",  # For polished console output
    "streamlit>=1.41.0",  # For pipeline visualization
    
    # Vector stores and RAG
    "chromadb>=0.5.20",
    "qdrant-client>=1.12.0",
    "pgvector>=0.3.6",
    
    # Deployment
    "modal>=0.72.0",  # Serverless deployment
    "ray[default]>=2.40.0",  # Distributed computing
    
    # Production essentials
    "uvicorn[standard]>=0.35.0",
    "redis[hiredis]>=5.3.0",
    "asyncpg>=0.30.0",
    "alembic>=1.16.0",
    "psycopg2-binary>=2.9.10",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.9.0",
    "mypy>=1.13.0",
    "pre-commit>=4.0.0",
    "ipykernel>=6.29.0",
    "types-aiofiles>=24.1.0.20250606",
    "types-pyyaml>=6.0.12.20250516",
    "types-cachetools>=6.0.0.20250525",
]

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "SIM", "ARG", "PTH", "UP", "RUF", "ASYNC", "FA", "TID", "RUF100"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
docstring-code-format = true
line-ending = "lf"
skip-magic-trailing-comma = false
docstring-code-line-length = "dynamic"

[tool.mypy]
python_version = "3.13"
warn_return_any = true
strict = true
ignore_missing_imports = true

# Performance optimizations (mypy 1.16+)
sqlite_cache = true
cache_fine_grained = true
incremental = true
```

### Environment Setup with uv

```bash
# Create Python 3.13 environment with uv
uv venv --python 3.13
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
uv sync --all-extras

# Lock dependencies for reproducibility
uv lock --upgrade

# Set up environment variables using python-decouple pattern
cp .env.example .env
```

### Automated Setup Script

```bash
#!/bin/bash
# setup.sh - Automated setup for Ubuntu 25

set -euo pipefail  # Fail fast on errors

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up DSPy LLM Pipeline...${NC}"

# Check Python version
if ! python3 --version | grep -q "3.13"; then
    echo -e "${RED}Error: Python 3.13+ required${NC}"
    echo "Install with: sudo apt install python3.13 python3.13-venv"
    exit 1
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
uv venv --python 3.13

# Activate venv
source .venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
uv sync --all-extras

# Set up environment file
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cp .env.example .env
    echo -e "${RED}Please edit .env with your API keys!${NC}"
fi

# Set up direnv (optional but recommended)
if command -v direnv &> /dev/null; then
    echo "source .venv/bin/activate" > .envrc
    direnv allow
fi

# Initialize database if needed
if [ -n "${DATABASE_URL:-}" ]; then
    echo -e "${YELLOW}Running database migrations...${NC}"
    alembic upgrade head
fi

echo -e "${GREEN}Setup complete! Activate with: source .venv/bin/activate${NC}"
```

```bash
# .env
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
COHERE_API_KEY=...

# DSPy Configuration
DSPY_CACHEDIR=.dspy_cache
DSPY_ASYNC_MAX_WORKERS=10  # Defaults to CPU×4
DSPY_EXPERIMENTAL_ENABLED=false  # v3.0 features are now stable

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=dspy-production
MLFLOW_DSPY_AUTOLOG=true

# Observability
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
DSPY_LOG_LEVEL=INFO

# Cost Management
MONTHLY_LLM_BUDGET_USD=1000
COST_ALERT_THRESHOLD=0.8
ENABLE_FALLBACK_MODELS=true

# Database Configuration
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/db
REDIS_URL=redis://localhost:6379
```

### Configuration Management

```python
# src/config/settings.py
from decouple import Config as DecoupleConfig, RepositoryEnv

# Always use this pattern for configuration
decouple_config = DecoupleConfig(RepositoryEnv(".env"))

# Core settings
OPENAI_API_KEY = decouple_config("OPENAI_API_KEY")
DATABASE_URL = decouple_config("DATABASE_URL")
REDIS_URL = decouple_config("REDIS_URL")
DSPY_CACHEDIR = decouple_config("DSPY_CACHEDIR", default=".dspy_cache")
```

---

## 1. DSPy Architecture & Core Concepts

DSPy treats prompting as a programming problem, not a string manipulation exercise. Understanding its architecture is crucial for building maintainable pipelines. With DSPy 3.0, the framework has matured significantly with first-class async support and built-in production features.

### ✅ DO: Structure Your Project with Clear Separation of Concerns

Balance is key - avoid extreme fragmentation with dozens of tiny files, but also don't create monolithic modules. Group related functionality together while maintaining clear boundaries.

```
/src
├── signatures/          # DSPy signatures (input/output contracts)
│   ├── __init__.py
│   └── core.py         # All core signatures in one file (if < 500 lines)
├── modules/            # DSPy modules (reusable components)
│   ├── __init__.py
│   ├── chains.py       # Chain-of-thought modules
│   ├── agents.py       # Agent-based modules
│   └── tools.py        # Tool-calling modules
├── pipelines/          # Complete pipelines combining modules
│   ├── __init__.py
│   ├── research.py     # Research pipeline
│   └── analysis.py     # Analysis pipeline
├── optimizers/         # Custom optimizers and metrics
│   ├── __init__.py
│   └── metrics.py
├── data/              # Data loading and preprocessing
│   ├── __init__.py
│   └── loaders.py
├── evaluation/        # Evaluation harnesses
│   ├── __init__.py
│   └── benchmarks.py
├── config/           # Configuration management
│   ├── __init__.py
│   └── settings.py
└── utils/            # Shared utilities
    ├── __init__.py
    ├── console.py    # Rich console output utilities
    └── async_helpers.py
```

### Core Components Example with Rich Console Output

```python
# src/signatures/core.py
from typing import Literal
import dspy

class DocumentAnalysis(dspy.Signature):
    """Analyze a document and extract key insights."""
    
    document: str = dspy.InputField(
        desc="The document to analyze",
        max_length=8000
    )
    analysis_type: Literal["summary", "sentiment", "entities"] = dspy.InputField(
        desc="Type of analysis to perform"
    )
    
    insights: str = dspy.OutputField(
        desc="Key insights from the analysis",
        min_length=100,
        max_length=1000
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    metadata: dict = dspy.OutputField(
        desc="Additional structured metadata",
        format="json"
    )
```

```python
# src/modules/chains.py
import dspy
from src.signatures.core import DocumentAnalysis
from src.utils.console import console, create_progress_panel
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

class AnalysisChain(dspy.Module):
    """Chain-of-thought analysis module with self-reflection and rich output."""
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.1):
        super().__init__()
        self.analyze = dspy.ChainOfThought(DocumentAnalysis)
        self.reflect = dspy.Predict(
            "analysis, requirements -> improved_analysis, reasoning"
        )
        self.model = model
        self.temperature = temperature
        
        # Configure LM
        self.lm = dspy.LM(
            model=model,
            temperature=temperature,
            cache=True  # Built-in caching in v3.0
        )
    
    def forward(self, document: str, analysis_type: str) -> dspy.Prediction:
        """Synchronous forward for backward compatibility."""
        # Set context for this module
        with dspy.context(lm=self.lm):
            # First pass analysis
            initial = self.analyze(
                document=document,
                analysis_type=analysis_type
            )
            
            # Self-reflection for improvement
            requirements = f"Ensure the {analysis_type} is comprehensive and actionable"
            improved = self.reflect(
                analysis=initial.insights,
                requirements=requirements
            )
            
            return dspy.Prediction(
                insights=improved.improved_analysis,
                confidence=initial.confidence,
                metadata=initial.metadata,
                reasoning=improved.reasoning
            )
    
    async def aforward(self, document: str, analysis_type: str) -> dspy.Prediction:
        """Native async forward method with rich console output."""
        # Create progress display
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        )
        
        with Live(progress, console=console, transient=True) as live:
            # Set context for this module
            with dspy.context(lm=self.lm):
                # First pass analysis
                task = progress.add_task("[cyan]Analyzing document...", total=None)
                initial = await self.analyze.acall(
                    document=document,
                    analysis_type=analysis_type
                )
                progress.update(task, completed=True)
                
                # Show intermediate results
                console.print(Panel(
                    f"[green]Initial confidence: {initial.confidence:.2%}[/green]\n"
                    f"Tokens used: {getattr(initial, '_tokens_used', 'N/A')}",
                    title="[bold]First Pass Complete[/bold]",
                    border_style="blue"
                ))
                
                # Async self-reflection
                task = progress.add_task("[yellow]Improving analysis...", total=None)
                requirements = f"Ensure the {analysis_type} is comprehensive and actionable"
                improved = await self.reflect.acall(
                    analysis=initial.insights,
                    requirements=requirements
                )
                progress.update(task, completed=True)
            
            # Display final results with rich formatting
            console.print(Panel(
                improved.improved_analysis,
                title=f"[bold green]Enhanced {analysis_type.title()} Analysis[/bold green]",
                border_style="green",
                padding=(1, 2)
            ))
            
            return dspy.Prediction(
                insights=improved.improved_analysis,
                confidence=initial.confidence,
                metadata=initial.metadata,
                reasoning=improved.reasoning
            )
```

```python
# src/utils/console.py
from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.syntax import Syntax
import json

# Custom theme for LLM pipelines
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "processing": "magenta",
    "cost": "bold yellow on red",
})

console = Console(theme=custom_theme)

def create_progress_panel(title: str, content: str, style: str = "blue") -> Panel:
    """Create a formatted panel for progress updates."""
    return Panel(
        content,
        title=f"[bold]{title}[/bold]",
        border_style=style,
        padding=(1, 2)
    )

def display_cost_tracking(tokens_used: int, cost_usd: float):
    """Display cost information with rich formatting."""
    console.print(
        Panel(
            f"[cost]Tokens: {tokens_used:,} | Cost: ${cost_usd:.4f}[/cost]",
            title="[bold]API Usage[/bold]",
            border_style="yellow"
        )
    )

def display_json_output(data: dict, title: str = "Output"):
    """Display JSON data with syntax highlighting."""
    json_str = json.dumps(data, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=title, border_style="green"))
```

---

## 2. Advanced Pipeline Patterns

### ✅ DO: Build Composable, Async-First Pipelines with Fail-Fast Error Handling

DSPy 3.0 has true native async support (not just `asyncio.to_thread`), making it significantly more efficient for production workloads. Follow the **async-first design rule**: prototype synchronously, then flip to async for high-QPS paths. For key functionality, implement **fail-fast** patterns.

```python
# src/pipelines/research.py
import asyncio
from typing import List, Dict, Any
import dspy
import structlog
from src.utils.console import console, display_cost_tracking

logger = structlog.get_logger()

class ResearchPipeline(dspy.Module):
    """Multi-stage research pipeline with parallel processing and fail-fast patterns."""
    
    def __init__(self, 
                 primary_model: str = "gpt-4o",
                 fast_model: str = "gpt-4o-mini",
                 max_concurrent: int = 5,
                 fail_fast: bool = True):  # Fail fast for production
        super().__init__()
        
        self.fail_fast = fail_fast
        
        # Configure models
        self.primary_lm = dspy.LM(
            model=primary_model,
            max_tokens=2000,
            temperature=0.1,
            cache=True
        )
        self.fast_lm = dspy.LM(
            model=fast_model,
            max_tokens=500,
            temperature=0.0,
            cache=True
        )
        
        # Initialize modules
        self.decompose = dspy.ChainOfThought(
            "complex_query -> sub_queries: list[str], query_plan: str"
        )
        self.search = dspy.Predict(
            "query -> search_results: list[dict], relevance_scores: list[float]"
        )
        self.synthesize = dspy.ChainOfThought(
            "query, search_results, query_plan -> synthesis, confidence, citations"
        )
        
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def aforward(self, query: str) -> Dict[str, Any]:
        """Execute research pipeline with parallel sub-query processing and rich output."""
        
        console.rule(f"[bold blue]Research Pipeline: {query[:50]}...[/bold blue]")
        logger.info("research_pipeline_started", query=query)
        
        try:
            # Decompose complex query
            with console.status("[bold green]Decomposing query..."):
                with dspy.context(lm=self.primary_lm):
                    decomposition = await self.decompose.acall(complex_query=query)
            
            console.print(f"[info]Generated {len(decomposition.sub_queries)} sub-queries[/info]")
            
            # Parallel search for sub-queries with progress tracking
            from rich.progress import Progress, TaskID
            
            with Progress(console=console) as progress:
                main_task = progress.add_task(
                    "[cyan]Searching sub-queries...", 
                    total=len(decomposition.sub_queries)
                )
                
                search_tasks = []
                for sub_query in decomposition.sub_queries:
                    task = self._search_with_limit(sub_query, progress, main_task)
                    search_tasks.append(task)
                
                if self.fail_fast:
                    # Use TaskGroup for fail-fast behavior (Python 3.11+)
                    try:
                        async with asyncio.TaskGroup() as tg:
                            tasks = [tg.create_task(task) for task in search_tasks]
                        search_results = [task.result() for task in tasks]
                    except* Exception as eg:
                        # Handle the exception group
                        console.print("[error]Critical error in search phase - failing fast[/error]")
                        for error in eg.exceptions:
                            logger.error("search_error", error=str(error))
                        raise eg.exceptions[0]  # Re-raise first error
                else:
                    # Graceful degradation mode
                    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
                    # Filter out errors
                    search_results = [r for r in search_results if not isinstance(r, Exception)]
            
            # Synthesize results
            console.print("[processing]Synthesizing results...[/processing]")
            with dspy.context(lm=self.primary_lm):
                synthesis = await self.synthesize.acall(
                    query=query,
                    search_results=search_results,
                    query_plan=decomposition.query_plan
                )
            
            # Display results and costs
            console.print(Panel(
                synthesis.synthesis,
                title="[bold green]Research Synthesis[/bold green]",
                border_style="green"
            ))
            
            # Track costs
            total_cost = getattr(self.primary_lm, '_session_cost', 0) + \
                        getattr(self.fast_lm, '_session_cost', 0)
            display_cost_tracking(
                tokens_used=getattr(synthesis, '_total_tokens', 0),
                cost_usd=total_cost
            )
            
            logger.info("research_pipeline_completed", 
                       query=query,
                       confidence=synthesis.confidence)
            
            return {
                "query": query,
                "synthesis": synthesis.synthesis,
                "confidence": synthesis.confidence,
                "citations": synthesis.citations,
                "sub_queries": decomposition.sub_queries,
                "search_results": search_results
            }
            
        except Exception as e:
            console.print(f"[error]Pipeline failed: {str(e)}[/error]")
            if self.fail_fast:
                raise
            return {"error": str(e), "query": query}
    
    async def _search_with_limit(self, query: str, progress: Progress, task_id: TaskID) -> Dict[str, Any]:
        """Rate-limited search with semaphore and progress update."""
        async with self.semaphore:
            with dspy.context(lm=self.fast_lm):
                result = await self.search.acall(query=query)
                progress.advance(task_id)
                return result.dict()
    
    def forward(self, query: str) -> Dict[str, Any]:
        """Synchronous wrapper for backward compatibility."""
        return asyncio.run(self.aforward(query))
```

### ✅ DO: Use Native Async Tools and Streaming

```python
# src/modules/tools/async_tools.py
import dspy
import asyncio
from typing import List, AsyncIterator
import httpx
from src.utils.console import console

# Define async tool function with proper error handling
async def web_search(query: str) -> List[str]:
    """Async web search tool with fail-fast pattern."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.search.com/v1/search",
                params={"q": query},
                headers={"Authorization": f"Bearer {SEARCH_API_KEY}"}
            )
            response.raise_for_status()  # Fail fast on HTTP errors
            return response.json()["results"]
    except httpx.HTTPError as e:
        console.print(f"[error]Search API error: {e}[/error]")
        raise  # Fail fast for critical functionality

# Create tool with DSPy 3.0 syntax
web_search_tool = dspy.Tool(
    func=web_search,
    name="web_search",
    desc="Search the web for information"
)

# Use with ReAct agent
class AsyncResearchAgent(dspy.Module):
    """Agent with async tool support and rich console output."""
    
    def __init__(self):
        super().__init__()
        self.agent = dspy.ReAct(
            signature="question -> answer",
            tools=[web_search_tool],
            max_iters=3
        )
    
    async def aforward(self, question: str) -> dspy.Prediction:
        """Native async execution with progress tracking."""
        with console.status(f"[bold green]Researching: {question}[/bold green]"):
            # acall() automatically uses tool's async methods
            result = await self.agent.acall(question=question)
            
        # Display trace of tool calls
        if hasattr(result, '_tool_calls'):
            console.print("[dim]Tool calls:[/dim]")
            for call in result._tool_calls:
                console.print(f"  → {call['tool']}: {call['input'][:50]}...")
                
        return result
```

### ✅ DO: Implement Streaming for Real-time Output

```python
# src/modules/streaming/stream_module.py
import dspy
from typing import AsyncIterator
from rich.live import Live
from rich.text import Text

class StreamingModule(dspy.Module):
    """Module with streaming support and live console display."""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought("topic -> article")
    
    async def stream(self, topic: str) -> AsyncIterator[str]:
        """Stream output tokens as they're generated with live display."""
        # Convert to streaming module
        streaming_generate = dspy.streamify(self.generate)
        
        # Create live display
        output_text = Text()
        
        with Live(output_text, console=console, refresh_per_second=10) as live:
            async for chunk in streaming_generate(topic=topic):
                if isinstance(chunk, str):
                    output_text.append(chunk)
                    yield chunk
                elif isinstance(chunk, dspy.Prediction):
                    # Final prediction object
                    confidence_text = f"\n\n[bold green]Confidence: {chunk.confidence:.2%}[/bold green]"
                    output_text.append(confidence_text)
                    yield confidence_text
```

### ✅ DO: Implement Fallback and Retry Strategies with Exponential Backoff

```python
# src/modules/tools/resilient_module.py
import asyncio
from typing import Any, Optional, List
import dspy
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import structlog
from src.utils.console import console

logger = structlog.get_logger()

class ResilientModule(dspy.Module):
    """Module with automatic fallbacks, retries, and rich error reporting."""
    
    def __init__(self, 
                 primary_model: str = "gpt-4o",
                 fallback_models: List[str] = ["claude-3-sonnet", "gpt-4o-mini"],
                 max_retries: int = 3):
        super().__init__()
        
        self.models = [primary_model] + fallback_models
        self.max_retries = max_retries
        self.predict = dspy.Predict("input -> output, reasoning")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    async def aforward(self, input_text: str) -> dspy.Prediction:
        """Execute with automatic retries and model fallbacks."""
        
        last_error = None
        
        for i, model in enumerate(self.models):
            try:
                console.print(f"[cyan]Trying model {i+1}/{len(self.models)}: {model}[/cyan]")
                
                lm = dspy.LM(model=model, cache=True)
                with dspy.context(lm=lm):
                    result = await self.predict.acall(input=input_text)
                    
                    # Validate output
                    if self._validate_output(result):
                        console.print(f"[success]✓ Success with {model}[/success]")
                        return result
                    else:
                        raise ValueError(f"Invalid output from {model}")
                        
            except Exception as e:
                console.print(f"[warning]✗ Model {model} failed: {str(e)}[/warning]")
                logger.warning(f"Model {model} failed", error=str(e))
                last_error = e
                
                if i < len(self.models) - 1:
                    # Add jitter to avoid thundering herd
                    await asyncio.sleep(0.1 * (i + 1))
                continue
        
        console.print("[error]All models failed![/error]")
        raise last_error or Exception("All models failed")
    
    def _validate_output(self, result: dspy.Prediction) -> bool:
        """Validate output meets quality criteria."""
        return (
            len(result.output) > 10 and
            result.reasoning and
            len(result.reasoning) > 20
        )
```

---

## 3. Testing and Evaluation

### ✅ DO: Implement REAL Integration Tests Only (No Mocks!)

Following the principle of **no unit tests or mocks**, all tests should use real API calls and real data.

```python
# tests/test_pipelines.py
import pytest
import dspy
from dspy.evaluate import Evaluate
from src.pipelines.research import ResearchPipeline
import asyncio
from src.utils.console import console
from rich.table import Table

class TestResearchPipeline:
    """REAL integration tests for research pipeline - no mocks!"""
    
    @pytest.fixture
    def pipeline(self):
        """Initialize pipeline with real models."""
        return ResearchPipeline(
            primary_model="gpt-4o",  # Use real model for integration tests
            max_concurrent=3,
            fail_fast=True  # Test fail-fast behavior
        )
    
    @pytest.fixture
    def test_examples(self):
        """Load REAL test examples - no fake data!"""
        return [
            dspy.Example(
                query="What are the latest developments in quantum computing as of 2025?",
                expected_topics=["quantum supremacy", "error correction", "qubits"],
                min_confidence=0.7
            ),
            dspy.Example(
                query="Explain the environmental impact of lithium mining in Chile",
                expected_topics=["water usage", "habitat destruction", "carbon footprint"],
                min_confidence=0.8
            )
        ]
    
    @pytest.mark.asyncio
    async def test_pipeline_with_real_apis(self, pipeline, test_examples):
        """Test pipeline accuracy with REAL API calls."""
        
        console.rule("[bold blue]Running Integration Tests[/bold blue]")
        
        def accuracy_metric(example, prediction):
            # Check if expected topics are covered
            output_lower = prediction['synthesis'].lower()
            topics_found = sum(
                1 for topic in example.expected_topics
                if topic.lower() in output_lower
            )
            topic_coverage = topics_found / len(example.expected_topics)
            
            # Check confidence threshold
            confidence_met = prediction['confidence'] >= example.min_confidence
            
            return (topic_coverage + float(confidence_met)) / 2
        
        # Create results table
        table = Table(title="Integration Test Results")
        table.add_column("Query", style="cyan", no_wrap=False)
        table.add_column("Score", style="magenta")
        table.add_column("Confidence", style="green")
        table.add_column("Cost", style="yellow")
        
        # DSPy 3.0 Evaluate API returns object with .score and .results
        evaluator = Evaluate(
            devset=test_examples,
            metric=accuracy_metric,
            num_threads=1,
            display_progress=True
        )
        
        eval_result = evaluator(pipeline)
        
        # Display results with rich formatting
        for idx, result in enumerate(eval_result.results):
            table.add_row(
                test_examples[idx].query[:50] + "...",
                f"{result.score:.2%}",
                f"{result.prediction.get('confidence', 0):.2%}",
                f"${getattr(result, 'cost', 0):.4f}"
            )
        
        console.print(table)
        
        assert eval_result.score >= 0.8, f"Pipeline accuracy {eval_result.score} below threshold"
    
    @pytest.mark.asyncio
    async def test_pipeline_latency_real_world(self, pipeline):
        """Test pipeline latency with REAL API calls."""
        import time
        
        query = "What is the capital of France and its population?"
        
        console.print("[cyan]Testing real-world latency...[/cyan]")
        start = time.time()
        result = await pipeline.aforward(query)
        latency = time.time() - start
        
        console.print(f"[info]Latency: {latency:.2f}s[/info]")
        console.print(f"[info]Result length: {len(result['synthesis'])} chars[/info]")
        
        assert latency < 10.0, f"Pipeline latency {latency}s exceeds 10s limit"
        assert result['synthesis'], "Pipeline returned empty synthesis"
    
    @pytest.mark.asyncio
    async def test_fail_fast_behavior(self, pipeline):
        """Test fail-fast error handling with real scenarios."""
        
        console.print("[yellow]Testing fail-fast behavior...[/yellow]")
        
        # Test with invalid API key scenario
        with pytest.raises(Exception) as exc_info:
            # Temporarily break the API key
            original_key = pipeline.primary_lm.api_key
            pipeline.primary_lm.api_key = "invalid_key"
            
            try:
                await pipeline.aforward("Test query")
            finally:
                # Restore key
                pipeline.primary_lm.api_key = original_key
        
        console.print(f"[success]✓ Failed fast as expected: {exc_info.value}[/success]")
```

### ✅ DO: Use Rich Console for Detailed Test Output

```python
# tests/conftest.py
import pytest
from rich.console import Console
from rich.logging import RichHandler
import logging

# Configure rich logging for all tests
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=Console(stderr=True))]
)

@pytest.fixture(autouse=True)
def setup_console_for_tests():
    """Ensure rich console works properly in tests."""
    from src.utils.console import console
    # Force color output in tests
    console.force_terminal = True
    console.force_interactive = True
    yield
    # Cleanup
    console.force_terminal = None
    console.force_interactive = None
```

---

## 4. Optimization and Fine-tuning

### ✅ DO: Use DSPy's Latest Optimizers with Real Data

```python
# src/optimizers/pipeline_optimizer.py
import dspy
from dspy.teleprompt import MIPROv2, SIMBA, BetterTogether, BootstrapFewShot
from typing import List, Callable, Optional
import json
from src.utils.console import console, display_json_output
from rich.progress import track

class PipelineOptimizer:
    """Optimize DSPy pipelines with state-of-the-art optimizers and rich output."""
    
    def __init__(self, 
                 pipeline: dspy.Module,
                 metric: Callable,
                 trainset: List[dspy.Example],
                 valset: Optional[List[dspy.Example]] = None):
        self.pipeline = pipeline
        self.metric = metric
        self.trainset = trainset
        self.valset = valset or trainset[:len(trainset)//5]
        
        # Validate data quality
        console.print(f"[info]Training set size: {len(self.trainset)}[/info]")
        console.print(f"[info]Validation set size: {len(self.valset)}[/info]")
    
    async def optimize_with_miprov2(self, 
                                    mode: str = "light",
                                    max_iterations: Optional[int] = None) -> dspy.Module:
        """
        MIPROv2: The current best optimizer for joint instruction + few-shot search.
        Modes: 
        - "light" (5-10 min): Quick optimization for prototyping
        - "medium" (20-30 min): Balanced for most production cases  
        - "heavy" (1+ hours): Maximum performance with extensive search
        """
        
        console.rule(f"[bold blue]Optimizing with MIPROv2 ({mode} mode)[/bold blue]")
        
        optimizer = MIPROv2(
            metric=self.metric,
            auto=mode,  # Automatic configuration based on mode
            prompt_model=dspy.LM("gpt-4o-mini"),  # Fast model for generating prompts
            task_model=self.pipeline.lm,  # Your task model
            num_iterations=max_iterations  # Override auto if needed
        )
        
        # Show optimization progress with rich
        with console.status("[bold green]Optimizing pipeline...") as status:
            optimized_pipeline = await optimizer.acompile(
                self.pipeline,
                trainset=self.trainset,
                valset=self.valset,
                max_bootstrapped_demos=4,
                max_labeled_demos=4,
                requires_permission_to_run=False
            )
        
        # Save and display optimization results
        self._save_optimization_report(optimizer)
        
        console.print("[success]✓ Optimization complete![/success]")
        return optimized_pipeline
    
    def _save_optimization_report(self, optimizer):
        """Save detailed optimization report with rich display."""
        report = {
            "optimizer_type": type(optimizer).__name__,
            "best_score": getattr(optimizer, 'best_score', None),
            "optimization_history": getattr(optimizer, 'history', []),
            "final_prompts": self._extract_prompts(
                optimizer.best_program if hasattr(optimizer, 'best_program') else self.pipeline
            )
        }
        
        # Save to file
        with open("optimization_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Display summary
        display_json_output(
            {
                "optimizer": report["optimizer_type"],
                "best_score": report["best_score"],
                "iterations": len(report["optimization_history"])
            },
            title="Optimization Summary"
        )
        
        # Show best prompts
        if report["final_prompts"]:
            console.print("\n[bold]Optimized Prompts:[/bold]")
            for name, prompt in report["final_prompts"].items():
                console.print(f"\n[cyan]{name}:[/cyan]")
                console.print(Panel(prompt[:200] + "...", border_style="blue"))
    
    def _extract_prompts(self, program: dspy.Module) -> dict:
        """Extract all prompts from optimized program."""
        prompts = {}
        for name, module in program.named_modules():
            if hasattr(module, 'extended_signature'):
                prompts[name] = str(module.extended_signature)
        return prompts
```

---

## 5. Production Deployment with Modern Patterns

### ✅ DO: Use Typer CLI with Rich Output

```python
# src/cli/main.py
import typer
from typing import Optional
from rich.console import Console
from rich.table import Table
import asyncio
from src.pipelines.research import ResearchPipeline
from src.config.settings import decouple_config

app = typer.Typer(
    name="dspy-pipeline",
    help="Production DSPy Pipeline CLI",
    add_completion=True,
    rich_markup_mode="rich"
)

console = Console()

@app.command()
def research(
    query: str = typer.Argument(..., help="Research query to process"),
    model: str = typer.Option("gpt-4o", help="Primary model to use"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Execute research pipeline with rich console output."""
    
    async def run():
        pipeline = ResearchPipeline(primary_model=model)
        
        if verbose:
            console.print(f"[dim]Using model: {model}[/dim]")
            console.print(f"[dim]Query: {query}[/dim]")
        
        try:
            result = await pipeline.aforward(query)
            
            # Display results in a nice table
            table = Table(title="Research Results", show_header=True)
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Confidence", f"{result['confidence']:.2%}")
            table.add_row("Sub-queries", str(len(result['sub_queries'])))
            table.add_row("Citations", str(len(result.get('citations', []))))
            
            console.print(table)
            
            # Show synthesis
            console.print("\n[bold]Synthesis:[/bold]")
            console.print(Panel(result['synthesis'], border_style="green"))
            
        except Exception as e:
            console.print(f"[error]Error: {str(e)}[/error]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    # Use asyncio.Runner for Python 3.13+
    with asyncio.Runner() as runner:
        runner.run(run())

@app.command()
def status():
    """Check system status and configuration."""
    
    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    
    # Check API keys
    openai_key = decouple_config("OPENAI_API_KEY", default="")
    table.add_row(
        "OpenAI API", 
        "✓ Configured" if openai_key else "✗ Missing",
        "Key ending in ..." + openai_key[-4:] if openai_key else "Not set"
    )
    
    # Check cache
    cache_dir = decouple_config("DSPY_CACHEDIR", default=".dspy_cache")
    import os
    cache_exists = os.path.exists(cache_dir)
    table.add_row(
        "DSPy Cache",
        "✓ Active" if cache_exists else "✗ Not initialized",
        cache_dir
    )
    
    console.print(table)

if __name__ == "__main__":
    app()
```

### ✅ DO: Use Modern Async Patterns (Python 3.13+)

```python
# src/utils/async_helpers.py
import asyncio
from typing import Coroutine, TypeVar, List, Any
from contextlib import asynccontextmanager

T = TypeVar('T')

def run_async(coro: Coroutine[None, None, T]) -> T:
    """Run async code with proper context propagation (Python 3.13+)"""
    with asyncio.Runner() as runner:
        return runner.run(coro)

async def parallel_map(func: Callable, items: List[Any], max_concurrent: int = 10) -> List[Any]:
    """Map function over items with controlled parallelism using TaskGroup."""
    results = [None] * len(items)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_item(idx: int, item: Any):
        async with semaphore:
            results[idx] = await func(item)
    
    # Use TaskGroup for better error handling (Python 3.11+)
    async with asyncio.TaskGroup() as tg:
        for idx, item in enumerate(items):
            tg.create_task(process_item(idx, item))
    
    return results

@asynccontextmanager
async def timed_operation(name: str):
    """Context manager for timing async operations with rich output."""
    from src.utils.console import console
    import time
    
    start = time.time()
    console.print(f"[cyan]Starting {name}...[/cyan]")
    try:
        yield
    finally:
        duration = time.time() - start
        console.print(f"[green]✓ {name} completed in {duration:.2f}s[/green]")
```

### ✅ DO: Deploy with Modern Docker Patterns

```dockerfile
# Dockerfile
# Stage 1: Builder
FROM python:3.13-slim-bookworm as builder

# Set up the environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    UV_SYSTEM_PYTHON=true

WORKDIR /app

# Install uv
RUN pip install uv

# Copy only the dependency configuration files
COPY pyproject.toml uv.lock* ./

# Install dependencies into a virtual environment
RUN uv venv .venv && \
    source .venv/bin/activate && \
    uv sync --all-extras

# ---
# Stage 2: Final Image
FROM python:3.13-slim-bookworm as final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-root user for security
RUN addgroup --system app && adduser --system --group app
USER app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv ./.venv

# Copy the application source code
COPY ./src ./src
COPY ./setup.sh .

# Make the virtual environment's executables available
ENV PATH="/app/.venv/bin:$PATH"

# Expose the port (if running API server)
EXPOSE 8000

# The command to run the application
CMD ["python", "-m", "src.cli.main"]
```

---

## Common Pitfalls and Solutions

### ❌ DON'T: Use Mocks or Fake Data in Tests

```python
# Bad - Using mocks
def test_pipeline_with_mocks():
    mock_llm = Mock()
    mock_llm.complete.return_value = "Mocked response"
    # This tells you nothing about real behavior!

# Good - Always use real APIs
@pytest.mark.asyncio
async def test_pipeline_real():
    pipeline = ResearchPipeline()
    result = await pipeline.aforward("Real query")
    assert result['confidence'] > 0.5  # Real validation
```

### ❌ DON'T: Hide Errors - Fail Fast for Critical Components

```python
# Bad - Swallowing errors
try:
    result = await critical_api_call()
except Exception:
    result = None  # Hide the problem

# Good - Fail fast with rich error output
try:
    result = await critical_api_call()
except Exception as e:
    console.print(f"[error]Critical API failed: {e}[/error]")
    console.print_exception()  # Full traceback with syntax highlighting
    raise  # Fail fast
```

### ✅ DO: Use Rich for All Console Output

```python
# Bad - Plain print statements
print(f"Processing {len(items)} items...")
for i, item in enumerate(items):
    print(f"Item {i}: {item}")

# Good - Rich console with progress
from rich.progress import track

console.print(f"[info]Processing {len(items)} items...[/info]")
for item in track(items, description="Processing..."):
    process_item(item)
```

### ✅ DO: Track Costs with Rich Display

```python
# src/utils/cost_tracker.py
from collections import defaultdict
import tiktoken
from rich.table import Table
from src.utils.console import console

class CostTracker:
    """Track and display LLM costs with rich formatting."""
    
    def __init__(self):
        self.costs = defaultdict(float)
        self.tokens = defaultdict(int)
        
    def track_call(self, model: str, tokens: int, cost: float):
        """Track a single API call."""
        self.costs[model] += cost
        self.tokens[model] += tokens
        
    def display_summary(self):
        """Display cost summary with rich table."""
        table = Table(title="API Cost Summary", show_footer=True)
        table.add_column("Model", style="cyan")
        table.add_column("Tokens", style="magenta", justify="right")
        table.add_column("Cost", style="yellow", justify="right")
        
        total_cost = 0
        total_tokens = 0
        
        for model in sorted(self.costs.keys()):
            table.add_row(
                model,
                f"{self.tokens[model]:,}",
                f"${self.costs[model]:.4f}"
            )
            total_cost += self.costs[model]
            total_tokens += self.tokens[model]
        
        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{total_tokens:,}[/bold]",
            f"[bold]${total_cost:.4f}[/bold]",
            style="bold green"
        )
        
        console.print(table)
        
        # Add run rate estimate
        console.print(f"\n[dim]Estimated daily cost at current rate: ${total_cost * 24:.2f}[/dim]")
```

## Conclusion

This guide provides a comprehensive framework for building production-grade LLM applications with DSPy 3.0 using modern Python 3.13+ patterns. Key takeaways:

1. **Python 3.13+ and uv**: Always use the latest Python with uv for dependency management
2. **Async Everything**: Use native async support for better performance  
3. **Rich Console Output**: Leverage rich library for polished, informative output
4. **Real Integration Tests**: No mocks or fake data - test with real APIs
5. **Fail Fast**: Don't hide critical errors - fail fast and fix issues
6. **Optimize Systematically**: Use DSPy's optimizers with real data
7. **Monitor Everything**: Track costs, performance, and errors with rich displays

The field of LLM engineering is rapidly evolving. Stay current with the DSPy community and contribute your learnings back to help others build better AI systems.

For the latest updates and patterns, refer to:
- Official DSPy Documentation: https://dspy.ai
- DSPy GitHub: https://github.com/stanfordnlp/dspy
- Community Forum: https://discord.gg/dspy
</rewritten_file>