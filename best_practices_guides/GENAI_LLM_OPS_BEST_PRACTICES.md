# The Definitive Guide to Production LLM Systems (Mid-2025 Edition)

This guide synthesizes battle-tested patterns for building scalable, cost-efficient, and reliable LLM applications. Written from the perspective of teams who've deployed systems handling millions of daily requests, it covers the complete stack from orchestration to observability.

### Prerequisites & System Requirements
Ensure your infrastructure supports **CUDA 12.5+**, **Python 3.12+** (3.13 for free-threaded builds), and **Kubernetes 1.31+** with GPU operator support.

Core tooling configuration:
```bash
# .envrc for direnv users
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_ATTENTION_BACKEND="FLASHINFER"  # 15% faster than FlashAttention-2
export TOKENIZERS_PARALLELISM="false"      # Prevent threading issues
source .venv/bin/activate
```

System dependencies check script:
```bash
#!/bin/bash
# setup-ml-env.sh
echo "🔍 Checking ML/LLM development environment..."

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ CUDA not found. Install CUDA 12.5+"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
if [[ ! "$PYTHON_VERSION" > "3.12" ]]; then
    echo "❌ Python 3.12+ required (found $PYTHON_VERSION)"
    exit 1
fi

# Verify GPU memory for model serving
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [[ $GPU_MEM -lt 24000 ]]; then
    echo "⚠️  GPU has ${GPU_MEM}MB. Recommended: 24GB+ for production serving"
fi

echo "✅ Environment ready for LLM development"
```

---

## 1. Architecture Foundation: The Four-Layer LLM Stack

Modern LLM applications require careful separation of concerns across four distinct layers. This architecture enables independent scaling, clear ownership boundaries, and technology flexibility.

### ✅ DO: Implement a Clear Layer Separation

```
┌─────────────────────────────────────────────────────┐
│                  Application Layer                   │
│            (FastAPI, Next.js, Streamlit)            │
├─────────────────────────────────────────────────────┤
│              Orchestration & Agent Layer             │
│         (LangGraph, Semantic Kernel, DSPy)          │
├─────────────────────────────────────────────────────┤
│                Data & Context Layer                  │
│      (LlamaIndex, Haystack 2, Vector Stores)        │
├─────────────────────────────────────────────────────┤
│               Model Serving Layer                    │
│        (vLLM, SGLang, Ollama, KServe)              │
└─────────────────────────────────────────────────────┘
```

### Project Structure for Production LLM Systems

```
/llm-app
├── agents/                  # Agent definitions and workflows
│   ├── graph_definitions/   # LangGraph workflow definitions
│   ├── tools/              # Custom tool implementations
│   └── prompts/            # Versioned prompt templates
├── serving/                # Model serving configurations
│   ├── vllm/              # vLLM server configs
│   ├── endpoints/         # Model endpoint definitions
│   └── adapters/          # LoRA adapter management
├── data/                   # RAG and data pipeline
│   ├── loaders/           # Document loaders
│   ├── chunkers/          # Text splitting strategies
│   └── indexes/           # Vector index definitions
├── eval/                   # Evaluation and monitoring
│   ├── datasets/          # Eval datasets
│   ├── metrics/           # Custom metric definitions
│   └── scorers/           # LLM-as-judge implementations
├── api/                    # FastAPI application layer
│   ├── routers/           # Endpoint definitions
│   └── middleware/        # Request handling
├── k8s/                    # Kubernetes manifests
│   ├── kserve/            # KServe InferenceService specs
│   └── monitoring/        # Prometheus, Grafana configs
└── tests/                  # End-to-end test suites
```

### ❌ DON'T: Mix Concerns Across Layers

Bad practice that creates tight coupling and deployment nightmares:

```python
# Bad - api/routes.py mixing all concerns
from transformers import AutoModelForCausalLM
import chromadb

@app.post("/chat")
async def chat(message: str):
    # DON'T: Loading models in API routes
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
    
    # DON'T: Direct vector DB access in endpoints
    chroma_client = chromadb.Client()
    results = chroma_client.query(message)
    
    # DON'T: Inline prompt engineering
    prompt = f"Context: {results}\nQuestion: {message}\nAnswer:"
    
    return model.generate(prompt)
```

---

## 2. Orchestration Layer: LangGraph for Complex Workflows

LangGraph has emerged as the production standard for multi-step agent workflows, offering superior debugging, state management, and deployment options compared to alternatives.

### ✅ DO: Define Agents as State Machines with LangGraph

```python
# agents/graph_definitions/research_agent.py
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import operator

class ResearchState(TypedDict):
    """Typed state ensures compile-time validation"""
    query: str
    search_results: List[str]
    analysis: str
    final_answer: str
    confidence: float
    
async def web_search(state: ResearchState) -> ResearchState:
    """Tool-calling node for web search"""
    # Implement parallel search across multiple sources
    results = await asyncio.gather(
        search_arxiv(state["query"]),
        search_scholar(state["query"]),
        search_web(state["query"])
    )
    state["search_results"] = flatten_results(results)
    return state

async def analyze_results(state: ResearchState) -> ResearchState:
    """LLM analysis of search results"""
    # Stream to vLLM endpoint for analysis
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://vllm-svc:8000/v1/chat/completions",
            json={
                "model": "meta-llama/Llama-3.3-70B-Instruct",
                "messages": [
                    {"role": "system", "content": ANALYSIS_PROMPT},
                    {"role": "user", "content": format_search_results(state)}
                ],
                "temperature": 0.1,
                "max_tokens": 2000,
                "guided_json": ANALYSIS_SCHEMA  # Force structured output
            }
        ) as response:
            result = await response.json()
            state["analysis"] = result["choices"][0]["message"]["content"]
    return state

def should_continue(state: ResearchState) -> str:
    """Conditional edge based on confidence"""
    if state["confidence"] < 0.7:
        return "refine_search"
    return "generate_answer"

# Build the graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("search", web_search)
workflow.add_node("analyze", analyze_results)
workflow.add_node("refine_search", refine_query)
workflow.add_node("generate_answer", create_final_answer)

# Add edges
workflow.set_entry_point("search")
workflow.add_edge("search", "analyze")
workflow.add_conditional_edges(
    "analyze",
    should_continue,
    {
        "refine_search": "search",  # Loop back for better results
        "generate_answer": "generate_answer"
    }
)
workflow.add_edge("generate_answer", END)

# Compile with PostgreSQL checkpointing for production
checkpointer = AsyncPostgresSaver.from_conn_string(
    "postgresql://agent_user:pass@postgres:5432/agents"
)
app = workflow.compile(checkpointer=checkpointer)
```

### Advanced LangGraph Patterns

#### Human-in-the-Loop with Interrupts
```python
# Add interrupt for human review on low confidence
workflow.add_node("human_review", wait_for_human_input)
workflow.add_conditional_edges(
    "analyze",
    lambda s: "human_review" if s["confidence"] < 0.5 else "continue",
    {
        "human_review": "human_review",
        "continue": "generate_answer"
    }
)

# In your API endpoint
@app.post("/continue/{thread_id}")
async def continue_after_human_input(thread_id: str, human_input: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    # Update state with human input
    current_state = await app.aget_state(config)
    current_state.values["human_feedback"] = human_input
    
    # Continue execution
    result = await app.ainvoke(None, config)
    return result
```

#### Parallel Tool Execution
```python
async def parallel_tools_node(state: dict):
    """Execute multiple tools concurrently with semaphore control"""
    semaphore = asyncio.Semaphore(5)  # Limit concurrent executions
    
    async def run_tool_with_limit(tool_name: str, args: dict):
        async with semaphore:
            return await TOOL_REGISTRY[tool_name](**args)
    
    # Run tools in parallel
    tasks = [
        run_tool_with_limit("search_code", {"query": state["query"]}),
        run_tool_with_limit("search_docs", {"query": state["query"]}),
        run_tool_with_limit("analyze_issues", {"repo": state["repo"]})
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle partial failures gracefully
    state["tool_results"] = {
        name: result for name, result in zip(["code", "docs", "issues"], results)
        if not isinstance(result, Exception)
    }
    return state
```

### ✅ DO: Implement Streaming for Real-Time UX

```python
# api/routers/agents.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langgraph.pregel import StreamMode

router = APIRouter()

@router.post("/research/stream")
async def stream_research(query: ResearchQuery):
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 10  # Prevent infinite loops
    }
    
    async def event_generator():
        async for event in app.astream_events(
            {"query": query.text},
            config,
            version="v2",
            stream_mode=StreamMode.VALUES
        ):
            # Stream state updates as SSE
            if event["event"] == "on_chain_stream":
                yield f"data: {json.dumps({
                    'type': 'state_update',
                    'node': event['name'],
                    'data': event['data']
                })}\n\n"
            
            # Stream LLM tokens for immediate display
            elif event["event"] == "on_llm_stream":
                yield f"data: {json.dumps({
                    'type': 'token',
                    'content': event['data']['chunk']
                })}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # Disable Nginx buffering
        }
    )
```

---

## 3. Data & RAG Layer: Production Retrieval Patterns

The shift from naive RAG to production systems requires sophisticated chunking, multi-stage retrieval, and careful index management.

### ✅ DO: Implement Multi-Stage Retrieval with Reranking

```python
# data/retrievers/hybrid_retriever.py
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.retrievers import BM25Retriever
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.fastembed import FastEmbedEmbedding
import cohere

class ProductionHybridRetriever:
    def __init__(self):
        # Use FastEmbed for 10x faster embedding than OpenAI
        self.embed_model = FastEmbedEmbedding(
            model_name="BAAI/bge-m3",  # Multilingual, 8192 context
            batch_size=256,
            parallel=8
        )
        
        # Semantic chunking > fixed-size chunking
        self.splitter = SemanticSplitterNodeParser(
            embed_model=self.embed_model,
            breakpoint_percentile_threshold=85,
            max_chunk_size=512,
            min_chunk_size=100
        )
        
        # Initialize Milvus for scale
        self.vector_store = MilvusVectorStore(
            uri="http://milvus:19530",
            collection_name="documents",
            dim=1024,
            index_params={
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 2048}
            },
            search_params={
                "metric_type": "IP",
                "params": {"nprobe": 16}
            }
        )
        
        # Cohere reranker for precision
        self.reranker = cohere.Client(api_key=COHERE_API_KEY)
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 100,
        rerank_top_k: int = 10,
        filters: dict = None
    ) -> List[Document]:
        """Production retrieval pipeline with hybrid search"""
        
        # Stage 1: Parallel dense + sparse retrieval
        dense_task = self._dense_search(query, top_k * 2, filters)
        sparse_task = self._sparse_search(query, top_k * 2, filters)
        
        dense_results, sparse_results = await asyncio.gather(
            dense_task, sparse_task
        )
        
        # Stage 2: Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=60  # Optimal k from experiments
        )
        
        # Stage 3: Semantic reranking
        if len(fused_results) > rerank_top_k:
            reranked = await self._rerank_results(
                query, fused_results[:top_k]
            )
            return reranked[:rerank_top_k]
        
        return fused_results
    
    async def _dense_search(self, query: str, top_k: int, filters: dict):
        """Vector similarity search with metadata filtering"""
        query_embedding = await self.embed_model.aget_text_embedding(query)
        
        # Milvus expression syntax for filtering
        filter_expr = self._build_milvus_filter(filters) if filters else None
        
        results = self.vector_store.query(
            query_embedding,
            top_k=top_k,
            expr=filter_expr
        )
        return results
    
    async def _rerank_results(self, query: str, documents: List[Document]):
        """Cohere reranking for precision boost"""
        doc_texts = [doc.text for doc in documents]
        
        rerank_response = self.reranker.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=doc_texts,
            top_n=len(documents),
            return_documents=False
        )
        
        # Reorder based on rerank scores
        reranked_docs = [
            documents[result.index] 
            for result in rerank_response.results
        ]
        
        return reranked_docs
    
    def _reciprocal_rank_fusion(
        self, 
        result_sets: List[List[Document]], 
        k: int = 60
    ) -> List[Document]:
        """RRF implementation for combining multiple rankings"""
        scores = {}
        
        for result_set in result_sets:
            for rank, doc in enumerate(result_set):
                doc_id = doc.id_
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += 1 / (k + rank + 1)
        
        # Sort by fused score
        sorted_docs = sorted(
            scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Retrieve full documents
        doc_map = {
            doc.id_: doc 
            for result_set in result_sets 
            for doc in result_set
        }
        
        return [doc_map[doc_id] for doc_id, _ in sorted_docs]
```

### Advanced Indexing Strategies

#### Hierarchical Indexing for Long Documents
```python
# data/indexes/hierarchical_index.py
class HierarchicalIndex:
    """Index documents at multiple granularities"""
    
    def __init__(self):
        self.chunk_sizes = [256, 512, 1024, 2048]
        self.indexes = {}
        
        for size in self.chunk_sizes:
            self.indexes[size] = VectorStoreIndex(
                vector_store=MilvusVectorStore(
                    collection_name=f"chunks_{size}",
                    dim=1024
                )
            )
    
    async def index_document(self, document: Document):
        """Index at multiple chunk sizes for flexible retrieval"""
        tasks = []
        
        for size in self.chunk_sizes:
            # Create chunks at this size
            chunks = self.create_chunks(document, size)
            
            # Add parent-child relationships
            for chunk in chunks:
                chunk.metadata["parent_doc_id"] = document.id_
                chunk.metadata["chunk_size"] = size
            
            # Index chunks asynchronously
            task = self.indexes[size].ainsert_nodes(chunks)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def retrieve_contextual(
        self, 
        query: str, 
        initial_chunk_size: int = 512,
        expand_context: bool = True
    ):
        """Retrieve with expanding context windows"""
        # Start with smaller chunks for precision
        initial_results = await self.indexes[initial_chunk_size].aretrieve(
            query, top_k=10
        )
        
        if not expand_context:
            return initial_results
        
        # Expand to larger chunks for more context
        expanded_results = []
        for result in initial_results[:5]:  # Top 5 only
            parent_id = result.metadata["parent_doc_id"]
            
            # Get larger chunk containing this segment
            larger_chunk = await self.indexes[2048].aretrieve(
                f"parent_doc_id:{parent_id}",
                top_k=1
            )
            expanded_results.extend(larger_chunk)
        
        return self._deduplicate_results(initial_results + expanded_results)
```

#### Graph-Enhanced RAG with Knowledge Graphs
```python
# data/indexes/knowledge_graph_rag.py
from neo4j import AsyncGraphDatabase
import networkx as nx

class GraphRAG:
    def __init__(self):
        self.driver = AsyncGraphDatabase.driver(
            "bolt://neo4j:7687",
            auth=("neo4j", "password")
        )
        self.vector_index = VectorStoreIndex(...)
    
    async def extract_and_index_entities(self, document: Document):
        """Extract entities and relationships using LLM"""
        # Use structured output for consistency
        extraction_prompt = """
        Extract entities and relationships from this text.
        Return JSON with schema:
        {
            "entities": [{"name": str, "type": str, "properties": dict}],
            "relationships": [{"source": str, "target": str, "type": str}]
        }
        """
        
        extraction = await llm_extract(extraction_prompt, document.text)
        
        # Store in Neo4j
        async with self.driver.session() as session:
            # Create entities
            for entity in extraction["entities"]:
                await session.run(
                    f"MERGE (n:{entity['type']} {{name: $name}}) "
                    "SET n += $properties",
                    name=entity["name"],
                    properties=entity["properties"]
                )
            
            # Create relationships
            for rel in extraction["relationships"]:
                await session.run(
                    "MATCH (a {name: $source}), (b {name: $target}) "
                    f"MERGE (a)-[:{rel['type']}]->(b)",
                    source=rel["source"],
                    target=rel["target"]
                )
    
    async def graph_retrieve(self, query: str, hops: int = 2):
        """Combine vector search with graph traversal"""
        # Step 1: Vector search for relevant entities
        entities = await self._extract_query_entities(query)
        
        # Step 2: Graph traversal from seed entities
        async with self.driver.session() as session:
            subgraph_query = """
            MATCH path = (n)-[*1..{hops}]-(m)
            WHERE n.name IN $entities
            RETURN path
            """.replace("{hops}", str(hops))
            
            result = await session.run(subgraph_query, entities=entities)
            paths = [record["path"] async for record in result]
        
        # Step 3: Convert subgraph to context
        context = self._paths_to_context(paths)
        
        # Step 4: Combine with vector search results
        vector_results = await self.vector_index.aretrieve(query)
        
        return {
            "graph_context": context,
            "vector_results": vector_results,
            "merged_context": self._merge_contexts(context, vector_results)
        }
```

---

## 4. Model Serving Layer: vLLM for Production Scale

vLLM has become the de facto standard for high-throughput LLM serving, offering 2-10x better performance than naive implementations through PagedAttention and continuous batching.

### ✅ DO: Configure vLLM for Optimal Performance

```python
# serving/vllm/server_config.py
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
import ray

class ProductionVLLMServer:
    def __init__(self):
        # Calculate optimal configuration based on hardware
        gpu_memory = get_gpu_memory_gb()
        model_size_gb = 140  # Llama-3.3-70B
        
        # Leave 10% GPU memory headroom
        gpu_memory_utilization = min(0.9, (gpu_memory - 2) / gpu_memory)
        
        # KV cache configuration
        # Rule of thumb: 5-10% of model size for KV cache
        max_num_seqs = 256 if gpu_memory >= 80 else 64
        
        self.engine_args = AsyncEngineArgs(
            model="meta-llama/Llama-3.3-70B-Instruct",
            
            # Tensor parallelism across GPUs
            tensor_parallel_size=4,  # For 4xA100-80GB
            
            # Pipeline parallelism for very large models
            pipeline_parallel_size=1,
            
            # Memory configuration
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=32768,
            
            # Enable advanced optimizations
            enable_prefix_caching=True,  # Cache common prefixes
            enable_chunked_prefill=True,  # Better latency
            max_num_batched_tokens=8192,
            
            # Use FlashInfer backend (15% faster than FlashAttention-2)
            attention_backend="flashinfer",
            
            # Speculative decoding for 2x speedup on greedy
            speculative_model="meta-llama/Llama-3.2-3B-Instruct",
            num_speculative_tokens=5,
            
            # Quantization for memory efficiency
            quantization="awq",  # Or "gptq" for GPTQ models
            
            # LoRA adapter support
            enable_lora=True,
            max_lora_rank=64,
            max_loras=4,  # Serve multiple adapters
            
            # Ray for distributed serving
            use_ray=True,
            ray_workers_use_nsight=False
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
    
    async def generate_stream(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        lora_request: Optional[LoRARequest] = None
    ):
        """Stream tokens with minimal latency"""
        request_id = str(uuid.uuid4())
        
        # Add request to engine
        await self.engine.add_request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            lora_request=lora_request
        )
        
        # Stream results
        async for output in self.engine.generate(request_id):
            if output.finished:
                # Include usage stats in final message
                yield {
                    "choices": [{
                        "delta": {"content": ""},
                        "finish_reason": output.outputs[0].finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": len(output.prompt_token_ids),
                        "completion_tokens": len(output.outputs[0].token_ids),
                        "total_tokens": len(output.prompt_token_ids) + 
                                      len(output.outputs[0].token_ids)
                    }
                }
            else:
                # Stream each token
                for completion in output.outputs:
                    if completion.text:
                        yield {
                            "choices": [{
                                "delta": {"content": completion.text},
                                "index": 0
                            }]
                        }
    
    async def benchmark_throughput(self):
        """Measure serving performance"""
        import time
        from vllm.utils import random_sample
        
        # Generate test prompts
        test_prompts = [
            random_sample(prompt_len=random.randint(100, 1000))
            for _ in range(100)
        ]
        
        start_time = time.time()
        tasks = []
        
        for prompt in test_prompts:
            sampling_params = SamplingParams(
                temperature=0.8,
                max_tokens=200,
                top_p=0.95
            )
            task = self.generate_stream(prompt, sampling_params)
            tasks.append(task)
        
        # Process all requests
        responses = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_tokens = sum(
            r["usage"]["total_tokens"] 
            for batch in responses 
            for r in batch
        )
        
        throughput = total_tokens / (end_time - start_time)
        print(f"Throughput: {throughput:.2f} tokens/second")
        
        return throughput
```

### Advanced vLLM Patterns

#### Multi-LoRA Serving
```python
# serving/vllm/multi_lora_server.py
class MultiLoRAServer:
    """Serve multiple fine-tuned adapters on single base model"""
    
    def __init__(self):
        self.lora_registry = {
            "finance": LoRARequest(
                lora_name="finance",
                lora_int_id=1,
                lora_local_path="/models/lora/finance-adapter"
            ),
            "medical": LoRARequest(
                lora_name="medical",
                lora_int_id=2,
                lora_local_path="/models/lora/medical-adapter"
            ),
            "legal": LoRARequest(
                lora_name="legal",
                lora_int_id=3,
                lora_local_path="/models/lora/legal-adapter"
            )
        }
    
    async def route_request(self, request: ChatRequest):
        """Route to appropriate LoRA based on domain"""
        # Use classifier to determine domain
        domain = await self.classify_domain(request.messages)
        
        # Select appropriate LoRA
        lora_request = self.lora_registry.get(domain)
        
        # Generate with domain-specific adapter
        return await self.engine.generate(
            prompt=request.to_prompt(),
            sampling_params=request.to_sampling_params(),
            lora_request=lora_request
        )
```

#### Structured Output with Guided Generation
```python
# serving/vllm/guided_generation.py
from pydantic import BaseModel
from typing import List, Optional
import jsonschema

class AnalysisOutput(BaseModel):
    summary: str
    sentiment: Literal["positive", "negative", "neutral"]
    key_points: List[str]
    confidence: float
    
async def generate_structured(
    prompt: str,
    schema: Type[BaseModel],
    engine: AsyncLLMEngine
):
    """Force LLM to generate valid JSON matching schema"""
    
    # Convert Pydantic to JSON Schema
    json_schema = schema.model_json_schema()
    
    # vLLM's guided generation via Outlines
    sampling_params = SamplingParams(
        temperature=0.1,  # Lower temp for structured output
        max_tokens=1000,
        guided_json=json_schema,
        guided_decoding_backend="lm-format-enforcer"  # Or "outlines"
    )
    
    result = await engine.generate(prompt, sampling_params)
    
    # Parse and validate
    output_text = result.outputs[0].text
    return schema.model_validate_json(output_text)
```

---

## 5. Kubernetes Deployment with KServe

KServe (formerly KFServing) provides production-grade model serving with autoscaling, canary deployments, and multi-model serving.

### ✅ DO: Deploy Models with KServe InferenceService

```yaml
# k8s/kserve/llama-inference-service.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama-70b-vllm
  annotations:
    serving.kserve.io/deploymentMode: "RawDeployment"  # More control
spec:
  predictor:
    # Model serving configuration
    model:
      modelFormat:
        name: vllm
      runtime: kserve-vllm-runtime
      storageUri: "s3://models/llama-3.3-70b"  # Or pvc://
      
      # Resource requirements
      resources:
        requests:
          memory: "160Gi"
          cpu: "16"
          nvidia.com/gpu: "4"  # 4x A100-40GB
        limits:
          memory: "180Gi"
          nvidia.com/gpu: "4"
      
      # Container configuration
      env:
      - name: VLLM_ATTENTION_BACKEND
        value: "FLASHINFER"
      - name: TENSOR_PARALLEL_SIZE
        value: "4"
      - name: GPU_MEMORY_UTILIZATION
        value: "0.95"
      - name: MAX_MODEL_LEN
        value: "8192"
      - name: ENABLE_PREFIX_CACHING
        value: "true"
      
      # Node selection for GPU nodes
      nodeSelector:
        node.kubernetes.io/gpu-type: "a100"
      
      # Tolerations for GPU nodes
      tolerations:
      - key: nvidia.com/gpu
        operator: "Exists"
        effect: "NoSchedule"
    
    # Autoscaling configuration
    scaleTarget: 50  # Target GPU utilization %
    scaleMetric: gpu  
    minReplicas: 1
    maxReplicas: 8
    
    # Canary deployment configuration
    canaryTrafficPercent: 20
    
  # Optional transformer for pre/post processing
  transformer:
    containers:
    - name: transformer
      image: myregistry/llm-transformer:latest
      env:
      - name: ENABLE_GUARDRAILS
        value: "true"
      resources:
        requests:
          memory: "4Gi"
          cpu: "2"

---
# k8s/kserve/runtime.yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ClusterServingRuntime
metadata:
  name: kserve-vllm-runtime
spec:
  supportedModelFormats:
  - name: vllm
    version: "1"
    
  containers:
  - name: kserve-container
    image: vllm/vllm-openai:v0.6.4
    command: ["python", "-m", "vllm.entrypoints.openai.api_server"]
    args:
    - --port=8080
    - --model=/mnt/models
    - --served-model-name={{.Name}}
    - --trust-remote-code
    
    # Probes for production stability
    livenessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 90
      periodSeconds: 30
      
    readinessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 30
      periodSeconds: 10
```

### Advanced KServe Patterns

#### Multi-Model Serving on Single GPU
```yaml
# k8s/kserve/multi-model-service.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: multi-model-server
spec:
  predictor:
    model:
      modelFormat:
        name: custom
      runtime: multi-model-runtime
      
      # Mount multiple models
      storageUri: "pvc://model-storage"
      
      env:
      - name: MODEL_PATHS
        value: |
          llama-8b:/models/llama-3.2-8b
          mistral-7b:/models/mistral-7b-v0.3
          gemma-2b:/models/gemma-2-2b
          
      - name: LOAD_MODELS_ON_STARTUP
        value: "gemma-2b"  # Only load small model initially
        
      - name: MODEL_CACHE_SIZE
        value: "2"  # Keep 2 models in GPU memory
        
      resources:
        requests:
          nvidia.com/gpu: "1"
          memory: "48Gi"
          
    # Custom autoscaling based on queue depth
    scaleMetric: custom
    scaleTarget: 10  # Target queue depth
```

#### Model A/B Testing with Traffic Splitting
```python
# k8s/kserve/model_ab_test.py
from kubernetes import client, config
import random

class ModelABTester:
    def __init__(self):
        config.load_incluster_config()
        self.custom_api = client.CustomObjectsApi()
        
    async def create_ab_test(
        self,
        model_a: str,
        model_b: str,
        traffic_split: float = 0.5
    ):
        """Create A/B test between two models"""
        
        # Create InferenceService with traffic split
        inference_service = {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {
                "name": "ab-test-models",
                "annotations": {
                    "serving.kserve.io/enable-prometheus-scraping": "true"
                }
            },
            "spec": {
                "predictor": {
                    "model": {
                        "modelFormat": {"name": "vllm"},
                        "runtime": "kserve-vllm-runtime",
                        "storageUri": f"s3://models/{model_a}"
                    }
                },
                "canaryRevision": {
                    "model": {
                        "modelFormat": {"name": "vllm"},
                        "runtime": "kserve-vllm-runtime", 
                        "storageUri": f"s3://models/{model_b}"
                    }
                },
                "canaryTrafficPercent": int(traffic_split * 100)
            }
        }
        
        # Deploy
        self.custom_api.create_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace="default",
            plural="inferenceservices",
            body=inference_service
        )
        
    async def update_traffic_split(self, name: str, new_split: float):
        """Gradually shift traffic between models"""
        body = {
            "spec": {
                "canaryTrafficPercent": int(new_split * 100)
            }
        }
        
        self.custom_api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace="default",
            plural="inferenceservices",
            name=name,
            body=body
        )
```

---

## 6. Observability & Evaluation: Production Monitoring

Langfuse has emerged as the standard for LLM observability, providing tracing, analytics, and human feedback collection in a single platform.

### ✅ DO: Implement Comprehensive Observability

```python
# eval/observability/langfuse_integration.py
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc import OTLPSpanExporter
import structlog

# Initialize Langfuse client
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host="https://cloud.langfuse.com",  # Or self-hosted
    release=os.getenv("GIT_COMMIT_SHA"),
    
    # Performance optimizations
    flush_at=100,  # Batch size
    flush_interval=10,  # Seconds
    max_retries=3,
    timeout=30
)

# Configure structured logging
logger = structlog.get_logger()

class ObservableAgent:
    @observe(as_type="agent")
    async def process_request(
        self,
        request_id: str,
        user_query: str,
        user_id: str
    ):
        """Fully traced agent execution"""
        
        # Set trace metadata
        langfuse_context.update_current_trace(
            name="research_agent",
            user_id=user_id,
            metadata={
                "request_id": request_id,
                "agent_version": "1.2.0",
                "deployment": "production"
            },
            tags=["research", "gpt-4"]
        )
        
        # Trace each step
        with langfuse_context.observe(
            name="retrieve_context",
            as_type="retrieval"
        ) as span:
            start_time = time.time()
            
            # Your retrieval logic
            documents = await self.retriever.retrieve(user_query)
            
            # Log retrieval metrics
            span.update(
                output={"num_documents": len(documents)},
                metadata={
                    "retrieval_latency": time.time() - start_time,
                    "index_name": "production_v2"
                }
            )
            
            # Sample documents for inspection
            if random.random() < 0.1:  # 10% sampling
                span.update(
                    metadata={"sampled_docs": [d.text[:200] for d in documents[:3]]}
                )
        
        # LLM generation with automatic tracing
        with langfuse_context.observe(
            name="generate_response",
            as_type="generation"
        ) as span:
            
            messages = self.build_messages(user_query, documents)
            
            # Track token usage
            response = await self.llm.agenerate(
                messages=messages,
                model="gpt-4-turbo-preview",
                temperature=0.7
            )
            
            span.update(
                input=messages,
                output=response.content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_cost": self.calculate_cost(response.usage)
                },
                model="gpt-4-turbo-preview",
                model_parameters={
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
            )
            
        # Score the interaction
        await self.score_interaction(request_id, response)
        
        return response
    
    async def score_interaction(self, request_id: str, response):
        """Add automated quality scores"""
        
        # Sentiment analysis
        sentiment_score = await self.analyze_sentiment(response.content)
        
        # Hallucination detection
        hallucination_score = await self.check_hallucination(
            response.content,
            response.source_documents
        )
        
        # Response quality (using another LLM as judge)
        quality_score = await self.judge_quality(response)
        
        # Send scores to Langfuse
        langfuse.score(
            trace_id=langfuse_context.get_current_trace_id(),
            name="sentiment",
            value=sentiment_score,
            data_type="NUMERIC"
        )
        
        langfuse.score(
            trace_id=langfuse_context.get_current_trace_id(),
            name="hallucination",
            value=hallucination_score,
            data_type="NUMERIC",
            comment="Lower is better"
        )
        
        langfuse.score(
            trace_id=langfuse_context.get_current_trace_id(),
            name="quality",
            value=quality_score,
            data_type="NUMERIC"
        )
```

### Production Evaluation Pipelines

#### Automated Evaluation with RAGAS
```python
# eval/pipelines/ragas_evaluation.py
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_similarity,
    answer_correctness
)
from datasets import Dataset
import asyncio

class ProductionRAGEvaluator:
    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
            answer_correctness
        ]
        
        # Configure LLM for evaluation
        self.eval_llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1  # Low temperature for consistency
        )
    
    async def evaluate_batch(
        self,
        queries: List[str],
        retrieved_contexts: List[List[str]],
        generated_answers: List[str],
        ground_truths: List[str] = None
    ):
        """Evaluate a batch of RAG outputs"""
        
        # Prepare dataset
        eval_data = {
            "question": queries,
            "contexts": retrieved_contexts,
            "answer": generated_answers
        }
        
        if ground_truths:
            eval_data["ground_truth"] = ground_truths
            
        dataset = Dataset.from_dict(eval_data)
        
        # Run evaluation
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.eval_llm,
            embeddings=FastEmbedEmbedding(model_name="BAAI/bge-m3"),
            raise_exceptions=False  # Continue on individual failures
        )
        
        # Send to monitoring
        await self.send_to_monitoring(results)
        
        return results
    
    async def continuous_evaluation(self):
        """Run evaluation on production traffic samples"""
        
        while True:
            # Sample recent production queries
            samples = await self.sample_recent_queries(
                sample_rate=0.01,  # 1% of traffic
                min_samples=100,
                max_samples=1000
            )
            
            if len(samples) >= 100:
                # Prepare data
                queries = [s["query"] for s in samples]
                contexts = [s["retrieved_contexts"] for s in samples]
                answers = [s["generated_answer"] for s in samples]
                
                # Evaluate
                results = await self.evaluate_batch(
                    queries, contexts, answers
                )
                
                # Alert on degradation
                if results["faithfulness"] < 0.8:
                    await self.alert_on_metric_degradation(
                        metric="faithfulness",
                        value=results["faithfulness"],
                        threshold=0.8
                    )
                
                logger.info(
                    "evaluation_complete",
                    sample_size=len(samples),
                    metrics=results
                )
            
            # Run every hour
            await asyncio.sleep(3600)
```

#### LLM-as-Judge for Complex Evaluation
```python
# eval/pipelines/llm_judge.py
class LLMJudge:
    """Use GPT-4 to evaluate complex criteria"""
    
    def __init__(self):
        self.judge_model = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1
        )
        
        self.criteria = {
            "helpfulness": """
                Score from 1-5 how helpful this response is.
                Consider: completeness, clarity, actionability.
                """,
            "harmlessness": """
                Score from 1-5 how safe this response is.
                Consider: bias, toxicity, misinformation.
                """,
            "honesty": """
                Score from 1-5 how honest this response is.
                Consider: uncertainty expression, hallucination.
                """
        }
    
    async def judge_response(
        self,
        query: str,
        response: str,
        context: List[str] = None
    ) -> Dict[str, float]:
        """Judge response on multiple criteria"""
        
        scores = {}
        
        for criterion, description in self.criteria.items():
            prompt = f"""
            Evaluate this AI response on {criterion}.
            
            {description}
            
            Query: {query}
            Response: {response}
            {"Context: " + str(context) if context else ""}
            
            Provide your evaluation as JSON:
            {{
                "score": <1-5>,
                "reasoning": "<brief explanation>",
                "specific_issues": ["<issue1>", "<issue2>", ...]
            }}
            """
            
            judgment = await self.judge_model.ainvoke(prompt)
            result = json.loads(judgment.content)
            
            scores[criterion] = {
                "score": result["score"] / 5.0,  # Normalize to 0-1
                "reasoning": result["reasoning"],
                "issues": result.get("specific_issues", [])
            }
            
            # Log concerning responses
            if result["score"] < 3:
                logger.warning(
                    "low_quality_response",
                    criterion=criterion,
                    score=result["score"],
                    reasoning=result["reasoning"],
                    query_preview=query[:100]
                )
        
        return scores
```

---

## 7. Cost Optimization & Performance Tuning

Production LLM systems can quickly become expensive. Here are battle-tested strategies for optimization.

### ✅ DO: Implement Intelligent Caching

```python
# optimization/caching/semantic_cache.py
from typing import Optional, List, Tuple
import hashlib
import numpy as np
from redis import asyncio as redis
import faiss

class SemanticCache:
    """Cache LLM responses with semantic similarity matching"""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.redis_client = redis.Redis(
            host="localhost",
            decode_responses=True,
            max_connections=100
        )
        
        self.embed_model = FastEmbedEmbedding(
            model_name="BAAI/bge-small-en-v1.5",  # Fast & good
            batch_size=32
        )
        
        self.similarity_threshold = similarity_threshold
        
        # FAISS index for similarity search
        self.dimension = 384  # bge-small dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product
        
        # Metadata storage
        self.cache_metadata = {}
    
    async def get_or_generate(
        self,
        prompt: str,
        generate_fn: Callable,
        ttl: int = 3600,
        use_semantic: bool = True
    ) -> Tuple[str, bool]:
        """Get from cache or generate new response"""
        
        # Try exact match first (fastest)
        exact_key = self._get_exact_key(prompt)
        cached = await self.redis_client.get(exact_key)
        
        if cached:
            await self._increment_stats("exact_hit")
            return cached, True
        
        # Try semantic match if enabled
        if use_semantic and self.index.ntotal > 0:
            similar_response = await self._semantic_search(prompt)
            if similar_response:
                await self._increment_stats("semantic_hit")
                
                # Cache the exact match for next time
                await self.redis_client.setex(
                    exact_key, ttl, similar_response
                )
                return similar_response, True
        
        # Generate new response
        await self._increment_stats("cache_miss")
        response = await generate_fn(prompt)
        
        # Store in cache
        await self._store_response(prompt, response, ttl)
        
        return response, False
    
    async def _semantic_search(self, prompt: str) -> Optional[str]:
        """Find semantically similar cached response"""
        
        # Embed the prompt
        embedding = await self.embed_model.aget_text_embedding(prompt)
        embedding_np = np.array([embedding]).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embedding_np)
        
        # Search
        distances, indices = self.index.search(embedding_np, k=5)
        
        # Check if any result meets threshold
        for distance, idx in zip(distances[0], indices[0]):
            if distance >= self.similarity_threshold:
                cache_id = self.cache_metadata.get(idx)
                if cache_id:
                    response = await self.redis_client.get(
                        f"response:{cache_id}"
                    )
                    if response:
                        logger.info(
                            "semantic_cache_hit",
                            similarity=float(distance),
                            cache_id=cache_id
                        )
                        return response
        
        return None
    
    async def _store_response(
        self,
        prompt: str,
        response: str,
        ttl: int
    ):
        """Store response with embeddings"""
        
        # Store exact match
        exact_key = self._get_exact_key(prompt)
        await self.redis_client.setex(exact_key, ttl, response)
        
        # Generate embedding for semantic search
        embedding = await self.embed_model.aget_text_embedding(prompt)
        embedding_np = np.array([embedding]).astype('float32')
        faiss.normalize_L2(embedding_np)
        
        # Add to FAISS index
        idx = self.index.ntotal
        self.index.add(embedding_np)
        
        # Store metadata
        cache_id = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        self.cache_metadata[idx] = cache_id
        
        # Store response separately for semantic matches
        await self.redis_client.setex(
            f"response:{cache_id}", ttl, response
        )
    
    def _get_exact_key(self, prompt: str) -> str:
        """Generate cache key for exact matching"""
        # Include model config in hash for safety
        prompt_hash = hashlib.sha256(
            f"{prompt}:gpt-4-turbo:0.7".encode()
        ).hexdigest()
        return f"llm_cache:{prompt_hash}"
    
    async def get_stats(self) -> dict:
        """Get cache performance statistics"""
        stats = {}
        for key in ["exact_hit", "semantic_hit", "cache_miss"]:
            stats[key] = int(
                await self.redis_client.get(f"cache_stats:{key}") or 0
            )
        
        total = sum(stats.values())
        if total > 0:
            stats["hit_rate"] = (
                (stats["exact_hit"] + stats["semantic_hit"]) / total
            )
        
        return stats
```

### Model Cascade for Cost Efficiency

```python
# optimization/model_cascade.py
class ModelCascade:
    """Route queries to appropriate models based on complexity"""
    
    def __init__(self):
        self.models = [
            {
                "name": "gemma-2b",
                "endpoint": "http://vllm-gemma:8000",
                "cost_per_1k_tokens": 0.0001,
                "capabilities": ["simple_qa", "classification"]
            },
            {
                "name": "llama-3.2-8b",
                "endpoint": "http://vllm-llama-8b:8000",
                "cost_per_1k_tokens": 0.001,
                "capabilities": ["reasoning", "summarization", "simple_qa"]
            },
            {
                "name": "llama-3.3-70b",
                "endpoint": "http://vllm-llama-70b:8000",
                "cost_per_1k_tokens": 0.01,
                "capabilities": ["complex_reasoning", "coding", "analysis"]
            }
        ]
        
        # Complexity classifier (small BERT model)
        self.classifier = ComplexityClassifier()
    
    async def route_query(
        self,
        query: str,
        required_capabilities: List[str] = None,
        max_cost_per_query: float = None
    ) -> dict:
        """Route to cheapest capable model"""
        
        # Classify query complexity
        complexity_score = await self.classifier.predict(query)
        
        # Filter models by capabilities
        capable_models = self.models
        if required_capabilities:
            capable_models = [
                m for m in self.models
                if all(cap in m["capabilities"] for cap in required_capabilities)
            ]
        
        # Try models from cheapest to most expensive
        for model in capable_models:
            # Skip if too expensive
            if max_cost_per_query:
                estimated_cost = self._estimate_cost(query, model)
                if estimated_cost > max_cost_per_query:
                    continue
            
            # Skip if model too simple for query
            if model["name"] == "gemma-2b" and complexity_score > 0.7:
                continue
            
            try:
                response = await self._call_model(
                    model["endpoint"],
                    query,
                    timeout=30 if model["name"] != "llama-70b" else 60
                )
                
                # Validate response quality
                if await self._validate_response(response, query):
                    logger.info(
                        "model_cascade_success",
                        model=model["name"],
                        complexity=complexity_score,
                        estimated_cost=estimated_cost
                    )
                    return response
                    
            except asyncio.TimeoutError:
                logger.warning(
                    "model_timeout",
                    model=model["name"],
                    query_preview=query[:100]
                )
                continue
        
        # Fallback to most capable model
        return await self._call_model(
            self.models[-1]["endpoint"],
            query
        )
```

---

## 8. Production Deployment Best Practices

### Complete Docker Setup for LLM Stack

```dockerfile
# Dockerfile.vllm - Optimized for NVIDIA GPUs
FROM nvcr.io/nvidia/pytorch:24.12-py3 as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install vLLM from source for latest optimizations
WORKDIR /workspace
RUN git clone https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir \
        flashinfer \
        vllm-flash-attn \
        xformers

# Production stage
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Copy vLLM installation
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /workspace/vllm /workspace/vllm

# Install runtime dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    prometheus-client \
    numpy<2.0 \
    pydantic>=2.0

# Model cache directory
ENV HF_HOME=/models/cache
ENV VLLM_USAGE_SOURCE=production

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

WORKDIR /workspace
EXPOSE 8000

# Use exec form to handle signals properly
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
```

### Helm Chart for Complete LLM Platform

```yaml
# helm/llm-platform/values.yaml
global:
  registry: myregistry.io
  imagePullSecrets:
    - name: regcred

# vLLM serving configuration
vllm:
  enabled: true
  replicas: 2
  
  model:
    name: "meta-llama/Llama-3.3-70B-Instruct"
    revision: "main"
    
  resources:
    requests:
      cpu: "16"
      memory: "160Gi"
      nvidia.com/gpu: "4"
    limits:
      nvidia.com/gpu: "4"
      
  config:
    tensorParallelSize: 4
    gpuMemoryUtilization: 0.95
    maxModelLen: 8192
    maxNumSeqs: 256
    
  storage:
    # Model cache PVC
    modelCache:
      enabled: true
      size: "500Gi"
      storageClass: "fast-nvme"
      
  monitoring:
    enabled: true
    serviceMonitor:
      enabled: true
      interval: 30s
      
  # Node affinity for GPU nodes
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: node.kubernetes.io/gpu-type
            operator: In
            values: ["a100", "h100"]

# LangGraph agents
langraph:
  enabled: true
  replicas: 3
  
  postgres:
    enabled: true
    auth:
      database: "langraph"
      username: "langraph"
      existingSecret: "langraph-postgres"
      
  redis:
    enabled: true
    sentinel:
      enabled: true
      
# Vector database
milvus:
  enabled: true
  mode: "cluster"
  
  minio:
    enabled: true
    persistence:
      size: "1Ti"
      
  pulsar:
    enabled: true
    
  dataNode:
    replicas: 3
    resources:
      requests:
        cpu: "4"
        memory: "16Gi"
        
# Observability stack
monitoring:
  prometheus:
    enabled: true
    retention: "30d"
    
  grafana:
    enabled: true
    dashboards:
      - llm-metrics
      - gpu-utilization
      - model-performance
      
  langfuse:
    enabled: true
    postgresql:
      enabled: true
      
# Ingress configuration
ingress:
  enabled: true
  className: "nginx"
  
  hosts:
    - host: api.llm-platform.example.com
      paths:
        - path: /v1/completions
          service: vllm
          port: 8000
        - path: /agents
          service: langraph
          port: 8080
          
  tls:
    - secretName: llm-platform-tls
      hosts:
        - api.llm-platform.example.com
```

### GitOps with ArgoCD

```yaml
# argocd/llm-platform-app.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: llm-platform
  namespace: argocd
spec:
  project: default
  
  source:
    repoURL: https://github.com/myorg/llm-platform
    targetRevision: HEAD
    path: helm/llm-platform
    
    helm:
      valueFiles:
        - values.yaml
        - values-prod.yaml
        
      # Dynamic values from secrets
      parameters:
      - name: "langfuse.secretKey"
        value: "$ARGOCD_APP_NAMESPACE:langfuse-secret:key"
        
  destination:
    server: https://kubernetes.default.svc
    namespace: llm-platform
    
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
        
  # Health checks
  health:
    ignoredConditions:
    - type: ProgressDeadlineExceeded
      message: "ReplicaSet .* has timed out progressing"
```

---

## 9. Security & Compliance

### Secure API Gateway with Rate Limiting

```python
# api/middleware/security.py
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from cryptography.fernet import Fernet
import redis.asyncio as redis
from typing import Optional

security = HTTPBearer()

class SecurityMiddleware:
    def __init__(self):
        self.redis_client = redis.Redis(
            host="localhost",
            decode_responses=True
        )
        
        # Encryption for sensitive data
        self.cipher = Fernet(os.getenv("ENCRYPTION_KEY"))
        
        # JWT configuration
        self.jwt_secret = os.getenv("JWT_SECRET")
        self.jwt_algorithm = "HS256"
    
    async def verify_token(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> dict:
        """Verify JWT tokens with caching"""
        token = credentials.credentials
        
        # Check if token is revoked
        is_revoked = await self.redis_client.get(f"revoked_token:{token}")
        if is_revoked:
            raise HTTPException(
                status_code=401,
                detail="Token has been revoked"
            )
        
        try:
            # Decode and verify
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )
            
            # Check expiration
            if payload.get("exp", 0) < time.time():
                raise HTTPException(
                    status_code=401,
                    detail="Token has expired"
                )
            
            return payload
            
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )
    
    async def rate_limit(
        self,
        request: Request,
        user: dict = Depends(verify_token)
    ):
        """User-based rate limiting with sliding window"""
        
        user_id = user["sub"]
        endpoint = f"{request.method}:{request.url.path}"
        
        # Different limits per endpoint
        limits = {
            "POST:/v1/completions": (100, 3600),      # 100/hour
            "POST:/v1/embeddings": (1000, 3600),      # 1000/hour
            "POST:/agents/research": (20, 3600),      # 20/hour
            "POST:/v1/images/generations": (50, 86400) # 50/day
        }
        
        limit, window = limits.get(endpoint, (1000, 3600))
        
        # Sliding window counter
        key = f"rate_limit:{user_id}:{endpoint}"
        current = int(time.time())
        
        pipe = self.redis_client.pipeline()
        pipe.zremrangebyscore(key, 0, current - window)
        pipe.zadd(key, {str(current): current})
        pipe.zcard(key)
        pipe.expire(key, window)
        results = await pipe.execute()
        
        request_count = results[2]
        
        if request_count > limit:
            retry_after = window - (current - int(
                await self.redis_client.zrange(key, 0, 0, withscores=True)[0][1]
            ))
            
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {limit} requests per {window}s",
                headers={"Retry-After": str(retry_after)}
            )
    
    async def log_request(
        self,
        request: Request,
        user: dict,
        response_time: float,
        tokens_used: Optional[int] = None
    ):
        """Structured audit logging"""
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user["sub"],
            "method": request.method,
            "path": request.url.path,
            "ip": request.client.host,
            "user_agent": request.headers.get("user-agent"),
            "response_time_ms": response_time * 1000,
            "tokens_used": tokens_used,
            "status_code": 200  # Update based on actual response
        }
        
        # Send to audit log
        await self.redis_client.xadd(
            "audit_log",
            {"data": json.dumps(log_entry)}
        )
        
        # Also log high-value requests
        if tokens_used and tokens_used > 10000:
            logger.warning(
                "high_token_usage",
                **log_entry
            )
```

### PII Detection and Masking

```python
# security/pii_detector.py
import presidio_analyzer
import presidio_anonymizer
from typing import List, Dict
import re

class PIIProtection:
    def __init__(self):
        # Initialize Presidio
        self.analyzer = presidio_analyzer.AnalyzerEngine()
        self.anonymizer = presidio_anonymizer.AnonymizerEngine()
        
        # Custom recognizers for domain-specific PII
        self.add_custom_recognizers()
        
        # Sensitive patterns
        self.sensitive_patterns = {
            "api_key": re.compile(r'(sk-[a-zA-Z0-9]{48}|api[_-]?key["\s:=]+["\']?[a-zA-Z0-9]{32,})'),
            "jwt": re.compile(r'eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*'),
            "credit_card": re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')
        }
    
    async def scan_and_mask(
        self,
        text: str,
        entities_to_mask: List[str] = None
    ) -> Dict:
        """Scan for PII and return masked version"""
        
        # Default entities
        if entities_to_mask is None:
            entities_to_mask = [
                "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                "CREDIT_CARD", "SSN", "LOCATION", "IP_ADDRESS"
            ]
        
        # Analyze
        results = self.analyzer.analyze(
            text=text,
            entities=entities_to_mask,
            language="en"
        )
        
        # Check custom patterns
        for pattern_name, pattern in self.sensitive_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                results.append(
                    presidio_analyzer.RecognizerResult(
                        entity_type=pattern_name.upper(),
                        start=match.start(),
                        end=match.end(),
                        score=1.0
                    )
                )
        
        # Anonymize if PII found
        if results:
            masked_text = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results
            ).text
            
            # Log detection for compliance
            logger.warning(
                "pii_detected",
                entities_found=[r.entity_type for r in results],
                count=len(results)
            )
            
            return {
                "contains_pii": True,
                "original_text": text,
                "masked_text": masked_text,
                "entities_found": [
                    {
                        "type": r.entity_type,
                        "start": r.start,
                        "end": r.end,
                        "score": r.score
                    }
                    for r in results
                ]
            }
        
        return {
            "contains_pii": False,
            "original_text": text,
            "masked_text": text,
            "entities_found": []
        }
    
    def add_custom_recognizers(self):
        """Add domain-specific PII recognizers"""
        
        # Employee ID pattern
        employee_pattern = presidio_analyzer.Pattern(
            name="employee_id_pattern",
            regex=r"EMP\d{6}",
            score=0.9
        )
        
        employee_recognizer = presidio_analyzer.PatternRecognizer(
            supported_entity="EMPLOYEE_ID",
            patterns=[employee_pattern]
        )
        
        self.analyzer.registry.add_recognizer(employee_recognizer)
```

---

## 10. Advanced Patterns & Future-Proofing

### Multi-Agent Collaboration with LangGraph

```python
# agents/multi_agent_system.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
import operator

class TeamState(TypedDict):
    """Shared state for agent collaboration"""
    task: str
    research_complete: bool
    code_complete: bool
    review_complete: bool
    research_findings: List[str]
    generated_code: str
    review_feedback: str
    final_output: str

class MultiAgentTeam:
    def __init__(self):
        # Specialized agents
        self.researcher = ResearchAgent()
        self.coder = CodingAgent()
        self.reviewer = ReviewAgent()
        self.coordinator = CoordinatorAgent()
        
        # Build collaboration graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        workflow = StateGraph(TeamState)
        
        # Add agent nodes
        workflow.add_node("coordinator", self.coordinator_node)
        workflow.add_node("researcher", self.researcher_node)
        workflow.add_node("coder", self.coder_node)
        workflow.add_node("reviewer", self.reviewer_node)
        workflow.add_node("synthesizer", self.synthesizer_node)
        
        # Define the flow
        workflow.set_entry_point("coordinator")
        
        # Coordinator decides which agents to activate
        workflow.add_conditional_edges(
            "coordinator",
            self.route_from_coordinator,
            {
                "research": "researcher",
                "code": "coder",
                "review": "reviewer",
                "done": "synthesizer"
            }
        )
        
        # Agents report back to coordinator
        workflow.add_edge("researcher", "coordinator")
        workflow.add_edge("coder", "coordinator")
        workflow.add_edge("reviewer", "coordinator")
        
        # Final synthesis
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()
    
    async def coordinator_node(self, state: TeamState) -> TeamState:
        """Coordinator decides task allocation"""
        
        # Analyze task and current progress
        task_analysis = await self.coordinator.analyze_task(
            task=state["task"],
            research_done=state.get("research_complete", False),
            code_done=state.get("code_complete", False),
            review_done=state.get("review_complete", False)
        )
        
        # Update routing decision
        state["next_agent"] = task_analysis["next_agent"]
        state["agent_instructions"] = task_analysis["instructions"]
        
        return state
    
    async def researcher_node(self, state: TeamState) -> TeamState:
        """Research agent gathers information"""
        
        findings = await self.researcher.research(
            query=state["task"],
            instructions=state.get("agent_instructions", "")
        )
        
        state["research_findings"] = findings
        state["research_complete"] = True
        
        return state
    
    def route_from_coordinator(self, state: TeamState) -> str:
        """Dynamic routing based on task needs"""
        
        next_agent = state.get("next_agent", "done")
        
        # Complex routing logic
        if next_agent == "auto":
            # Determine based on task type
            if "research" in state["task"].lower() and not state.get("research_complete"):
                return "research"
            elif "code" in state["task"].lower() and not state.get("code_complete"):
                return "code"
            elif state.get("generated_code") and not state.get("review_complete"):
                return "review"
            else:
                return "done"
        
        return next_agent
```

### Edge Deployment with Ollama

```python
# serving/edge/ollama_edge_server.py
import ollama
from fastapi import FastAPI, HTTPException
from typing import AsyncGenerator
import asyncio

class OllamaEdgeServer:
    """Lightweight edge deployment for small models"""
    
    def __init__(self):
        self.client = ollama.AsyncClient(
            host="http://localhost:11434"
        )
        
        # Pre-load models on startup
        self.loaded_models = set()
        
        # Model registry with capabilities
        self.model_registry = {
            "llama3.2:3b": {
                "capabilities": ["chat", "simple_qa"],
                "context_length": 128000,
                "quantization": "Q4_K_M"
            },
            "phi-3.5:3.8b": {
                "capabilities": ["chat", "code", "reasoning"],
                "context_length": 128000,
                "quantization": "Q4_K_M"
            },
            "gemma2:2b": {
                "capabilities": ["chat", "summarization"],
                "context_length": 8192,
                "quantization": "Q4_0"
            }
        }
    
    async def ensure_model_loaded(self, model: str):
        """Pull and load model if not already loaded"""
        
        if model not in self.loaded_models:
            try:
                # Pull model if not exists
                await self.client.pull(model)
                
                # Warm up the model
                await self.client.generate(
                    model=model,
                    prompt="Hello",
                    options={"num_predict": 1}
                )
                
                self.loaded_models.add(model)
                
            except Exception as e:
                logger.error(f"Failed to load model {model}: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Model {model} unavailable"
                )
    
    async def generate_stream(
        self,
        prompt: str,
        model: str = "llama3.2:3b",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> AsyncGenerator[str, None]:
        """Stream generation from edge model"""
        
        # Ensure model is loaded
        await self.ensure_model_loaded(model)
        
        # Stream response
        async for chunk in await self.client.generate(
            model=model,
            prompt=prompt,
            stream=True,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.model_registry[model]["context_length"],
                "repeat_penalty": 1.1,
                "top_k": 40,
                "top_p": 0.9
            }
        ):
            if chunk.get("response"):
                yield chunk["response"]
    
    async def create_embedding(
        self,
        text: str,
        model: str = "nomic-embed-text"
    ) -> List[float]:
        """Generate embeddings locally"""
        
        await self.ensure_model_loaded(model)
        
        response = await self.client.embeddings(
            model=model,
            prompt=text
        )
        
        return response["embedding"]
```

### Function Calling & Tool Use with SGLang

```python
# serving/sglang/function_calling.py
import sglang as sgl
from typing import List, Dict, Any

class SGLangFunctionCaller:
    """Structured generation and function calling with SGLang"""
    
    def __init__(self):
        # Initialize runtime
        self.runtime = sgl.Runtime(
            model_path="meta-llama/Llama-3.3-70B-Instruct",
            tp_size=4,
            mem_fraction_static=0.85
        )
        
        # Tool registry
        self.tools = {
            "search_web": self.search_web,
            "execute_code": self.execute_code,
            "query_database": self.query_database
        }
    
    @sgl.function
    def agent_with_tools(s, task: str, tools: List[Dict]):
        """SGLang program for tool-using agent"""
        
        s += sgl.system("You are a helpful assistant with access to tools.")
        s += sgl.user(task)
        
        # Generate initial response
        s += sgl.assistant(sgl.gen("response", max_tokens=200))
        
        # Check if tool use is needed
        s += sgl.user("Do you need to use any tools? If yes, specify which tool and arguments in JSON format.")
        
        # Force JSON output for tool call
        s += sgl.assistant(
            sgl.gen(
                "tool_call",
                max_tokens=200,
                regex=r'\{"tool": "[^"]+", "arguments": \{[^}]*\}\}'
            )
        )
        
        # Parse tool call
        with s.if_(s["tool_call"] != ""):
            # This will be executed if tool is called
            s += sgl.user(f"Tool output: {{tool_output}}")
            s += sgl.assistant(sgl.gen("final_response", max_tokens=500))
    
    async def execute_with_tools(self, task: str) -> str:
        """Execute task with automatic tool calling"""
        
        # Initial generation
        state = self.agent_with_tools.run(
            task=task,
            tools=list(self.tools.keys())
        )
        
        # Check if tool was called
        if state.get("tool_call"):
            tool_data = json.loads(state["tool_call"])
            tool_name = tool_data["tool"]
            tool_args = tool_data["arguments"]
            
            # Execute tool
            if tool_name in self.tools:
                tool_output = await self.tools[tool_name](**tool_args)
                
                # Continue generation with tool output
                state = self.agent_with_tools.run(
                    task=task,
                    tools=list(self.tools.keys()),
                    tool_output=str(tool_output),
                    # Resume from previous state
                    **state
                )
                
                return state["final_response"]
        
        return state["response"]
    
    @sgl.function
    def structured_extraction(s, text: str, schema: Dict):
        """Extract structured data matching a schema"""
        
        s += sgl.system(
            "Extract information from the text according to the schema. "
            "Output valid JSON matching the schema exactly."
        )
        
        s += sgl.user(f"Text: {text}\n\nSchema: {json.dumps(schema)}")
        
        # SGLang's constrained generation ensures valid JSON
        s += sgl.assistant(
            sgl.gen(
                "extraction",
                max_tokens=1000,
                regex=sgl.json_schema_to_regex(schema)
            )
        )
    
    async def extract_structured(
        self,
        text: str,
        output_schema: Type[BaseModel]
    ) -> BaseModel:
        """Extract and validate structured data"""
        
        schema = output_schema.model_json_schema()
        
        state = self.structured_extraction.run(
            text=text,
            schema=schema
        )
        
        # Parse and validate
        extracted = json.loads(state["extraction"])
        return output_schema.model_validate(extracted)
```

---

## 11. Testing & Validation

### End-to-End LLM Testing

```python
# tests/e2e/test_llm_pipeline.py
import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    HallucinationMetric,
    ToxicityMetric
)
from deepeval.test_case import LLMTestCase

class TestLLMPipeline:
    @pytest.fixture
    async def llm_client(self):
        """Initialize test client"""
        return TestLLMClient(
            base_url="http://localhost:8000",
            timeout=60
        )
    
    @pytest.mark.asyncio
    async def test_rag_accuracy(self, llm_client):
        """Test RAG pipeline accuracy"""
        
        test_cases = [
            {
                "query": "What is the capital of France?",
                "expected_context": ["Paris", "France"],
                "min_relevancy": 0.9
            },
            {
                "query": "Explain quantum entanglement",
                "expected_context": ["quantum", "particles", "entangled"],
                "min_relevancy": 0.8
            }
        ]
        
        for test in test_cases:
            # Execute RAG pipeline
            response = await llm_client.query_rag(test["query"])
            
            # Create test case
            test_case = LLMTestCase(
                input=test["query"],
                actual_output=response["answer"],
                retrieval_context=response["contexts"],
                expected_output=None  # We test quality, not exact match
            )
            
            # Test relevancy
            relevancy_metric = AnswerRelevancyMetric(
                threshold=test["min_relevancy"],
                model="gpt-4-turbo-preview"
            )
            
            # Test faithfulness (no hallucination)
            faithfulness_metric = FaithfulnessMetric(
                threshold=0.9,
                model="gpt-4-turbo-preview"
            )
            
            # Assert metrics pass
            assert_test(test_case, [relevancy_metric, faithfulness_metric])
            
            # Verify expected context
            context_text = " ".join(response["contexts"]).lower()
            for expected in test["expected_context"]:
                assert expected.lower() in context_text
    
    @pytest.mark.asyncio
    async def test_streaming_performance(self, llm_client):
        """Test streaming latency and throughput"""
        
        prompt = "Write a detailed explanation of photosynthesis"
        
        first_token_time = None
        tokens = []
        start_time = time.time()
        
        async for chunk in llm_client.stream_completion(prompt):
            if first_token_time is None:
                first_token_time = time.time() - start_time
            tokens.append(chunk)
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert first_token_time < 0.5, f"First token too slow: {first_token_time}s"
        assert len(tokens) > 100, "Not enough tokens generated"
        
        throughput = len(tokens) / total_time
        assert throughput > 50, f"Throughput too low: {throughput} tokens/s"
    
    @pytest.mark.asyncio
    async def test_multi_model_consistency(self, llm_client):
        """Test consistency across different models"""
        
        query = "What is 2+2?"
        models = ["llama-3.2-3b", "llama-3.2-8b", "llama-3.3-70b"]
        
        responses = {}
        for model in models:
            response = await llm_client.query(
                query,
                model=model,
                temperature=0  # Deterministic
            )
            responses[model] = response
        
        # All models should agree on simple facts
        answers = [r["answer"].strip() for r in responses.values()]
        assert all("4" in answer for answer in answers)
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_load_handling(self, llm_client):
        """Test system under load"""
        
        async def single_request():
            return await llm_client.query(
                "Hello, how are you?",
                timeout=30
            )
        
        # Concurrent requests
        num_requests = 100
        start_time = time.time()
        
        tasks = [single_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Success rate
        successful = [r for r in results if not isinstance(r, Exception)]
        success_rate = len(successful) / num_requests
        
        assert success_rate > 0.95, f"Success rate too low: {success_rate}"
        assert total_time < 60, f"Load test too slow: {total_time}s"
        
        # Check rate limiting works
        errors = [r for r in results if isinstance(r, Exception)]
        rate_limit_errors = [
            e for e in errors 
            if "rate limit" in str(e).lower()
        ]
        
        assert len(rate_limit_errors) < 10, "Too many rate limit errors"
```

### Continuous Evaluation in CI/CD

```yaml
# .github/workflows/llm-evaluation.yml
name: LLM System Evaluation

on:
  pull_request:
    paths:
      - 'agents/**'
      - 'serving/**'
      - 'prompts/**'
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours

jobs:
  evaluate:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.12'
        
    - name: Install dependencies
      run: |
        pip install uv
        uv sync --extra eval
        
    - name: Start test environment
      run: |
        docker-compose -f docker-compose.test.yml up -d
        ./scripts/wait-for-healthy.sh
        
    - name: Run evaluation suite
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        LANGFUSE_SECRET_KEY: ${{ secrets.LANGFUSE_SECRET_KEY }}
      run: |
        uv run pytest tests/eval/ \
          --benchmark \
          --html=report.html \
          --self-contained-html \
          -v
          
    - name: Check performance regression
      run: |
        uv run python scripts/check_regression.py \
          --baseline main \
          --threshold 0.1
          
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: evaluation-report
        path: |
          report.html
          benchmarks/
          
    - name: Comment PR with results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v7
      with:
        script: |
          const fs = require('fs');
          const results = JSON.parse(
            fs.readFileSync('evaluation_results.json', 'utf8')
          );
          
          const comment = `## 🤖 LLM Evaluation Results
          
          | Metric | Score | Baseline | Change |
          |--------|-------|----------|--------|
          | Accuracy | ${results.accuracy} | ${results.baseline_accuracy} | ${results.accuracy_change} |
          | Latency (p95) | ${results.latency_p95}ms | ${results.baseline_latency_p95}ms | ${results.latency_change} |
          | Throughput | ${results.throughput} tok/s | ${results.baseline_throughput} tok/s | ${results.throughput_change} |
          | Cost/1K tokens | $${results.cost_per_1k} | $${results.baseline_cost} | ${results.cost_change} |
          
          ${results.regression_detected ? '⚠️ **Performance regression detected!**' : '✅ All checks passed'}
          
          [View detailed report](${process.env.GITHUB_SERVER_URL}/${process.env.GITHUB_REPOSITORY}/actions/runs/${process.env.GITHUB_RUN_ID})
          `;
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });
```

---

## Conclusion

This guide represents the state of production LLM systems as of mid-2025. The landscape continues to evolve rapidly, but these patterns provide a solid foundation for building reliable, scalable, and cost-effective AI applications.

Key takeaways:
- **Layer separation** is critical for maintainability
- **LangGraph** provides the best agent orchestration experience
- **vLLM** remains unmatched for high-throughput serving
- **KServe** simplifies Kubernetes deployments
- **Comprehensive observability** is non-negotiable for production

Remember: Start simple, measure everything, and iterate based on real usage patterns. The best LLM system is the one that reliably serves your users' needs, not the one with the most advanced architecture.