I'll search for the latest information about PostgreSQL 17, Ubuntu 25, and modern best practices for using these technologies together in 2025.# The Definitive Guide to PostgreSQL 17 with FastAPI, SQLModel, and Modern Python (mid-2025)

This guide synthesizes production-grade best practices for building scalable, high-performance applications with PostgreSQL 17, FastAPI, SQLModel, and the modern Python async ecosystem. It moves beyond basic tutorials to provide battle-tested patterns for real-world applications.

### Prerequisites & System Requirements
Ensure your environment meets these specifications:
- **Ubuntu 25.04 Server** (latest LTS)
- **PostgreSQL 17.5+** (apply minor updates within 7 days of release)
- **Python 3.13+** (with optional free-threaded build support)
- **uv 0.4+** for dependency management
- **FastAPI 0.115+**, **SQLModel 0.0.25+**, **SQLAlchemy 2.0.41+**, **psycopg[binary,pool] 3.2.1+**

> **Note**: This guide assumes you're building for production. Development shortcuts are explicitly marked.

---

## 1. PostgreSQL 17 Installation & Advanced Configuration

PostgreSQL 17 brings game-changing features: 20x more efficient VACUUM memory management, incremental backups, JSON_TABLE support, and significantly improved parallel query performance.

### ✅ DO: Install PostgreSQL 17 from Official APT Repository

The latest version of PostgreSQL is not included in the Ubuntu default repository, so you will need to add the PostgreSQL official repository to the APT.

```bash
# Add PostgreSQL official APT repository
sudo sh -c 'echo "deb [signed-by=/etc/apt/keyrings/postgresql-keyring.gpg] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'

# Add repository signing key (using signed-by method, not deprecated apt-key)
curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo gpg --dearmor -o /etc/apt/keyrings/postgresql-keyring.gpg

# Update package list and install PostgreSQL 17
sudo apt update
sudo apt install -y postgresql-17 postgresql-client-17 postgresql-contrib-17

# Install essential extensions
sudo apt install -y postgresql-17-pgvector postgresql-17-pg-stat-statements
```

### ✅ DO: Create a Custom Data Directory on Separate Storage

PostgreSQL stores its data in the $PGDATA directory, typically located at /var/lib/postgresql/17/main. However, you can specify a custom location if necessary.

For production systems, always separate your database storage from the OS disk:

```bash
# Create custom data directory (assuming /data is your dedicated storage mount)
sudo mkdir -p /data/postgresql/17/main
sudo chown postgres:postgres /data/postgresql/17/main
sudo chmod 700 /data/postgresql/17/main

# Initialize the new data directory
sudo -u postgres /usr/lib/postgresql/17/bin/initdb -D /data/postgresql/17/main

# Update PostgreSQL configuration
sudo sed -i "s|data_directory = '/var/lib/postgresql/17/main'|data_directory = '/data/postgresql/17/main'|" \
    /etc/postgresql/17/main/postgresql.conf

# Restart PostgreSQL
sudo systemctl restart postgresql@17-main

# Schedule managed-service minor upgrades
# For cloud providers (AWS RDS, Google Cloud SQL, Azure Database), 
# enable automatic minor version upgrades to apply patches within 7 days
```

### Production-Optimized Configuration

PostgreSQL has a wide array of configuration parameters. These parameters control almost every aspect of the database's behavior: memory usage, write-ahead logging (WAL), autovacuum, planner behavior

Edit `/etc/postgresql/17/main/postgresql.conf` with these production settings:

```ini
# Memory Configuration (for 64GB RAM server)
shared_buffers = 16GB              # 25% of total RAM
effective_cache_size = 48GB        # 75% of total RAM
work_mem = 128MB                   # Per-operation memory
maintenance_work_mem = 1GB         # PG17 vacuum now uses TidStore (20x more efficient)

# PostgreSQL 17 Parallel Query Optimization
max_parallel_workers_per_gather = 4
max_parallel_workers = 16
max_parallel_maintenance_workers = 4
parallel_leader_participation = on

# Write-Ahead Logging (optimized for NVMe SSDs)
wal_compression = zstd             # New in PG15+, better than pglz
checkpoint_completion_target = 0.9
max_wal_size = 8GB
min_wal_size = 2GB

# PostgreSQL 17 Vacuum Improvements
autovacuum_vacuum_cost_limit = 3000    # Increased for faster cleanup with TidStore
autovacuum_naptime = 10s                # More frequent checks
autovacuum_vacuum_threshold = 50
autovacuum_analyze_threshold = 50
autovacuum_freeze_max_age = 1000000000  # 1 billion

# Connection Management
max_connections = 200                    # Adjust based on workload
superuser_reserved_connections = 5
statement_timeout = 30s                  # Prevent runaway queries

# Query Planner (PostgreSQL 17 optimizations)
enable_partitionwise_aggregate = on
enable_partitionwise_join = on
jit = on                                # Just-In-Time compilation
jit_above_cost = 100000

# Statistics and Monitoring
# Use ALTER SYSTEM SET for easier container deployments
# ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements,pgvector';
pg_stat_statements.track = all
pg_stat_statements.max = 10000

# SSL/TLS Configuration
ssl = on
ssl_cert_file = '/etc/postgresql/17/main/server.crt'
ssl_key_file = '/etc/postgresql/17/main/server.key'
ssl_ciphers = 'HIGH:MEDIUM:!LOW:!MD5:!RC4:!3DES'
ssl_prefer_server_ciphers = on
ssl_min_protocol_version = 'TLSv1.3'
```

### Client Authentication Security

PostgreSQL uses its own memory buffer (shared_buffers) and relies on the operating system's kernel cache (leading to potential "double buffering")

Configure `/etc/postgresql/17/main/pg_hba.conf` for secure authentication:

```conf
# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   all             postgres                                peer
local   all             all                                     peer map=local_users

# Require SSL for all remote connections
hostssl all             all             10.0.0.0/8              scram-sha-256
hostssl all             all             172.16.0.0/12           scram-sha-256
hostssl all             all             192.168.0.0/16          scram-sha-256

# Reject non-SSL connections
hostnossl all           all             0.0.0.0/0               reject
```

### Create Application Database and User

```sql
-- Connect as postgres superuser
sudo -u postgres psql

-- Create application database with optimal settings
CREATE DATABASE fastapi_app
    WITH 
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;

-- Create application user with strong password
CREATE USER app_user WITH 
    PASSWORD 'use-a-very-strong-password-here'
    CONNECTION LIMIT 100;

-- Grant necessary privileges
GRANT CONNECT ON DATABASE fastapi_app TO app_user;
GRANT CREATE ON DATABASE fastapi_app TO app_user;

-- Switch to app database
\c fastapi_app

-- Create schema and set permissions
CREATE SCHEMA IF NOT EXISTS app AUTHORIZATION app_user;
ALTER USER app_user SET search_path TO app, public;

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "vector";  -- pgvector 0.7+ with HNSW, half-precision, L1/Jaccard

-- PostgreSQL 17: Grant MAINTAIN privilege for non-superuser maintenance
GRANT MAINTAIN ON SCHEMA app TO app_user;

-- PostgreSQL 17: New EXPLAIN options for better visibility
-- Use EXPLAIN (ANALYZE, MEMORY, SERIALIZE) to see memory usage and serialization costs
-- Example: EXPLAIN (ANALYZE, MEMORY, SERIALIZE) SELECT * FROM users;
```

---

## 2. Modern Python Project Setup with uv

### ✅ DO: Use uv for Lightning-Fast Dependency Management

Create a new FastAPI project with modern tooling:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project directory
mkdir fastapi-postgres-app && cd fastapi-postgres-app

# Initialize Python 3.13 project
uv init
uv python pin 3.13

# Create virtual environment with specific Python version
uv venv --python 3.13

# For experimental free-threaded Python (GIL-less)
# uv python install 3.13-freethreaded
# uv venv --python 3.13-freethreaded

# Activate virtual environment
source .venv/bin/activate  # On Unix/macOS
```

### Project Structure for Scale

```
fastapi-postgres-app/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dependencies.py      # Shared FastAPI dependencies
│   │   └── middleware.py        # Custom ASGI middleware
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # Settings with pydantic-settings
│   │   ├── database.py         # Database connection management
│   │   └── security.py         # Auth and security utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py            # SQLModel base configuration
│   │   └── user.py            # Domain models
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── user.py            # Database access layer
│   ├── routers/
│   │   ├── __init__.py
│   │   └── users.py           # API endpoints
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── user.py            # Pydantic schemas for API
│   └── main.py                # FastAPI app creation
├── alembic/                   # Database migrations
├── tests/                     # Test suite
├── scripts/                   # Utility scripts
├── .env.example              # Environment variables template
├── pyproject.toml            # Project configuration
└── docker-compose.yml        # Local development setup
```

### Modern pyproject.toml Configuration

```toml
[project]
name = "fastapi-postgres-app"
version = "0.1.0"
description = "Production-grade FastAPI with PostgreSQL 17"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    # Core web framework
    "fastapi[standard]>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "gunicorn>=23.0.0",
    
    # Database - Latest async stack
    "sqlmodel>=0.0.25",
    "sqlalchemy[asyncio]>=2.0.41",
    "alembic>=1.13.0",
    
    # PostgreSQL drivers (choose based on needs)
    "psycopg[binary,pool]>=3.2.1",  # Modern psycopg3
    "asyncpg>=0.29.0",              # High-performance async
    
    # Data validation and settings
    "pydantic>=2.9.0",
    "pydantic-settings>=2.5.0",
    "email-validator>=2.2.0",
    
    # Async HTTP client
    "httpx>=0.27.0",
    
    # Security
    "passlib[bcrypt]>=1.7.4",
    "python-jose[cryptography]>=3.3.0",
    "python-multipart>=0.0.9",
    
    # Utilities
    "python-decouple>=3.8",
    "structlog>=24.4.0",
    "orjson>=3.10.0",
    
    # Monitoring
    "prometheus-client>=0.20.0",
    "opentelemetry-api>=1.25.0",
    "opentelemetry-instrumentation-fastapi>=0.45b0",
    "opentelemetry-instrumentation-sqlalchemy>=0.45b0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0.0",
    "httpx>=0.27.0",
    "factory-boy>=3.3.0",
    "faker>=28.0.0",
    "ruff>=0.7.0",
    "mypy>=1.11.0",
    "pre-commit>=3.8.0",
    "types-passlib>=1.7.7",
    "types-python-jose>=3.3.4",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 120
target-version = "py313"
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "ARG",    # flake8-unused-arguments
    "SIM",    # flake8-simplify
    "PTH",    # flake8-use-pathlib
    "ASYNC",  # flake8-async
]

[tool.mypy]
python_version = "3.13"
strict = true
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = false

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
```

### Install Dependencies

```bash
# Install all dependencies including dev tools
uv sync --all-extras

# Or just production dependencies
uv sync
```

---

## 3. Database Connection Management: The Path to Performance

The choice between psycopg3 and asyncpg is crucial for performance. Here's when to use each:

### psycopg3 vs asyncpg: Making the Right Choice

Psycopg 3 is a newly designed PostgreSQL database adapter for the Python programming language. Psycopg 3 presents a familiar interface for everyone who has used Psycopg 2 or any other DB-API 2.0 database adapter

**Use psycopg3 when:**
- You need Django compatibility
- You're migrating from psycopg2
- You want a familiar DB-API 2.0 interface
- You need comprehensive PostgreSQL type support out of the box

**Use asyncpg when:**
- Raw performance is critical (3-5x faster than psycopg3)
- You're building a high-throughput API
- You don't need Django compatibility
- You can handle the lower-level API

### ✅ DO: Use psycopg3 Pipeline Mode for Batch Operations

For high-throughput OLTP workloads with psycopg3, enable pipeline mode (stable in psycopg 3.2):

```python
# Using psycopg3 pipeline mode
import psycopg
from psycopg import AsyncConnection

async def batch_insert_with_pipeline(conn: AsyncConnection, records: List[Dict]):
    """Execute multiple statements efficiently using pipeline mode."""
    async with conn.pipeline():
        async with conn.cursor() as cur:
            for record in records:
                await cur.execute(
                    "INSERT INTO users (email, username) VALUES (%s, %s)",
                    (record['email'], record['username'])
                )
    # All statements are sent in a single batch when the pipeline exits
```

This feature is now stable and recommended by AWS and community guides for 2025 deployments.

### ✅ DO: Connection Pool Sizing Best Practices

Set `max_overflow=0` and monitor `.pool_status()` to prevent connection exhaustion:

```python
# Recommended pool configuration
engine = create_async_engine(
    "postgresql+asyncpg://...",
    pool_size=20,
    max_overflow=0,  # Prevent going over pool_size
    pool_timeout=30,
    pool_pre_ping=True,
)

# Monitor pool status
pool_status = engine.pool.status()
logger.info(f"Pool size: {pool_status['size']}, checked out: {pool_status['checked_out_connections']}")
```

### ✅ DO: Implement a Robust Async Database Manager (asyncpg)

For maximum performance, here's a production-grade asyncpg implementation:

```python
# src/core/database.py
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
import asyncio
import asyncpg
from asyncpg import Pool, Connection
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import settings

logger = structlog.get_logger()

class DatabaseManager:
    """
    Production-grade async database manager with connection pooling,
    health checks, and graceful shutdown.
    """
    
    def __init__(self):
        self._pool: Optional[Pool] = None
        self._connection_lock = asyncio.Lock()
        
    @property
    def pool(self) -> Pool:
        if self._pool is None:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")
        return self._pool
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def initialize(self) -> None:
        """Initialize the connection pool with retry logic."""
        async with self._connection_lock:
            if self._pool is not None:
                return
                
            logger.info("Initializing database connection pool")
            
            try:
                self._pool = await asyncpg.create_pool(
                    settings.database_url,
                    min_size=settings.db_pool_min_size,
                    max_size=settings.db_pool_max_size,
                    max_queries=50000,
                    max_inactive_connection_lifetime=300,
                    timeout=60,
                    command_timeout=60,
                    server_settings={
                        'application_name': settings.app_name,
                        'jit': 'off',  # Disable JIT for OLTP workloads
                    },
                )
                
                # Verify connection
                async with self._pool.acquire() as conn:
                    version = await conn.fetchval("SELECT version()")
                    logger.info("Database connected", version=version)
                    
            except Exception as e:
                logger.error("Failed to initialize database pool", error=str(e))
                raise
    
    async def close(self) -> None:
        """Gracefully close all database connections."""
        if self._pool:
            logger.info("Closing database connection pool")
            await self._pool.close()
            self._pool = None
    
    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[Connection, None]:
        """Acquire a connection from the pool with automatic cleanup."""
        async with self.pool.acquire() as connection:
            yield connection
    
    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[Connection, None]:
        """Provide a database connection with automatic transaction management."""
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                yield connection
    
    async def execute(self, query: str, *args, timeout: float = None) -> str:
        """Execute a query that doesn't return data."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args, timeout=timeout)
    
    async def fetch(self, query: str, *args, timeout: float = None) -> list:
        """Execute a query and fetch all results."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args, timeout=timeout)
    
    async def fetchrow(self, query: str, *args, timeout: float = None) -> Optional[asyncpg.Record]:
        """Execute a query and fetch a single row."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args, timeout=timeout)
    
    async def fetchval(self, query: str, *args, column: int = 0, timeout: float = None):
        """Execute a query and fetch a single value."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args, column=column, timeout=timeout)
    
    async def health_check(self) -> bool:
        """Perform a health check on the database connection."""
        try:
            async with asyncio.timeout(5):
                result = await self.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False

# Global instance
db_manager = DatabaseManager()

# Dependency for FastAPI
async def get_db() -> DatabaseManager:
    """FastAPI dependency to get database manager."""
    return db_manager
```

### ✅ DO: Alternative Implementation with SQLModel + psycopg3

If you prefer SQLModel's ORM capabilities with psycopg3's modern features:

```python
# src/core/database_sqlmodel.py
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import structlog
from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    create_async_engine,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool
from sqlmodel import SQLModel

from src.core.config import settings

logger = structlog.get_logger()

class SQLModelManager:
    """SQLModel + psycopg3 async database manager."""
    
    def __init__(self):
        # Use psycopg3 (note the psycopg driver, not psycopg2)
        self.engine: AsyncEngine = create_async_engine(
            settings.database_url_psycopg3,  # postgresql+psycopg://...
            echo=settings.debug,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
            poolclass=AsyncAdaptedQueuePool,
            connect_args={
                "server_settings": {
                    "application_name": settings.app_name,
                    "jit": "off"
                },
                "connect_timeout": 10,
                # psycopg3 specific optimizations
                "prepared_statement_cache_size": 256,
                "pipeline_mode": True,  # Enable pipeline mode
            }
        )
        
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False
        )
    
    async def create_all(self):
        """Create all tables (development only)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
    
    async def close(self):
        """Close all connections."""
        await self.engine.dispose()
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Provide a transactional scope around a series of operations."""
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

# Global instance
sqlmodel_manager = SQLModelManager()

# FastAPI dependency
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    async with sqlmodel_manager.get_session() as session:
        yield session
```

### Environment Configuration

```python
# src/core/config.py
from typing import Optional
from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )
    
    # Application
    app_name: str = "FastAPI PostgreSQL App"
    debug: bool = False
    environment: str = Field(default="development", validation_alias="ENV")
    
    # PostgreSQL
    postgres_user: str
    postgres_password: str
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str
    
    # Connection URLs (computed)
    database_url: Optional[str] = None
    database_url_psycopg3: Optional[str] = None
    
    # Connection Pool Settings
    db_pool_size: int = 20
    db_pool_min_size: int = 10
    db_max_overflow: int = 10
    db_pool_timeout: float = 30.0
    
    # Security
    secret_key: str
    access_token_expire_minutes: int = 30
    
    @field_validator("database_url", mode="before")
    def assemble_db_connection(cls, v: Optional[str], values) -> str:
        if v:
            return v
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=values.data.get("postgres_user"),
            password=values.data.get("postgres_password"),
            host=values.data.get("postgres_host"),
            port=values.data.get("postgres_port"),
            path=values.data.get("postgres_db"),
        ).unicode_string()
    
    @field_validator("database_url_psycopg3", mode="before")
    def assemble_psycopg3_connection(cls, v: Optional[str], values) -> str:
        if v:
            return v
        return PostgresDsn.build(
            scheme="postgresql+psycopg",  # Note: psycopg not psycopg2
            username=values.data.get("postgres_user"),
            password=values.data.get("postgres_password"),
            host=values.data.get("postgres_host"),
            port=values.data.get("postgres_port"),
            path=values.data.get("postgres_db"),
        ).unicode_string()

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
```

---

## 4. SQLModel: Production-Grade Data Modeling

SQLModel is based on Python type annotations, and powered by Pydantic and SQLAlchemy

### ⚠️ IMPORTANT: SQLModel Indexing Changes

As of SQLModel 0.0.25+, indexes are no longer created by default. You must explicitly use `Field(index=True)` or add `Index()` in `__table_args__`:

```python
# Old behavior (pre-0.0.25): This would create an index automatically
email: str = Field(unique=True)

# New behavior (0.0.25+): Must explicitly request index
email: str = Field(unique=True, index=True)
```

### 📌 Note on SQLModel Project Status

SQLModel is maintained by a single developer and updates can be slower than SQLAlchemy. The project now has Pydantic v2.11 compatibility (0.0.25+), but some advanced SQLAlchemy 2.0 features may not be fully supported. For production applications requiring cutting-edge features, consider using SQLAlchemy directly with separate Pydantic models.

### ✅ DO: Design Models with PostgreSQL 17 Features in Mind

```python
# src/models/base.py
from datetime import datetime, timezone
from typing import Optional
import uuid
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, DateTime, func, text
from sqlalchemy.dialects.postgresql import UUID

class TimestampMixin(SQLModel):
    """Mixin for automatic timestamp management."""
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.current_timestamp(),
            nullable=False
        )
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
            nullable=False
        )
    )

class UUIDMixin(SQLModel):
    """Mixin for UUID primary keys."""
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(
            UUID(as_uuid=True),
            primary_key=True,
            server_default=text("gen_random_uuid()"),
            nullable=False
        )
    )

# src/models/user.py
from typing import Optional, List
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field, Relationship, Column, Index
from sqlalchemy import String, func
from pydantic import EmailStr

from src.models.base import UUIDMixin, TimestampMixin

class UserBase(SQLModel):
    """Base user attributes."""
    email: EmailStr = Field(
        sa_column=Column(String(255), unique=True, nullable=False)
    )
    username: str = Field(
        min_length=3,
        max_length=50,
        sa_column=Column(String(50), unique=True, nullable=False)
    )
    full_name: Optional[str] = Field(default=None, max_length=255)
    is_active: bool = Field(default=True)
    is_superuser: bool = Field(default=False)

class User(UserBase, UUIDMixin, TimestampMixin, table=True):
    """User database model with all mixins."""
    __tablename__ = "users"
    __table_args__ = (
        # PostgreSQL 17: Create indexes for common query patterns
        Index("idx_user_email_active", "email", "is_active"),
        Index("idx_user_username_lower", func.lower("username")),
        # Use BRIN index for timestamp columns (PostgreSQL 17 optimization)
        Index("idx_user_created_at_brin", "created_at", postgresql_using="brin"),
        {"schema": "app"}  # Use custom schema
    )
    
    hashed_password: str = Field(sa_column=Column(String, nullable=False))
    last_login: Optional[datetime] = Field(default=None)
    
    # Relationships
    posts: List["Post"] = Relationship(back_populates="author")
    
    # PostgreSQL 17: JSON column for flexible user metadata
    metadata_json: Optional[dict] = Field(
        default=None,
        sa_column=Column(
            "metadata",
            String,
            server_default=text("'{}'::jsonb")
        )
    )

class Post(UUIDMixin, TimestampMixin, table=True):
    """Example related model."""
    __tablename__ = "posts"
    __table_args__ = (
        # PostgreSQL 17: Use covering index for better performance
        Index(
            "idx_post_author_created",
            "author_id",
            "created_at",
            postgresql_include=["title", "is_published"]
        ),
        {"schema": "app"}
    )
    
    title: str = Field(max_length=255, nullable=False)
    content: str = Field(nullable=False)
    is_published: bool = Field(default=False)
    
    # Foreign key with proper naming
    author_id: uuid.UUID = Field(
        foreign_key="app.users.id",
        nullable=False
    )
    
    # Relationships
    author: User = Relationship(back_populates="posts")
```

### ✅ DO: Create Separate Pydantic Schemas for API

```python
# src/schemas/user.py
from typing import Optional
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, EmailStr, Field, ConfigDict

class UserCreate(BaseModel):
    """Schema for creating a new user."""
    email: EmailStr
    username: str = Field(min_length=3, max_length=50)
    full_name: Optional[str] = Field(None, max_length=255)
    password: str = Field(min_length=8, max_length=128)

class UserUpdate(BaseModel):
    """Schema for updating user information."""
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    full_name: Optional[str] = Field(None, max_length=255)
    is_active: Optional[bool] = None

class UserResponse(BaseModel):
    """Schema for user responses (excludes sensitive data)."""
    model_config = ConfigDict(from_attributes=True)  # Pydantic v2 syntax
    
    id: uuid.UUID
    email: EmailStr
    username: str
    full_name: Optional[str]
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime]

class UserInDB(UserResponse):
    """Schema including hashed password (internal use only)."""
    hashed_password: str
```

---

## 5. Repository Pattern with Advanced PostgreSQL 17 Features

### ✅ DO: Implement a Type-Safe Repository Layer

```python
# src/repositories/base.py
from typing import TypeVar, Generic, Type, Optional, List, Dict, Any
from uuid import UUID
from sqlmodel import SQLModel, select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, and_, or_
import structlog

from src.core.database import db_manager

ModelType = TypeVar("ModelType", bound=SQLModel)
logger = structlog.get_logger()

class BaseRepository(Generic[ModelType]):
    """
    Base repository with common CRUD operations and PostgreSQL 17 features.
    """
    def __init__(self, model: Type[ModelType]):
        self.model = model
    
    async def create(self, *, obj_in: Dict[str, Any]) -> ModelType:
        """Create a new record."""
        db_obj = self.model(**obj_in)
        
        async with db_manager.transaction() as conn:
            # Use PostgreSQL 17's improved RETURNING clause
            query = f"""
                INSERT INTO app.{self.model.__tablename__} ({', '.join(obj_in.keys())})
                VALUES ({', '.join(f'${i+1}' for i in range(len(obj_in)))})
                RETURNING *
            """
            row = await conn.fetchrow(query, *obj_in.values())
            return self.model(**dict(row))
    
    async def get(self, id: UUID) -> Optional[ModelType]:
        """Get a single record by ID."""
        async with db_manager.acquire() as conn:
            query = f"SELECT * FROM app.{self.model.__tablename__} WHERE id = $1"
            row = await conn.fetchrow(query, id)
            return self.model(**dict(row)) if row else None
    
    async def get_multi(
        self,
        *,
        skip: int = 0,
        limit: int = 100,
        order_by: str = "created_at DESC"
    ) -> List[ModelType]:
        """Get multiple records with pagination."""
        async with db_manager.acquire() as conn:
            query = f"""
                SELECT * FROM app.{self.model.__tablename__}
                ORDER BY {order_by}
                LIMIT $1 OFFSET $2
            """
            rows = await conn.fetch(query, limit, skip)
            return [self.model(**dict(row)) for row in rows]
    
    async def update(self, *, id: UUID, obj_in: Dict[str, Any]) -> Optional[ModelType]:
        """Update a record."""
        if not obj_in:
            return await self.get(id)
        
        async with db_manager.transaction() as conn:
            # Build dynamic UPDATE query
            set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(obj_in.keys()))
            query = f"""
                UPDATE app.{self.model.__tablename__}
                SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                WHERE id = $1
                RETURNING *
            """
            values = [id] + list(obj_in.values())
            row = await conn.fetchrow(query, *values)
            return self.model(**dict(row)) if row else None
    
    async def delete(self, *, id: UUID) -> bool:
        """Delete a record."""
        async with db_manager.transaction() as conn:
            query = f"DELETE FROM app.{self.model.__tablename__} WHERE id = $1"
            result = await conn.execute(query, id)
            return result.split()[-1] != "0"
    
    async def count(self, *, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count records with optional filters."""
        async with db_manager.acquire() as conn:
            where_clause = ""
            values = []
            
            if filters:
                conditions = []
                for i, (key, value) in enumerate(filters.items(), 1):
                    conditions.append(f"{key} = ${i}")
                    values.append(value)
                where_clause = f"WHERE {' AND '.join(conditions)}"
            
            query = f"SELECT COUNT(*) FROM app.{self.model.__tablename__} {where_clause}"
            return await conn.fetchval(query, *values)
    
    async def exists(self, *, id: UUID) -> bool:
        """Check if a record exists."""
        async with db_manager.acquire() as conn:
            query = f"""
                SELECT EXISTS(
                    SELECT 1 FROM app.{self.model.__tablename__} WHERE id = $1
                )
            """
            return await conn.fetchval(query, id)

# src/repositories/user.py
from typing import Optional, List
from uuid import UUID
from datetime import datetime, timezone
import structlog

from src.models.user import User
from src.repositories.base import BaseRepository
from src.core.database import db_manager
from src.core.security import verify_password, get_password_hash

logger = structlog.get_logger()

class UserRepository(BaseRepository[User]):
    """User-specific repository with additional methods."""
    
    def __init__(self):
        super().__init__(model=User)
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email address."""
        async with db_manager.acquire() as conn:
            query = """
                SELECT * FROM app.users 
                WHERE lower(email) = lower($1)
            """
            row = await conn.fetchrow(query, email)
            return User(**dict(row)) if row else None
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username (case-insensitive)."""
        async with db_manager.acquire() as conn:
            # PostgreSQL 17: Use index on lower(username)
            query = """
                SELECT * FROM app.users 
                WHERE lower(username) = lower($1)
            """
            row = await conn.fetchrow(query, username)
            return User(**dict(row)) if row else None
    
    async def create_user(self, *, email: str, username: str, password: str, **kwargs) -> User:
        """Create a new user with hashed password."""
        user_data = {
            "email": email,
            "username": username,
            "hashed_password": get_password_hash(password),
            **kwargs
        }
        return await self.create(obj_in=user_data)
    
    async def authenticate(self, *, email: str, password: str) -> Optional[User]:
        """Authenticate user by email and password."""
        user = await self.get_by_email(email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        await self.update_last_login(user.id)
        return user
    
    async def update_last_login(self, user_id: UUID) -> None:
        """Update user's last login timestamp."""
        async with db_manager.execute(
            "UPDATE app.users SET last_login = $1 WHERE id = $2",
            datetime.now(timezone.utc),
            user_id
        )
    
    async def search_users(
        self,
        *,
        query: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """
        Search users using PostgreSQL 17's improved full-text search.
        """
        async with db_manager.acquire() as conn:
            # Use PostgreSQL's full-text search with ranking
            search_query = """
                SELECT *, 
                       ts_rank(
                           to_tsvector('english', username || ' ' || COALESCE(full_name, '')),
                           plainto_tsquery('english', $1)
                       ) as rank
                FROM app.users
                WHERE to_tsvector('english', username || ' ' || COALESCE(full_name, '')) 
                      @@ plainto_tsquery('english', $1)
                   OR username ILIKE $2
                   OR full_name ILIKE $2
                ORDER BY rank DESC, created_at DESC
                LIMIT $3 OFFSET $4
            """
            pattern = f"%{query}%"
            rows = await conn.fetch(search_query, query, pattern, limit, skip)
            return [User(**dict(row)) for row in rows]
    
    async def get_active_users_count(self) -> int:
        """Get count of active users using PostgreSQL 17's optimized COUNT."""
        return await self.count(filters={"is_active": True})
    
    async def bulk_create_users(self, users_data: List[dict]) -> List[User]:
        """
        Bulk create users using PostgreSQL 17's improved COPY performance.
        """
        if not users_data:
            return []
        
        # Prepare data with hashed passwords
        for user_data in users_data:
            if "password" in user_data:
                user_data["hashed_password"] = get_password_hash(user_data.pop("password"))
        
        async with db_manager.transaction() as conn:
            # Use PostgreSQL's COPY for bulk insert (2x faster in PG17)
            columns = users_data[0].keys()
            
            # Create temporary table
            await conn.execute("""
                CREATE TEMP TABLE temp_users (LIKE app.users INCLUDING ALL)
            """)
            
            # Copy data
            await conn.copy_records_to_table(
                "temp_users",
                records=[tuple(user[col] for col in columns) for user in users_data],
                columns=columns
            )
            
            # Insert from temp table with conflict handling
            query = """
                INSERT INTO app.users
                SELECT * FROM temp_users
                ON CONFLICT (email) DO NOTHING
                RETURNING *
            """
            rows = await conn.fetch(query)
            return [User(**dict(row)) for row in rows]
```

---

## 6. FastAPI Application Structure with Production Patterns

### ✅ DO: Implement Proper Application Lifecycle Management

```python
# src/main.py
from contextlib import asynccontextmanager
import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_client import make_asgi_app

from src.core.config import settings
from src.core.database import db_manager
from src.api.middleware import TimingMiddleware, RequestIDMiddleware
from src.routers import users, auth, health
from src.core.logging import configure_logging

# Configure structured logging
configure_logging()
logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting up application", environment=settings.environment)
    await db_manager.initialize()
    
    # Run migrations in production
    if settings.environment == "production":
        logger.info("Running database migrations")
        # Run alembic migrations programmatically
        
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    await db_manager.close()

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan,
)

# Add middleware in correct order (bottom to top execution)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )

app.add_middleware(TimingMiddleware)
app.add_middleware(RequestIDMiddleware)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "FastAPI + PostgreSQL 17 API",
        "version": "0.1.0",
        "environment": settings.environment
    }
```

### Custom Middleware for Production

```python
# src/api/middleware.py
import time
import uuid
from typing import Callable
import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = structlog.get_logger()

class TimingMiddleware(BaseHTTPMiddleware):
    """Add request timing to responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        
        response = await call_next(request)
        
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        
        # Log slow requests
        if process_time > 1.0:
            logger.warning(
                "Slow request detected",
                path=request.url.path,
                method=request.method,
                process_time=process_time
            )
        
        return response

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add request ID for tracing."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        
        # Add to structlog context
        structlog.contextvars.bind_contextvars(request_id=request_id)
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        structlog.contextvars.unbind_contextvars("request_id")
        return response
```

### API Endpoints with Async Excellence

```python
# src/routers/users.py
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.user import UserCreate, UserUpdate, UserResponse
from src.repositories.user import UserRepository
from src.core.database import get_db
from src.core.auth import get_current_active_user
from src.models.user import User

router = APIRouter()

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_in: UserCreate,
    current_user: User = Depends(get_current_active_user),
) -> UserResponse:
    """Create a new user (admin only)."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    user_repo = UserRepository()
    
    # Check if user exists
    if await user_repo.get_by_email(user_in.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    if await user_repo.get_by_username(user_in.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create user
    user = await user_repo.create_user(
        email=user_in.email,
        username=user_in.username,
        password=user_in.password,
        full_name=user_in.full_name
    )
    
    return UserResponse.model_validate(user)

@router.get("/", response_model=List[UserResponse])
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
) -> List[UserResponse]:
    """List users with pagination and search."""
    user_repo = UserRepository()
    
    if search:
        users = await user_repo.search_users(
            query=search,
            skip=skip,
            limit=limit
        )
    else:
        users = await user_repo.get_multi(skip=skip, limit=limit)
    
    return [UserResponse.model_validate(user) for user in users]

@router.get("/me", response_model=UserResponse)
async def read_current_user(
    current_user: User = Depends(get_current_active_user),
) -> UserResponse:
    """Get current user information."""
    return UserResponse.model_validate(current_user)

@router.get("/{user_id}", response_model=UserResponse)
async def read_user(
    user_id: UUID,
    current_user: User = Depends(get_current_active_user),
) -> UserResponse:
    """Get a specific user by ID."""
    user_repo = UserRepository()
    user = await user_repo.get(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse.model_validate(user)

@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: UUID,
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
) -> UserResponse:
    """Update user information."""
    # Users can only update themselves unless they're superuser
    if user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    user_repo = UserRepository()
    
    # Check for conflicts if updating email/username
    if user_update.email:
        existing = await user_repo.get_by_email(user_update.email)
        if existing and existing.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    if user_update.username:
        existing = await user_repo.get_by_username(user_update.username)
        if existing and existing.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
    
    # Update user
    update_data = user_update.model_dump(exclude_unset=True)
    updated_user = await user_repo.update(id=user_id, obj_in=update_data)
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse.model_validate(updated_user)

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: UUID,
    current_user: User = Depends(get_current_active_user),
) -> None:
    """Delete a user (admin only)."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    user_repo = UserRepository()
    
    if not await user_repo.delete(id=user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
```

---

## 7. Database Migrations with Alembic

### ✅ DO: Set Up Alembic for Production-Grade Migrations

```bash
# Initialize Alembic
alembic init -t async alembic

# Update alembic.ini
sed -i 's|sqlalchemy.url = driver://user:pass@localhost/dbname|sqlalchemy.url = postgresql+asyncpg://user:pass@localhost/dbname|' alembic.ini
```

Configure Alembic for async operations:

```python
# alembic/env.py
import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context
from src.models import *  # Import all models
from src.core.config import settings
from sqlmodel import SQLModel

# this is the Alembic Config object
config = context.config

# Set the database URL from settings
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema="app",  # Use app schema
    )

    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table_schema="app",
    )

    with context.begin_transaction():
        context.run_migrations()

async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### Create Initial Migration

```bash
# Generate first migration
alembic revision --autogenerate -m "Initial user model with PostgreSQL 17 features"

# Review the generated migration file
# Apply migration
alembic upgrade head
```

Example migration with PostgreSQL 17 features:

```python
# alembic/versions/xxx_initial_user_model.py
"""Initial user model with PostgreSQL 17 features

Revision ID: xxx
Revises: 
Create Date: 2025-07-01 10:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
import sqlmodel
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'xxx'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Create app schema if not exists
    op.execute("CREATE SCHEMA IF NOT EXISTS app")
    
    # Create users table with PostgreSQL 17 features
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), 
                  server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('hashed_password', sa.String(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('is_superuser', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column('created_at', sa.DateTime(timezone=True), 
                  server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), 
                  server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('username'),
        schema='app'
    )
    
    # Create indexes using PostgreSQL 17 features
    op.create_index('idx_user_email_active', 'users', ['email', 'is_active'], 
                    unique=False, schema='app')
    op.create_index('idx_user_username_lower', 'users', 
                    [sa.text('lower(username)')], unique=False, schema='app')
    
    # Create BRIN index for timestamp (PostgreSQL 17 optimization)
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_created_at_brin 
        ON app.users USING brin(created_at)
    """)
    
    # Create trigger for updated_at
    op.execute("""
        CREATE OR REPLACE FUNCTION app.update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        CREATE TRIGGER update_users_updated_at 
        BEFORE UPDATE ON app.users 
        FOR EACH ROW EXECUTE FUNCTION app.update_updated_at_column();
    """)

def downgrade() -> None:
    op.drop_table('users', schema='app')
    op.execute("DROP FUNCTION IF EXISTS app.update_updated_at_column() CASCADE")
```

---

## 8. Advanced PostgreSQL 17 Performance Patterns

### ✅ DO: Leverage PostgreSQL 17's New Features

PostgreSQL 17 significantly improves performance, query handling, and database management, making it more efficient for high-demand systems

```python
# src/repositories/advanced.py
from typing import List, Dict, Any
import json
from src.core.database import db_manager

class AdvancedPatterns:
    """Showcase PostgreSQL 17's advanced features."""
    
    async def use_json_table(self, json_data: List[Dict[str, Any]]) -> List[Dict]:
        """
        Use PostgreSQL 17's JSON_TABLE feature for efficient JSON processing.
        """
        async with db_manager.acquire() as conn:
            # Convert Python dict to JSON string
            json_str = json.dumps(json_data)
            
            # Use JSON_TABLE to convert JSON to relational format
            query = """
                SELECT *
                FROM JSON_TABLE(
                    $1::jsonb,
                    '$[*]' COLUMNS (
                        id UUID PATH '$.id',
                        name TEXT PATH '$.name',
                        email TEXT PATH '$.email',
                        created_at TIMESTAMP PATH '$.created_at'
                    )
                ) AS jt
                WHERE jt.email IS NOT NULL
                ORDER BY jt.created_at DESC
            """
            
            rows = await conn.fetch(query, json_str)
            return [dict(row) for row in rows]
    
    async def bulk_upsert_optimized(self, table: str, records: List[Dict]) -> int:
        """
        Perform bulk upsert using PostgreSQL 17's improved MERGE performance.
        """
        if not records:
            return 0
        
        async with db_manager.transaction() as conn:
            # Create temporary table
            temp_table = f"temp_{table}_{uuid.uuid4().hex[:8]}"
            
            # Get columns from first record
            columns = list(records[0].keys())
            
            # Create temp table with same structure
            await conn.execute(f"""
                CREATE TEMP TABLE {temp_table} 
                (LIKE app.{table} INCLUDING ALL)
            """)
            
            # Use COPY for ultra-fast bulk insert into temp table
            await conn.copy_records_to_table(
                temp_table,
                records=[tuple(r.get(col) for col in columns) for r in records],
                columns=columns
            )
            
            # Use MERGE for upsert (PostgreSQL 15+ feature, optimized in 17)
            merge_query = f"""
                MERGE INTO app.{table} AS target
                USING {temp_table} AS source
                ON target.id = source.id
                WHEN MATCHED THEN
                    UPDATE SET {', '.join(f'{col} = source.{col}' for col in columns if col != 'id')},
                               updated_at = CURRENT_TIMESTAMP
                WHEN NOT MATCHED THEN
                    INSERT ({', '.join(columns)})
                    VALUES ({', '.join(f'source.{col}' for col in columns)})
            """
            
            result = await conn.execute(merge_query)
            
            # Extract affected row count
            return int(result.split()[-1])
    
    async def parallel_aggregate_query(self, start_date: str, end_date: str) -> Dict:
        """
        Use PostgreSQL 17's improved parallel query execution for aggregations.
        """
        async with db_manager.acquire() as conn:
            # Force parallel execution for complex aggregation
            await conn.execute("SET max_parallel_workers_per_gather = 4")
            await conn.execute("SET parallel_setup_cost = 10")
            await conn.execute("SET parallel_tuple_cost = 0.01")
            
            query = """
                WITH daily_stats AS (
                    SELECT 
                        date_trunc('day', created_at) AS day,
                        COUNT(*) AS user_count,
                        COUNT(DISTINCT email) AS unique_emails,
                        AVG(CASE WHEN is_active THEN 1 ELSE 0 END)::numeric(5,2) AS active_rate
                    FROM app.users
                    WHERE created_at BETWEEN $1 AND $2
                    GROUP BY date_trunc('day', created_at)
                    -- PostgreSQL 17: Parallel aggregation
                )
                SELECT 
                    to_json(array_agg(daily_stats ORDER BY day)) AS daily_data,
                    SUM(user_count) AS total_users,
                    AVG(active_rate)::numeric(5,2) AS avg_active_rate
                FROM daily_stats
            """
            
            row = await conn.fetchrow(query, start_date, end_date)
            return dict(row) if row else {}
    
    async def use_covering_index_scan(self, user_ids: List[UUID]) -> List[Dict]:
        """
        Demonstrate PostgreSQL 17's improved index-only scans with covering indexes.
        """
        async with db_manager.acquire() as conn:
            # This query can use the covering index we created
            query = """
                SELECT id, username, email, is_active
                FROM app.users
                WHERE id = ANY($1::uuid[])
                ORDER BY created_at DESC
            """
            
            rows = await conn.fetch(query, user_ids)
            
            # Check if index-only scan was used
            explain = await conn.fetch(f"EXPLAIN (ANALYZE, BUFFERS) {query}", user_ids)
            
            # Log whether covering index was used
            explain_text = '\n'.join(row['QUERY PLAN'] for row in explain)
            if 'Index Only Scan' in explain_text:
                logger.info("Covering index successfully used for query")
            
            return [dict(row) for row in rows]
```

### Monitoring and Performance Analysis

```python
# src/core/monitoring.py
from typing import List, Dict
import structlog
from prometheus_client import Counter, Histogram, Gauge
from src.core.database import db_manager

logger = structlog.get_logger()

# Prometheus metrics
db_query_duration = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['query_type', 'table']
)

db_connection_pool_size = Gauge(
    'db_connection_pool_size',
    'Current size of database connection pool'
)

class DatabaseMonitor:
    """Monitor PostgreSQL 17 performance metrics."""
    
    async def get_slow_queries(self, threshold_ms: int = 1000) -> List[Dict]:
        """Get slow queries from pg_stat_statements."""
        query = """
            SELECT 
                query,
                calls,
                total_exec_time,
                mean_exec_time,
                stddev_exec_time,
                rows,
                100.0 * shared_blks_hit / 
                    NULLIF(shared_blks_hit + shared_blks_read, 0) AS hit_percent
            FROM pg_stat_statements
            WHERE mean_exec_time > $1
            ORDER BY mean_exec_time DESC
            LIMIT 20
        """
        
        async with db_manager.acquire() as conn:
            rows = await conn.fetch(query, threshold_ms)
            return [dict(row) for row in rows]
    
    async def analyze_table_bloat(self) -> List[Dict]:
        """Analyze table bloat using PostgreSQL 17's improved statistics."""
        query = """
            WITH constants AS (
                SELECT current_setting('block_size')::numeric AS bs,
                       23 AS hdr, 8 AS ma
            ),
            bloat_info AS (
                SELECT
                    schemaname,
                    tablename,
                    cc.reltuples,
                    cc.relpages,
                    bs,
                    CEIL((cc.reltuples * 
                          ((datahdr + ma - (CASE WHEN datahdr%ma=0 THEN ma ELSE datahdr%ma END)) + 
                           nullhdr + 4)) / (bs-20::float)) AS otta
                FROM (
                    SELECT
                        ns.nspname AS schemaname,
                        tbl.relname AS tablename,
                        tbl.reltuples,
                        tbl.relpages,
                        hdr + AVG(COALESCE(null_frac,0) * COALESCE(avg_width, 1024))::numeric AS datahdr,
                        MAX(COALESCE(null_frac,0)) AS nullhdr
                    FROM pg_class tbl
                    JOIN pg_namespace ns ON ns.oid = tbl.relnamespace
                    JOIN (
                        SELECT
                            starelid,
                            SUM((1-stanullfrac)*stawidth) AS avg_width,
                            MAX(stanullfrac) AS null_frac
                        FROM pg_statistic
                        GROUP BY starelid
                    ) ss ON ss.starelid = tbl.oid
                    CROSS JOIN constants
                    WHERE tbl.relkind = 'r'
                    AND ns.nspname = 'app'
                    GROUP BY 1,2,3,4,hdr
                ) cc
                CROSS JOIN constants
            )
            SELECT
                schemaname,
                tablename,
                reltuples::bigint AS row_count,
                relpages::bigint * bs::bigint AS real_size,
                otta::bigint * bs::bigint AS optimal_size,
                CASE WHEN relpages > 0 AND otta > 0
                    THEN (100 * (relpages - otta)::numeric / relpages)::numeric(5,2)
                    ELSE 0
                END AS bloat_pct
            FROM bloat_info
            WHERE relpages > 10
            ORDER BY bloat_pct DESC
        """
        
        async with db_manager.acquire() as conn:
            rows = await conn.fetch(query)
            return [dict(row) for row in rows]
    
    async def get_connection_stats(self) -> Dict:
        """Get PostgreSQL connection statistics."""
        query = """
            SELECT
                count(*) FILTER (WHERE state = 'active') AS active,
                count(*) FILTER (WHERE state = 'idle') AS idle,
                count(*) FILTER (WHERE state = 'idle in transaction') AS idle_in_transaction,
                count(*) FILTER (WHERE waiting) AS waiting,
                count(*) AS total
            FROM pg_stat_activity
            WHERE pid != pg_backend_pid()
        """
        
        async with db_manager.acquire() as conn:
            row = await conn.fetchrow(query)
            stats = dict(row) if row else {}
            
            # Update Prometheus metrics
            if self._pool:
                db_connection_pool_size.set(self._pool.get_size())
            
            return stats
    
    async def get_io_statistics(self) -> List[Dict]:
        """Get I/O statistics from pg_stat_io (PostgreSQL 17 feature)."""
        query = """
            SELECT 
                backend_type,
                object,
                context,
                reads,
                writes,
                writebacks,
                extends,
                hits,
                evictions,
                reuses,
                fsyncs
            FROM pg_stat_io
            WHERE backend_type IN ('client backend', 'autovacuum worker', 'background writer')
            ORDER BY (reads + writes) DESC
            LIMIT 20
        """
        
        async with db_manager.acquire() as conn:
            rows = await conn.fetch(query)
            return [dict(row) for row in rows]
```

---

## 8. Vector Search with pgvector 0.7+ and Horizontal Scaling with Citus 13

### ✅ DO: Leverage pgvector 0.7+ Advanced Features

PostgreSQL with pgvector 0.7+ now supports HNSW indexes, half-precision vectors, and multiple distance metrics:

```python
# src/repositories/vector.py
from typing import List, Dict, Any
import numpy as np
from src.core.database import db_manager

class VectorRepository:
    """Repository for vector similarity search operations."""
    
    async def create_vector_table(self):
        """Create table with vector column and HNSW index."""
        async with db_manager.transaction() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS app.embeddings (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    content TEXT NOT NULL,
                    embedding vector(768),  -- For BERT-like models
                    embedding_half halfvec(768),  -- Half-precision for storage efficiency
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create HNSW index (much faster than IVFFlat for queries)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw 
                ON app.embeddings 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
            
            # Alternative distance metrics now supported
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_l1
                ON app.embeddings 
                USING hnsw (embedding vector_l1_ops);
            """)
    
    async def insert_embeddings(self, items: List[Dict[str, Any]]):
        """Bulk insert embeddings with half-precision storage."""
        async with db_manager.transaction() as conn:
            # Prepare data for COPY
            records = []
            for item in items:
                embedding = np.array(item['embedding'], dtype=np.float32)
                records.append((
                    item['content'],
                    embedding.tolist(),
                    item.get('metadata', {})
                ))
            
            # Use COPY for bulk insert
            await conn.copy_records_to_table(
                'embeddings',
                records=records,
                columns=['content', 'embedding', 'metadata'],
                schema_name='app'
            )
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        distance_metric: str = 'cosine'
    ) -> List[Dict]:
        """Perform similarity search using specified distance metric."""
        
        # Map distance metrics to operators
        operators = {
            'cosine': '<=>',
            'l1': '<+>',
            'l2': '<->',
            'inner_product': '<#>'
        }
        
        operator = operators.get(distance_metric, '<=>')
        
        query = f"""
            SELECT 
                id,
                content,
                metadata,
                embedding {operator} $1::vector AS distance
            FROM app.embeddings
            ORDER BY embedding {operator} $1::vector
            LIMIT $2
        """
        
        async with db_manager.acquire() as conn:
            rows = await conn.fetch(query, query_embedding, limit)
            return [dict(row) for row in rows]
```

### ✅ DO: Implement Horizontal Scaling with Citus 13

Citus 13 brings distributed PostgreSQL 17 support with automatic sharding:

```python
# src/repositories/distributed.py
class DistributedRepository:
    """Repository for Citus distributed operations."""
    
    async def setup_distributed_tables(self):
        """Set up Citus distributed tables."""
        async with db_manager.transaction() as conn:
            # Enable Citus extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS citus;")
            
            # Create a distributed table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS app.events (
                    tenant_id UUID NOT NULL,
                    event_id UUID NOT NULL DEFAULT gen_random_uuid(),
                    event_type TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (tenant_id, event_id)
                )
            """)
            
            # Distribute the table by tenant_id
            await conn.execute("""
                SELECT create_distributed_table('app.events', 'tenant_id');
            """)
            
            # Create distributed indexes
            await conn.execute("""
                CREATE INDEX idx_events_created_at 
                ON app.events (tenant_id, created_at DESC);
            """)
    
    async def create_reference_table(self):
        """Create a reference table replicated to all nodes."""
        async with db_manager.transaction() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS app.lookup_data (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL
                )
            """)
            
            # Replicate to all worker nodes
            await conn.execute("""
                SELECT create_reference_table('app.lookup_data');
            """)
    
    async def distributed_explain(self, query: str, params: List[Any]) -> str:
        """Use Citus 13's distributed EXPLAIN."""
        async with db_manager.acquire() as conn:
            explain_query = f"EXPLAIN (ANALYZE, VERBOSE, BUFFERS) {query}"
            rows = await conn.fetch(explain_query, *params)
            return '\n'.join(row['QUERY PLAN'] for row in rows)
    
    async def get_shard_info(self, table_name: str) -> List[Dict]:
        """Get information about table shards."""
        query = """
            SELECT 
                shardid,
                shardstate,
                shardlength,
                nodename,
                nodeport
            FROM pg_dist_shard_placement
            JOIN pg_dist_shard USING (shardid)
            WHERE logicalrelid = $1::regclass
            ORDER BY shardid
        """
        
        async with db_manager.acquire() as conn:
            rows = await conn.fetch(query, f'app.{table_name}')
            return [dict(row) for row in rows]
```

### Vector Search with Citus Distribution

```python
async def setup_distributed_vectors():
    """Set up distributed vector search with Citus."""
    async with db_manager.transaction() as conn:
        # Create distributed vector table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS app.distributed_embeddings (
                tenant_id UUID NOT NULL,
                id UUID NOT NULL DEFAULT gen_random_uuid(),
                embedding vector(768),
                content TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (tenant_id, id)
            )
        """)
        
        # Distribute by tenant_id
        await conn.execute("""
            SELECT create_distributed_table(
                'app.distributed_embeddings', 
                'tenant_id'
            );
        """)
        
        # Citus 13 automatically creates HNSW indexes on all shards
        await conn.execute("""
            CREATE INDEX idx_dist_embeddings_hnsw 
            ON app.distributed_embeddings 
            USING hnsw (embedding vector_cosine_ops);
        """)
```

---

## 9. Logical Replication Improvements in PostgreSQL 17

PostgreSQL 17 removes the need to drop logical replication slots during major version upgrades. Slot restarts are now monotonic, making blue-green deployments safer:

```python
# src/repositories/replication.py
class ReplicationManager:
    """Manage logical replication for zero-downtime deployments."""
    
    async def setup_logical_replication(self, publication_name: str = "app_publication"):
        """Set up logical replication publication."""
        async with db_manager.transaction() as conn:
            # Create publication for all tables in app schema
            await conn.execute(f"""
                CREATE PUBLICATION {publication_name}
                FOR ALL TABLES IN SCHEMA app
                WITH (publish_via_partition_root = true);
            """)
    
    async def monitor_replication_lag(self) -> Dict:
        """Monitor replication lag for all subscribers."""
        query = """
            SELECT 
                slot_name,
                active,
                restart_lsn,
                confirmed_flush_lsn,
                pg_wal_lsn_diff(pg_current_wal_lsn(), confirmed_flush_lsn) AS lag_bytes,
                pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), confirmed_flush_lsn)) AS lag_size
            FROM pg_replication_slots
            WHERE slot_type = 'logical'
        """
        
        async with db_manager.acquire() as conn:
            rows = await conn.fetch(query)
            return [dict(row) for row in rows]
```

---

## 10. Testing Strategies for Async PostgreSQL

### ✅ DO: Implement Comprehensive Async Testing

```python
# tests/conftest.py
import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
import asyncpg
from httpx import AsyncClient, ASGITransport

from src.main import app
from src.core.database import db_manager
from src.models.base import SQLModel
from src.core.config import settings

# Override settings for testing
settings.database_url = "postgresql+asyncpg://postgres:postgres@localhost:5432/test_db"
settings.environment = "testing"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="session")
async def setup_database():
    """Create test database and tables."""
    # Create test database
    conn = await asyncpg.connect(
        user='postgres',
        password='postgres',
        host='localhost',
        port=5432,
        database='postgres'
    )
    
    try:
        await conn.execute('DROP DATABASE IF EXISTS test_db')
        await conn.execute('CREATE DATABASE test_db')
    finally:
        await conn.close()
    
    # Initialize tables
    await db_manager.initialize()
    
    # Run migrations or create tables
    from src.models import *  # Import all models
    async with db_manager._engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    
    yield
    
    # Cleanup
    await db_manager.close()

@pytest_asyncio.fixture
async def db_session(setup_database) -> AsyncGenerator[AsyncSession, None]:
    """Get a test database session with transaction rollback."""
    async with db_manager.transaction() as conn:
        # Start a transaction
        trans = conn.transaction()
        await trans.start()
        
        yield conn
        
        # Rollback the transaction
        await trans.rollback()

@pytest_asyncio.fixture
async def client(setup_database) -> AsyncGenerator[AsyncClient, None]:
    """Get a test client for the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac

@pytest_asyncio.fixture
async def authenticated_client(client: AsyncClient, test_user) -> AsyncClient:
    """Get an authenticated test client."""
    # Login to get token
    response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": test_user.email,
            "password": "testpassword123"
        }
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    
    # Set authorization header
    client.headers["Authorization"] = f"Bearer {token}"
    return client

# tests/test_users.py
import pytest
from uuid import uuid4

from src.repositories.user import UserRepository

@pytest.mark.asyncio
async def test_create_user(db_session):
    """Test user creation."""
    user_repo = UserRepository()
    
    user = await user_repo.create_user(
        email="test@example.com",
        username="testuser",
        password="securepassword123",
        full_name="Test User"
    )
    
    assert user.id is not None
    assert user.email == "test@example.com"
    assert user.username == "testuser"
    assert user.hashed_password != "securepassword123"  # Should be hashed

@pytest.mark.asyncio
async def test_bulk_create_performance(db_session):
    """Test bulk user creation performance."""
    user_repo = UserRepository()
    
    # Create 1000 test users
    users_data = [
        {
            "email": f"user{i}@example.com",
            "username": f"user{i}",
            "password": "password123",
            "full_name": f"User {i}"
        }
        for i in range(1000)
    ]
    
    import time
    start = time.perf_counter()
    
    created_users = await user_repo.bulk_create_users(users_data)
    
    elapsed = time.perf_counter() - start
    
    assert len(created_users) == 1000
    assert elapsed < 2.0  # Should complete in under 2 seconds
    
    # Verify data integrity
    assert all(user.id is not None for user in created_users)
    assert all(user.email.startswith("user") for user in created_users)

@pytest.mark.asyncio
async def test_parallel_queries(db_session):
    """Test PostgreSQL 17's parallel query capabilities."""
    import asyncio
    from src.repositories.advanced import AdvancedPatterns
    
    patterns = AdvancedPatterns()
    
    # Run multiple aggregation queries in parallel
    tasks = [
        patterns.parallel_aggregate_query("2024-01-01", "2024-12-31"),
        patterns.parallel_aggregate_query("2023-01-01", "2023-12-31"),
        patterns.parallel_aggregate_query("2022-01-01", "2022-12-31"),
    ]
    
    start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start
    
    assert len(results) == 3
    assert elapsed < 1.0  # Parallel execution should be fast

@pytest.mark.asyncio
async def test_api_user_endpoints(authenticated_client: AsyncClient):
    """Test user API endpoints."""
    # Get current user
    response = await authenticated_client.get("/api/v1/users/me")
    assert response.status_code == 200
    current_user = response.json()
    
    # List users
    response = await authenticated_client.get("/api/v1/users/")
    assert response.status_code == 200
    users = response.json()
    assert isinstance(users, list)
    
    # Search users
    response = await authenticated_client.get("/api/v1/users/?search=test")
    assert response.status_code == 200
    search_results = response.json()
    assert all("test" in user["username"].lower() or 
              "test" in user.get("full_name", "").lower() 
              for user in search_results)
```

---

## 10. Production Deployment Patterns

### ✅ DO: Use Gunicorn with Uvicorn Workers

```python
# gunicorn.conf.py
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '8007')}"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 10000
max_requests_jitter = 1000
timeout = 60
keepalive = 5

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'fastapi-postgres-app'

# Server mechanics
daemon = False
pidfile = '/tmp/fastapi-postgres-app.pid'
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
# keyfile = '/path/to/keyfile'
# certfile = '/path/to/certfile'

# Server hooks
def when_ready(server):
    server.log.info("Server is ready. Spawning workers")

def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def pre_exec(server):
    server.log.info("Forked child, re-executing.")

def on_exit(server):
    server.log.info("Shutting down: Master")
```

### Dockerfile for Production

```dockerfile
# Dockerfile
FROM python:3.13-slim-bookworm as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Final stage
FROM python:3.13-slim-bookworm

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser . .

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8007/health || exit 1

EXPOSE 8007

CMD ["gunicorn", "-c", "gunicorn.conf.py", "src.main:app"]
```

### Docker Compose for Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:17-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: fastapi_app
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    command: |
      postgres
      -c shared_buffers=256MB
      -c effective_cache_size=1GB
      -c maintenance_work_mem=64MB
      -c wal_compression=zstd
      -c max_connections=100
      -c random_page_cost=1.1
      -c effective_io_concurrency=200
      -c work_mem=4MB
      -c min_wal_size=1GB
      -c max_wal_size=4GB
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  app:
    build: .
    ports:
      - "8007:8007"
    environment:
      DATABASE_URL: postgresql+asyncpg://postgres:postgres@postgres:5432/fastapi_app
      REDIS_URL: redis://redis:6379
      ENVIRONMENT: development
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    volumes:
      - ./src:/app/src
      - ./alembic:/app/alembic
    command: |
      sh -c "
        alembic upgrade head &&
        uvicorn src.main:app --reload --host 0.0.0.0 --port 8007
      "

volumes:
  postgres_data:
  redis_data:
```

### Production Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-app
  labels:
    app: fastapi-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fastapi-app
  template:
    metadata:
      labels:
        app: fastapi-app
    spec:
      containers:
      - name: app
        image: your-registry/fastapi-app:latest
        ports:
        - containerPort: 8007
          name: http
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: secret-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - fastapi-app
              topologyKey: kubernetes.io/hostname
```

---

## 11. Monitoring, Observability, and Performance Optimization

### ✅ DO: Implement Comprehensive Monitoring

```python
# src/core/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time
import psutil
import asyncio

# Application metrics
app_info = Info('app_info', 'Application information')
app_info.info({
    'version': '0.1.0',
    'environment': settings.environment,
    'python_version': sys.version,
})

# HTTP metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# Database metrics
db_queries_total = Counter(
    'db_queries_total',
    'Total database queries',
    ['query_type', 'table']
)

db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['query_type', 'table']
)

db_pool_size = Gauge(
    'db_pool_size',
    'Database connection pool size'
)

db_pool_used = Gauge(
    'db_pool_used',
    'Database connections in use'
)

# System metrics
system_cpu_percent = Gauge('system_cpu_percent', 'System CPU usage')
system_memory_percent = Gauge('system_memory_percent', 'System memory usage')
system_disk_usage_percent = Gauge('system_disk_usage_percent', 'Disk usage')

# Background task to update system metrics
async def collect_system_metrics():
    """Collect system metrics every 15 seconds."""
    while True:
        try:
            # CPU usage
            system_cpu_percent.set(psutil.cpu_percent(interval=1))
            
            # Memory usage
            memory = psutil.virtual_memory()
            system_memory_percent.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            system_disk_usage_percent.set(disk.percent)
            
            # Database pool metrics
            if db_manager._pool:
                db_pool_size.set(db_manager._pool._size)
                db_pool_used.set(db_manager._pool._size - db_manager._pool._free_size)
            
            await asyncio.sleep(15)
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
            await asyncio.sleep(60)

# Decorator for tracking function metrics
def track_time(metric: Histogram):
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                metric.observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                metric.observe(duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
```

### Health Check Endpoints

```python
# src/routers/health.py
from fastapi import APIRouter, Depends
from typing import Dict
import asyncio
from datetime import datetime, timezone

from src.core.database import db_manager
from src.core.config import settings

router = APIRouter()

@router.get("/health")
async def health_check() -> Dict:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": settings.environment
    }

@router.get("/health/ready")
async def readiness_check() -> Dict:
    """Detailed readiness check for all dependencies."""
    checks = {}
    overall_status = "ready"
    
    # Check database
    try:
        async with asyncio.timeout(5):
            db_healthy = await db_manager.health_check()
            checks["database"] = "healthy" if db_healthy else "unhealthy"
    except Exception as e:
        checks["database"] = "unhealthy"
        overall_status = "not ready"
    
    # Check Redis (if applicable)
    try:
        # Add Redis health check here
        checks["redis"] = "healthy"
    except Exception:
        checks["redis"] = "unhealthy"
    
    # Check disk space
    import shutil
    stat = shutil.disk_usage("/")
    disk_percent = (stat.used / stat.total) * 100
    checks["disk_space"] = "healthy" if disk_percent < 90 else "unhealthy"
    
    return {
        "status": overall_status,
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@router.get("/health/startup")
async def startup_check() -> Dict:
    """Startup probe for Kubernetes."""
    # Simple check that the app has started
    return {"status": "started"}
```

---

## 12. Security Best Practices

### ✅ DO: Implement Comprehensive Security

```python
# src/core/security.py
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import secrets
from passlib.context import CryptContext
from jose import jwt, JWTError
import structlog

from src.core.config import settings

logger = structlog.get_logger()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
ALGORITHM = "HS256"

def create_access_token(
    subject: str,
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[Dict[str, Any]] = None
) -> str:
    """Create a JWT token with optional additional claims."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.access_token_expire_minutes
        )
    
    to_encode = {
        "exp": expire,
        "sub": subject,
        "iat": datetime.now(timezone.utc),
        "jti": secrets.token_urlsafe(16),  # JWT ID for token revocation
    }
    
    if additional_claims:
        to_encode.update(additional_claims)
    
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        logger.warning("Invalid JWT token", error=str(e))
        return None

# Rate limiting
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, calls: int = 10, period: timedelta = timedelta(minutes=1)):
        self.calls = calls
        self.period = period
        self.calls_made = defaultdict(list)
        self._cleanup_task = None
    
    async def __call__(self, key: str) -> bool:
        """Check if request is allowed."""
        now = datetime.now(timezone.utc)
        
        # Clean old entries
        cutoff = now - self.period
        self.calls_made[key] = [
            call_time for call_time in self.calls_made[key]
            if call_time > cutoff
        ]
        
        # Check rate limit
        if len(self.calls_made[key]) >= self.calls:
            return False
        
        # Record call
        self.calls_made[key].append(now)
        return True
    
    async def start_cleanup(self):
        """Start background cleanup task."""
        while True:
            await asyncio.sleep(300)  # Clean every 5 minutes
            cutoff = datetime.now(timezone.utc) - self.period
            
            # Remove old entries
            keys_to_remove = []
            for key, calls in self.calls_made.items():
                self.calls_made[key] = [
                    call_time for call_time in calls
                    if call_time > cutoff
                ]
                if not self.calls_made[key]:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.calls_made[key]

# Create rate limiter instances
login_rate_limiter = RateLimiter(calls=5, period=timedelta(minutes=15))
api_rate_limiter = RateLimiter(calls=100, period=timedelta(minutes=1))
```

### Security Headers Middleware

```python
# src/api/security_middleware.py
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Remove server header
        response.headers.pop("server", None)
        
        return response
```

---

## 13. Common Pitfalls and How to Avoid Them

### ❌ DON'T: Use Synchronous Database Drivers in Async Code

```python
# Bad - Blocks the event loop
import psycopg2  # Synchronous driver

def get_user_sync(user_id: int):
    conn = psycopg2.connect(database_url)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    return cursor.fetchone()

# Good - Non-blocking async
async def get_user_async(user_id: int):
    async with db_manager.acquire() as conn:
        return await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
```

### ❌ DON'T: Forget to Handle Connection Pool Exhaustion

```python
# Bad - No timeout or error handling
async def bad_long_running_task():
    conn = await db_manager.pool.acquire()
    # If this fails, connection is never released
    await some_operation()
    await conn.close()

# Good - Use context manager with timeout
async def good_long_running_task():
    async with asyncio.timeout(30):
        async with db_manager.acquire() as conn:
            await some_operation()
```

### ❌ DON'T: Use ORM Lazy Loading in Async Context

```python
# Bad - Causes errors with async SQLAlchemy
user = await session.get(User, user_id)
posts = user.posts  # Lazy loading fails in async

# Good - Eager load relationships
from sqlalchemy.orm import selectinload

stmt = select(User).options(selectinload(User.posts)).where(User.id == user_id)
result = await session.execute(stmt)
user = result.scalar_one()
posts = user.posts  # Already loaded
```

---

## Conclusion

This comprehensive guide provides a production-ready foundation for building high-performance applications with PostgreSQL 17, FastAPI, and modern Python async patterns. The combination of PostgreSQL 17's performance improvements, FastAPI's speed, and proper async patterns creates a powerful stack for modern applications.

### Key Takeaways:

1. **PostgreSQL 17** brings significant performance improvements that should be leveraged
2. **Choose the right async driver** (asyncpg for performance, psycopg3 for compatibility)
3. **Use SQLModel** for clean, type-safe models while maintaining flexibility
4. **Implement proper connection pooling** and lifecycle management
5. **Monitor everything** - you can't optimize what you don't measure
6. **Test async code properly** with appropriate fixtures and patterns
7. **Deploy with scalability in mind** using proper containerization and orchestration

### Next Steps:

- Implement caching with Redis for frequently accessed data
- Add background task processing with Celery or Arq
- Integrate with observability platforms (Datadog, New Relic, etc.)
- Implement API versioning for backward compatibility
- Add comprehensive API documentation with examples

Remember: Performance is not just about speed—it's about reliability, maintainability, and scalability. This guide provides patterns that achieve all three.

---

## Final Checklist to Keep Your Guide Evergreen

1. **Track minor releases** – Subscribe to the PGDG RSS feed or set up `apt-get --just-print upgrade` in CI to monitor PostgreSQL updates
2. **Pin SQLAlchemy/SQLModel/Pydantic** as a coherent trio (`sqlalchemy>=2.0.41`, `sqlmodel>=0.0.25`, `pydantic>=2.11`)
3. **Benchmark pgvector indexes** – Tune `lists` & `probes` parameters per workload for optimal HNSW performance
4. **Observe first, then tune** – Use `EXPLAIN (MEMORY,SERIALIZE)` and `pg_stat_io` to make invisible costs visible
5. **Enable psycopg3 pipeline mode** for batch operations to reduce round-trip latency
6. **Set max_overflow=0** in connection pools and monitor `.pool_status()` to prevent exhaustion
7. **Apply PostgreSQL minor updates within 7 days** – They often contain critical security fixes and performance improvements

These updates reflect the newest community wisdom and production experiences as of mid-2025, ensuring your stack remains performant, secure, and maintainable.
