# The Definitive Guide to High-Performance Rust Web APIs with Axum, SeaORM, and PostgreSQL (Late 2025 Edition)

This guide synthesizes battle-tested patterns for building production-grade, high-performance web APIs in Rust. It leverages the ecosystem's most mature libraries while avoiding common pitfalls that plague Rust web services at scale.

## Prerequisites & Core Dependencies

Your project should target **Rust 1.85+** (stable channel with Rust 2024 edition), using **Axum 0.8+**, **SeaORM 1.2+**, and **PostgreSQL 16+**. The async runtime is **Tokio 1.45+** with all features enabled.

### Base `Cargo.toml` Configuration

```toml
[package]
name = "api-service"
version = "0.1.0"
edition = "2024"
rust-version = "1.85"

[dependencies]
# Web Framework
axum = { version = "0.8", features = ["macros", "ws", "multipart"] }
axum-extra = { version = "0.10", features = ["typed-header", "cookie", "form"] }
tower = { version = "0.5", features = ["full"] }
tower-http = { version = "0.6", features = ["fs", "cors", "compression-full", "trace", "timeout", "limit", "request-id", "sensitive-headers"] }
hyper = { version = "1.5", features = ["full"] }
hyper-util = "0.1"

# Async Runtime
tokio = { version = "1.45", features = ["full"] }
tokio-util = { version = "0.7", features = ["io"] }

# Database
sea-orm = { version = "1.2", features = ["sqlx-postgres", "runtime-tokio-rustls", "macros", "mock", "with-chrono", "with-json", "with-uuid"] }
sea-orm-migration = "1.2"
sqlx = { version = "0.8", features = ["runtime-tokio-rustls", "postgres", "uuid", "chrono", "json"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
serde_with = "3.12"

# Validation
validator = { version = "0.19", features = ["derive"] }
garde = { version = "0.21", features = ["derive", "email", "url"] }

# Error Handling
thiserror = "2.0"
anyhow = "1.0"
color-eyre = "0.6"

# Logging & Tracing
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json", "fmt", "ansi"] }
tracing-opentelemetry = "0.28"
opentelemetry = { version = "0.27", features = ["trace", "metrics"] }
opentelemetry-otlp = { version = "0.27", features = ["tonic", "trace", "metrics"] }
opentelemetry_sdk = { version = "0.27", features = ["rt-tokio"] }

# Security & Auth
argon2 = "0.5"
jsonwebtoken = "9.3"
uuid = { version = "1.11", features = ["v4", "v7", "serde"] }
ring = "0.17"
rustls = "0.23"

# Configuration
config = { version = "0.14", features = ["toml"] }
dotenvy = "0.15"

# Utilities
chrono = { version = "0.4", features = ["serde"] }
regex = "1.11"
once_cell = "1.20"
arc-swap = "1.7"
bytes = "1.9"
futures = "0.3"
pin-project = "1.1"
dashmap = "6.1"

# HTTP Client
reqwest = { version = "0.12", features = ["rustls-tls", "json", "stream", "multipart"] }

# Development Dependencies
[dev-dependencies]
mockall = "0.13"
fake = { version = "2.10", features = ["derive"] }
proptest = "1.6"
test-case = "3.3"
wiremock = "0.6"
testcontainers = { version = "0.24", features = ["postgres"] }
criterion = { version = "0.6", features = ["html_reports", "async_tokio"] }
insta = { version = "1.42", features = ["json", "yaml"] }
rstest = "0.24"

# Build optimizations
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
panic = "abort"

[profile.release-with-debug]
inherits = "release"
strip = false
debug = true

# Faster builds in development
[profile.dev]
opt-level = 0

[profile.dev.package."*"]
opt-level = 3
```

### Workspace Configuration for Larger Projects

```toml
# Cargo.toml (workspace root)
[workspace]
members = ["api", "entity", "migration", "common", "jobs"]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2024"
rust-version = "1.85"
authors = ["Your Team <team@example.com>"]

[workspace.dependencies]
# Define versions once, use everywhere
axum = "0.8"
sea-orm = "1.2"
tokio = { version = "1.45", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
```

## 1. Project Architecture & Code Organization

### ✅ DO: Use a Domain-Driven Layered Architecture

This structure scales with your team and clearly separates concerns:

```
/src
├── main.rs              # Application entry point
├── config.rs            # Configuration structs and loading
├── error.rs             # Global error types and handling
├── state.rs             # Application state definition
├── routes/              # HTTP route definitions
│   ├── mod.rs          # Route registration
│   ├── health.rs       # Health check endpoints
│   ├── auth.rs         # Authentication endpoints
│   └── users.rs        # User management endpoints
├── handlers/            # Request handlers (controllers)
│   ├── mod.rs
│   ├── auth.rs
│   └── users.rs
├── services/            # Business logic layer
│   ├── mod.rs
│   ├── auth.rs
│   ├── user.rs
│   └── email.rs
├── repositories/        # Data access layer
│   ├── mod.rs
│   ├── user.rs
│   └── session.rs
├── models/              # Domain models
│   ├── mod.rs
│   ├── user.rs
│   └── auth.rs
├── middleware/          # Custom middleware
│   ├── mod.rs
│   ├── auth.rs
│   ├── logging.rs
│   └── rate_limit.rs
├── utils/               # Shared utilities
│   ├── mod.rs
│   ├── validation.rs
│   └── crypto.rs
└── telemetry.rs         # Observability setup
```

### ✅ DO: Separate Entity Definitions in a Workspace

For medium to large projects, use a workspace structure:

```
workspace/
├── Cargo.toml           # Workspace definition
├── api/                 # Main API service
│   ├── Cargo.toml
│   └── src/
├── entity/              # SeaORM entities
│   ├── Cargo.toml
│   └── src/
├── migration/           # Database migrations
│   ├── Cargo.toml
│   └── src/
└── common/              # Shared utilities
    ├── Cargo.toml
    └── src/
```

## 2. Application State & Dependency Injection

### ✅ DO: Use a Single, Cloneable Application State

Axum's state management works best with a single `Arc<AppState>` that contains all shared resources:

```rust
// src/state.rs
use std::sync::Arc;
use arc_swap::ArcSwap;
use sea_orm::DatabaseConnection;
use tokio::sync::RwLock;
use dashmap::DashMap;

#[derive(Clone)]
pub struct AppState {
    pub db: DatabaseConnection,
    pub config: Arc<Config>,
    pub http_client: reqwest::Client,
    pub cache: Arc<DashMap<String, CachedValue>>,
    pub rate_limiter: Arc<RateLimiter>,
    pub jwt_keys: Arc<JwtKeys>,
    // Use ArcSwap for values that change at runtime
    pub feature_flags: Arc<ArcSwap<FeatureFlags>>,
}

impl AppState {
    pub async fn new(config: Config) -> Result<Self> {
        // Initialize database connection with optimized pool settings
        let db = sea_orm::Database::connect(&config.database.url)
            .await?
            .max_connections(config.database.max_connections)
            .min_connections(config.database.min_connections)
            .connect_timeout(Duration::from_secs(5))
            .idle_timeout(Duration::from_secs(600))
            .max_lifetime(Duration::from_secs(1800))
            .sqlx_logging(false) // Disable SQLx logs, use SeaORM's
            .build()?;
        
        // HTTP client with connection pooling
        let http_client = reqwest::Client::builder()
            .pool_max_idle_per_host(32)
            .timeout(Duration::from_secs(30))
            .build()?;
        
        Ok(Self {
            db,
            config: Arc::new(config),
            http_client,
            cache: Arc::new(DashMap::with_capacity(10_000)),
            rate_limiter: Arc::new(RateLimiter::new()),
            jwt_keys: Arc::new(JwtKeys::from_config(&config)?),
            feature_flags: Arc::new(ArcSwap::from_pointee(FeatureFlags::default())),
        })
    }
}
```

### ❌ DON'T: Create Multiple State Types or Use Globals

```rust
// Bad - Multiple states lead to confusion
let app = Router::new()
    .with_state(db_state)
    .with_state(config_state)
    .with_state(cache_state);

// Bad - Global state with lazy_static or once_cell
lazy_static! {
    static ref DB: DatabaseConnection = ...;
}
```

## 3. Error Handling: The Foundation of Reliability

### ✅ DO: Create a Unified Error Type with Rich Context

```rust
// src/error.rs
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Database error")]
    Database(#[from] sea_orm::DbErr),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Authentication required")]
    Unauthorized,
    
    #[error("Insufficient permissions")]
    Forbidden,
    
    #[error("Resource not found: {0}")]
    NotFound(String),
    
    #[error("Conflict: {0}")]
    Conflict(String),
    
    #[error("Rate limit exceeded")]
    TooManyRequests,
    
    #[error("External service error: {0}")]
    ExternalService(String),
    
    #[error("Internal server error")]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message, error_code) = match self {
            AppError::Database(ref e) => {
                tracing::error!("Database error: {:?}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error", "DB_ERROR")
            }
            AppError::Validation(ref msg) => {
                (StatusCode::BAD_REQUEST, msg.as_str(), "VALIDATION_ERROR")
            }
            AppError::Unauthorized => {
                (StatusCode::UNAUTHORIZED, "Authentication required", "UNAUTHORIZED")
            }
            AppError::Forbidden => {
                (StatusCode::FORBIDDEN, "Insufficient permissions", "FORBIDDEN")
            }
            AppError::NotFound(ref resource) => {
                (StatusCode::NOT_FOUND, resource.as_str(), "NOT_FOUND")
            }
            AppError::Conflict(ref msg) => {
                (StatusCode::CONFLICT, msg.as_str(), "CONFLICT")
            }
            AppError::TooManyRequests => {
                (StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded", "RATE_LIMITED")
            }
            AppError::ExternalService(ref service) => {
                tracing::error!("External service error: {}", service);
                (StatusCode::BAD_GATEWAY, "External service unavailable", "EXTERNAL_ERROR")
            }
            AppError::Internal(ref e) => {
                tracing::error!("Internal error: {:?}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error", "INTERNAL_ERROR")
            }
        };

        let body = Json(json!({
            "error": {
                "code": error_code,
                "message": error_message,
                "request_id": RequestId::get(),
            }
        }));

        (status, body).into_response()
    }
}

// Convenience type alias
pub type Result<T> = std::result::Result<T, AppError>;
```

### ✅ DO: Use the `?` Operator Liberally with Context

```rust
use anyhow::Context;

async fn get_user_by_email(db: &DatabaseConnection, email: &str) -> Result<User> {
    let user = Users::find()
        .filter(users::Column::Email.eq(email))
        .one(db)
        .await
        .context("Failed to query user by email")?
        .ok_or_else(|| AppError::NotFound(format!("User with email {} not found", email)))?;
    
    Ok(user)
}
```

## 4. Database Patterns with SeaORM

### ✅ DO: Use SeaORM's Migration System

```rust
// migration/src/m20250101_000001_create_users_table.rs
use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(Users::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(Users::Id)
                            .uuid()
                            .not_null()
                            .primary_key()
                            .default(Expr::cust("gen_random_uuid()"))
                    )
                    .col(
                        ColumnDef::new(Users::Email)
                            .string()
                            .not_null()
                            .unique_key()
                    )
                    .col(
                        ColumnDef::new(Users::PasswordHash)
                            .string()
                            .not_null()
                    )
                    .col(
                        ColumnDef::new(Users::CreatedAt)
                            .timestamp_with_time_zone()
                            .not_null()
                            .default(Expr::current_timestamp())
                    )
                    .col(
                        ColumnDef::new(Users::UpdatedAt)
                            .timestamp_with_time_zone()
                            .not_null()
                            .default(Expr::current_timestamp())
                    )
                    .to_owned(),
            )
            .await?;

        // Create indexes for common queries
        manager
            .create_index(
                Index::create()
                    .name("idx_users_email")
                    .table(Users::Table)
                    .col(Users::Email)
                    .to_owned(),
            )
            .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(Users::Table).to_owned())
            .await
    }
}

#[derive(Iden)]
enum Users {
    Table,
    Id,
    Email,
    PasswordHash,
    CreatedAt,
    UpdatedAt,
}
```

### ✅ DO: Use Repository Pattern for Complex Queries

```rust
// src/repositories/user.rs
use sea_orm::*;
use crate::entities::{users, prelude::*};

pub struct UserRepository<'a> {
    db: &'a DatabaseConnection,
}

impl<'a> UserRepository<'a> {
    pub fn new(db: &'a DatabaseConnection) -> Self {
        Self { db }
    }

    pub async fn find_by_email(&self, email: &str) -> Result<Option<users::Model>> {
        Users::find()
            .filter(users::Column::Email.eq(email))
            .one(self.db)
            .await
    }

    pub async fn find_active_with_roles(&self, user_id: Uuid) -> Result<Option<UserWithRoles>> {
        let user = Users::find_by_id(user_id)
            .filter(users::Column::IsActive.eq(true))
            .find_with_related(UserRoles)
            .all(self.db)
            .await?
            .into_iter()
            .next()
            .map(|(user, roles)| UserWithRoles { user, roles });
        
        Ok(user)
    }

    pub async fn create_with_transaction(&self, email: String, password_hash: String) -> Result<users::Model> {
        let txn = self.db.begin().await?;
        
        let user = users::ActiveModel {
            email: Set(email),
            password_hash: Set(password_hash),
            ..Default::default()
        };
        
        let user = user.insert(&txn).await?;
        
        // Perform other operations in the same transaction
        UserAuditLog::create(&txn, &user, "USER_CREATED").await?;
        
        txn.commit().await?;
        Ok(user)
    }

    // Optimized batch operations
    pub async fn update_last_login_batch(&self, user_ids: &[Uuid]) -> Result<u64> {
        let result = Users::update_many()
            .col_expr(users::Column::LastLoginAt, Expr::current_timestamp())
            .filter(users::Column::Id.is_in(user_ids))
            .exec(self.db)
            .await?;
        
        Ok(result.rows_affected)
    }
}
```

### ✅ DO: Use Database Transactions for Data Integrity

```rust
// src/services/user.rs
pub async fn transfer_ownership(
    db: &DatabaseConnection,
    from_user_id: Uuid,
    to_user_id: Uuid,
    resource_ids: Vec<Uuid>,
) -> Result<()> {
    let txn = db.begin().await?;
    
    // Verify both users exist and are active
    let from_user = Users::find_by_id(from_user_id)
        .filter(users::Column::IsActive.eq(true))
        .one(&txn)
        .await?
        .ok_or_else(|| AppError::NotFound("Source user not found".into()))?;
    
    let to_user = Users::find_by_id(to_user_id)
        .filter(users::Column::IsActive.eq(true))
        .one(&txn)
        .await?
        .ok_or_else(|| AppError::NotFound("Target user not found".into()))?;
    
    // Update all resources atomically
    let update_result = Resources::update_many()
        .col_expr(resources::Column::OwnerId, Expr::value(to_user_id))
        .col_expr(resources::Column::UpdatedAt, Expr::current_timestamp())
        .filter(resources::Column::Id.is_in(resource_ids.clone()))
        .filter(resources::Column::OwnerId.eq(from_user_id))
        .exec(&txn)
        .await?;
    
    if update_result.rows_affected != resource_ids.len() as u64 {
        return Err(AppError::Conflict("Some resources could not be transferred".into()));
    }
    
    // Audit log
    let audit_entry = audit_logs::ActiveModel {
        user_id: Set(from_user_id),
        action: Set("OWNERSHIP_TRANSFERRED".to_string()),
        details: Set(json!({
            "to_user_id": to_user_id,
            "resource_count": resource_ids.len(),
            "resource_ids": resource_ids,
        })),
        ..Default::default()
    };
    audit_entry.insert(&txn).await?;
    
    txn.commit().await?;
    Ok(())
}
```

## 5. High-Performance API Design with Axum

### ✅ DO: Use Extractors for Clean Handler Functions

```rust
// src/handlers/users.rs
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use garde::Validate;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Deserialize, Validate)]
pub struct CreateUserRequest {
    #[garde(email)]
    pub email: String,
    #[garde(length(min = 8, max = 128))]
    pub password: String,
    #[garde(length(min = 1, max = 100))]
    pub name: String,
}

#[derive(Debug, Serialize)]
pub struct UserResponse {
    pub id: Uuid,
    pub email: String,
    pub name: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Deserialize, Validate)]
pub struct ListUsersQuery {
    #[garde(range(min = 1, max = 100))]
    pub limit: Option<u64>,
    #[garde(range(min = 0))]
    pub offset: Option<u64>,
    pub search: Option<String>,
}

// Clean handler with multiple extractors
pub async fn create_user(
    State(state): State<AppState>,
    ValidatedJson(payload): ValidatedJson<CreateUserRequest>,
) -> Result<impl IntoResponse> {
    // Validation is already done by ValidatedJson extractor
    let user_service = UserService::new(&state);
    let user = user_service.create_user(payload).await?;
    
    Ok((StatusCode::CREATED, Json(UserResponse::from(user))))
}

pub async fn list_users(
    State(state): State<AppState>,
    Query(params): Query<ListUsersQuery>,
    Extension(current_user): Extension<CurrentUser>,
) -> Result<Json<Vec<UserResponse>>> {
    // Validate query parameters
    params.validate(&()).map_err(|e| AppError::Validation(e.to_string()))?;
    
    let user_repo = UserRepository::new(&state.db);
    let users = user_repo
        .find_paginated(params.limit.unwrap_or(20), params.offset.unwrap_or(0))
        .await?;
    
    Ok(Json(users.into_iter().map(UserResponse::from).collect()))
}
```

### ✅ DO: Create Custom Extractors for Common Patterns

```rust
// src/extractors/mod.rs
use axum::{
    async_trait,
    extract::{FromRequestParts, FromRequest},
    http::{request::Parts, Request},
    RequestPartsExt,
};
use garde::Validate;

// Custom JSON extractor with validation
pub struct ValidatedJson<T>(pub T);

#[async_trait]
impl<T, S> FromRequest<S> for ValidatedJson<T>
where
    T: DeserializeOwned + Validate,
    S: Send + Sync,
{
    type Rejection = AppError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let Json(value) = Json::<T>::from_request(req, state)
            .await
            .map_err(|_| AppError::Validation("Invalid JSON".into()))?;
        
        value.validate(&()).map_err(|e| AppError::Validation(e.to_string()))?;
        
        Ok(ValidatedJson(value))
    }
}

// Authenticated user extractor
#[derive(Clone)]
pub struct CurrentUser {
    pub id: Uuid,
    pub email: String,
    pub roles: Vec<String>,
}

#[async_trait]
impl<S> FromRequestParts<S> for CurrentUser
where
    S: Send + Sync,
{
    type Rejection = AppError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let Extension(user) = parts
            .extract::<Extension<CurrentUser>>()
            .await
            .map_err(|_| AppError::Unauthorized)?;
        
        Ok(user)
    }
}
```

## 6. Authentication & Authorization

### ✅ DO: Use JWT with Refresh Tokens

```rust
// src/services/auth.rs
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: Uuid,        // user_id
    pub email: String,
    pub roles: Vec<String>,
    pub exp: i64,         // expiration
    pub iat: i64,         // issued at
    pub jti: Uuid,        // JWT ID for revocation
}

pub struct AuthService {
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    access_token_duration: Duration,
    refresh_token_duration: Duration,
}

impl AuthService {
    pub fn generate_token_pair(&self, user: &User) -> Result<TokenPair> {
        let now = Utc::now();
        let access_exp = now + self.access_token_duration;
        let refresh_exp = now + self.refresh_token_duration;
        
        let access_claims = Claims {
            sub: user.id,
            email: user.email.clone(),
            roles: user.roles.clone(),
            exp: access_exp.timestamp(),
            iat: now.timestamp(),
            jti: Uuid::new_v4(),
        };
        
        let access_token = encode(&Header::default(), &access_claims, &self.encoding_key)?;
        
        let refresh_claims = Claims {
            sub: user.id,
            email: user.email.clone(),
            roles: vec!["refresh".to_string()], // Minimal claims for refresh
            exp: refresh_exp.timestamp(),
            iat: now.timestamp(),
            jti: Uuid::new_v4(),
        };
        
        let refresh_token = encode(&Header::default(), &refresh_claims, &self.encoding_key)?;
        
        // Store refresh token in database for revocation support
        RefreshTokens::insert(refresh_tokens::ActiveModel {
            jti: Set(refresh_claims.jti),
            user_id: Set(user.id),
            expires_at: Set(refresh_exp.naive_utc()),
            ..Default::default()
        })
        .exec(&self.db)
        .await?;
        
        Ok(TokenPair {
            access_token,
            refresh_token,
            expires_in: self.access_token_duration.num_seconds(),
        })
    }
}
```

### ✅ DO: Implement Secure Password Handling

```rust
// src/utils/crypto.rs
use argon2::{
    password_hash::{rand_core::OsRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2,
};

pub struct PasswordService {
    argon2: Argon2<'static>,
}

impl PasswordService {
    pub fn new() -> Self {
        // Configure Argon2 with secure defaults
        let argon2 = Argon2::new(
            argon2::Algorithm::Argon2id,
            argon2::Version::V0x13,
            argon2::Params::new(19456, 2, 1, None).unwrap(),
        );
        
        Self { argon2 }
    }
    
    pub fn hash_password(&self, password: &str) -> Result<String> {
        let salt = SaltString::generate(&mut OsRng);
        let password_hash = self.argon2
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Failed to hash password: {}", e)))?;
        
        Ok(password_hash.to_string())
    }
    
    pub fn verify_password(&self, password: &str, hash: &str) -> Result<bool> {
        let parsed_hash = PasswordHash::new(hash)
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Invalid password hash: {}", e)))?;
        
        Ok(self.argon2.verify_password(password.as_bytes(), &parsed_hash).is_ok())
    }
}
```

## 7. Middleware & Request Processing

### ✅ DO: Layer Middleware in the Correct Order

```rust
// src/routes/mod.rs
use axum::{Router, middleware};
use tower_http::{
    compression::CompressionLayer,
    cors::CorsLayer,
    limit::RequestBodyLimitLayer,
    timeout::TimeoutLayer,
    trace::{DefaultMakeSpan, DefaultOnRequest, DefaultOnResponse, TraceLayer},
};

pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(state.config.cors.allowed_origins.clone())
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
        .max_age(Duration::from_secs(3600));

    Router::new()
        .merge(auth_routes())
        .merge(user_routes())
        .merge(health_routes())
        // The order matters! Apply in this sequence:
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
                .on_request(DefaultOnRequest::new().level(Level::INFO))
                .on_response(DefaultOnResponse::new().level(Level::INFO))
        )
        .layer(TimeoutLayer::new(Duration::from_secs(30)))
        .layer(CompressionLayer::new())
        .layer(RequestBodyLimitLayer::new(10 * 1024 * 1024)) // 10MB
        .layer(cors)
        .layer(middleware::from_fn_with_state(state.clone(), rate_limit_middleware))
        .layer(middleware::from_fn(request_id_middleware))
        .with_state(state)
}
```

### ✅ DO: Implement Smart Rate Limiting

```rust
// src/middleware/rate_limit.rs
use axum::{
    extract::{State, ConnectInfo},
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};
use std::net::SocketAddr;
use governor::{Quota, RateLimiter as Gov};

pub async fn rate_limit_middleware<B>(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    request: Request<B>,
    next: Next<B>,
) -> Result<Response, StatusCode> {
    let ip = addr.ip().to_string();
    
    // Check if IP is rate limited
    let rate_limiter = &state.rate_limiter;
    let key = format!("rate_limit:{}", ip);
    
    match rate_limiter.check_key(&key) {
        Ok(_) => Ok(next.run(request).await),
        Err(_) => {
            let retry_after = rate_limiter.time_until_ready(&key);
            let mut response = (
                StatusCode::TOO_MANY_REQUESTS,
                "Rate limit exceeded",
            ).into_response();
            
            response.headers_mut().insert(
                "Retry-After",
                retry_after.as_secs().to_string().parse().unwrap(),
            );
            
            Ok(response)
        }
    }
}

// Advanced rate limiter with different tiers
pub struct TieredRateLimiter {
    anonymous: Arc<Gov<String, DefaultKeyedStateStore<String>>>,
    authenticated: Arc<Gov<String, DefaultKeyedStateStore<String>>>,
    premium: Arc<Gov<String, DefaultKeyedStateStore<String>>>,
}

impl TieredRateLimiter {
    pub fn new() -> Self {
        Self {
            anonymous: Arc::new(Gov::new(
                Quota::per_minute(NonZeroU32::new(30).unwrap()),
                DefaultKeyedStateStore::default(),
            )),
            authenticated: Arc::new(Gov::new(
                Quota::per_minute(NonZeroU32::new(100).unwrap()),
                DefaultKeyedStateStore::default(),
            )),
            premium: Arc::new(Gov::new(
                Quota::per_minute(NonZeroU32::new(1000).unwrap()),
                DefaultKeyedStateStore::default(),
            )),
        }
    }
}
```

## 8. Async Patterns & Performance Optimization

### ✅ DO: Use Async Streams for Large Data Sets

```rust
// src/handlers/export.rs
use axum::response::sse::{Event, Sse};
use futures::stream::{Stream, StreamExt};
use tokio_stream::wrappers::IntervalStream;

pub async fn export_users_stream(
    State(state): State<AppState>,
    Extension(user): Extension<CurrentUser>,
) -> Sse<impl Stream<Item = Result<Event, std::io::Error>>> {
    let stream = async_stream::stream! {
        let mut offset = 0u64;
        let limit = 1000u64;
        
        loop {
            match Users::find()
                .order_by_asc(users::Column::Id)
                .limit(limit)
                .offset(offset)
                .all(&state.db)
                .await
            {
                Ok(users) if users.is_empty() => break,
                Ok(users) => {
                    for user in users {
                        yield Ok(Event::default()
                            .event("user")
                            .json_data(UserExportDto::from(user))
                            .unwrap());
                    }
                    offset += limit;
                }
                Err(e) => {
                    yield Ok(Event::default()
                        .event("error")
                        .data(format!("Error: {}", e)));
                    break;
                }
            }
            
            // Prevent overwhelming the client
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        
        yield Ok(Event::default().event("complete").data(""));
    };
    
    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(30))
            .text("keep-alive"),
    )
}
```

### ✅ DO: Use Concurrent Processing with Controlled Parallelism

```rust
// src/services/batch_processor.rs
use futures::{stream, StreamExt};
use tokio::sync::Semaphore;

pub struct BatchProcessor {
    semaphore: Arc<Semaphore>,
    max_concurrent: usize,
}

impl BatchProcessor {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            max_concurrent,
        }
    }
    
    pub async fn process_users_batch<F, Fut>(
        &self,
        user_ids: Vec<Uuid>,
        processor: F,
    ) -> Vec<Result<ProcessResult>>
    where
        F: Fn(Uuid) -> Fut + Clone,
        Fut: Future<Output = Result<ProcessResult>>,
    {
        stream::iter(user_ids)
            .map(|user_id| {
                let semaphore = self.semaphore.clone();
                let processor = processor.clone();
                
                async move {
                    let _permit = semaphore.acquire().await.unwrap();
                    processor(user_id).await
                }
            })
            .buffer_unordered(self.max_concurrent)
            .collect()
            .await
    }
}

// Usage
let processor = BatchProcessor::new(10); // Max 10 concurrent operations
let results = processor.process_users_batch(user_ids, |user_id| async move {
    // Process each user
    expensive_operation(user_id).await
}).await;
```

### ✅ DO: Implement Connection Pooling for External Services

```rust
// src/services/external_api.rs
use bb8::{Pool, RunError};
use bb8_redis::RedisConnectionManager;

pub struct ExternalApiClient {
    http_client: reqwest::Client,
    redis_pool: Pool<RedisConnectionManager>,
    circuit_breaker: Arc<CircuitBreaker>,
}

impl ExternalApiClient {
    pub async fn call_with_cache<T>(&self, endpoint: &str) -> Result<T>
    where
        T: DeserializeOwned + Serialize,
    {
        let cache_key = format!("api_cache:{}", endpoint);
        
        // Try cache first
        if let Ok(cached) = self.get_from_cache::<T>(&cache_key).await {
            return Ok(cached);
        }
        
        // Check circuit breaker
        if !self.circuit_breaker.is_closed() {
            return Err(AppError::ExternalService("Service unavailable".into()));
        }
        
        // Make the actual call with retry logic
        let response = self.call_with_retry(endpoint, 3).await?;
        
        // Cache the successful response
        self.set_cache(&cache_key, &response, 300).await?;
        
        Ok(response)
    }
    
    async fn call_with_retry<T>(&self, endpoint: &str, max_retries: u32) -> Result<T>
    where
        T: DeserializeOwned,
    {
        let mut retries = 0;
        let mut last_error = None;
        
        while retries < max_retries {
            match self.http_client
                .get(endpoint)
                .timeout(Duration::from_secs(10))
                .send()
                .await
            {
                Ok(response) if response.status().is_success() => {
                    self.circuit_breaker.record_success();
                    return response.json::<T>().await
                        .map_err(|e| AppError::ExternalService(e.to_string()));
                }
                Ok(response) => {
                    let status = response.status();
                    if status.is_server_error() && retries < max_retries - 1 {
                        // Retry on 5xx errors
                        retries += 1;
                        let backoff = Duration::from_millis(100 * 2u64.pow(retries));
                        tokio::time::sleep(backoff).await;
                        continue;
                    }
                    self.circuit_breaker.record_failure();
                    return Err(AppError::ExternalService(format!("API returned {}", status)));
                }
                Err(e) => {
                    last_error = Some(e);
                    if retries < max_retries - 1 {
                        retries += 1;
                        let backoff = Duration::from_millis(100 * 2u64.pow(retries));
                        tokio::time::sleep(backoff).await;
                        continue;
                    }
                    self.circuit_breaker.record_failure();
                }
            }
        }
        
        Err(AppError::ExternalService(
            last_error.map(|e| e.to_string()).unwrap_or_else(|| "Unknown error".into())
        ))
    }
}
```

## 9. Testing Strategies

### ✅ DO: Use Test Containers for Integration Tests

```rust
// tests/common/mod.rs
use testcontainers::{clients, images::postgres::Postgres, Container};
use sea_orm::{Database, DatabaseConnection};

pub struct TestContext<'a> {
    pub db: DatabaseConnection,
    pub app: Router,
    _container: Container<'a, Postgres>,
}

pub async fn setup_test_context<'a>(
    docker: &'a clients::Cli,
) -> TestContext<'a> {
    // Start PostgreSQL container
    let container = docker.run(Postgres::default());
    let port = container.get_host_port_ipv4(5432);
    
    let database_url = format!(
        "postgres://postgres:postgres@localhost:{}/postgres",
        port
    );
    
    // Connect and run migrations
    let db = Database::connect(&database_url).await.unwrap();
    Migrator::up(&db, None).await.unwrap();
    
    // Create test app state
    let config = Config::test_default();
    let state = AppState::new_with_db(config, db.clone()).await.unwrap();
    let app = create_router(state);
    
    TestContext { db, app, _container: container }
}

// tests/api/users_test.rs
use axum::http::{StatusCode, header};
use tower::ServiceExt;

#[tokio::test]
async fn test_create_user() {
    let docker = clients::Cli::default();
    let ctx = setup_test_context(&docker).await;
    
    let payload = json!({
        "email": "test@example.com",
        "password": "secure_password123",
        "name": "Test User"
    });
    
    let response = ctx.app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/users")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::CREATED);
    
    let body: Value = serde_json::from_slice(&hyper::body::to_bytes(response.into_body()).await.unwrap()).unwrap();
    assert_eq!(body["email"], "test@example.com");
    assert!(body["id"].is_string());
}
```

### ✅ DO: Use Property-Based Testing for Business Logic

```rust
// tests/properties/user_service_test.rs
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_password_hash_properties(password in "[a-zA-Z0-9!@#$%^&*()]{8,128}") {
        let service = PasswordService::new();
        
        // Hash should always succeed for valid passwords
        let hash = service.hash_password(&password).unwrap();
        
        // Hash should be different from the password
        prop_assert_ne!(&hash, &password);
        
        // Hash should be verifiable
        prop_assert!(service.verify_password(&password, &hash).unwrap());
        
        // Different password should not verify
        let different_password = format!("{}x", password);
        prop_assert!(!service.verify_password(&different_password, &hash).unwrap());
        
        // Hash should be consistent format (Argon2 PHC string)
        prop_assert!(hash.starts_with("$argon2"));
    }
}
```

## 10. Observability & Monitoring

### ✅ DO: Implement Comprehensive Tracing

```rust
// src/telemetry.rs
use opentelemetry::{global, sdk::propagation::TraceContextPropagator};
use opentelemetry_otlp::WithExportConfig;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

pub fn init_telemetry(service_name: &str) -> Result<()> {
    // Set up OTLP exporter
    let otlp_exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_endpoint("http://localhost:4317");
    
    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(otlp_exporter)
        .with_trace_config(
            opentelemetry_sdk::trace::config()
                .with_sampler(opentelemetry_sdk::trace::Sampler::AlwaysOn)
                .with_id_generator(opentelemetry_sdk::trace::RandomIdGenerator::default())
                .with_resource(opentelemetry_sdk::Resource::new(vec![
                    opentelemetry::KeyValue::new("service.name", service_name.to_string()),
                    opentelemetry::KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
                ]))
        )
        .install_batch(opentelemetry_sdk::runtime::Tokio)?;
    
    // Create a tracing layer with the tracer
    let telemetry_layer = tracing_opentelemetry::layer().with_tracer(tracer);
    
    // Set up log formatting
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(false)
        .with_thread_ids(true)
        .with_level(true)
        .json();
    
    // Combine layers
    let subscriber = tracing_subscriber::Registry::default()
        .with(EnvFilter::from_default_env())
        .with(telemetry_layer)
        .with(fmt_layer);
    
    tracing::subscriber::set_global_default(subscriber)?;
    
    // Set up propagator
    global::set_text_map_propagator(TraceContextPropagator::new());
    
    Ok(())
}

// Instrument key operations
#[tracing::instrument(skip(db))]
pub async fn find_user_with_roles(db: &DatabaseConnection, user_id: Uuid) -> Result<UserWithRoles> {
    let span = tracing::Span::current();
    span.record("user.id", &user_id.to_string());
    
    let user = Users::find_by_id(user_id)
        .find_with_related(UserRoles)
        .all(db)
        .await?;
    
    span.record("roles.count", user.1.len());
    
    Ok(UserWithRoles::from(user))
}
```

### ✅ DO: Export Prometheus Metrics

```rust
// src/metrics.rs
use axum::{extract::MatchedPath, http::Request, middleware::Next, response::Response};
use metrics::{counter, histogram, increment_counter, Unit};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};

pub fn setup_metrics_recorder() -> PrometheusHandle {
    const EXPONENTIAL_SECONDS: &[f64] = &[
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    ];
    
    PrometheusBuilder::new()
        .set_buckets_for_metric(
            Matcher::Full("http_request_duration_seconds".to_string()),
            EXPONENTIAL_SECONDS,
        )
        .set_buckets_for_metric(
            Matcher::Full("db_query_duration_seconds".to_string()),
            EXPONENTIAL_SECONDS,
        )
        .install_recorder()
        .unwrap()
}

pub async fn track_metrics<B>(
    req: Request<B>,
    next: Next<B>,
) -> Response {
    let start = std::time::Instant::now();
    let path = if let Some(matched_path) = req.extensions().get::<MatchedPath>() {
        matched_path.as_str().to_owned()
    } else {
        "unknown".to_string()
    };
    let method = req.method().to_string();
    
    let response = next.run(req).await;
    
    let status = response.status().as_u16().to_string();
    let duration = start.elapsed().as_secs_f64();
    
    // Record metrics
    increment_counter!(
        "http_requests_total",
        "method" => method.clone(),
        "path" => path.clone(),
        "status" => status.clone(),
    );
    
    histogram!(
        "http_request_duration_seconds",
        duration,
        "method" => method,
        "path" => path,
        "status" => status,
    );
    
    response
}

// Database metrics middleware for SeaORM
pub struct MetricsDatabase {
    inner: DatabaseConnection,
}

#[async_trait::async_trait]
impl ConnectionTrait for MetricsDatabase {
    async fn execute(&self, stmt: Statement) -> Result<ExecResult, DbErr> {
        let start = std::time::Instant::now();
        let result = self.inner.execute(stmt).await;
        let duration = start.elapsed().as_secs_f64();
        
        histogram!("db_query_duration_seconds", duration, "operation" => "execute");
        
        result
    }
    
    // Implement other trait methods similarly...
}
```

## 11. Production Deployment

### ✅ DO: Use Multi-Stage Docker Builds

```dockerfile
# Dockerfile
# Build stage
FROM rust:1.82-bookworm as builder

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY entity/Cargo.toml entity/
COPY migration/Cargo.toml migration/
COPY api/Cargo.toml api/

# Build dependencies (this is cached as long as Cargo.toml files don't change)
RUN mkdir -p api/src entity/src migration/src && \
    echo "fn main() {}" > api/src/main.rs && \
    touch entity/src/lib.rs migration/src/lib.rs && \
    cargo build --release --bin api-service && \
    rm -rf api/src entity/src migration/src

# Copy source code
COPY . .

# Build the application
RUN touch api/src/main.rs && \
    cargo build --release --bin api-service

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1001 appuser

WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/target/release/api-service /app/

# Copy any static assets or configuration files
COPY --from=builder /app/config /app/config

# Change ownership
RUN chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

EXPOSE 3000

CMD ["./api-service"]
```

### ✅ DO: Implement Graceful Shutdown

```rust
// src/main.rs
use tokio::signal;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize telemetry
    init_telemetry("api-service")?;
    
    // Load configuration
    let config = Config::from_env()?;
    
    // Create application state
    let state = AppState::new(config.clone()).await?;
    
    // Set up metrics
    let metrics_handle = setup_metrics_recorder();
    
    // Create router
    let app = create_router(state.clone())
        .route("/metrics", get(move || async move { metrics_handle.render() }));
    
    // Create server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.server.port));
    let listener = TcpListener::bind(addr).await?;
    
    tracing::info!("Server listening on {}", addr);
    
    // Serve with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    
    // Cleanup
    state.shutdown().await;
    opentelemetry::global::shutdown_tracer_provider();
    
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("Shutdown signal received, starting graceful shutdown");
}
```

## 12. Advanced Patterns

### Connection Pool Monitoring

```rust
// src/database/pool_monitor.rs
use sea_orm::{ConnectOptions, DatabaseConnection};
use std::time::Duration;

pub async fn create_monitored_pool(database_url: &str) -> Result<DatabaseConnection> {
    let mut opt = ConnectOptions::new(database_url);
    
    opt.max_connections(100)
        .min_connections(10)
        .connect_timeout(Duration::from_secs(8))
        .acquire_timeout(Duration::from_secs(8))
        .idle_timeout(Duration::from_secs(600))
        .max_lifetime(Duration::from_secs(1800))
        .sqlx_logging(true)
        .sqlx_logging_level(log::LevelFilter::Trace)
        .set_schema_search_path("public");
    
    let db = sea_orm::Database::connect(opt).await?;
    
    // Spawn a monitoring task
    let db_clone = db.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        loop {
            interval.tick().await;
            
            // This is a hack to get pool metrics from SQLx through SeaORM
            // In a real app, you might need to use SQLx directly for this
            if let Ok(pool) = db_clone.get_postgres_connection_pool() {
                let size = pool.size();
                let idle = pool.num_idle();
                
                metrics::gauge!("db_pool_connections_total", size as f64);
                metrics::gauge!("db_pool_connections_idle", idle as f64);
                metrics::gauge!("db_pool_connections_active", (size - idle) as f64);
            }
        }
    });
    
    Ok(db)
}
```

### Event Sourcing Pattern

```rust
// src/events/mod.rs
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DomainEvent {
    UserCreated {
        user_id: Uuid,
        email: String,
        created_at: DateTime<Utc>,
    },
    UserUpdated {
        user_id: Uuid,
        changes: serde_json::Value,
        updated_at: DateTime<Utc>,
    },
    UserDeleted {
        user_id: Uuid,
        deleted_at: DateTime<Utc>,
    },
}

pub struct EventStore {
    db: DatabaseConnection,
}

impl EventStore {
    pub async fn append(&self, aggregate_id: Uuid, events: Vec<DomainEvent>) -> Result<()> {
        let txn = self.db.begin().await?;
        
        for event in events {
            let event_data = serde_json::to_value(&event)?;
            
            event_store::ActiveModel {
                aggregate_id: Set(aggregate_id),
                event_type: Set(event.event_type()),
                event_data: Set(event_data),
                event_version: Set(self.next_version(aggregate_id).await?),
                occurred_at: Set(Utc::now()),
                ..Default::default()
            }
            .insert(&txn)
            .await?;
        }
        
        txn.commit().await?;
        Ok(())
    }
    
    pub async fn get_events(&self, aggregate_id: Uuid) -> Result<Vec<DomainEvent>> {
        let events = EventStore::find()
            .filter(event_store::Column::AggregateId.eq(aggregate_id))
            .order_by_asc(event_store::Column::EventVersion)
            .all(&self.db)
            .await?;
        
        events.into_iter()
            .map(|e| serde_json::from_value(e.event_data).map_err(Into::into))
            .collect()
    }
}
```

### CQRS Implementation

```rust
// src/cqrs/mod.rs
#[async_trait::async_trait]
pub trait Command: Send + Sync {
    type Result;
    async fn execute(&self, state: &AppState) -> Result<Self::Result>;
}

#[async_trait::async_trait]
pub trait Query: Send + Sync {
    type Result;
    async fn execute(&self, state: &AppState) -> Result<Self::Result>;
}

// Command example
pub struct CreateUserCommand {
    pub email: String,
    pub password: String,
    pub name: String,
}

#[async_trait::async_trait]
impl Command for CreateUserCommand {
    type Result = User;
    
    async fn execute(&self, state: &AppState) -> Result<Self::Result> {
        // Validate
        if Users::find()
            .filter(users::Column::Email.eq(&self.email))
            .one(&state.db)
            .await?
            .is_some()
        {
            return Err(AppError::Conflict("Email already exists".into()));
        }
        
        // Hash password
        let password_hash = state.password_service.hash(&self.password)?;
        
        // Create user
        let user = users::ActiveModel {
            email: Set(self.email.clone()),
            password_hash: Set(password_hash),
            name: Set(self.name.clone()),
            ..Default::default()
        }
        .insert(&state.db)
        .await?;
        
        // Emit event
        state.event_bus.publish(DomainEvent::UserCreated {
            user_id: user.id,
            email: user.email.clone(),
            created_at: user.created_at,
        }).await?;
        
        Ok(user)
    }
}
```

### WebSocket Support

```rust
// src/websocket/mod.rs
use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::Response,
};
use futures::{sink::SinkExt, stream::StreamExt};

pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Extension(user): Extension<CurrentUser>,
) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state, user))
}

async fn handle_socket(socket: WebSocket, state: AppState, user: CurrentUser) {
    let (mut sender, mut receiver) = socket.split();
    
    // Subscribe to user's channel
    let mut rx = state.broadcast.subscribe(user.id).await;
    
    // Spawn task to forward messages from broadcast to websocket
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if sender
                .send(Message::Text(serde_json::to_string(&msg).unwrap()))
                .await
                .is_err()
            {
                break;
            }
        }
    });
    
    // Handle incoming messages
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    // Parse and handle message
                    if let Ok(cmd) = serde_json::from_str::<WsCommand>(&text) {
                        handle_ws_command(cmd, &state, &user).await;
                    }
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
    });
    
    // Wait for either task to finish
    tokio::select! {
        _ = (&mut send_task) => recv_task.abort(),
        _ = (&mut recv_task) => send_task.abort(),
    }
}
```

## Conclusion

This guide represents the state of the art for building high-performance Rust web services in mid-2025. The patterns presented here have been battle-tested in production environments handling millions of requests per day.

Key takeaways:
- **Type safety everywhere**: Leverage Rust's type system to catch errors at compile time
- **Async all the way down**: Use Tokio and async patterns consistently
- **Layer your architecture**: Clear separation between routes, handlers, services, and repositories
- **Monitor everything**: Comprehensive observability is non-negotiable in production
- **Test at every level**: From unit tests to integration tests with real databases

The Rust ecosystem continues to evolve rapidly. Stay current with updates to core libraries like Axum and SeaORM, and don't hesitate to explore new patterns as they emerge. The combination of Rust's performance and safety guarantees with modern web development practices creates systems that are both fast and reliable.

Remember: premature optimization is still the root of all evil. Start with clean, idiomatic code and optimize based on real-world metrics. Rust gives you the tools to write blazingly fast code when you need it, but clarity and correctness should always come first.