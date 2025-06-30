# The Definitive Guide to Go Web Development with Fiber/Echo, pgx, and Ent (Late 2025 Edition)

This guide synthesizes modern best practices for building scalable, secure, and performant web applications with Go, focusing on the Fiber/Echo frameworks, pgx PostgreSQL driver, and Ent ORM. It provides production-grade architectural patterns and moves beyond basic tutorials.

### Prerequisites & Configuration

Ensure your project uses **Go 1.24+** with support for generic type aliases, Swiss Tables map implementation, and improved performance. 

```bash
# Check Go version
go version  # Should show go1.24.x or higher

# Initialize module with modern Go
go mod init github.com/yourorg/yourapp
```

Create a modern `go.mod` with proper toolchain directives:

```go
module github.com/yourorg/yourapp

go 1.24

toolchain go1.24.0

require (
    github.com/gofiber/fiber/v3 v3.0.0-beta.3
    github.com/labstack/echo/v5 v5.0.0-alpha.1
    github.com/jackc/pgx/v5 v5.6.0
    entgo.io/ent v0.14.0
    github.com/google/wire v0.6.0
)
```

> **Note**: Go 1.24 introduces full support for generic type aliases, Swiss Tables-based map implementation (2-3% performance improvement), and `go:wasmexport` directive for WebAssembly.

### Development Tools Configuration

```makefile
# Makefile
.PHONY: tools
tools:
	@echo "Installing development tools..."
	@go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	@go install github.com/google/wire/cmd/wire@latest
	@go install entgo.io/ent/cmd/ent@latest
	@go install github.com/air-verse/air@latest
	@go install github.com/pressly/goose/v3/cmd/goose@latest
	@go install github.com/gotesttools/gotestfmt/v2/cmd/gotestfmt@latest
	@go install golang.org/x/vuln/cmd/govulncheck@latest

.PHONY: lint
lint:
	golangci-lint run --fix

.PHONY: test
test:
	go test -v -race -coverprofile=coverage.out ./... | gotestfmt

.PHONY: bench
bench:
	go test -run=^$ -bench=. -benchmem ./...
```

**.golangci.yml** configuration for strict linting:

```yaml
# .golangci.yml
run:
  timeout: 5m
  go: "1.24"

linters:
  enable-all: true
  disable:
    - exhaustruct  # Too noisy for partial struct initialization
    - depguard     # Overly restrictive
    - gochecknoglobals  # Some globals are fine
    - wsl          # Too opinionated on whitespace
    - varnamelen   # Short names are idiomatic in Go

linters-settings:
  revive:
    enable-all-rules: true
    rules:
      - name: cognitive-complexity
        arguments: [15]
  govet:
    enable-all: true
  errcheck:
    check-type-assertions: true
    check-blank: true
  gocyclo:
    min-complexity: 15
  
issues:
  exclude-rules:
    - path: _test\.go
      linters:
        - dupl
        - gosec
```

---

## 1. Foundational Architecture & Project Structure

A well-organized project structure is crucial for maintainability and team collaboration. Go's conventions favor flat structures, but larger applications benefit from domain-driven organization.

### ✅ DO: Use Domain-Driven Design with Clear Boundaries

```
/cmd
├── api/              # API server entry point
│   └── main.go
├── worker/           # Background job worker
│   └── main.go
└── migrate/          # Database migration tool
    └── main.go

/internal             # Private application code
├── api/              # HTTP layer
│   ├── handler/      # HTTP handlers
│   ├── middleware/   # HTTP middleware
│   └── router/       # Route definitions
├── domain/           # Core business logic
│   ├── user/         # User domain
│   │   ├── entity.go
│   │   ├── repository.go
│   │   └── service.go
│   └── billing/      # Billing domain
├── infrastructure/   # External services
│   ├── database/     # Database implementation
│   ├── cache/        # Redis client
│   └── email/        # Email service
└── pkg/              # Shared utilities
    ├── logger/
    ├── validator/
    └── crypto/

/ent                  # Ent ORM schemas and generated code
├── schema/
│   ├── user.go
│   └── billing.go
└── generate.go

/migrations           # SQL migrations
/scripts              # Build and deployment scripts
/docs                 # Documentation
/api                  # OpenAPI/Swagger specs
```

### ✅ DO: Use Dependency Injection with Wire

Google's Wire provides compile-time dependency injection, eliminating runtime reflection overhead.

```go
// internal/infrastructure/database/provider.go
package database

import (
    "context"
    "fmt"
    
    "github.com/jackc/pgx/v5/pgxpool"
    "github.com/google/wire"
)

// ProviderSet is the Wire provider set for database
var ProviderSet = wire.NewSet(
    NewPgxPool,
    NewEntClient,
)

// Config holds database configuration
type Config struct {
    Host     string
    Port     int
    User     string
    Password string
    DBName   string
    SSLMode  string
}

// NewPgxPool creates a new pgx connection pool
func NewPgxPool(ctx context.Context, cfg Config) (*pgxpool.Pool, error) {
    dsn := fmt.Sprintf(
        "host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
        cfg.Host, cfg.Port, cfg.User, cfg.Password, cfg.DBName, cfg.SSLMode,
    )
    
    config, err := pgxpool.ParseConfig(dsn)
    if err != nil {
        return nil, fmt.Errorf("parse config: %w", err)
    }
    
    // Performance optimizations
    config.MaxConns = 25
    config.MinConns = 5
    config.MaxConnLifetime = time.Hour
    config.MaxConnIdleTime = time.Minute * 30
    
    pool, err := pgxpool.NewWithConfig(ctx, config)
    if err != nil {
        return nil, fmt.Errorf("create pool: %w", err)
    }
    
    return pool, nil
}
```

### Framework Choice: Fiber vs Echo

Choose based on your needs:

**Fiber (v3)** - When raw performance is critical:
- Built on fasthttp, ~10x faster than net/http
- Zero memory allocation on hot paths
- Express-like API for Node.js developers
- Better for high-throughput microservices

**Echo (v5)** - When ecosystem compatibility matters:
- Built on standard net/http
- Better middleware ecosystem
- More mature and stable
- Better for complex applications

---

## 2. High-Performance HTTP Handlers

### ✅ DO: Use Fiber v3 for Maximum Performance

```go
// internal/api/handler/user_handler.go
package handler

import (
    "github.com/gofiber/fiber/v3"
    "github.com/yourorg/yourapp/internal/domain/user"
    "github.com/yourorg/yourapp/internal/pkg/logger"
)

type UserHandler struct {
    userService user.Service
    log         logger.Logger
}

func NewUserHandler(userService user.Service, log logger.Logger) *UserHandler {
    return &UserHandler{
        userService: userService,
        log:         log,
    }
}

// GetUser handles GET /users/:id
func (h *UserHandler) GetUser(c fiber.Ctx) error {
    ctx := c.Context()
    
    // Fiber v3 provides type-safe param parsing
    id, err := c.Params().Int("id")
    if err != nil {
        return fiber.NewError(fiber.StatusBadRequest, "invalid user ID")
    }
    
    user, err := h.userService.GetByID(ctx, id)
    if err != nil {
        h.log.Error("failed to get user", "error", err, "id", id)
        return fiber.NewError(fiber.StatusInternalServerError, "internal error")
    }
    
    if user == nil {
        return fiber.NewError(fiber.StatusNotFound, "user not found")
    }
    
    // Use Fiber's optimized JSON encoder
    return c.JSON(user)
}

// CreateUser handles POST /users with zero-allocation parsing
func (h *UserHandler) CreateUser(c fiber.Ctx) error {
    var req CreateUserRequest
    
    // BodyParser reuses buffers for zero allocation
    if err := c.Bind().Body(&req); err != nil {
        return fiber.NewError(fiber.StatusBadRequest, "invalid request body")
    }
    
    // Validate using struct tags
    if err := c.Bind().Validate(&req); err != nil {
        return fiber.NewError(fiber.StatusBadRequest, err.Error())
    }
    
    user, err := h.userService.Create(c.Context(), &req)
    if err != nil {
        return handleServiceError(err)
    }
    
    return c.Status(fiber.StatusCreated).JSON(user)
}
```

### ✅ DO: Use Echo v5 for Standard Compatibility

```go
// internal/api/handler/user_handler_echo.go
package handler

import (
    "net/http"
    
    "github.com/labstack/echo/v5"
    "github.com/yourorg/yourapp/internal/domain/user"
)

type UserHandler struct {
    userService user.Service
}

// GetUser handles GET /users/:id
func (h *UserHandler) GetUser(c echo.Context) error {
    id, err := strconv.Atoi(c.PathParam("id"))
    if err != nil {
        return echo.NewHTTPError(http.StatusBadRequest, "invalid user ID")
    }
    
    user, err := h.userService.GetByID(c.Request().Context(), id)
    if err != nil {
        return echo.NewHTTPError(http.StatusInternalServerError, err.Error())
    }
    
    return c.JSON(http.StatusOK, user)
}
```

### ❌ DON'T: Mix Concerns in Handlers

```go
// Bad - handler doing too much
func (h *UserHandler) CreateUser(c fiber.Ctx) error {
    var req CreateUserRequest
    c.BodyParser(&req)
    
    // DON'T: Business logic in handler
    if req.Email == "" || !strings.Contains(req.Email, "@") {
        return errors.New("invalid email")
    }
    
    // DON'T: Direct database access
    db := c.Locals("db").(*sql.DB)
    _, err := db.Exec("INSERT INTO users...")
    
    // DON'T: External service calls
    emailClient.SendWelcomeEmail(req.Email)
    
    return c.JSON(user)
}
```

---

## 3. Database Layer with pgx and Ent

### ✅ DO: Use Ent for Type-Safe ORM with pgx Driver

**Define Ent Schema:**

```go
// ent/schema/user.go
package schema

import (
    "time"
    
    "entgo.io/ent"
    "entgo.io/ent/dialect/entsql"
    "entgo.io/ent/schema/edge"
    "entgo.io/ent/schema/field"
    "entgo.io/ent/schema/index"
    "entgo.io/ent/schema/mixin"
)

// TimeMixin provides created_at and updated_at fields
type TimeMixin struct {
    mixin.Schema
}

func (TimeMixin) Fields() []ent.Field {
    return []ent.Field{
        field.Time("created_at").
            Default(time.Now).
            Immutable().
            Annotations(entsql.Default("CURRENT_TIMESTAMP")),
        field.Time("updated_at").
            Default(time.Now).
            UpdateDefault(time.Now).
            Annotations(entsql.Default("CURRENT_TIMESTAMP")),
    }
}

// User holds the schema definition for the User entity
type User struct {
    ent.Schema
}

// Mixin of the User
func (User) Mixin() []ent.Mixin {
    return []ent.Mixin{
        TimeMixin{},
    }
}

// Fields of the User
func (User) Fields() []ent.Field {
    return []ent.Field{
        field.String("email").
            Unique().
            NotEmpty().
            Annotations(
                entsql.Annotation{Size: 255},
            ),
        field.String("password_hash").
            Sensitive(), // Won't be included in JSON
        field.Enum("status").
            Values("active", "inactive", "suspended").
            Default("active"),
        field.JSON("metadata", map[string]interface{}{}).
            Optional(),
    }
}

// Edges of the User
func (User) Edges() []ent.Edge {
    return []ent.Edge{
        edge.To("orders", Order.Type),
        edge.To("sessions", Session.Type),
    }
}

// Indexes of the User
func (User) Indexes() []ent.Index {
    return []ent.Index{
        index.Fields("email"),
        index.Fields("status", "created_at"),
    }
}
```

**Initialize Ent with pgx:**

```go
// internal/infrastructure/database/ent.go
package database

import (
    "context"
    "database/sql"
    "fmt"
    
    "entgo.io/ent/dialect"
    entsql "entgo.io/ent/dialect/sql"
    "github.com/jackc/pgx/v5/pgxpool"
    "github.com/jackc/pgx/v5/stdlib"
    
    "github.com/yourorg/yourapp/ent"
)

// NewEntClient creates a new Ent client with pgx driver
func NewEntClient(pool *pgxpool.Pool) (*ent.Client, error) {
    // Use pgx through database/sql interface
    db := stdlib.OpenDBFromPool(pool)
    
    // Create Ent driver
    drv := entsql.OpenDB(dialect.Postgres, db)
    
    client := ent.NewClient(ent.Driver(drv))
    
    // Enable debug in development
    if isDevelopment() {
        client = client.Debug()
    }
    
    return client, nil
}
```

### ✅ DO: Use Repository Pattern with Ent

```go
// internal/domain/user/repository.go
package user

import (
    "context"
    "fmt"
    
    "github.com/yourorg/yourapp/ent"
    "github.com/yourorg/yourapp/ent/user"
)

type Repository interface {
    Create(ctx context.Context, params CreateParams) (*ent.User, error)
    GetByID(ctx context.Context, id int) (*ent.User, error)
    GetByEmail(ctx context.Context, email string) (*ent.User, error)
    Update(ctx context.Context, id int, params UpdateParams) (*ent.User, error)
    Delete(ctx context.Context, id int) error
    List(ctx context.Context, params ListParams) ([]*ent.User, int, error)
}

type repository struct {
    client *ent.Client
}

func NewRepository(client *ent.Client) Repository {
    return &repository{client: client}
}

func (r *repository) Create(ctx context.Context, params CreateParams) (*ent.User, error) {
    return r.client.User.
        Create().
        SetEmail(params.Email).
        SetPasswordHash(params.PasswordHash).
        SetStatus(user.StatusActive).
        Save(ctx)
}

func (r *repository) GetByEmail(ctx context.Context, email string) (*ent.User, error) {
    return r.client.User.
        Query().
        Where(user.EmailEQ(email)).
        Only(ctx)
}

// List with pagination and filtering
func (r *repository) List(ctx context.Context, params ListParams) ([]*ent.User, int, error) {
    query := r.client.User.Query()
    
    // Apply filters
    if params.Status != "" {
        query = query.Where(user.StatusEQ(params.Status))
    }
    
    // Get total count
    count, err := query.Count(ctx)
    if err != nil {
        return nil, 0, err
    }
    
    // Apply pagination
    users, err := query.
        Limit(params.Limit).
        Offset(params.Offset).
        Order(ent.Desc(user.FieldCreatedAt)).
        All(ctx)
        
    return users, count, err
}
```

### ✅ DO: Use Raw pgx for Performance-Critical Queries

```go
// internal/infrastructure/database/analytics_repo.go
package database

import (
    "context"
    "time"
    
    "github.com/jackc/pgx/v5"
    "github.com/jackc/pgx/v5/pgxpool"
)

type AnalyticsRepository struct {
    pool *pgxpool.Pool
}

// GetUserActivityStats uses raw SQL for complex aggregations
func (r *AnalyticsRepository) GetUserActivityStats(
    ctx context.Context, 
    startDate, endDate time.Time,
) ([]UserActivityStat, error) {
    query := `
        WITH daily_stats AS (
            SELECT 
                DATE(created_at) as activity_date,
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(*) as total_events,
                SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchases
            FROM user_events
            WHERE created_at >= $1 AND created_at < $2
            GROUP BY DATE(created_at)
        )
        SELECT 
            activity_date,
            unique_users,
            total_events,
            purchases,
            SUM(unique_users) OVER (ORDER BY activity_date) as cumulative_users
        FROM daily_stats
        ORDER BY activity_date
    `
    
    rows, err := r.pool.Query(ctx, query, startDate, endDate)
    if err != nil {
        return nil, fmt.Errorf("query stats: %w", err)
    }
    defer rows.Close()
    
    var stats []UserActivityStat
    for rows.Next() {
        var stat UserActivityStat
        err := rows.Scan(
            &stat.Date,
            &stat.UniqueUsers,
            &stat.TotalEvents,
            &stat.Purchases,
            &stat.CumulativeUsers,
        )
        if err != nil {
            return nil, fmt.Errorf("scan row: %w", err)
        }
        stats = append(stats, stat)
    }
    
    return stats, rows.Err()
}

// BulkInsertEvents uses COPY for maximum performance
func (r *AnalyticsRepository) BulkInsertEvents(
    ctx context.Context,
    events []Event,
) error {
    _, err := r.pool.CopyFrom(
        ctx,
        pgx.Identifier{"user_events"},
        []string{"user_id", "event_type", "metadata", "created_at"},
        pgx.CopyFromSlice(len(events), func(i int) ([]interface{}, error) {
            return []interface{}{
                events[i].UserID,
                events[i].EventType,
                events[i].Metadata,
                events[i].CreatedAt,
            }, nil
        }),
    )
    return err
}
```

### ✅ DO: Use Database Transactions Properly

```go
// internal/domain/billing/service.go
package billing

import (
    "context"
    "fmt"
    
    "github.com/yourorg/yourapp/ent"
)

type Service struct {
    client *ent.Client
}

// ProcessPayment handles payment with proper transaction management
func (s *Service) ProcessPayment(ctx context.Context, userID int, amount float64) error {
    // Use Ent's transaction support
    tx, err := s.client.Tx(ctx)
    if err != nil {
        return fmt.Errorf("start transaction: %w", err)
    }
    
    // Ensure rollback on error
    defer func() {
        if err != nil {
            if rerr := tx.Rollback(); rerr != nil {
                err = fmt.Errorf("%w: rollback failed: %v", err, rerr)
            }
        }
    }()
    
    // Lock user for update to prevent race conditions
    user, err := tx.User.
        Query().
        Where(user.IDEQ(userID)).
        ForUpdate().
        Only(ctx)
    if err != nil {
        return fmt.Errorf("get user: %w", err)
    }
    
    // Create payment record
    payment, err := tx.Payment.
        Create().
        SetUserID(userID).
        SetAmount(amount).
        SetStatus("pending").
        Save(ctx)
    if err != nil {
        return fmt.Errorf("create payment: %w", err)
    }
    
    // Process with external service
    if err := s.processExternalPayment(ctx, payment); err != nil {
        return fmt.Errorf("external payment: %w", err)
    }
    
    // Update payment status
    _, err = tx.Payment.
        UpdateOneID(payment.ID).
        SetStatus("completed").
        Save(ctx)
    if err != nil {
        return fmt.Errorf("update payment: %w", err)
    }
    
    // Commit transaction
    if err := tx.Commit(); err != nil {
        return fmt.Errorf("commit: %w", err)
    }
    
    return nil
}
```

---

## 4. Error Handling and Observability

### ✅ DO: Use Structured Error Handling

```go
// internal/pkg/apperror/errors.go
package apperror

import (
    "errors"
    "fmt"
)

// Error types
var (
    ErrNotFound     = errors.New("not found")
    ErrUnauthorized = errors.New("unauthorized")
    ErrForbidden    = errors.New("forbidden")
    ErrValidation   = errors.New("validation failed")
    ErrInternal     = errors.New("internal error")
)

// Error represents an application error with context
type Error struct {
    Code    string
    Message string
    Err     error
    Details map[string]interface{}
}

func (e *Error) Error() string {
    if e.Err != nil {
        return fmt.Sprintf("%s: %v", e.Message, e.Err)
    }
    return e.Message
}

func (e *Error) Unwrap() error {
    return e.Err
}

// New creates a new application error
func New(code string, message string) *Error {
    return &Error{
        Code:    code,
        Message: message,
        Details: make(map[string]interface{}),
    }
}

// Wrap wraps an error with additional context
func Wrap(err error, message string) *Error {
    return &Error{
        Code:    "INTERNAL_ERROR",
        Message: message,
        Err:     err,
        Details: make(map[string]interface{}),
    }
}

// WithDetails adds details to the error
func (e *Error) WithDetails(key string, value interface{}) *Error {
    e.Details[key] = value
    return e
}
```

### ✅ DO: Implement Comprehensive Middleware

```go
// internal/api/middleware/logging.go
package middleware

import (
    "time"
    
    "github.com/gofiber/fiber/v3"
    "github.com/gofiber/fiber/v3/middleware/requestid"
    "go.uber.org/zap"
)

// NewLoggingMiddleware creates structured logging middleware
func NewLoggingMiddleware(logger *zap.Logger) fiber.Handler {
    return func(c fiber.Ctx) error {
        start := time.Now()
        
        // Get or generate request ID
        reqID := c.Locals(requestid.ConfigDefault.ContextKey).(string)
        
        // Add request ID to context logger
        ctxLogger := logger.With(
            zap.String("request_id", reqID),
            zap.String("method", c.Method()),
            zap.String("path", c.Path()),
            zap.String("ip", c.IP()),
        )
        
        // Store logger in context for handlers
        c.Locals("logger", ctxLogger)
        
        // Process request
        err := c.Next()
        
        // Log response
        duration := time.Since(start)
        ctxLogger.Info("request completed",
            zap.Int("status", c.Response().StatusCode()),
            zap.Duration("duration", duration),
            zap.Int("bytes", len(c.Response().Body())),
            zap.Error(err),
        )
        
        return err
    }
}

// NewErrorMiddleware handles errors consistently
func NewErrorMiddleware(logger *zap.Logger) fiber.Handler {
    return func(c fiber.Ctx) error {
        err := c.Next()
        
        if err != nil {
            // Check if it's a Fiber error
            var fiberErr *fiber.Error
            if errors.As(err, &fiberErr) {
                return c.Status(fiberErr.Code).JSON(fiber.Map{
                    "error": fiber.Map{
                        "code":    fiberErr.Code,
                        "message": fiberErr.Message,
                    },
                })
            }
            
            // Check if it's an app error
            var appErr *apperror.Error
            if errors.As(err, &appErr) {
                status := fiber.StatusInternalServerError
                switch {
                case errors.Is(appErr.Err, apperror.ErrNotFound):
                    status = fiber.StatusNotFound
                case errors.Is(appErr.Err, apperror.ErrUnauthorized):
                    status = fiber.StatusUnauthorized
                case errors.Is(appErr.Err, apperror.ErrValidation):
                    status = fiber.StatusBadRequest
                }
                
                return c.Status(status).JSON(fiber.Map{
                    "error": fiber.Map{
                        "code":    appErr.Code,
                        "message": appErr.Message,
                        "details": appErr.Details,
                    },
                })
            }
            
            // Log unexpected errors
            logger.Error("unexpected error",
                zap.Error(err),
                zap.String("request_id", c.Locals("request_id").(string)),
            )
            
            // Return generic error
            return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
                "error": fiber.Map{
                    "code":    "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                },
            })
        }
        
        return nil
    }
}
```

---

## 5. Authentication and Authorization

### ✅ DO: Use JWT with Refresh Tokens

```go
// internal/pkg/auth/jwt.go
package auth

import (
    "crypto/rand"
    "encoding/base64"
    "time"
    
    "github.com/golang-jwt/jwt/v5"
)

type TokenService struct {
    accessSecret  []byte
    refreshSecret []byte
    accessTTL     time.Duration
    refreshTTL    time.Duration
}

type Claims struct {
    UserID int    `json:"user_id"`
    Email  string `json:"email"`
    Role   string `json:"role"`
    jwt.RegisteredClaims
}

func NewTokenService(accessSecret, refreshSecret string) *TokenService {
    return &TokenService{
        accessSecret:  []byte(accessSecret),
        refreshSecret: []byte(refreshSecret),
        accessTTL:     15 * time.Minute,
        refreshTTL:    7 * 24 * time.Hour,
    }
}

func (s *TokenService) GenerateTokenPair(userID int, email, role string) (string, string, error) {
    // Generate access token
    accessClaims := Claims{
        UserID: userID,
        Email:  email,
        Role:   role,
        RegisteredClaims: jwt.RegisteredClaims{
            ExpiresAt: jwt.NewNumericDate(time.Now().Add(s.accessTTL)),
            IssuedAt:  jwt.NewNumericDate(time.Now()),
            NotBefore: jwt.NewNumericDate(time.Now()),
        },
    }
    
    accessToken := jwt.NewWithClaims(jwt.SigningMethodHS256, accessClaims)
    accessTokenString, err := accessToken.SignedString(s.accessSecret)
    if err != nil {
        return "", "", err
    }
    
    // Generate refresh token
    refreshClaims := jwt.RegisteredClaims{
        Subject:   fmt.Sprintf("%d", userID),
        ExpiresAt: jwt.NewNumericDate(time.Now().Add(s.refreshTTL)),
        IssuedAt:  jwt.NewNumericDate(time.Now()),
    }
    
    refreshToken := jwt.NewWithClaims(jwt.SigningMethodHS256, refreshClaims)
    refreshTokenString, err := refreshToken.SignedString(s.refreshSecret)
    if err != nil {
        return "", "", err
    }
    
    return accessTokenString, refreshTokenString, nil
}

// ValidateAccessToken validates and returns claims
func (s *TokenService) ValidateAccessToken(tokenString string) (*Claims, error) {
    token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
        if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
            return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
        }
        return s.accessSecret, nil
    })
    
    if err != nil {
        return nil, err
    }
    
    if claims, ok := token.Claims.(*Claims); ok && token.Valid {
        return claims, nil
    }
    
    return nil, jwt.ErrTokenInvalidClaims
}
```

### ✅ DO: Implement Auth Middleware

```go
// internal/api/middleware/auth.go
package middleware

import (
    "strings"
    
    "github.com/gofiber/fiber/v3"
    "github.com/yourorg/yourapp/internal/pkg/auth"
)

func NewAuthMiddleware(tokenService *auth.TokenService) fiber.Handler {
    return func(c fiber.Ctx) error {
        // Get token from header
        authHeader := c.Get("Authorization")
        if authHeader == "" {
            return fiber.NewError(fiber.StatusUnauthorized, "missing authorization header")
        }
        
        // Extract bearer token
        parts := strings.Split(authHeader, " ")
        if len(parts) != 2 || parts[0] != "Bearer" {
            return fiber.NewError(fiber.StatusUnauthorized, "invalid authorization header")
        }
        
        // Validate token
        claims, err := tokenService.ValidateAccessToken(parts[1])
        if err != nil {
            return fiber.NewError(fiber.StatusUnauthorized, "invalid token")
        }
        
        // Store user info in context
        c.Locals("user_id", claims.UserID)
        c.Locals("user_email", claims.Email)
        c.Locals("user_role", claims.Role)
        
        return c.Next()
    }
}

// RequireRole checks if user has required role
func RequireRole(roles ...string) fiber.Handler {
    return func(c fiber.Ctx) error {
        userRole := c.Locals("user_role").(string)
        
        for _, role := range roles {
            if userRole == role {
                return c.Next()
            }
        }
        
        return fiber.NewError(fiber.StatusForbidden, "insufficient permissions")
    }
}
```

---

## 6. Caching Strategy

### ✅ DO: Implement Multi-Level Caching

```go
// internal/infrastructure/cache/redis.go
package cache

import (
    "context"
    "encoding/json"
    "fmt"
    "time"
    
    "github.com/redis/go-redis/v9"
)

type RedisCache struct {
    client *redis.Client
}

func NewRedisCache(addr string) (*RedisCache, error) {
    client := redis.NewClient(&redis.Options{
        Addr:         addr,
        PoolSize:     10,
        MinIdleConns: 5,
        MaxRetries:   3,
    })
    
    // Test connection
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    if err := client.Ping(ctx).Err(); err != nil {
        return nil, fmt.Errorf("redis ping failed: %w", err)
    }
    
    return &RedisCache{client: client}, nil
}

// Get retrieves and unmarshals data
func (c *RedisCache) Get(ctx context.Context, key string, dest interface{}) error {
    data, err := c.client.Get(ctx, key).Bytes()
    if err != nil {
        if err == redis.Nil {
            return ErrCacheMiss
        }
        return fmt.Errorf("get key %s: %w", key, err)
    }
    
    if err := json.Unmarshal(data, dest); err != nil {
        return fmt.Errorf("unmarshal: %w", err)
    }
    
    return nil
}

// Set marshals and stores data
func (c *RedisCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
    data, err := json.Marshal(value)
    if err != nil {
        return fmt.Errorf("marshal: %w", err)
    }
    
    if err := c.client.Set(ctx, key, data, ttl).Err(); err != nil {
        return fmt.Errorf("set key %s: %w", key, err)
    }
    
    return nil
}

// SetNX sets value only if key doesn't exist (for distributed locks)
func (c *RedisCache) SetNX(ctx context.Context, key string, value interface{}, ttl time.Duration) (bool, error) {
    data, err := json.Marshal(value)
    if err != nil {
        return false, fmt.Errorf("marshal: %w", err)
    }
    
    return c.client.SetNX(ctx, key, data, ttl).Result()
}

// Delete removes keys
func (c *RedisCache) Delete(ctx context.Context, keys ...string) error {
    return c.client.Del(ctx, keys...).Err()
}
```

### ✅ DO: Use Cache-Aside Pattern with Proper Invalidation

```go
// internal/domain/user/service.go
package user

import (
    "context"
    "fmt"
    "time"
    
    "github.com/yourorg/yourapp/internal/infrastructure/cache"
    "github.com/yourorg/yourapp/ent"
)

type Service struct {
    repo  Repository
    cache cache.Cache
}

func (s *Service) GetByID(ctx context.Context, id int) (*ent.User, error) {
    // Try cache first
    key := fmt.Sprintf("user:%d", id)
    var user ent.User
    
    err := s.cache.Get(ctx, key, &user)
    if err == nil {
        return &user, nil
    }
    
    // Cache miss - get from database
    dbUser, err := s.repo.GetByID(ctx, id)
    if err != nil {
        return nil, err
    }
    
    // Cache for future requests
    if dbUser != nil {
        _ = s.cache.Set(ctx, key, dbUser, 5*time.Minute)
    }
    
    return dbUser, nil
}

func (s *Service) Update(ctx context.Context, id int, params UpdateParams) (*ent.User, error) {
    // Update in database
    user, err := s.repo.Update(ctx, id, params)
    if err != nil {
        return nil, err
    }
    
    // Invalidate cache
    key := fmt.Sprintf("user:%d", id)
    _ = s.cache.Delete(ctx, key)
    
    return user, nil
}
```

---

## 7. Testing Strategies

### ✅ DO: Use Table-Driven Tests

```go
// internal/domain/user/service_test.go
package user_test

import (
    "context"
    "errors"
    "testing"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/mock"
    "github.com/yourorg/yourapp/internal/domain/user"
)

func TestUserService_Create(t *testing.T) {
    tests := []struct {
        name    string
        input   user.CreateParams
        setup   func(*mockRepository, *mockCache)
        wantErr error
    }{
        {
            name: "successful creation",
            input: user.CreateParams{
                Email:    "test@example.com",
                Password: "securepass123",
            },
            setup: func(repo *mockRepository, cache *mockCache) {
                repo.On("GetByEmail", mock.Anything, "test@example.com").
                    Return(nil, nil)
                repo.On("Create", mock.Anything, mock.Anything).
                    Return(&ent.User{ID: 1, Email: "test@example.com"}, nil)
            },
            wantErr: nil,
        },
        {
            name: "email already exists",
            input: user.CreateParams{
                Email:    "existing@example.com",
                Password: "password123",
            },
            setup: func(repo *mockRepository, cache *mockCache) {
                repo.On("GetByEmail", mock.Anything, "existing@example.com").
                    Return(&ent.User{ID: 1}, nil)
            },
            wantErr: user.ErrEmailExists,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // Setup
            repo := new(mockRepository)
            cache := new(mockCache)
            tt.setup(repo, cache)
            
            service := user.NewService(repo, cache)
            
            // Execute
            _, err := service.Create(context.Background(), tt.input)
            
            // Assert
            if tt.wantErr != nil {
                assert.ErrorIs(t, err, tt.wantErr)
            } else {
                assert.NoError(t, err)
            }
            
            repo.AssertExpectations(t)
            cache.AssertExpectations(t)
        })
    }
}
```

### ✅ DO: Use Integration Tests with Real Database

```go
// internal/domain/user/repository_integration_test.go
//go:build integration

package user_test

import (
    "context"
    "testing"
    
    "github.com/stretchr/testify/suite"
    "github.com/testcontainers/testcontainers-go"
    "github.com/testcontainers/testcontainers-go/modules/postgres"
    "github.com/yourorg/yourapp/ent"
    "github.com/yourorg/yourapp/ent/enttest"
)

type RepositoryTestSuite struct {
    suite.Suite
    container *postgres.PostgresContainer
    client    *ent.Client
    repo      user.Repository
}

func (s *RepositoryTestSuite) SetupSuite() {
    ctx := context.Background()
    
    // Start PostgreSQL container
    container, err := postgres.Run(ctx,
        "docker.io/postgres:16-alpine",
        postgres.WithDatabase("testdb"),
        postgres.WithUsername("test"),
        postgres.WithPassword("test"),
        testcontainers.WithWaitStrategy(
            wait.ForLog("database system is ready to accept connections").
                WithOccurrence(2).
                WithStartupTimeout(30 * time.Second),
        ),
    )
    s.Require().NoError(err)
    s.container = container
    
    // Get connection string
    connStr, err := container.ConnectionString(ctx, "sslmode=disable")
    s.Require().NoError(err)
    
    // Create Ent client
    s.client = enttest.Open(s.T(), "postgres", connStr)
    
    // Run migrations
    err = s.client.Schema.Create(ctx)
    s.Require().NoError(err)
    
    // Create repository
    s.repo = user.NewRepository(s.client)
}

func (s *RepositoryTestSuite) TearDownSuite() {
    s.client.Close()
    s.container.Terminate(context.Background())
}

func (s *RepositoryTestSuite) TestCreate() {
    ctx := context.Background()
    
    // Create user
    params := user.CreateParams{
        Email:        "test@example.com",
        PasswordHash: "hashedpassword",
    }
    
    created, err := s.repo.Create(ctx, params)
    s.NoError(err)
    s.NotNil(created)
    s.Equal(params.Email, created.Email)
    
    // Verify user exists
    found, err := s.repo.GetByEmail(ctx, params.Email)
    s.NoError(err)
    s.Equal(created.ID, found.ID)
}

func TestRepositoryTestSuite(t *testing.T) {
    suite.Run(t, new(RepositoryTestSuite))
}
```

---

## 8. Performance Optimization

### ✅ DO: Use Connection Pooling and Prepared Statements

```go
// internal/infrastructure/database/optimized.go
package database

import (
    "context"
    "sync"
    
    "github.com/jackc/pgx/v5"
    "github.com/jackc/pgx/v5/pgxpool"
)

type OptimizedRepo struct {
    pool         *pgxpool.Pool
    preparedStmts map[string]*pgconn.StatementDescription
    mu           sync.RWMutex
}

func NewOptimizedRepo(pool *pgxpool.Pool) *OptimizedRepo {
    return &OptimizedRepo{
        pool:          pool,
        preparedStmts: make(map[string]*pgconn.StatementDescription),
    }
}

// PrepareStatements prepares commonly used statements
func (r *OptimizedRepo) PrepareStatements(ctx context.Context) error {
    statements := map[string]string{
        "getUserByID":    "SELECT id, email, status FROM users WHERE id = $1",
        "getUserByEmail": "SELECT id, email, status FROM users WHERE email = $1",
        "updateUserStatus": "UPDATE users SET status = $2, updated_at = NOW() WHERE id = $1",
    }
    
    conn, err := r.pool.Acquire(ctx)
    if err != nil {
        return err
    }
    defer conn.Release()
    
    for name, sql := range statements {
        sd, err := conn.Conn().Prepare(ctx, name, sql)
        if err != nil {
            return fmt.Errorf("prepare %s: %w", name, err)
        }
        
        r.mu.Lock()
        r.preparedStmts[name] = sd
        r.mu.Unlock()
    }
    
    return nil
}

// GetUserByID uses prepared statement
func (r *OptimizedRepo) GetUserByID(ctx context.Context, id int) (*User, error) {
    var user User
    
    err := r.pool.QueryRow(ctx, "getUserByID", id).Scan(
        &user.ID,
        &user.Email,
        &user.Status,
    )
    
    if err == pgx.ErrNoRows {
        return nil, nil
    }
    
    return &user, err
}
```

### ✅ DO: Implement Request Coalescing

```go
// internal/pkg/singleflight/singleflight.go
package singleflight

import (
    "context"
    "sync"
    
    "golang.org/x/sync/singleflight"
)

// Group wraps singleflight.Group with context support
type Group struct {
    sf singleflight.Group
}

// Do executes and returns the results of the given function, making sure that
// only one execution is in-flight for a given key at a time
func (g *Group) Do(ctx context.Context, key string, fn func() (interface{}, error)) (interface{}, error) {
    result := make(chan singleflight.Result, 1)
    
    go func() {
        v, err, _ := g.sf.Do(key, fn)
        result <- singleflight.Result{Val: v, Err: err}
    }()
    
    select {
    case <-ctx.Done():
        return nil, ctx.Err()
    case r := <-result:
        return r.Val, r.Err
    }
}

// Usage in service
type UserService struct {
    repo Repository
    sf   singleflight.Group
}

func (s *UserService) GetByID(ctx context.Context, id int) (*User, error) {
    key := fmt.Sprintf("user:%d", id)
    
    result, err := s.sf.Do(ctx, key, func() (interface{}, error) {
        return s.repo.GetByID(ctx, id)
    })
    
    if err != nil {
        return nil, err
    }
    
    return result.(*User), nil
}
```

### ✅ DO: Use Object Pools for Frequently Allocated Objects

```go
// internal/pkg/pool/buffer_pool.go
package pool

import (
    "bytes"
    "sync"
)

var bufferPool = sync.Pool{
    New: func() interface{} {
        return new(bytes.Buffer)
    },
}

// GetBuffer gets a buffer from the pool
func GetBuffer() *bytes.Buffer {
    buf := bufferPool.Get().(*bytes.Buffer)
    buf.Reset()
    return buf
}

// PutBuffer returns a buffer to the pool
func PutBuffer(buf *bytes.Buffer) {
    if buf.Cap() > 64*1024 { // Don't pool large buffers
        return
    }
    bufferPool.Put(buf)
}

// Usage example
func ProcessData(data []byte) ([]byte, error) {
    buf := GetBuffer()
    defer PutBuffer(buf)
    
    // Use buffer for processing
    if err := processIntoBuffer(data, buf); err != nil {
        return nil, err
    }
    
    // Return copy of data
    result := make([]byte, buf.Len())
    copy(result, buf.Bytes())
    return result, nil
}
```

---

## 9. Configuration Management

### ✅ DO: Use Environment-Based Configuration

```go
// internal/config/config.go
package config

import (
    "fmt"
    "time"
    
    "github.com/caarlos0/env/v11"
    "github.com/joho/godotenv"
)

type Config struct {
    App      AppConfig
    Server   ServerConfig
    Database DatabaseConfig
    Redis    RedisConfig
    Auth     AuthConfig
    Logger   LoggerConfig
}

type AppConfig struct {
    Name        string `env:"APP_NAME" envDefault:"myapp"`
    Environment string `env:"APP_ENV" envDefault:"development"`
    Version     string `env:"APP_VERSION" envDefault:"unknown"`
}

type ServerConfig struct {
    Host         string        `env:"SERVER_HOST" envDefault:"0.0.0.0"`
    Port         int           `env:"SERVER_PORT" envDefault:"8080"`
    ReadTimeout  time.Duration `env:"SERVER_READ_TIMEOUT" envDefault:"10s"`
    WriteTimeout time.Duration `env:"SERVER_WRITE_TIMEOUT" envDefault:"10s"`
    IdleTimeout  time.Duration `env:"SERVER_IDLE_TIMEOUT" envDefault:"120s"`
}

type DatabaseConfig struct {
    Host            string        `env:"DB_HOST" envDefault:"localhost"`
    Port            int           `env:"DB_PORT" envDefault:"5432"`
    User            string        `env:"DB_USER,required"`
    Password        string        `env:"DB_PASSWORD,required"`
    Name            string        `env:"DB_NAME,required"`
    SSLMode         string        `env:"DB_SSLMODE" envDefault:"disable"`
    MaxConnections  int           `env:"DB_MAX_CONNECTIONS" envDefault:"25"`
    MaxIdleConns    int           `env:"DB_MAX_IDLE_CONNS" envDefault:"5"`
    ConnMaxLifetime time.Duration `env:"DB_CONN_MAX_LIFETIME" envDefault:"1h"`
}

type RedisConfig struct {
    Address  string `env:"REDIS_ADDRESS" envDefault:"localhost:6379"`
    Password string `env:"REDIS_PASSWORD"`
    DB       int    `env:"REDIS_DB" envDefault:"0"`
}

type AuthConfig struct {
    AccessTokenSecret  string        `env:"AUTH_ACCESS_SECRET,required"`
    RefreshTokenSecret string        `env:"AUTH_REFRESH_SECRET,required"`
    AccessTokenTTL     time.Duration `env:"AUTH_ACCESS_TTL" envDefault:"15m"`
    RefreshTokenTTL    time.Duration `env:"AUTH_REFRESH_TTL" envDefault:"168h"`
}

type LoggerConfig struct {
    Level      string `env:"LOG_LEVEL" envDefault:"info"`
    Format     string `env:"LOG_FORMAT" envDefault:"json"`
    AddCaller  bool   `env:"LOG_ADD_CALLER" envDefault:"true"`
    StackTrace bool   `env:"LOG_STACK_TRACE" envDefault:"false"`
}

// Load loads configuration from environment
func Load() (*Config, error) {
    // Load .env file in development
    if err := godotenv.Load(); err != nil {
        // Ignore error if .env doesn't exist
    }
    
    cfg := &Config{}
    if err := env.Parse(cfg); err != nil {
        return nil, fmt.Errorf("parse env: %w", err)
    }
    
    return cfg, nil
}

// Validate validates the configuration
func (c *Config) Validate() error {
    if c.App.Environment != "development" && 
       c.App.Environment != "staging" && 
       c.App.Environment != "production" {
        return fmt.Errorf("invalid environment: %s", c.App.Environment)
    }
    
    if c.Database.MaxConnections < c.Database.MaxIdleConns {
        return fmt.Errorf("max connections must be >= max idle connections")
    }
    
    return nil
}
```

---

## 10. Graceful Shutdown

### ✅ DO: Implement Proper Shutdown Handling

```go
// cmd/api/main.go
package main

import (
    "context"
    "errors"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"
    
    "github.com/gofiber/fiber/v3"
    "go.uber.org/zap"
)

func main() {
    // Initialize dependencies
    cfg, err := config.Load()
    if err != nil {
        log.Fatal("load config", err)
    }
    
    logger, err := logger.New(cfg.Logger)
    if err != nil {
        log.Fatal("init logger", err)
    }
    
    // Initialize app with dependency injection
    app, cleanup, err := initializeApp(cfg, logger)
    if err != nil {
        logger.Fatal("initialize app", zap.Error(err))
    }
    defer cleanup()
    
    // Create server
    server := &http.Server{
        Addr:         fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port),
        Handler:      app.Handler(),
        ReadTimeout:  cfg.Server.ReadTimeout,
        WriteTimeout: cfg.Server.WriteTimeout,
        IdleTimeout:  cfg.Server.IdleTimeout,
    }
    
    // Start server in goroutine
    serverErrors := make(chan error, 1)
    go func() {
        logger.Info("starting server",
            zap.String("address", server.Addr),
            zap.String("version", cfg.App.Version),
        )
        serverErrors <- server.ListenAndServe()
    }()
    
    // Create shutdown channel
    shutdown := make(chan os.Signal, 1)
    signal.Notify(shutdown, os.Interrupt, syscall.SIGTERM)
    
    // Wait for shutdown signal or server error
    select {
    case err := <-serverErrors:
        if !errors.Is(err, http.ErrServerClosed) {
            logger.Error("server error", zap.Error(err))
        }
        
    case sig := <-shutdown:
        logger.Info("shutdown signal received", zap.String("signal", sig.String()))
        
        // Create shutdown context with timeout
        ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
        defer cancel()
        
        // Gracefully shutdown server
        if err := server.Shutdown(ctx); err != nil {
            logger.Error("server shutdown error", zap.Error(err))
            server.Close()
        }
    }
    
    logger.Info("server stopped")
}

// initializeApp uses Wire for dependency injection
func initializeApp(cfg *config.Config, logger *zap.Logger) (*fiber.App, func(), error) {
    // This would be generated by Wire
    db, err := database.NewPgxPool(context.Background(), cfg.Database)
    if err != nil {
        return nil, nil, err
    }
    
    entClient, err := database.NewEntClient(db)
    if err != nil {
        return nil, nil, err
    }
    
    redisCache, err := cache.NewRedisCache(cfg.Redis.Address)
    if err != nil {
        return nil, nil, err
    }
    
    // ... initialize other dependencies
    
    app := fiber.New(fiber.Config{
        ErrorHandler: middleware.NewErrorMiddleware(logger),
        ReadTimeout:  cfg.Server.ReadTimeout,
        WriteTimeout: cfg.Server.WriteTimeout,
    })
    
    // Setup routes
    router.Setup(app, handlers, middlewares)
    
    // Cleanup function
    cleanup := func() {
        db.Close()
        entClient.Close()
        redisCache.Close()
        logger.Sync()
    }
    
    return app, cleanup, nil
}
```

---

## 11. Monitoring and Observability

### ✅ DO: Implement OpenTelemetry

```go
// internal/pkg/telemetry/telemetry.go
package telemetry

import (
    "context"
    "time"
    
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/otel/exporters/otlp/otlptrace"
    "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
    "go.opentelemetry.io/otel/propagation"
    "go.opentelemetry.io/otel/sdk/resource"
    sdktrace "go.opentelemetry.io/otel/sdk/trace"
    semconv "go.opentelemetry.io/otel/semconv/v1.24.0"
    "go.opentelemetry.io/otel/trace"
)

// InitTracer initializes OpenTelemetry tracer
func InitTracer(ctx context.Context, serviceName, version string) (func(), error) {
    // Create OTLP exporter
    exporter, err := otlptrace.New(
        ctx,
        otlptracegrpc.NewClient(
            otlptracegrpc.WithEndpoint("localhost:4317"),
            otlptracegrpc.WithInsecure(),
        ),
    )
    if err != nil {
        return nil, err
    }
    
    // Create resource
    res, err := resource.New(ctx,
        resource.WithAttributes(
            semconv.ServiceName(serviceName),
            semconv.ServiceVersion(version),
        ),
    )
    if err != nil {
        return nil, err
    }
    
    // Create tracer provider
    tp := sdktrace.NewTracerProvider(
        sdktrace.WithBatcher(exporter),
        sdktrace.WithResource(res),
        sdktrace.WithSampler(sdktrace.AlwaysSample()),
    )
    
    otel.SetTracerProvider(tp)
    otel.SetTextMapPropagator(
        propagation.NewCompositeTextMapPropagator(
            propagation.TraceContext{},
            propagation.Baggage{},
        ),
    )
    
    // Return cleanup function
    return func() {
        ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
        defer cancel()
        tp.Shutdown(ctx)
    }, nil
}

// Middleware for Fiber
func TracingMiddleware() fiber.Handler {
    return func(c fiber.Ctx) error {
        tracer := otel.Tracer("fiber")
        
        // Extract context from headers
        ctx := otel.GetTextMapPropagator().Extract(
            c.Context(),
            propagation.HeaderCarrier(c.GetReqHeaders()),
        )
        
        // Start span
        ctx, span := tracer.Start(ctx, c.Route().Path,
            trace.WithAttributes(
                semconv.HTTPMethod(c.Method()),
                semconv.HTTPTarget(c.OriginalURL()),
                semconv.HTTPRoute(c.Route().Path),
            ),
        )
        defer span.End()
        
        // Store context
        c.SetUserContext(ctx)
        
        // Process request
        err := c.Next()
        
        // Set status
        statusCode := c.Response().StatusCode()
        span.SetAttributes(semconv.HTTPStatusCode(statusCode))
        
        if err != nil {
            span.RecordError(err)
        }
        
        return err
    }
}
```

### ✅ DO: Export Prometheus Metrics

```go
// internal/pkg/metrics/metrics.go
package metrics

import (
    "github.com/gofiber/fiber/v3"
    "github.com/gofiber/fiber/v3/middleware/adaptor"
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
    httpRequestsTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total number of HTTP requests",
        },
        []string{"method", "route", "status"},
    )
    
    httpRequestDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "http_request_duration_seconds",
            Help:    "HTTP request duration in seconds",
            Buckets: prometheus.DefBuckets,
        },
        []string{"method", "route"},
    )
    
    dbQueriesTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "db_queries_total",
            Help: "Total number of database queries",
        },
        []string{"query_type", "table"},
    )
    
    cacheHitsTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "cache_hits_total",
            Help: "Total number of cache hits",
        },
        []string{"cache_type"},
    )
)

// PrometheusMiddleware collects HTTP metrics
func PrometheusMiddleware() fiber.Handler {
    return func(c fiber.Ctx) error {
        start := time.Now()
        
        // Process request
        err := c.Next()
        
        // Record metrics
        duration := time.Since(start).Seconds()
        status := strconv.Itoa(c.Response().StatusCode())
        route := c.Route().Path
        method := c.Method()
        
        httpRequestsTotal.WithLabelValues(method, route, status).Inc()
        httpRequestDuration.WithLabelValues(method, route).Observe(duration)
        
        return err
    }
}

// Handler returns Prometheus metrics handler
func Handler() fiber.Handler {
    return adaptor.HTTPHandler(promhttp.Handler())
}

// RecordDBQuery records database query metrics
func RecordDBQuery(queryType, table string) {
    dbQueriesTotal.WithLabelValues(queryType, table).Inc()
}

// RecordCacheHit records cache hit
func RecordCacheHit(cacheType string) {
    cacheHitsTotal.WithLabelValues(cacheType).Inc()
}
```

---

## 12. Production Deployment

### ✅ DO: Use Multi-Stage Docker Builds

```dockerfile
# Dockerfile
# Stage 1: Build
FROM golang:1.23-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git make

WORKDIR /build

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source
COPY . .

# Build binary
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s -X main.version=$(git describe --tags --always)" \
    -o app ./cmd/api

# Stage 2: Runtime
FROM gcr.io/distroless/static-debian12:nonroot

# Copy binary
COPY --from=builder /build/app /app

# Copy migrations
COPY --from=builder /build/migrations /migrations

# Use non-root user
USER nonroot:nonroot

EXPOSE 8080

ENTRYPOINT ["/app"]
```

### ✅ DO: Use Kubernetes for Orchestration

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  labels:
    app: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: api
        image: myapp/api:latest
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: APP_ENV
          value: "production"
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: password
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /health/live
            port: http
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: api
spec:
  selector:
    app: api
  ports:
  - port: 80
    targetPort: http
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 13. Advanced Patterns

### Context Propagation for Distributed Systems

```go
// internal/pkg/context/context.go
package context

import (
    "context"
    
    "github.com/google/uuid"
)

type contextKey string

const (
    requestIDKey contextKey = "request_id"
    userIDKey    contextKey = "user_id"
    traceIDKey   contextKey = "trace_id"
)

// WithRequestID adds request ID to context
func WithRequestID(ctx context.Context, requestID string) context.Context {
    return context.WithValue(ctx, requestIDKey, requestID)
}

// RequestID gets request ID from context
func RequestID(ctx context.Context) string {
    if v, ok := ctx.Value(requestIDKey).(string); ok {
        return v
    }
    return ""
}

// WithUserID adds user ID to context
func WithUserID(ctx context.Context, userID int) context.Context {
    return context.WithValue(ctx, userIDKey, userID)
}

// UserID gets user ID from context
func UserID(ctx context.Context) int {
    if v, ok := ctx.Value(userIDKey).(int); ok {
        return v
    }
    return 0
}
```

### Circuit Breaker Pattern

```go
// internal/pkg/circuitbreaker/breaker.go
package circuitbreaker

import (
    "context"
    "errors"
    "sync"
    "time"
)

var ErrCircuitOpen = errors.New("circuit breaker is open")

type State int

const (
    StateClosed State = iota
    StateOpen
    StateHalfOpen
)

type CircuitBreaker struct {
    mu              sync.RWMutex
    name            string
    maxFailures     int
    resetTimeout    time.Duration
    state           State
    failures        int
    lastFailureTime time.Time
    successCount    int
}

func New(name string, maxFailures int, resetTimeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        name:         name,
        maxFailures:  maxFailures,
        resetTimeout: resetTimeout,
        state:        StateClosed,
    }
}

func (cb *CircuitBreaker) Execute(ctx context.Context, fn func() error) error {
    if err := cb.canExecute(); err != nil {
        return err
    }
    
    err := fn()
    cb.recordResult(err)
    return err
}

func (cb *CircuitBreaker) canExecute() error {
    cb.mu.RLock()
    defer cb.mu.RUnlock()
    
    switch cb.state {
    case StateOpen:
        if time.Since(cb.lastFailureTime) > cb.resetTimeout {
            cb.mu.RUnlock()
            cb.mu.Lock()
            cb.state = StateHalfOpen
            cb.mu.Unlock()
            cb.mu.RLock()
            return nil
        }
        return ErrCircuitOpen
    case StateHalfOpen:
        return nil
    default:
        return nil
    }
}

func (cb *CircuitBreaker) recordResult(err error) {
    cb.mu.Lock()
    defer cb.mu.Unlock()
    
    if err != nil {
        cb.failures++
        cb.lastFailureTime = time.Now()
        
        if cb.state == StateHalfOpen {
            cb.state = StateOpen
        } else if cb.failures >= cb.maxFailures {
            cb.state = StateOpen
        }
    } else {
        if cb.state == StateHalfOpen {
            cb.successCount++
            if cb.successCount >= cb.maxFailures {
                cb.state = StateClosed
                cb.failures = 0
                cb.successCount = 0
            }
        } else {
            cb.failures = 0
        }
    }
}
```

### Feature Flags

```go
// internal/pkg/feature/flags.go
package feature

import (
    "context"
    "sync"
    "time"
)

type Flag string

const (
    FlagNewPaymentFlow Flag = "new_payment_flow"
    FlagBetaFeatures   Flag = "beta_features"
)

type Manager struct {
    mu    sync.RWMutex
    flags map[Flag]FlagConfig
}

type FlagConfig struct {
    Enabled    bool
    Percentage int // 0-100 for gradual rollout
    UserIDs    []int
}

func NewManager() *Manager {
    return &Manager{
        flags: make(map[Flag]FlagConfig),
    }
}

func (m *Manager) IsEnabled(ctx context.Context, flag Flag) bool {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    config, exists := m.flags[flag]
    if !exists || !config.Enabled {
        return false
    }
    
    // Check user-specific enablement
    userID := UserID(ctx)
    for _, id := range config.UserIDs {
        if id == userID {
            return true
        }
    }
    
    // Check percentage rollout
    if config.Percentage > 0 && config.Percentage < 100 {
        // Simple hash-based rollout
        return (userID % 100) < config.Percentage
    }
    
    return config.Percentage >= 100
}
```

---

## 14. Security Best Practices

### ✅ DO: Implement Rate Limiting

```go
// internal/api/middleware/ratelimit.go
package middleware

import (
    "fmt"
    "time"
    
    "github.com/gofiber/fiber/v3"
    "github.com/gofiber/fiber/v3/middleware/limiter"
    "github.com/redis/go-redis/v9"
)

// NewRateLimiter creates a rate limiting middleware with Redis backend
func NewRateLimiter(redisClient *redis.Client) fiber.Handler {
    return limiter.New(limiter.Config{
        Max:        100,
        Expiration: 1 * time.Minute,
        KeyGenerator: func(c fiber.Ctx) string {
            // Use user ID if authenticated, otherwise IP
            if userID := c.Locals("user_id"); userID != nil {
                return fmt.Sprintf("user:%v", userID)
            }
            return c.IP()
        },
        Storage: newRedisStorage(redisClient),
        LimitReached: func(c fiber.Ctx) error {
            return fiber.NewError(
                fiber.StatusTooManyRequests,
                "Rate limit exceeded",
            )
        },
        SkipFailedRequests:     false,
        SkipSuccessfulRequests: false,
    })
}

// Custom rate limiter for specific endpoints
func NewEndpointRateLimiter(limit int, window time.Duration) fiber.Handler {
    return func(c fiber.Ctx) error {
        key := fmt.Sprintf("endpoint:%s:%s", c.Route().Path, c.IP())
        
        // Implementation using sliding window
        // ...
        
        return c.Next()
    }
}
```

### ✅ DO: Validate All Input

```go
// internal/pkg/validator/validator.go
package validator

import (
    "fmt"
    "strings"
    
    "github.com/go-playground/validator/v10"
)

var validate *validator.Validate

func init() {
    validate = validator.New()
    
    // Register custom validators
    validate.RegisterValidation("password", validatePassword)
    validate.RegisterValidation("username", validateUsername)
}

// Validate validates a struct
func Validate(s interface{}) error {
    if err := validate.Struct(s); err != nil {
        return formatValidationError(err)
    }
    return nil
}

// Custom validators
func validatePassword(fl validator.FieldLevel) bool {
    password := fl.Field().String()
    
    // At least 8 chars, 1 upper, 1 lower, 1 digit
    if len(password) < 8 {
        return false
    }
    
    var hasUpper, hasLower, hasDigit bool
    for _, char := range password {
        switch {
        case 'A' <= char && char <= 'Z':
            hasUpper = true
        case 'a' <= char && char <= 'z':
            hasLower = true
        case '0' <= char && char <= '9':
            hasDigit = true
        }
    }
    
    return hasUpper && hasLower && hasDigit
}

func validateUsername(fl validator.FieldLevel) bool {
    username := fl.Field().String()
    
    // 3-20 chars, alphanumeric and underscore only
    if len(username) < 3 || len(username) > 20 {
        return false
    }
    
    for _, char := range username {
        if !((char >= 'a' && char <= 'z') ||
            (char >= 'A' && char <= 'Z') ||
            (char >= '0' && char <= '9') ||
            char == '_') {
            return false
        }
    }
    
    return true
}

// Usage in handler
type CreateUserRequest struct {
    Username string `json:"username" validate:"required,username"`
    Email    string `json:"email" validate:"required,email"`
    Password string `json:"password" validate:"required,password"`
}
```

---

## 15. Common Pitfalls and Solutions

### ❌ DON'T: Ignore Context Cancellation

```go
// Bad - ignores context
func (s *Service) ProcessBatch(items []Item) error {
    for _, item := range items {
        // This could run forever even if request is cancelled
        if err := s.processItem(item); err != nil {
            return err
        }
    }
    return nil
}

// Good - respects context
func (s *Service) ProcessBatch(ctx context.Context, items []Item) error {
    for _, item := range items {
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
            if err := s.processItem(ctx, item); err != nil {
                return err
            }
        }
    }
    return nil
}
```

### ❌ DON'T: Leak Goroutines

```go
// Bad - goroutine leak
func (s *Service) StartWorker() {
    go func() {
        for {
            // This runs forever with no way to stop
            s.doWork()
            time.Sleep(time.Second)
        }
    }()
}

// Good - controllable goroutine
func (s *Service) StartWorker(ctx context.Context) {
    go func() {
        ticker := time.NewTicker(time.Second)
        defer ticker.Stop()
        
        for {
            select {
            case <-ctx.Done():
                return
            case <-ticker.C:
                s.doWork()
            }
        }
    }()
}
```

### ❌ DON'T: Use `interface{}` Everywhere

```go
// Bad - loses type safety
func ProcessData(data interface{}) (interface{}, error) {
    // Lots of type assertions needed
    switch v := data.(type) {
    case string:
        return processString(v), nil
    case int:
        return processInt(v), nil
    default:
        return nil, errors.New("unsupported type")
    }
}

// Good - use generics (Go 1.18+)
func ProcessData[T Processable](data T) (T, error) {
    return data.Process()
}

type Processable interface {
    Process() error
}
```

### ✅ DO: Use Functional Options Pattern

```go
// internal/infrastructure/database/options.go
package database

type Option func(*Config)

func WithMaxConnections(n int) Option {
    return func(c *Config) {
        c.MaxConnections = n
    }
}

func WithTimeout(d time.Duration) Option {
    return func(c *Config) {
        c.Timeout = d
    }
}

// Usage
db, err := NewDatabase(
    WithMaxConnections(50),
    WithTimeout(30 * time.Second),
)
```

---

## Migration Guide: From Standard Library to This Stack

### From `net/http` to Fiber

```go
// Before - net/http
func handler(w http.ResponseWriter, r *http.Request) {
    userID := r.URL.Query().Get("id")
    
    user, err := getUser(userID)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

// After - Fiber
func handler(c fiber.Ctx) error {
    userID := c.Query("id")
    
    user, err := getUser(userID)
    if err != nil {
        return fiber.NewError(fiber.StatusInternalServerError, err.Error())
    }
    
    return c.JSON(user)
}
```

### From `database/sql` to Ent

```go
// Before - database/sql
func GetUser(db *sql.DB, id int) (*User, error) {
    var user User
    err := db.QueryRow(
        "SELECT id, email, created_at FROM users WHERE id = $1",
        id,
    ).Scan(&user.ID, &user.Email, &user.CreatedAt)
    
    if err == sql.ErrNoRows {
        return nil, nil
    }
    return &user, err
}

// After - Ent
func GetUser(client *ent.Client, id int) (*ent.User, error) {
    return client.User.Get(context.Background(), id)
}
```

---

## Conclusion

This guide provides a comprehensive foundation for building high-performance Go web applications in 2025. Remember:

1. **Performance**: Use Fiber for speed, Echo for compatibility
2. **Type Safety**: Leverage Ent ORM for compile-time guarantees
3. **Observability**: Implement tracing, metrics, and structured logging from day one
4. **Testing**: Write table-driven tests and integration tests with real databases
5. **Production**: Use proper configuration management, graceful shutdown, and container orchestration

The patterns and practices outlined here are battle-tested and scale from small services to large distributed systems. Adapt them to your specific needs, but always prioritize clarity, maintainability, and performance.

Happy coding! 🚀