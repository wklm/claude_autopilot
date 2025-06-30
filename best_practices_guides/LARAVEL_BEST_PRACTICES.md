# Laravel 12 Production Best Practices Guide (Late 2025 Edition)

A comprehensive, opinionated guide for building scalable, secure, and performant web applications with Laravel 12, emphasizing modern PHP 8.4+ features, type safety, and production-grade patterns.

## Prerequisites & Initial Setup

This guide assumes **Laravel 12.x**, **PHP 8.4+**, **Composer 2.9+**, and **Node.js 22+**. We'll leverage PHP 8.4's property hooks, improved performance, and enhanced type system throughout.

### Project Initialization with Modern Tooling

```bash
# Create new Laravel project with all the bells and whistles
composer create-project laravel/laravel myapp "12.*" --prefer-dist
cd myapp

# Install development dependencies
composer require --dev pestphp/pest pestphp/pest-plugin-laravel
composer require --dev phpstan/phpstan larastan/larastan
composer require --dev friendsofphp/php-cs-fixer
composer require --dev barryvdh/laravel-ide-helper

# Production performance tools
composer require laravel/octane spiral/roadrunner-laravel
composer require laravel/telescope --dev
composer require spatie/laravel-ray --dev

# Modern frontend (choose your path)
php artisan install:api --pest --no-interaction  # For API-only
# OR
php artisan install:inertia --vue --ssr --pest  # For Inertia.js
# OR  
php artisan livewire:install                     # For Livewire 3
```

### Essential Configuration Files

**`.env` with structured sections:**
```env
# Application
APP_NAME="MyApp"
APP_ENV=local
APP_KEY=base64:...
APP_DEBUG=true
APP_TIMEZONE=UTC
APP_URL=http://localhost:8000

# Security
SESSION_SECURE_COOKIE=true
SESSION_SAME_SITE=strict
SANCTUM_STATEFUL_DOMAINS=localhost:3000
TRUSTED_PROXIES=*
TRUSTED_HOSTS=

# Database with read replicas
DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=myapp
DB_USERNAME=root
DB_PASSWORD=
DB_READ_HOST=127.0.0.1  # Read replica for scaling

# Performance
OCTANE_SERVER=roadrunner
CACHE_DRIVER=redis
SESSION_DRIVER=redis
QUEUE_CONNECTION=redis
REDIS_CLIENT=phpredis

# Monitoring
TELESCOPE_ENABLED=true
DEBUGBAR_ENABLED=false
RAY_ENABLED=true
```

**`phpstan.neon` for maximum type safety:**
```neon
includes:
    - vendor/larastan/larastan/extension.neon

parameters:
    level: 9
    paths:
        - app
        - config
        - database
        - routes
    excludePaths:
        - database/migrations/*
    treatPhpDocTypesAsCertain: true
    strictRules:
        disallowedLooseComparison: true
        booleansInConditions: true
        uselessCast: true
        requireParentConstructorCall: true
```

**`composer.json` scripts section:**
```json
{
    "scripts": {
        "analyze": [
            "vendor/bin/phpstan analyse --memory-limit=2G"
        ],
        "test": [
            "vendor/bin/pest --parallel"
        ],
        "fix": [
            "vendor/bin/php-cs-fixer fix"
        ],
        "check": [
            "@fix",
            "@analyze", 
            "@test"
        ]
    }
}
```

---

## 1. Architecture & Project Structure

Laravel 12's default structure is good for small projects, but production applications need more organization. Adopt a domain-driven structure with clear boundaries.

### Laravel 12 New Features

Laravel 12, released February 24, 2025, brings minimal breaking changes while introducing powerful new capabilities:

**✅ New Application Starter Kits**
- React and Vue starters with Inertia 2, TypeScript, shadcn/ui, and Tailwind CSS
- Livewire starters with Flux UI component library and Laravel Volt
- WorkOS AuthKit integration for social auth, passkeys, and SSO (free up to 1M MAU)

**✅ Enhanced Dependency Compatibility**
- UUIDv7 by default with `HasUuids` trait (ordered UUIDs)
- Carbon 3.x as the default date library
- Improved concurrency with associative array key preservation
- Container now respects class property default values

**✅ Performance & Database Improvements**
- Multi-schema database inspection across all databases
- Improved query performance with schema-qualified table names
- Enhanced memory management for high-traffic applications

### ✅ DO: Use Domain-Driven Design (DDD) Structure

```
/app
├── Console/
│   └── Commands/
├── Domain/                    # Business logic organized by domain
│   ├── User/
│   │   ├── Actions/          # Single-purpose service classes
│   │   │   ├── CreateUser.php
│   │   │   └── UpdateUserProfile.php
│   │   ├── Data/             # DTOs for type safety
│   │   │   ├── UserData.php
│   │   │   └── UserProfileData.php
│   │   ├── Events/
│   │   ├── Models/
│   │   │   └── User.php
│   │   ├── Queries/          # Read-optimized query builders
│   │   │   └── UserQuery.php
│   │   └── Rules/            # Domain-specific validation
│   └── Product/
│       ├── Actions/
│       ├── Data/
│       ├── Models/
│       └── Services/         # Complex orchestration logic
├── Http/
│   ├── Controllers/          # Thin controllers
│   ├── Middleware/
│   ├── Requests/            # Form requests for validation
│   └── Resources/           # API transformations
├── Infrastructure/          # External service integrations
│   ├── Payment/
│   │   ├── PaymentGatewayInterface.php
│   │   └── StripePaymentGateway.php
│   └── Notification/
└── Support/                 # Framework-agnostic helpers
    ├── Enums/
    └── Traits/
```

### ✅ DO: Implement Action Classes for Business Logic

Actions provide a single point of entry for business operations, making testing and reuse trivial.

```php
<?php

declare(strict_types=1);

namespace App\Domain\User\Actions;

use App\Domain\User\Data\UserData;
use App\Domain\User\Events\UserCreated;
use App\Domain\User\Models\User;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Hash;

final readonly class CreateUser
{
    public function execute(UserData $data): User
    {
        return DB::transaction(function () use ($data) {
            $user = User::create([
                'name' => $data->name,
                'email' => $data->email,
                'password' => Hash::make($data->password),
            ]);

            $user->profile()->create([
                'bio' => $data->bio,
                'avatar_url' => $data->avatarUrl,
            ]);

            event(new UserCreated($user));

            return $user->fresh(['profile']);
        });
    }
}
```

### ✅ DO: Use Data Transfer Objects (DTOs) with PHP 8.4 Property Hooks

DTOs provide type safety and validation at the boundaries of your application. PHP 8.4's property hooks eliminate boilerplate.

```php
<?php

declare(strict_types=1);

namespace App\Domain\User\Data;

use Spatie\LaravelData\Attributes\Validation\Email;
use Spatie\LaravelData\Attributes\Validation\Max;
use Spatie\LaravelData\Attributes\Validation\Required;
use Spatie\LaravelData\Data;

final class UserData extends Data
{
    public function __construct(
        #[Required, Max(255)]
        public string $name,
        
        #[Required, Email]
        public string $email,
        
        #[Required, Max(255)]
        public string $password,
        
        public ?string $bio = null,
        
        public ?string $avatarUrl = null,
    ) {
        // PHP 8.4 property hooks for computed properties
        public string $displayName { 
            get => ucwords($this->name);
        }
        
        public bool $hasAvatar {
            get => $this->avatarUrl !== null;
        }
    }
    
    public static function fromRequest(Request $request): self
    {
        return self::from($request->validated());
    }
}
```

### ❌ DON'T: Use Anemic Models or God Controllers

```php
// Bad - Everything in the controller
class UserController extends Controller
{
    public function store(Request $request)
    {
        $validated = $request->validate([...]);
        
        $user = new User();
        $user->name = $validated['name'];
        $user->email = $validated['email'];
        $user->password = bcrypt($validated['password']);
        $user->save();
        
        // More business logic...
        // Email sending...
        // Event dispatching...
        
        return response()->json($user);
    }
}
```

---

## 2. Database & Eloquent Patterns

### ✅ DO: Use Query Objects for Complex Queries

Encapsulate complex queries in dedicated classes for reusability and testing.

```php
<?php

declare(strict_types=1);

namespace App\Domain\User\Queries;

use App\Domain\User\Models\User;
use Illuminate\Database\Eloquent\Builder;
use Illuminate\Support\Collection;

final class ActiveUsersQuery
{
    private Builder $query;
    
    public function __construct()
    {
        $this->query = User::query()
            ->where('active', true)
            ->whereNotNull('email_verified_at');
    }
    
    public function withPosts(): self
    {
        $this->query->with(['posts' => function ($query) {
            $query->published()->latest();
        }]);
        
        return $this;
    }
    
    public function fromCountry(string $countryCode): self
    {
        $this->query->where('country_code', $countryCode);
        
        return $this;
    }
    
    public function get(): Collection
    {
        return $this->query->get();
    }
    
    public function paginate(int $perPage = 15): LengthAwarePaginator
    {
        return $this->query->paginate($perPage);
    }
}

// Usage
$users = (new ActiveUsersQuery())
    ->fromCountry('US')
    ->withPosts()
    ->paginate(20);
```

### ✅ DO: Leverage Advanced Eloquent Features

```php
<?php

namespace App\Domain\Product\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Casts\AsArrayObject;
use Illuminate\Database\Eloquent\Casts\Attribute;

class Product extends Model
{
    protected $fillable = ['name', 'price', 'metadata'];
    
    protected $casts = [
        'price' => 'decimal:2',
        'metadata' => AsArrayObject::class, // Allows object access to JSON
        'published_at' => 'immutable_datetime',
    ];
    
    // PHP 8.4 property hooks in Eloquent accessors
    protected function displayPrice(): Attribute
    {
        return Attribute::make(
            get: fn () => '$' . number_format($this->price, 2)
        );
    }
    
    // Efficient scope with index hints
    public function scopePublished(Builder $query): void
    {
        $query->whereNotNull('published_at')
              ->where('published_at', '<=', now())
              ->orderBy('published_at', 'desc');
    }
    
    // Use database-level computations
    public function scopeWithRevenue(Builder $query): void
    {
        $query->selectRaw('products.*, 
            (SELECT COALESCE(SUM(quantity * price), 0) 
             FROM order_items 
             WHERE order_items.product_id = products.id
             AND order_items.created_at >= ?) as revenue', 
            [now()->subMonth()]
        );
    }
}
```

### ✅ DO: Optimize N+1 Queries with Eager Loading

```php
// Bad - N+1 query problem
$posts = Post::all();
foreach ($posts as $post) {
    echo $post->author->name; // Queries DB for each post
}

// Good - Eager load relationships
$posts = Post::with(['author', 'comments.user'])->get();

// Better - Use query-time loading for conditional relationships
$posts = Post::query()
    ->with(['author:id,name', 'tags:id,name']) // Select only needed columns
    ->withCount('comments') // Adds comments_count attribute
    ->when($request->include_stats, function ($query) {
        $query->withAvg('ratings', 'score')
              ->withSum('purchases', 'amount');
    })
    ->get();
```

### ✅ DO: Use Database Transactions Properly

```php
use Illuminate\Support\Facades\DB;

// Automatic transaction with callbacks
$user = DB::transaction(function () use ($data) {
    $user = User::create($data['user']);
    $user->roles()->attach($data['roles']);
    $user->profile()->create($data['profile']);
    
    return $user;
}, attempts: 5); // Retry on deadlock

// Manual transaction for more control
DB::beginTransaction();
try {
    // Complex operations...
    DB::commit();
} catch (\Exception $e) {
    DB::rollBack();
    throw $e;
}

// Use read replicas for heavy queries
$analytics = DB::connection('mysql_read')
    ->table('users')
    ->select(DB::raw('DATE(created_at) as date'), DB::raw('COUNT(*) as count'))
    ->groupBy('date')
    ->get();
```

---

## 3. Modern API Development

### ✅ DO: Use API Resources with Fractal-like Transformations

```php
<?php

namespace App\Http\Resources;

use Illuminate\Http\Request;
use Illuminate\Http\Resources\Json\JsonResource;

class UserResource extends JsonResource
{
    public function toArray(Request $request): array
    {
        return [
            'id' => $this->id,
            'name' => $this->name,
            'email' => $this->email,
            'avatar' => $this->when(
                $this->hasAvatar(),
                fn () => $this->avatar_url
            ),
            'stats' => $this->whenLoaded('stats', fn () => [
                'posts_count' => $this->stats->posts_count,
                'comments_count' => $this->stats->comments_count,
            ]),
            'posts' => PostResource::collection($this->whenLoaded('posts')),
            'created_at' => $this->created_at->toIso8601String(),
            'links' => [
                'self' => route('api.users.show', $this),
                'posts' => route('api.users.posts.index', $this),
            ],
        ];
    }
}
```

### ✅ DO: Implement Robust API Versioning

```php
// routes/api.php
use Illuminate\Support\Facades\Route;

Route::prefix('v1')->name('api.v1.')->group(function () {
    require __DIR__.'/api/v1.php';
});

Route::prefix('v2')->name('api.v2.')->group(function () {
    require __DIR__.'/api/v2.php';
});

// config/app.php
'api_version' => env('API_VERSION', 'v1'),

// Middleware for version negotiation
namespace App\Http\Middleware;

class ApiVersioning
{
    public function handle(Request $request, Closure $next)
    {
        $version = $request->header('X-API-Version', config('app.api_version'));
        
        if (!in_array($version, ['v1', 'v2'])) {
            return response()->json([
                'error' => 'Invalid API version'
            ], 400);
        }
        
        $request->attributes->set('api_version', $version);
        
        return $next($request);
    }
}
```

### ✅ DO: Use OpenAPI/Swagger Annotations

```php
<?php

namespace App\Http\Controllers\Api;

use OpenApi\Attributes as OA;

#[OA\Info(title: "MyApp API", version: "2.0")]
class UserController extends Controller
{
    #[OA\Get(
        path: "/api/v1/users/{id}",
        summary: "Get user by ID",
        tags: ["Users"],
        parameters: [
            new OA\Parameter(
                name: "id",
                in: "path",
                required: true,
                schema: new OA\Schema(type: "integer")
            )
        ],
        responses: [
            new OA\Response(
                response: 200,
                description: "Success",
                content: new OA\JsonContent(ref: "#/components/schemas/User")
            ),
            new OA\Response(
                response: 404,
                description: "User not found"
            )
        ]
    )]
    public function show(User $user): UserResource
    {
        return new UserResource($user->load(['profile', 'stats']));
    }
}
```

---

## 4. Performance Optimization with Laravel Octane

### ✅ DO: Configure Octane for Production

```php
// config/octane.php
return [
    'server' => env('OCTANE_SERVER', 'roadrunner'),
    
    'https' => env('OCTANE_HTTPS', true),
    
    'listeners' => [
        WorkerStarting::class => [
            EnsureFrontendRequestsAreStateful::class,
            WarmServices::class, // Custom warming
        ],
        
        RequestReceived::class => [
            ...Octane::prepareApplicationForNextOperation(),
            InjectCorrelationId::class,
        ],
    ],
    
    'warm' => [
        ...Octane::defaultServicesToWarm(),
        'cache',
        'cache.store',
        'config',
        'db',
        'view',
    ],
    
    'tables' => [
        'users:1000' => [
            'id', 'name', 'email',
        ],
    ],
];
```

### ✅ DO: Write Octane-Safe Code

```php
<?php

namespace App\Services;

use Illuminate\Contracts\Foundation\Application;

// Bad - Stateful service that causes memory leaks
class BadAnalyticsService
{
    private array $events = []; // State persists between requests!
    
    public function track(string $event): void
    {
        $this->events[] = $event;
    }
}

// Good - Stateless service safe for long-running processes
final readonly class AnalyticsService
{
    public function __construct(
        private Application $app,
        private string $apiKey,
    ) {}
    
    public function track(string $event, array $properties = []): void
    {
        // Dispatch to queue for processing
        dispatch(new TrackAnalyticsEvent($event, $properties));
    }
}

// Reset state between requests if needed
class ResettableService implements \Laravel\Octane\Contracts\ResetsState
{
    private array $requestData = [];
    
    public function reset(): void
    {
        $this->requestData = [];
    }
}
```

### ✅ DO: Implement Concurrent Task Processing

```php
use Laravel\Octane\Facades\Octane;
use Illuminate\Support\Facades\Http;

// Process multiple tasks concurrently
[$users, $posts, $stats] = Octane::concurrently([
    fn () => User::active()->get(),
    fn () => Post::published()->latest()->take(10)->get(),
    fn () => cache()->remember('stats', 3600, fn () => $this->calculateStats()),
]);

// Concurrent HTTP requests
$responses = Octane::concurrently([
    fn () => Http::get('https://api.service1.com/data'),
    fn () => Http::get('https://api.service2.com/data'),
    fn () => Http::get('https://api.service3.com/data'),
]);
```

---

## 5. Caching Strategies

### ✅ DO: Implement Multi-Layer Caching

```php
<?php

namespace App\Support\Cache;

use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Redis;

final class CacheManager
{
    // L1: In-memory cache for hot data
    private static array $memory = [];
    
    // L2: Redis cache
    // L3: Database cache (cache table)
    
    public function remember(string $key, \Closure $callback, ?int $ttl = 3600): mixed
    {
        // Check L1 cache
        if (isset(self::$memory[$key])) {
            return self::$memory[$key];
        }
        
        // Check L2 cache with stampede protection
        $value = Cache::store('redis')->remember($key, $ttl, function () use ($callback, $key) {
            // Add lock to prevent cache stampede
            $lock = Cache::lock("cache_lock:{$key}", 10);
            
            if ($lock->get()) {
                try {
                    return $callback();
                } finally {
                    $lock->release();
                }
            }
            
            // If can't get lock, wait and retry
            sleep(1);
            return Cache::get($key);
        });
        
        // Store in L1 for this request
        self::$memory[$key] = $value;
        
        return $value;
    }
    
    public function tags(array $tags): self
    {
        // Tagged caching for easy invalidation
        return new self(Cache::tags($tags));
    }
}
```

### ✅ DO: Use Redis Efficiently

```php
// Batch operations to reduce round trips
Redis::pipeline(function ($pipe) use ($userId) {
    $pipe->hset("user:{$userId}", 'last_seen', now());
    $pipe->zincrby('active_users', 1, $userId);
    $pipe->expire("session:{$userId}", 3600);
});

// Use Redis data structures appropriately
class LeaderboardService
{
    public function addScore(int $userId, int $score): void
    {
        Redis::zadd('leaderboard', $score, $userId);
    }
    
    public function getTop(int $count = 10): array
    {
        return Redis::zrevrange('leaderboard', 0, $count - 1, 'WITHSCORES');
    }
    
    public function getUserRank(int $userId): ?int
    {
        $rank = Redis::zrevrank('leaderboard', $userId);
        return $rank !== null ? $rank + 1 : null;
    }
}

// Implement cache warming
class CacheWarmingCommand extends Command
{
    protected $signature = 'cache:warm {--tables=*}';
    
    public function handle(): void
    {
        $this->info('Warming cache...');
        
        // Warm frequently accessed data
        User::active()->chunk(100, function ($users) {
            foreach ($users as $user) {
                Cache::remember("user:{$user->id}", 3600, fn () => $user);
            }
        });
        
        // Warm computed values
        Cache::remember('stats:daily', 3600, fn () => $this->computeDailyStats());
    }
}
```

---

## 6. Queue Processing Best Practices

### ✅ DO: Design Idempotent, Retryable Jobs

```php
<?php

namespace App\Jobs;

use App\Domain\Order\Models\Order;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;
use Illuminate\Support\Facades\Redis;

class ProcessOrderPayment implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;
    
    public int $tries = 3;
    public int $maxExceptions = 2;
    public int $timeout = 120;
    public array $backoff = [30, 60, 120]; // Exponential backoff
    
    public function __construct(
        public Order $order,
        public string $idempotencyKey,
    ) {}
    
    public function handle(): void
    {
        // Check idempotency
        $processed = Redis::get("payment:processed:{$this->idempotencyKey}");
        if ($processed) {
            $this->delete();
            return;
        }
        
        // Process payment
        $result = app(PaymentGateway::class)->charge(
            amount: $this->order->total,
            currency: $this->order->currency,
            customerId: $this->order->user->stripe_id,
            metadata: [
                'order_id' => $this->order->id,
                'idempotency_key' => $this->idempotencyKey,
            ],
        );
        
        // Mark as processed
        Redis::setex(
            "payment:processed:{$this->idempotencyKey}",
            86400, // 24 hours
            $result->id
        );
        
        // Update order
        $this->order->update([
            'payment_id' => $result->id,
            'status' => 'paid',
            'paid_at' => now(),
        ]);
    }
    
    public function failed(\Throwable $exception): void
    {
        // Notify monitoring
        report($exception);
        
        // Update order status
        $this->order->update(['status' => 'payment_failed']);
        
        // Notify customer
        $this->order->user->notify(new PaymentFailedNotification($this->order));
    }
    
    // Determine if job should retry
    public function shouldRetry(\Throwable $exception): bool
    {
        // Don't retry on permanent failures
        if ($exception instanceof InvalidCardException) {
            return false;
        }
        
        return true;
    }
}
```

### ✅ DO: Use Job Batching for Complex Workflows

```php
use Illuminate\Bus\Batch;
use Illuminate\Support\Facades\Bus;

$batch = Bus::batch([
    new ProcessProductImages($product),
    new GenerateProductThumbnails($product),
    new UpdateSearchIndex($product),
    new NotifySubscribers($product),
])
->then(function (Batch $batch) {
    // All jobs completed successfully
    $batch->product->update(['processing_status' => 'completed']);
})
->catch(function (Batch $batch, \Throwable $e) {
    // First job failure
    $batch->product->update(['processing_status' => 'failed']);
})
->finally(function (Batch $batch) {
    // Cleanup
    Cache::forget("processing:{$batch->product->id}");
})
->name('Process Product: ' . $product->id)
->dispatch();
```

### ✅ DO: Monitor Queue Health

```php
// Custom queue monitoring
class QueueHealthCheck
{
    public function check(): array
    {
        $metrics = [];
        
        // Check queue sizes
        foreach (['default', 'emails', 'webhooks'] as $queue) {
            $size = Redis::llen("queues:{$queue}");
            $metrics["queue_{$queue}_size"] = $size;
            
            if ($size > 1000) {
                $this->alert("Queue {$queue} has {$size} jobs pending");
            }
        }
        
        // Check failed jobs
        $failedCount = DB::table('failed_jobs')
            ->where('failed_at', '>', now()->subHour())
            ->count();
            
        $metrics['failed_jobs_last_hour'] = $failedCount;
        
        // Check processing rate
        $processed = Redis::get('jobs_processed_last_minute') ?? 0;
        $metrics['jobs_per_minute'] = $processed;
        
        return $metrics;
    }
}
```

---

## 7. Testing with Pest

### ✅ DO: Write Expressive Tests with Pest

```php
<?php

use App\Domain\User\Actions\CreateUser;
use App\Domain\User\Data\UserData;
use App\Domain\User\Models\User;
use Illuminate\Foundation\Testing\RefreshDatabase;

uses(RefreshDatabase::class);

beforeEach(function () {
    $this->userData = UserData::from([
        'name' => 'John Doe',
        'email' => 'john@example.com',
        'password' => 'password123',
    ]);
});

test('user can be created with valid data', function () {
    $user = app(CreateUser::class)->execute($this->userData);
    
    expect($user)
        ->toBeInstanceOf(User::class)
        ->name->toBe('John Doe')
        ->email->toBe('john@example.com');
        
    $this->assertDatabaseHas('users', [
        'email' => 'john@example.com',
    ]);
});

test('user creation triggers event', function () {
    Event::fake([UserCreated::class]);
    
    app(CreateUser::class)->execute($this->userData);
    
    Event::assertDispatched(UserCreated::class, function ($event) {
        return $event->user->email === 'john@example.com';
    });
});

// Dataset testing
it('validates user email formats', function (string $email, bool $valid) {
    $data = $this->userData->toArray();
    $data['email'] = $email;
    
    if ($valid) {
        expect(fn () => UserData::from($data))->not->toThrow();
    } else {
        expect(fn () => UserData::from($data))->toThrow(ValidationException::class);
    }
})->with([
    ['valid@email.com', true],
    ['invalid.email', false],
    ['test@', false],
    ['test@domain', false],
    ['test@domain.com', true],
]);
```

### ✅ DO: Use Higher-Order Tests for API Testing

```php
<?php

use function Pest\Laravel\{getJson, postJson, putJson, deleteJson};

test('api endpoints')
    ->group('api')
    ->beforeEach(function () {
        $this->user = User::factory()->create();
        $this->actingAs($this->user, 'sanctum');
    });

test('can list users')
    ->getJson('/api/v1/users')
    ->assertOk()
    ->assertJsonStructure([
        'data' => [
            '*' => ['id', 'name', 'email', 'created_at']
        ],
        'meta' => ['current_page', 'total']
    ]);

test('can create user with valid data')
    ->postJson('/api/v1/users', [
        'name' => 'New User',
        'email' => 'new@example.com',
        'password' => 'password123',
    ])
    ->assertCreated()
    ->assertJsonPath('data.email', 'new@example.com');

// Architecture testing
test('domain models extend base model')
    ->expect('App\Domain\*\Models\*')
    ->toExtend('Illuminate\Database\Eloquent\Model');

test('controllers are suffixed properly')
    ->expect('App\Http\Controllers\*')
    ->toHaveSuffix('Controller');

test('no debug code in production')
    ->expect(['dd', 'dump', 'var_dump'])
    ->not->toBeUsed();
```

---

## 8. Security Best Practices

### ✅ DO: Implement Comprehensive Security Headers

```php
<?php

namespace App\Http\Middleware;

class SecurityHeaders
{
    public function handle($request, \Closure $next)
    {
        $response = $next($request);
        
        $response->headers->set('X-Content-Type-Options', 'nosniff');
        $response->headers->set('X-Frame-Options', 'DENY');
        $response->headers->set('X-XSS-Protection', '1; mode=block');
        $response->headers->set('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
        $response->headers->set('Referrer-Policy', 'strict-origin-when-cross-origin');
        $response->headers->set('Permissions-Policy', 'geolocation=(), microphone=(), camera=()');
        
        // Content Security Policy
        $csp = "default-src 'self'; " .
               "script-src 'self' 'nonce-" . $request->attributes->get('csp_nonce') . "' https://cdn.jsdelivr.net; " .
               "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; " .
               "font-src 'self' https://fonts.gstatic.com; " .
               "img-src 'self' data: https:; " .
               "connect-src 'self' wss: https://api.stripe.com";
               
        $response->headers->set('Content-Security-Policy', $csp);
        
        return $response;
    }
}
```

### ✅ DO: Use Proper Authentication with Sanctum

```php
// For SPA authentication
Route::post('/login', function (Request $request) {
    $credentials = $request->validate([
        'email' => ['required', 'email'],
        'password' => ['required'],
    ]);
    
    if (!Auth::attempt($credentials)) {
        throw ValidationException::withMessages([
            'email' => ['The provided credentials are incorrect.'],
        ]);
    }
    
    $request->session()->regenerate();
    
    // Optional: Issue a token for mobile apps
    $token = $request->user()->createToken(
        name: 'auth-token',
        abilities: ['*'],
        expiresAt: now()->addWeek()
    );
    
    return response()->json([
        'user' => new UserResource($request->user()),
        'token' => $token->plainTextToken,
    ]);
});

// Protect routes
Route::middleware(['auth:sanctum', 'verified'])->group(function () {
    Route::apiResource('posts', PostController::class);
});

// Token abilities (scopes)
Route::post('/posts', function (Request $request) {
    if (!$request->user()->tokenCan('posts:create')) {
        abort(403);
    }
    
    // Create post...
})->middleware('auth:sanctum');
```

### ✅ DO: Implement Rate Limiting

```php
// In RouteServiceProvider
protected function configureRateLimiting(): void
{
    RateLimiter::for('api', function (Request $request) {
        return Limit::perMinute(60)->by($request->user()?->id ?: $request->ip());
    });
    
    RateLimiter::for('auth', function (Request $request) {
        return Limit::perMinute(5)->by($request->ip());
    });
    
    // Dynamic rate limiting based on user tier
    RateLimiter::for('api-premium', function (Request $request) {
        $user = $request->user();
        
        return match($user?->subscription_tier) {
            'premium' => Limit::perMinute(600),
            'pro' => Limit::perMinute(300),
            default => Limit::perMinute(60),
        };
    });
}

// Apply to routes
Route::middleware(['throttle:auth'])->group(function () {
    Route::post('/login', [AuthController::class, 'login']);
    Route::post('/register', [AuthController::class, 'register']);
});
```

---

## 9. Real-time Features with Laravel Reverb

### ✅ DO: Implement WebSocket Broadcasting

```php
// Install and configure Reverb (Laravel's built-in WebSocket server)
composer require laravel/reverb
php artisan reverb:install

// Broadcasting configuration
'connections' => [
    'reverb' => [
        'driver' => 'reverb',
        'app_id' => env('REVERB_APP_ID'),
        'app_key' => env('REVERB_APP_KEY'),
        'app_secret' => env('REVERB_APP_SECRET'),
        'host' => env('REVERB_HOST', 'localhost'),
        'port' => env('REVERB_PORT', 8080),
        'scheme' => env('REVERB_SCHEME', 'http'),
    ],
],

// Event broadcasting
<?php

namespace App\Events;

use App\Models\Message;
use Illuminate\Broadcasting\Channel;
use Illuminate\Broadcasting\InteractsWithSockets;
use Illuminate\Broadcasting\PresenceChannel;
use Illuminate\Contracts\Broadcasting\ShouldBroadcastNow;
use Illuminate\Foundation\Events\Dispatchable;
use Illuminate\Queue\SerializesModels;

class MessageSent implements ShouldBroadcastNow
{
    use Dispatchable, InteractsWithSockets, SerializesModels;
    
    public function __construct(
        public Message $message
    ) {}
    
    public function broadcastOn(): array
    {
        return [
            new PresenceChannel('chat.' . $this->message->room_id),
        ];
    }
    
    public function broadcastWith(): array
    {
        return [
            'id' => $this->message->id,
            'content' => $this->message->content,
            'user' => [
                'id' => $this->message->user->id,
                'name' => $this->message->user->name,
            ],
            'sent_at' => $this->message->created_at->toIso8601String(),
        ];
    }
}

// Frontend listening (Echo)
Echo.join(`chat.${roomId}`)
    .here((users) => {
        console.log('Users in room:', users);
    })
    .joining((user) => {
        console.log('User joined:', user);
    })
    .leaving((user) => {
        console.log('User left:', user);
    })
    .listen('MessageSent', (e) => {
        appendMessage(e);
    });
```

---

## 10. DevOps & Deployment

### ✅ DO: Use Docker for Consistent Environments

```dockerfile
# Production Dockerfile
FROM dunglas/frankenphp:latest-php8.4 AS frankenphp

# Install PHP extensions
RUN install-php-extensions \
    pcntl \
    pdo_mysql \
    redis \
    opcache \
    intl \
    zip \
    gd \
    bcmath

# Copy application
WORKDIR /app
COPY . .

# Install dependencies
RUN composer install --no-dev --optimize-autoloader --no-interaction

# Build assets
RUN npm ci && npm run build && rm -rf node_modules

# Configure FrankenPHP
ENV FRANKENPHP_CONFIG="worker ./public/index.php"
ENV APP_RUNTIME="octane"

# Run as non-root
RUN chown -R www-data:www-data /app
USER www-data

EXPOSE 8000
CMD ["frankenphp", "run", "--config", "/etc/caddy/Caddyfile"]
```

### ✅ DO: Implement Zero-Downtime Deployments

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup PHP
        uses: shivammathur/setup-php@v2
        with:
          php-version: '8.4'
          tools: composer:v2
          
      - name: Install dependencies
        run: |
          composer install --no-dev --optimize-autoloader
          npm ci && npm run build
          
      - name: Run tests
        run: |
          cp .env.testing .env
          php artisan test --parallel
          
      - name: Deploy to production
        run: |
          # Blue-green deployment
          php artisan down --retry=60
          rsync -avz --delete \
            --exclude='.env' \
            --exclude='storage/' \
            --exclude='bootstrap/cache/' \
            ./ user@server:/var/www/app-new/
            
          ssh user@server << 'EOF'
            cd /var/www/app-new
            php artisan migrate --force
            php artisan config:cache
            php artisan route:cache
            php artisan view:cache
            php artisan queue:restart
            php artisan octane:reload
            
            # Atomic switch
            ln -nfs /var/www/app-new /var/www/app
          EOF
          
          php artisan up
```

### ✅ DO: Monitor Everything

```php
// config/logging.php
'channels' => [
    'stack' => [
        'driver' => 'stack',
        'channels' => ['daily', 'slack', 'datadog'],
    ],
    
    'datadog' => [
        'driver' => 'monolog',
        'handler' => DatadogHandler::class,
        'with' => [
            'apiKey' => env('DATADOG_API_KEY'),
            'host' => env('DATADOG_HOST', 'app'),
            'tags' => ['env:' . env('APP_ENV')],
        ],
    ],
];

// Custom health checks
Route::get('/health', function () {
    $checks = [
        'database' => fn () => DB::connection()->getPdo() !== null,
        'redis' => fn () => Redis::ping() === 'PONG',
        'queue' => fn () => Queue::size() < 1000,
        'storage' => fn () => Storage::disk('local')->exists('health.check'),
    ];
    
    $results = [];
    foreach ($checks as $name => $check) {
        try {
            $results[$name] = $check() ? 'healthy' : 'unhealthy';
        } catch (\Exception $e) {
            $results[$name] = 'unhealthy';
        }
    }
    
    $healthy = !in_array('unhealthy', $results);
    
    return response()->json([
        'status' => $healthy ? 'healthy' : 'unhealthy',
        'checks' => $results,
        'timestamp' => now()->toIso8601String(),
    ], $healthy ? 200 : 503);
});
```

---

## Summary

This guide represents the cutting edge of Laravel development in mid-2025. Key takeaways:

1. **Architecture Matters**: Use DDD principles to organize code by business domains
2. **Type Safety**: Leverage PHP 8.4 features and static analysis for bulletproof code
3. **Performance**: Octane + proper caching + queue design = blazing fast apps
4. **Testing**: Pest makes tests readable and maintainable
5. **Security**: Defense in depth with proper authentication, authorization, and headers
6. **Modern Frontend**: Choose between Inertia.js for SPA-like apps or API-first with your favorite framework
7. **Real-time**: Laravel Reverb makes WebSockets trivial
8. **DevOps**: Containerize everything and automate deployments

Remember: these patterns are for production applications. Start simple and add complexity as your application grows. The best code is code that your team understands and can maintain.