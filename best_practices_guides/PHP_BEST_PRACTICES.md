# Modern PHP Web Development: The Definitive Guide (mid-2025 Edition)

This guide synthesizes production-grade best practices for building scalable, secure, and performant PHP applications in 2025. It assumes PHP 8.4+, embraces modern tooling, and provides battle-tested architectural patterns.

## Prerequisites & Core Philosophy

**Minimum Requirements:**
- PHP 8.4+ (with JIT and Fibers)
- Composer 2.8+
- Node.js 22+ (for asset building)
- Docker 26+ for development

**Key Principles:**
- Type-first development with native types and generics
- Async-by-default for I/O operations
- Zero-tolerance for deprecated patterns
- Security and performance baked in, not bolted on

## 1. Project Foundation & Modern Toolchain

### ✅ DO: Adopt a Consistent Project Structure

Modern PHP projects should follow PSR-4 autoloading with a clear separation of concerns:

```
/project-root
├── src/                    # Application source code
│   ├── Application/        # Application services and use cases
│   ├── Domain/            # Business logic and entities
│   ├── Infrastructure/    # External services, repositories
│   └── Presentation/      # Controllers, API endpoints
├── config/                # Configuration files
├── public/               # Web root (index.php only)
├── tests/                # All test files
├── var/                  # Cache, logs, temporary files
├── docker/               # Docker configuration
├── .github/              # CI/CD workflows
├── composer.json         # Dependencies
├── composer.lock         # Locked versions
├── phpunit.xml.dist      # Test configuration
├── psalm.xml             # Static analysis config
├── rector.php            # Automated refactoring
└── .php-cs-fixer.php     # Code style rules
```

### ✅ DO: Configure composer.json for Modern Development

```json
{
    "name": "company/project",
    "type": "project",
    "require": {
        "php": ">=8.4",
        "ext-apcu": "*",
        "ext-redis": "*",
        "ext-openswoole": "*",
        "revolt/event-loop": "^1.0",
        "symfony/framework-bundle": "^7.2",
        "api-platform/core": "^4.0",
        "doctrine/orm": "^3.3",
        "nette/php-generator": "^4.1",
        "symfony/messenger": "^7.2",
        "league/flysystem": "^3.30"
    },
    "require-dev": {
        "phpunit/phpunit": "^11.5",
        "psalm/plugin-symfony": "^5.2",
        "rector/rector": "^2.0",
        "friendsofphp/php-cs-fixer": "^3.70",
        "infection/infection": "^0.29",
        "phpstan/phpstan": "^2.0",
        "symfony/profiler-pack": "^1.0"
    },
    "autoload": {
        "psr-4": {
            "App\\": "src/"
        }
    },
    "autoload-dev": {
        "psr-4": {
            "App\\Tests\\": "tests/"
        }
    },
    "scripts": {
        "test": "phpunit",
        "analyze": "psalm --stats",
        "fix-style": "php-cs-fixer fix",
        "check-style": "php-cs-fixer fix --dry-run --diff",
        "refactor": "rector process",
        "mutation": "infection --threads=max"
    },
    "config": {
        "optimize-autoloader": true,
        "preferred-install": "dist",
        "sort-packages": true,
        "allow-plugins": {
            "composer/package-versions-deprecated": true,
            "infection/extension-installer": true
        }
    }
}
```

### ✅ DO: Embrace Static Analysis from Day One

**psalm.xml configuration for maximum safety:**

```xml
<?xml version="1.0"?>
<psalm
    errorLevel="1"
    resolveFromConfigFile="true"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns="https://getpsalm.org/schema/config"
    xsi:schemaLocation="https://getpsalm.org/schema/config vendor/vimeo/psalm/config.xsd"
    errorBaseline="psalm-baseline.xml"
    findUnusedBaselineEntry="true"
    findUnusedCode="true"
>
    <projectFiles>
        <directory name="src" />
        <ignoreFiles>
            <directory name="vendor" />
            <directory name="var" />
        </ignoreFiles>
    </projectFiles>
    
    <plugins>
        <pluginClass class="Psalm\SymfonyPsalmPlugin\Plugin" />
    </plugins>
    
    <issueHandlers>
        <PropertyNotSetInConstructor>
            <errorLevel type="suppress">
                <referencedClass name="App\Domain\Entity\*" />
            </errorLevel>
        </PropertyNotSetInConstructor>
    </issueHandlers>
</psalm>
```

## 2. Framework Selection & Architecture

### For API-First Applications: API Platform 4

API Platform 4 provides a complete toolkit for building hypermedia-driven REST and GraphQL APIs with built-in OpenAPI documentation.

```php
<?php
// src/Domain/Entity/Product.php
declare(strict_types=1);

namespace App\Domain\Entity;

use ApiPlatform\Metadata\ApiResource;
use ApiPlatform\Metadata\Get;
use ApiPlatform\Metadata\GetCollection;
use ApiPlatform\Metadata\Post;
use ApiPlatform\Metadata\Put;
use ApiPlatform\Metadata\Delete;
use App\Domain\State\ProductProcessor;
use Doctrine\ORM\Mapping as ORM;
use Symfony\Component\Validator\Constraints as Assert;
use Symfony\Component\Serializer\Annotation\Groups;

#[ORM\Entity]
#[ORM\Table(name: 'products')]
#[ORM\Index(columns: ['slug'], name: 'idx_product_slug')]
#[ORM\Index(columns: ['created_at'], name: 'idx_product_created')]
#[ORM\Cache(usage: 'NONSTRICT_READ_WRITE')]
#[ApiResource(
    shortName: 'Product',
    operations: [
        new GetCollection(
            paginationClientItemsPerPage: true,
            paginationMaximumItemsPerPage: 100
        ),
        new Get(),
        new Post(processor: ProductProcessor::class),
        new Put(processor: ProductProcessor::class),
        new Delete()
    ],
    normalizationContext: ['groups' => ['product:read']],
    denormalizationContext: ['groups' => ['product:write']],
    mercure: true, // Real-time updates via Mercure
    extraProperties: [
        'standard_put' => false, // Use PATCH for partial updates
    ]
)]
class Product
{
    #[ORM\Id]
    #[ORM\GeneratedValue]
    #[ORM\Column(type: 'integer')]
    #[Groups(['product:read'])]
    private ?int $id = null;

    #[ORM\Column(length: 255)]
    #[Assert\NotBlank]
    #[Assert\Length(min: 3, max: 255)]
    #[Groups(['product:read', 'product:write'])]
    private string $name;

    #[ORM\Column(type: 'text')]
    #[Assert\NotBlank]
    #[Groups(['product:read', 'product:write'])]
    private string $description;

    #[ORM\Column(type: 'decimal', precision: 10, scale: 2)]
    #[Assert\Positive]
    #[Groups(['product:read', 'product:write'])]
    private string $price;

    #[ORM\Column(type: 'datetime_immutable')]
    #[Groups(['product:read'])]
    private \DateTimeImmutable $createdAt;

    public function __construct()
    {
        $this->createdAt = new \DateTimeImmutable();
    }

    // Getters and setters with proper type hints...
}
```

### For Full-Stack Applications: Symfony 7.2+ with Symfony UX

```php
<?php
// src/Presentation/Controller/DashboardController.php
declare(strict_types=1);

namespace App\Presentation\Controller;

use App\Application\Query\DashboardDataQuery;
use App\Application\QueryBus;
use Symfony\Bundle\FrameworkBundle\Controller\AbstractController;
use Symfony\Component\HttpFoundation\Response;
use Symfony\Component\Routing\Attribute\Route;
use Symfony\Component\Security\Http\Attribute\IsGranted;
use Symfony\UX\Turbo\TurboBundle;

#[Route('/dashboard', name: 'dashboard_')]
#[IsGranted('ROLE_USER')]
class DashboardController extends AbstractController
{
    public function __construct(
        private readonly QueryBus $queryBus,
    ) {}

    #[Route('', name: 'index', methods: ['GET'])]
    public function index(): Response
    {
        $data = $this->queryBus->handle(new DashboardDataQuery(
            userId: $this->getUser()->getId()
        ));

        return $this->render('dashboard/index.html.twig', [
            'metrics' => $data->metrics,
            'recentActivity' => $data->recentActivity,
        ]);
    }

    #[Route('/metrics', name: 'metrics_stream', methods: ['GET'])]
    public function metricsStream(): Response
    {
        $response = $this->render('dashboard/_metrics.stream.html.twig', [
            'metrics' => $this->queryBus->handle(new DashboardDataQuery(
                userId: $this->getUser()->getId()
            ))->metrics,
        ]);

        // Enable Turbo Streams for real-time updates
        $response->headers->set('Content-Type', TurboBundle::STREAM_MEDIA_TYPE);
        
        return $response;
    }
}
```

## 3. Modern PHP 8.4 Features in Action

### ✅ DO: Leverage Property Hooks (PHP 8.4)

Property hooks provide computed properties with full type safety:

```php
<?php
declare(strict_types=1);

namespace App\Domain\ValueObject;

use App\Domain\Exception\InvalidPriceException;

final class Price
{
    private float $amount;
    
    public float $amountWithTax {
        get => $this->amount * 1.20;
    }
    
    public string $formatted {
        get => sprintf('$%.2f', $this->amount);
    }
    
    public float $netAmount {
        get => $this->amount;
        set(float $value) {
            if ($value < 0) {
                throw new InvalidPriceException('Price cannot be negative');
            }
            $this->amount = round($value, 2);
        }
    }
    
    public function __construct(float $amount)
    {
        $this->netAmount = $amount; // Uses the setter
    }
}

// Usage
$price = new Price(99.99);
echo $price->formatted;      // $99.99
echo $price->amountWithTax;  // 119.99
```

### ✅ DO: Use Asymmetric Visibility (PHP 8.4)

Control read/write access at the property level:

```php
<?php
declare(strict_types=1);

namespace App\Domain\Entity;

use Doctrine\ORM\Mapping as ORM;

#[ORM\Entity]
class User
{
    #[ORM\Column(type: 'string')]
    public private(set) string $email;
    
    #[ORM\Column(type: 'integer')]
    public private(set) int $loginAttempts = 0;
    
    #[ORM\Column(type: 'datetime_immutable', nullable: true)]
    public private(set) ?\DateTimeImmutable $lastLoginAt = null;
    
    public function __construct(string $email)
    {
        $this->email = $email;
    }
    
    public function recordLoginAttempt(): void
    {
        $this->loginAttempts++;
    }
    
    public function recordSuccessfulLogin(): void
    {
        $this->loginAttempts = 0;
        $this->lastLoginAt = new \DateTimeImmutable();
    }
}
```

### ✅ DO: Embrace Pattern Matching

```php
<?php
declare(strict_types=1);

namespace App\Application\Service;

use App\Domain\Event\DomainEvent;
use App\Domain\Event\OrderPlaced;
use App\Domain\Event\OrderShipped;
use App\Domain\Event\OrderCancelled;

final class EventProcessor
{
    public function process(DomainEvent $event): void
    {
        match (true) {
            $event instanceof OrderPlaced => $this->handleOrderPlaced($event),
            $event instanceof OrderShipped => $this->handleOrderShipped($event),
            $event instanceof OrderCancelled => $this->handleOrderCancelled($event),
            default => throw new \LogicException('Unhandled event type: ' . $event::class),
        };
    }
    
    private function handleOrderPlaced(OrderPlaced $event): void
    {
        // Send confirmation email
        // Update inventory
        // Trigger fulfillment
    }
    
    // Other handlers...
}
```

## 4. Async PHP with Fibers and Event Loops

### ✅ DO: Use Revolt Event Loop for Concurrent I/O

```php
<?php
declare(strict_types=1);

namespace App\Infrastructure\Http;

use Revolt\EventLoop;
use Psr\Http\Client\ClientInterface;
use Psr\Http\Message\RequestInterface;
use Psr\Http\Message\ResponseInterface;

final class AsyncHttpClient implements ClientInterface
{
    private array $pendingRequests = [];
    
    public function sendRequest(RequestInterface $request): ResponseInterface
    {
        $fiber = new \Fiber(function () use ($request) {
            $suspension = EventLoop::getSuspension();
            
            $this->pendingRequests[] = [
                'request' => $request,
                'suspension' => $suspension,
            ];
            
            return $suspension->suspend();
        });
        
        return $fiber->start();
    }
    
    public function sendConcurrent(array $requests): array
    {
        $fibers = [];
        
        foreach ($requests as $key => $request) {
            $fibers[$key] = new \Fiber(fn() => $this->sendRequest($request));
            $fibers[$key]->start();
        }
        
        // Process all requests concurrently
        EventLoop::run();
        
        $responses = [];
        foreach ($fibers as $key => $fiber) {
            $responses[$key] = $fiber->getReturn();
        }
        
        return $responses;
    }
}
```

### ✅ DO: Implement Async Database Queries

```php
<?php
declare(strict_types=1);

namespace App\Infrastructure\Persistence;

use Revolt\EventLoop;
use Swoole\Coroutine\PostgreSQL;

final class AsyncPostgresRepository
{
    private PostgreSQL $connection;
    
    public function __construct(string $dsn)
    {
        $this->connection = new PostgreSQL();
        $this->connection->connect($dsn);
    }
    
    public function findByIdAsync(int $id): \Fiber
    {
        return new \Fiber(function () use ($id) {
            $query = 'SELECT * FROM users WHERE id = $1';
            
            $result = EventLoop::getSuspension()->suspend(
                fn() => $this->connection->query($query, [$id])
            );
            
            return $result ? $result[0] : null;
        });
    }
    
    public function findMultipleAsync(array $ids): array
    {
        $fibers = [];
        
        foreach ($ids as $id) {
            $fibers[$id] = $this->findByIdAsync($id);
            $fibers[$id]->start();
        }
        
        $results = [];
        foreach ($fibers as $id => $fiber) {
            $results[$id] = $fiber->getReturn();
        }
        
        return $results;
    }
}
```

## 5. Database Layer with Doctrine 3

### ✅ DO: Use Read/Write Splitting

```yaml
# config/packages/doctrine.yaml
doctrine:
    dbal:
        default_connection: default
        connections:
            default:
                primary:
                    url: '%env(DATABASE_WRITE_URL)%'
                replicas:
                    replica1:
                        url: '%env(DATABASE_READ1_URL)%'
                    replica2:
                        url: '%env(DATABASE_READ2_URL)%'
                driver: 'pdo_pgsql'
                server_version: '16'
                charset: utf8mb4
                default_table_options:
                    charset: utf8mb4
                    collate: utf8mb4_unicode_ci
    orm:
        auto_generate_proxy_classes: false
        naming_strategy: doctrine.orm.naming_strategy.underscore_number_aware
        metadata_cache_driver:
            type: apcu
            namespace: doctrine_metadata_
        query_cache_driver:
            type: apcu
            namespace: doctrine_query_
        result_cache_driver:
            type: redis
            host: '%env(REDIS_HOST)%'
            namespace: doctrine_result_
```

### ✅ DO: Implement Repository Pattern with Specification

```php
<?php
declare(strict_types=1);

namespace App\Infrastructure\Persistence\Repository;

use App\Domain\Repository\ProductRepositoryInterface;
use App\Domain\Specification\SpecificationInterface;
use Doctrine\ORM\EntityManagerInterface;
use Doctrine\ORM\QueryBuilder;

final class DoctrineProductRepository implements ProductRepositoryInterface
{
    public function __construct(
        private readonly EntityManagerInterface $em,
    ) {}
    
    public function findBySpecification(SpecificationInterface $spec): array
    {
        $qb = $this->createQueryBuilder();
        $spec->apply($qb);
        
        return $qb->getQuery()
            ->setResultCacheId('products_' . $spec->getCacheKey())
            ->setResultCacheLifetime(3600)
            ->getResult();
    }
    
    public function add(Product $product): void
    {
        $this->em->persist($product);
    }
    
    public function remove(Product $product): void
    {
        $this->em->remove($product);
    }
    
    private function createQueryBuilder(): QueryBuilder
    {
        return $this->em->createQueryBuilder()
            ->select('p', 'c', 'i') // Eager load associations
            ->from(Product::class, 'p')
            ->leftJoin('p.category', 'c')
            ->leftJoin('p.images', 'i');
    }
}
```

## 6. Caching Strategy

### ✅ DO: Implement Multi-Tier Caching

```php
<?php
declare(strict_types=1);

namespace App\Infrastructure\Cache;

use Psr\Cache\CacheItemPoolInterface;
use Symfony\Contracts\Cache\ItemInterface;
use Symfony\Contracts\Cache\TagAwareCacheInterface;

final class CacheManager
{
    public function __construct(
        private readonly TagAwareCacheInterface $appCache,  // Redis
        private readonly CacheItemPoolInterface $localCache, // APCu
    ) {}
    
    public function remember(
        string $key, 
        callable $callback, 
        int $ttl = 3600,
        array $tags = []
    ): mixed {
        // Try local cache first (L1)
        $localItem = $this->localCache->getItem($key);
        if ($localItem->isHit()) {
            return $localItem->get();
        }
        
        // Try distributed cache (L2)
        $value = $this->appCache->get($key, function (ItemInterface $item) use ($callback, $ttl, $tags) {
            $item->expiresAfter($ttl);
            if ($tags) {
                $item->tag($tags);
            }
            
            return $callback();
        });
        
        // Store in local cache
        $localItem->set($value);
        $localItem->expiresAfter(min($ttl, 300)); // Max 5 minutes locally
        $this->localCache->save($localItem);
        
        return $value;
    }
    
    public function invalidateTags(array $tags): void
    {
        $this->appCache->invalidateTags($tags);
    }
}
```

### ✅ DO: Cache Warming Strategy

```php
<?php
declare(strict_types=1);

namespace App\Application\Command;

use App\Infrastructure\Cache\CacheWarmer;
use Symfony\Component\Console\Attribute\AsCommand;
use Symfony\Component\Console\Command\Command;
use Symfony\Component\Console\Input\InputInterface;
use Symfony\Component\Console\Output\OutputInterface;
use Symfony\Component\Console\Style\SymfonyStyle;

#[AsCommand(
    name: 'app:cache:warm',
    description: 'Warm application caches',
)]
final class CacheWarmCommand extends Command
{
    public function __construct(
        private readonly CacheWarmer $warmer,
    ) {
        parent::__construct();
    }
    
    protected function execute(InputInterface $input, OutputInterface $output): int
    {
        $io = new SymfonyStyle($input, $output);
        $io->title('Warming Application Caches');
        
        $io->section('Configuration Cache');
        $this->warmer->warmConfiguration();
        $io->success('Configuration cached');
        
        $io->section('Route Cache');
        $this->warmer->warmRoutes();
        $io->success('Routes cached');
        
        $io->section('Popular Queries Cache');
        $progress = $io->createProgressBar();
        $this->warmer->warmPopularQueries($progress);
        $progress->finish();
        
        $io->success('All caches warmed successfully');
        
        return Command::SUCCESS;
    }
}
```

## 7. Security Patterns

### ✅ DO: Implement Defense in Depth

```php
<?php
declare(strict_types=1);

namespace App\Infrastructure\Security;

use Symfony\Component\HttpFoundation\Request;
use Symfony\Component\HttpFoundation\Response;
use Symfony\Component\RateLimiter\RateLimiterFactory;
use Symfony\Component\Security\Core\Exception\AccessDeniedException;

final class SecurityMiddleware
{
    public function __construct(
        private readonly RateLimiterFactory $apiLimiter,
        private readonly RateLimiterFactory $loginLimiter,
        private readonly ContentSecurityPolicyManager $cspManager,
    ) {}
    
    public function handle(Request $request, callable $next): Response
    {
        // Rate limiting
        $this->enforceRateLimit($request);
        
        // CSRF protection for state-changing requests
        if (in_array($request->getMethod(), ['POST', 'PUT', 'DELETE', 'PATCH'])) {
            $this->validateCsrfToken($request);
        }
        
        // Process request
        $response = $next($request);
        
        // Security headers
        $this->addSecurityHeaders($response);
        
        return $response;
    }
    
    private function enforceRateLimit(Request $request): void
    {
        $limiter = $request->attributes->get('_route') === 'login' 
            ? $this->loginLimiter 
            : $this->apiLimiter;
            
        $limit = $limiter->create($request->getClientIp());
        
        if (!$limit->consume()->isAccepted()) {
            throw new AccessDeniedException('Rate limit exceeded');
        }
    }
    
    private function addSecurityHeaders(Response $response): void
    {
        $response->headers->set('X-Content-Type-Options', 'nosniff');
        $response->headers->set('X-Frame-Options', 'DENY');
        $response->headers->set('X-XSS-Protection', '1; mode=block');
        $response->headers->set('Referrer-Policy', 'strict-origin-when-cross-origin');
        $response->headers->set('Permissions-Policy', 'geolocation=(), microphone=(), camera=()');
        
        // Dynamic CSP based on route
        $response->headers->set(
            'Content-Security-Policy', 
            $this->cspManager->getPolicy($request->attributes->get('_route'))
        );
    }
}
```

### ✅ DO: Secure Session Configuration

```yaml
# config/packages/framework.yaml
framework:
    session:
        handler_id: 'redis://redis:6379'
        cookie_secure: true
        cookie_httponly: true
        cookie_samesite: 'strict'
        name: 'APP_SESSION'
        gc_probability: 0 # Disable file-based GC when using Redis
        sid_bits_per_character: 6
        sid_length: 48
```

## 8. Testing Strategy

### ✅ DO: Write Fast, Isolated Tests

```php
<?php
declare(strict_types=1);

namespace App\Tests\Application\Service;

use App\Application\Service\OrderService;
use App\Domain\Entity\Order;
use App\Domain\Repository\OrderRepositoryInterface;
use App\Domain\Service\PricingService;
use PHPUnit\Framework\TestCase;
use Prophecy\PhpUnit\ProphecyTrait;

final class OrderServiceTest extends TestCase
{
    use ProphecyTrait;
    
    private OrderService $service;
    private OrderRepositoryInterface $repository;
    private PricingService $pricingService;
    
    protected function setUp(): void
    {
        $this->repository = $this->prophesize(OrderRepositoryInterface::class);
        $this->pricingService = $this->prophesize(PricingService::class);
        
        $this->service = new OrderService(
            $this->repository->reveal(),
            $this->pricingService->reveal()
        );
    }
    
    /** @test */
    public function it_calculates_order_total_with_tax(): void
    {
        // Arrange
        $order = Order::create();
        $order->addItem('PROD-123', 2, 50.00);
        
        $this->pricingService
            ->calculateTax($order)
            ->willReturn(20.00);
        
        // Act
        $total = $this->service->calculateTotal($order);
        
        // Assert
        $this->assertSame(120.00, $total);
    }
}
```

### ✅ DO: Use Mutation Testing

```php
// infection.json
{
    "timeout": 30,
    "source": {
        "directories": ["src"]
    },
    "mutators": {
        "@default": true,
        "@function_signature": false
    },
    "logs": {
        "html": "var/infection.html",
        "summary": "var/infection-summary.log"
    },
    "minMsi": 85,
    "minCoveredMsi": 95,
    "testFramework": "phpunit",
    "testFrameworkOptions": "--testsuite=unit"
}
```

### ✅ DO: Contract Testing for APIs

```php
<?php
declare(strict_types=1);

namespace App\Tests\Contract;

use App\Tests\ApiTestCase;
use Symfony\Component\HttpFoundation\Response;

final class ProductApiContractTest extends ApiTestCase
{
    /** @test */
    public function get_product_returns_expected_schema(): void
    {
        // Arrange
        $this->loadFixtures(['products']);
        
        // Act
        $response = $this->request('GET', '/api/products/1');
        
        // Assert
        $this->assertResponseStatusCodeSame(Response::HTTP_OK);
        $this->assertMatchesJsonSchema($response, 'product.schema.json');
        
        // Verify specific fields
        $data = json_decode($response->getContent(), true);
        $this->assertArrayHasKey('id', $data);
        $this->assertArrayHasKey('name', $data);
        $this->assertArrayHasKey('price', $data);
        $this->assertIsFloat($data['price']);
    }
}
```

## 9. Performance Optimization

### ✅ DO: Implement Aggressive OpCode Caching

```ini
; PHP 8.4 production php.ini
opcache.enable=1
opcache.memory_consumption=512
opcache.interned_strings_buffer=64
opcache.max_accelerated_files=100000
opcache.validate_timestamps=0
opcache.save_comments=0
opcache.fast_shutdown=1
opcache.enable_file_override=1
opcache.jit=tracing
opcache.jit_buffer_size=256M
opcache.jit_max_recursive_calls=10
opcache.jit_max_recursive_returns=5

; Preload your framework
opcache.preload=/var/www/config/preload.php
opcache.preload_user=www-data
```

### ✅ DO: Profile and Optimize Database Queries

```php
<?php
declare(strict_types=1);

namespace App\Infrastructure\Persistence;

use Doctrine\DBAL\Logging\Middleware;
use Psr\Log\LoggerInterface;

final class QueryAnalyzer
{
    private array $queries = [];
    
    public function __construct(
        private readonly LoggerInterface $logger,
    ) {}
    
    public function analyze(): void
    {
        $problematicQueries = array_filter(
            $this->queries,
            fn($q) => $q['duration'] > 100 || $q['memory'] > 1048576
        );
        
        foreach ($problematicQueries as $query) {
            $this->logger->warning('Slow query detected', [
                'sql' => $query['sql'],
                'duration_ms' => $query['duration'],
                'memory_bytes' => $query['memory'],
                'explain' => $this->explainQuery($query['sql']),
            ]);
        }
    }
    
    private function explainQuery(string $sql): array
    {
        // Run EXPLAIN ANALYZE and return results
        // Identify missing indexes, table scans, etc.
    }
}
```

## 10. Real-time Features with Mercure

### ✅ DO: Implement Server-Sent Events

```php
<?php
declare(strict_types=1);

namespace App\Application\Service;

use Symfony\Component\Mercure\HubInterface;
use Symfony\Component\Mercure\Update;

final class RealtimeNotifier
{
    public function __construct(
        private readonly HubInterface $hub,
    ) {}
    
    public function notifyOrderUpdate(Order $order): void
    {
        $update = new Update(
            topics: [
                sprintf('https://example.com/orders/%d', $order->getId()),
                sprintf('https://example.com/users/%d/orders', $order->getUserId()),
            ],
            data: json_encode([
                'id' => $order->getId(),
                'status' => $order->getStatus(),
                'updatedAt' => $order->getUpdatedAt()->format(\DateTime::ATOM),
            ]),
            private: true, // Only for authenticated users
            id: (string) $order->getVersion(),
            type: 'order.updated'
        );
        
        $this->hub->publish($update);
    }
}
```

## 11. Message Queue Integration

### ✅ DO: Use Symfony Messenger for Async Processing

```php
<?php
declare(strict_types=1);

namespace App\Application\Message;

use App\Application\Message\Command\ProcessOrder;
use App\Domain\Repository\OrderRepositoryInterface;
use Symfony\Component\Messenger\Attribute\AsMessageHandler;
use Psr\Log\LoggerInterface;

#[AsMessageHandler]
final class ProcessOrderHandler
{
    public function __construct(
        private readonly OrderRepositoryInterface $repository,
        private readonly LoggerInterface $logger,
    ) {}
    
    public function __invoke(ProcessOrder $message): void
    {
        $order = $this->repository->find($message->orderId);
        
        if (!$order) {
            $this->logger->error('Order not found', ['orderId' => $message->orderId]);
            return;
        }
        
        try {
            // Process the order
            $order->process();
            $this->repository->save($order);
            
            $this->logger->info('Order processed', [
                'orderId' => $order->getId(),
                'processingTime' => $message->getProcessingTime(),
            ]);
        } catch (\Exception $e) {
            // Let it retry via messenger configuration
            throw $e;
        }
    }
}
```

### Message Bus Configuration

```yaml
# config/packages/messenger.yaml
framework:
    messenger:
        default_bus: command.bus
        
        buses:
            command.bus:
                middleware:
                    - doctrine_transaction
                    - validation
            
            query.bus:
                middleware:
                    - validation
            
            event.bus:
                default_middleware:
                    enabled: true
                    allow_no_handlers: true
                    
        transports:
            async:
                dsn: '%env(MESSENGER_TRANSPORT_DSN)%'
                serializer: messenger.transport.symfony_serializer
                options:
                    exchange:
                        type: topic
                        default_publish_routing_key: normal
                    queues:
                        messages_high:
                            binding_keys: [high]
                            max_retries: 5
                        messages_normal:
                            binding_keys: [normal]
                            max_retries: 3
                            
            failed:
                dsn: 'doctrine://default?queue_name=failed'
                
        routing:
            'App\Application\Message\Command\ProcessOrder': async
            'App\Application\Message\Event\*': event.bus
```

## 12. Docker Development Environment

### ✅ DO: Use Multi-Stage Builds

```dockerfile
# Dockerfile
# Stage 1: Dependencies
FROM php:8.4-fpm-alpine AS dependencies

RUN apk add --no-cache \
    postgresql-dev \
    redis \
    git \
    zip \
    unzip

# Install PHP extensions
RUN docker-php-ext-install \
    pdo_pgsql \
    opcache \
    pcntl \
    sockets

# Install OpenSwoole
RUN pecl install openswoole && \
    docker-php-ext-enable openswoole

# Install Composer
COPY --from=composer:2.8 /usr/bin/composer /usr/bin/composer

WORKDIR /app

# Copy composer files
COPY composer.json composer.lock ./

# Install dependencies
RUN composer install --no-scripts --no-autoloader

# Stage 2: Application
FROM php:8.4-fpm-alpine AS app

# Copy PHP extensions from dependencies stage
COPY --from=dependencies /usr/local/lib/php/extensions /usr/local/lib/php/extensions
COPY --from=dependencies /usr/local/etc/php/conf.d /usr/local/etc/php/conf.d

# Copy application
WORKDIR /app
COPY --from=dependencies /app/vendor ./vendor
COPY . .

# Generate optimized autoloader
RUN composer dump-autoload --optimize

# Stage 3: Production
FROM php:8.4-fpm-alpine AS production

# Security: Run as non-root user
RUN addgroup -g 1000 app && \
    adduser -D -u 1000 -G app app

# Copy from app stage
COPY --from=app --chown=app:app /app /app

# PHP configuration
COPY docker/php/php.ini /usr/local/etc/php/
COPY docker/php/www.conf /usr/local/etc/php-fpm.d/

USER app

EXPOSE 9000

CMD ["php-fpm"]
```

### Docker Compose for Development

```yaml
# docker-compose.yml
services:
  app:
    build:
      context: .
      target: app
    volumes:
      - .:/app
      - ./var:/app/var
    environment:
      - APP_ENV=dev
      - DATABASE_URL=postgresql://app:secret@postgres:5432/app_dev
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis

  nginx:
    image: nginx:alpine
    ports:
      - "8080:80"
    volumes:
      - ./docker/nginx/default.conf:/etc/nginx/conf.d/default.conf
      - .:/app
    depends_on:
      - app

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: app_dev
      POSTGRES_USER: app
      POSTGRES_PASSWORD: secret
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  mercure:
    image: dunglas/mercure
    environment:
      MERCURE_PUBLISHER_JWT_KEY: '!ChangeThisMercureHubJWTSecretKey!'
      MERCURE_SUBSCRIBER_JWT_KEY: '!ChangeThisMercureHubJWTSecretKey!'
    ports:
      - "3000:80"

volumes:
  postgres_data:
  redis_data:
```

## 13. CI/CD Pipeline

### ✅ DO: Implement Comprehensive GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        php-version: ['8.4']
        
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
          
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4
      
      - name: Setup PHP
        uses: shivammathur/setup-php@v2
        with:
          php-version: ${{ matrix.php-version }}
          extensions: mbstring, xml, ctype, iconv, intl, pdo_pgsql, redis, apcu
          coverage: pcov
          tools: composer:v2
          
      - name: Validate composer files
        run: composer validate --strict
        
      - name: Cache Composer packages
        uses: actions/cache@v3
        with:
          path: vendor
          key: ${{ runner.os }}-php-${{ hashFiles('**/composer.lock') }}
          restore-keys: |
            ${{ runner.os }}-php-
            
      - name: Install Dependencies
        run: composer install --prefer-dist --no-progress --optimize-autoloader
        
      - name: Run PHP CS Fixer
        run: composer check-style
        
      - name: Run Psalm
        run: composer analyze
        
      - name: Run PHPStan
        run: vendor/bin/phpstan analyse
        
      - name: Run Tests
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test
          REDIS_URL: redis://localhost:6379
        run: |
          bin/console doctrine:database:create --env=test
          bin/console doctrine:migrations:migrate --env=test --no-interaction
          composer test -- --coverage-xml
          
      - name: Run Mutation Tests
        run: composer mutation
        continue-on-error: true
        
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run security check
        uses: symfonycorp/security-checker-action@v5
```

## 14. Monitoring and Observability

### ✅ DO: Implement Comprehensive Logging

```php
<?php
declare(strict_types=1);

namespace App\Infrastructure\Logging;

use Monolog\Handler\HandlerInterface;
use Monolog\Handler\StreamHandler;
use Monolog\Handler\RedisHandler;
use Monolog\Processor\ProcessorInterface;
use Monolog\Processor\UidProcessor;
use Monolog\Processor\WebProcessor;
use Monolog\Processor\MemoryUsageProcessor;
use Monolog\Formatter\JsonFormatter;

final class LoggerFactory
{
    public function createApplicationLogger(): Logger
    {
        $handlers = [
            $this->createStreamHandler(),
            $this->createRedisHandler(),
        ];
        
        $processors = [
            new UidProcessor(),
            new WebProcessor(),
            new MemoryUsageProcessor(),
            new ContextProcessor(), // Custom processor for request context
        ];
        
        return new Logger('app', $handlers, $processors);
    }
    
    private function createStreamHandler(): HandlerInterface
    {
        $handler = new StreamHandler('php://stdout');
        $handler->setFormatter(new JsonFormatter());
        
        return $handler;
    }
    
    private function createRedisHandler(): HandlerInterface
    {
        $redis = new \Redis();
        $redis->connect($_ENV['REDIS_HOST']);
        
        $handler = new RedisHandler($redis, 'logs');
        $handler->setFormatter(new JsonFormatter());
        
        return $handler;
    }
}
```

### ✅ DO: Add OpenTelemetry Support

```php
<?php
declare(strict_types=1);

namespace App\Infrastructure\Telemetry;

use OpenTelemetry\API\Trace\TracerInterface;
use OpenTelemetry\API\Trace\SpanKind;
use OpenTelemetry\Context\Context;

final class TracingMiddleware
{
    public function __construct(
        private readonly TracerInterface $tracer,
    ) {}
    
    public function handle(Request $request, callable $next): Response
    {
        $span = $this->tracer->spanBuilder('http.request')
            ->setSpanKind(SpanKind::KIND_SERVER)
            ->setAttribute('http.method', $request->getMethod())
            ->setAttribute('http.url', $request->getUri())
            ->setAttribute('http.target', $request->getPathInfo())
            ->startSpan();
            
        $scope = $span->activate();
        
        try {
            $response = $next($request);
            
            $span->setAttribute('http.status_code', $response->getStatusCode());
            
            return $response;
        } catch (\Exception $e) {
            $span->recordException($e);
            $span->setStatus('ERROR', $e->getMessage());
            
            throw $e;
        } finally {
            $span->end();
            $scope->detach();
        }
    }
}
```

## 15. Advanced Patterns

### Domain Event Sourcing

```php
<?php
declare(strict_types=1);

namespace App\Domain\EventSourcing;

use App\Domain\Event\DomainEvent;
use Doctrine\DBAL\Connection;

final class EventStore
{
    public function __construct(
        private readonly Connection $connection,
        private readonly EventSerializer $serializer,
    ) {}
    
    public function append(string $aggregateId, array $events, int $expectedVersion): void
    {
        $this->connection->transactional(function () use ($aggregateId, $events, $expectedVersion) {
            // Check for concurrency conflicts
            $currentVersion = $this->getCurrentVersion($aggregateId);
            if ($currentVersion !== $expectedVersion) {
                throw new ConcurrencyException(
                    "Expected version {$expectedVersion}, but current version is {$currentVersion}"
                );
            }
            
            foreach ($events as $event) {
                $this->connection->insert('event_store', [
                    'aggregate_id' => $aggregateId,
                    'aggregate_type' => $event->getAggregateType(),
                    'event_type' => get_class($event),
                    'event_data' => $this->serializer->serialize($event),
                    'event_version' => ++$currentVersion,
                    'occurred_at' => $event->occurredAt()->format('Y-m-d H:i:s.u'),
                ]);
            }
        });
    }
    
    public function load(string $aggregateId, int $fromVersion = 0): array
    {
        $rows = $this->connection->fetchAllAssociative(
            'SELECT * FROM event_store WHERE aggregate_id = ? AND event_version > ? ORDER BY event_version',
            [$aggregateId, $fromVersion]
        );
        
        return array_map(
            fn($row) => $this->serializer->deserialize($row['event_data'], $row['event_type']),
            $rows
        );
    }
}
```

### Advanced Error Handling

```php
<?php
declare(strict_types=1);

namespace App\Infrastructure\Error;

use Symfony\Component\HttpKernel\Exception\HttpExceptionInterface;
use Symfony\Component\HttpFoundation\JsonResponse;

final class ErrorHandler
{
    private const ERROR_CODES = [
        ValidationException::class => 'VALIDATION_ERROR',
        AuthenticationException::class => 'AUTH_ERROR',
        ResourceNotFoundException::class => 'NOT_FOUND',
        DomainException::class => 'DOMAIN_ERROR',
    ];
    
    public function handle(\Throwable $exception): JsonResponse
    {
        $statusCode = $exception instanceof HttpExceptionInterface 
            ? $exception->getStatusCode() 
            : 500;
            
        $errorCode = self::ERROR_CODES[get_class($exception)] ?? 'INTERNAL_ERROR';
        
        $response = [
            'error' => [
                'code' => $errorCode,
                'message' => $this->getPublicMessage($exception),
                'timestamp' => (new \DateTimeImmutable())->format(\DateTime::ATOM),
                'trace_id' => Context::getCurrent()->getSpan()->getContext()->getTraceId(),
            ],
        ];
        
        if ($exception instanceof ValidationException) {
            $response['error']['violations'] = $exception->getViolations();
        }
        
        if ($_ENV['APP_DEBUG'] ?? false) {
            $response['error']['debug'] = [
                'exception' => get_class($exception),
                'file' => $exception->getFile(),
                'line' => $exception->getLine(),
                'trace' => $exception->getTraceAsString(),
            ];
        }
        
        return new JsonResponse($response, $statusCode);
    }
    
    private function getPublicMessage(\Throwable $exception): string
    {
        return match (true) {
            $exception instanceof DomainException => $exception->getMessage(),
            $exception instanceof ValidationException => 'Validation failed',
            $exception instanceof AuthenticationException => 'Authentication required',
            default => 'An error occurred',
        };
    }
}
```

## Conclusion

This guide represents the cutting edge of PHP development in 2025. Key takeaways:

1. **Embrace modern PHP features** - Property hooks, asymmetric visibility, and improved type system
2. **Think async-first** - Use Fibers and event loops for concurrent operations
3. **Layer your architecture** - Clear separation between domain, application, and infrastructure
4. **Cache aggressively** - Multi-tier caching with proper invalidation
5. **Test comprehensively** - Unit, integration, contract, and mutation testing
6. **Monitor everything** - Structured logging, distributed tracing, and metrics
7. **Automate relentlessly** - From code style to deployment

PHP has evolved into a mature, high-performance platform for building modern web applications. By following these patterns, you'll create applications that are maintainable, scalable, and ready for the challenges of production environments.