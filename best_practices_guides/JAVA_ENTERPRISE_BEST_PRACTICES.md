# The Definitive Guide to High-Performance Java Enterprise Development (Mid-2025 Edition)

This guide synthesizes cutting-edge patterns for building scalable, secure, and performant enterprise applications with Java 21+ LTS, Spring Boot 3.3+, and modern JVM technologies. It moves beyond basic tutorials to provide production-grade architectural blueprints proven in high-throughput systems.

## Prerequisites & Core Configuration

Ensure your project uses **Java 21 LTS** (or Java 23 for early adopters), **Spring Boot 3.3.4+**, and **Gradle 8.11+** or **Maven 3.9.10+**.

### Modern Gradle Configuration (Kotlin DSL)

```kotlin
// build.gradle.kts
import org.springframework.boot.gradle.tasks.bundling.BootBuildImage

plugins {
    java
    id("org.springframework.boot") version "3.3.4"
    id("io.spring.dependency-management") version "1.1.7"
    id("org.graalvm.buildtools.native") version "0.10.5"
    id("com.gorylenko.gradle-git-properties") version "2.4.2"
    id("org.openapi.generator") version "7.10.0"
}

group = "com.enterprise"
version = "1.0.0"

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
        vendor = JvmVendorSpec.BELLSOFT // Liberica NIK for native-image support
    }
}

configurations {
    compileOnly {
        extendsFrom(configurations.annotationProcessor.get())
    }
}

repositories {
    mavenCentral()
    maven { url = uri("https://repo.spring.io/milestone") }
}

dependencies {
    // Spring Boot Starters
    implementation("org.springframework.boot:spring-boot-starter-web")
    implementation("org.springframework.boot:spring-boot-starter-data-jpa")
    implementation("org.springframework.boot:spring-boot-starter-data-redis")
    implementation("org.springframework.boot:spring-boot-starter-actuator")
    implementation("org.springframework.boot:spring-boot-starter-validation")
    implementation("org.springframework.boot:spring-boot-starter-security")
    
    // Virtual Threads & Async Support
    implementation("org.springframework.boot:spring-boot-starter-webflux")
    
    // Modern Database Access
    implementation("org.jooq:jooq:3.19.15")
    implementation("com.zaxxer:HikariCP:6.2.0")
    runtimeOnly("org.postgresql:postgresql:42.7.5")
    
    // Performance & Monitoring
    implementation("io.micrometer:micrometer-tracing-bridge-otel")
    implementation("io.opentelemetry:opentelemetry-exporter-otlp")
    implementation("net.ttddyy.observation:datasource-micrometer-spring-boot:1.0.6")
    
    // Modern Utilities
    implementation("org.mapstruct:mapstruct:1.6.3")
    annotationProcessor("org.mapstruct:mapstruct-processor:1.6.3")
    compileOnly("org.projectlombok:lombok")
    annotationProcessor("org.projectlombok:lombok")
    
    // Testing
    testImplementation("org.springframework.boot:spring-boot-starter-test")
    testImplementation("org.springframework.security:spring-security-test")
    testImplementation("org.testcontainers:testcontainers:1.20.5")
    testImplementation("org.testcontainers:postgresql:1.20.5")
    testImplementation("io.rest-assured:rest-assured:5.6.0")
    testImplementation("org.awaitility:awaitility:4.2.2")
}

tasks.withType<JavaCompile> {
    options.compilerArgs.addAll(listOf(
        "--enable-preview",
        "-Xlint:all,-processing",
        "-parameters" // Preserve parameter names for Spring
    ))
}

tasks.withType<Test> {
    useJUnitPlatform()
    jvmArgs("--enable-preview", "-XX:+EnableDynamicAgentLoading")
    
    // Enable virtual threads for tests
    systemProperty("spring.threads.virtual.enabled", "true")
}

// Native Image Configuration
graalvmNative {
    binaries {
        named("main") {
            imageName = "enterprise-app"
            mainClass = "com.enterprise.Application"
            buildArgs.add("--enable-preview")
            buildArgs.add("--initialize-at-build-time=org.slf4j")
            buildArgs.add("-H:+ReportExceptionStackTraces")
            
            // Optimize for throughput (PGO-like optimization)
            buildArgs.add("-march=native")
            buildArgs.add("-O3")
        }
    }
}

// Container image building with Paketo Buildpacks
tasks.named<BootBuildImage>("bootBuildImage") {
    imageName = "enterprise/${project.name}:${project.version}"
    
    environment.put("BP_JVM_VERSION", "21")
    environment.put("BPE_DELIM_JAVA_TOOL_OPTIONS", " ")
    environment.put("BPE_APPEND_JAVA_TOOL_OPTIONS", "--enable-preview")
    
    // Enable CDS (Class Data Sharing) for faster startup
    environment.put("BPE_APPEND_JAVA_TOOL_OPTIONS", "-XX:SharedArchiveFile=app.jsa")
    
    docker {
        publishRegistry {
            username = System.getenv("DOCKER_USERNAME")
            password = System.getenv("DOCKER_PASSWORD")
        }
    }
}
```

### Application Properties for Performance

```yaml
# application.yml
spring:
  application:
    name: enterprise-service
  
  threads:
    virtual:
      enabled: true # Enable virtual threads globally
  
  datasource:
    url: jdbc:postgresql://localhost:5432/enterprise?reWriteBatchedInserts=true
    username: ${DB_USERNAME}
    password: ${DB_PASSWORD}
    hikari:
      maximum-pool-size: 20 # Much smaller with virtual threads
      minimum-idle: 5
      connection-timeout: 5000
      idle-timeout: 300000
      max-lifetime: 600000
      data-source-properties:
        prepStmtCacheSize: 250
        prepStmtCacheSqlLimit: 2048
        cachePrepStmts: true
        useServerPrepStmts: true
  
  jpa:
    hibernate:
      ddl-auto: validate
    properties:
      hibernate:
        # Java 21 performance optimizations
        jdbc:
          batch_size: 25
          order_inserts: true
          order_updates: true
        query:
          plan_cache_max_size: 256
          plan_parameter_metadata_max_size: 128
        # Enable statistics in dev only
        generate_statistics: false
  
  data:
    redis:
      repositories:
        enabled: false # Disable if not using Redis repositories
      lettuce:
        pool:
          enabled: true
          max-active: 8
          max-idle: 8
  
  # R2DBC for reactive data access
  r2dbc:
    url: r2dbc:postgresql://localhost:5432/enterprise
    pool:
      enabled: true
      initial-size: 5
      max-size: 20

# Virtual Thread Executor Configuration
management:
  endpoints:
    web:
      exposure:
        include: health,info,metrics,prometheus,threaddump,heapdump
  metrics:
    distribution:
      percentiles-histogram:
        http.server.requests: true
    tags:
      application: ${spring.application.name}
  tracing:
    sampling:
      probability: 0.1 # 10% sampling in production

# Security headers
server:
  compression:
    enabled: true
    mime-types: application/json,application/xml,text/html,text/xml,text/plain
  http2:
    enabled: true
  error:
    include-message: never
    include-stacktrace: never

# Logging
logging:
  level:
    root: INFO
    com.enterprise: DEBUG
    org.springframework.web: DEBUG
    org.hibernate.SQL: DEBUG # Development only
  pattern:
    console: "%d{ISO8601} %highlight(%-5level) [%blue(%t)] %yellow(%C{1}): %msg%n%throwable"
```

---

## 1. Project Structure & Architecture

Modern Java enterprise applications require clear separation of concerns and hexagonal architecture principles.

### ✅ DO: Use Domain-Driven Design with Hexagonal Architecture

```
/src/main/java/com/enterprise
├── application/           # Application services (use cases)
│   ├── port/             # Interfaces (ports)
│   │   ├── in/           # Driving ports (API interfaces)
│   │   └── out/          # Driven ports (SPI interfaces)
│   └── service/          # Service implementations
├── domain/               # Core business logic (entities, value objects)
│   ├── model/           # Domain entities and aggregates
│   ├── event/           # Domain events
│   └── exception/       # Domain-specific exceptions
├── infrastructure/       # Technical implementation details
│   ├── adapter/         # Port implementations
│   │   ├── in/          # REST controllers, GraphQL, gRPC
│   │   └── out/         # Database, messaging, external APIs
│   ├── config/          # Spring configuration classes
│   └── security/        # Security infrastructure
└── common/              # Shared utilities and constants
```

### ✅ DO: Use Java Records for DTOs and Value Objects

Records provide immutability, automatic equals/hashCode, and pattern matching support.

```java
// Domain value object
public record Money(
    BigDecimal amount,
    Currency currency
) {
    // Compact constructor for validation
    public Money {
        Objects.requireNonNull(amount, "Amount cannot be null");
        Objects.requireNonNull(currency, "Currency cannot be null");
        if (amount.scale() > currency.getDefaultFractionDigits()) {
            throw new IllegalArgumentException("Too many decimal places");
        }
    }
    
    // Factory methods for common cases
    public static Money zero(Currency currency) {
        return new Money(BigDecimal.ZERO, currency);
    }
    
    // Business logic methods
    public Money add(Money other) {
        if (!currency.equals(other.currency)) {
            throw new IllegalArgumentException("Currency mismatch");
        }
        return new Money(amount.add(other.amount), currency);
    }
}

// API DTO with validation
public record CreateOrderRequest(
    @NotNull @Size(min = 1, max = 50) String customerId,
    @NotEmpty List<@Valid OrderItemRequest> items,
    @Valid AddressDto shippingAddress
) {}
```

---

## 2. Virtual Threads: The New Concurrency Paradigm

Java 21's virtual threads eliminate the need for reactive programming in most cases while maintaining excellent performance.

### ✅ DO: Use Virtual Threads for I/O-Bound Operations

Virtual threads provide the simplicity of blocking code with the performance of async.

```java
@RestController
@RequestMapping("/api/v1/orders")
public class OrderController {
    private final OrderService orderService;
    private final InventoryClient inventoryClient;
    private final PaymentClient paymentClient;
    
    @PostMapping
    public ResponseEntity<OrderResponse> createOrder(@Valid @RequestBody CreateOrderRequest request) {
        // Each HTTP request runs on a virtual thread
        // These blocking calls don't waste OS threads
        var inventory = inventoryClient.checkInventory(request.items());
        var paymentAuth = paymentClient.authorizePayment(request.paymentDetails());
        
        // Even with "blocking" calls, thousands of concurrent requests are handled efficiently
        var order = orderService.createOrder(request, inventory, paymentAuth);
        
        return ResponseEntity.status(HttpStatus.CREATED).body(order);
    }
}
```

### ✅ DO: Use Structured Concurrency for Parallel Operations

Replace `CompletableFuture` with structured concurrency for cleaner error handling.

```java
@Service
public class OrderEnrichmentService {
    
    public EnrichedOrder enrichOrder(Order order) throws InterruptedException {
        try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
            // Launch parallel tasks
            Supplier<Customer> customerTask = scope.fork(() -> 
                customerService.findById(order.customerId())
            );
            
            Supplier<List<Product>> productsTask = scope.fork(() ->
                productService.findAllById(order.productIds())
            );
            
            Supplier<ShippingInfo> shippingTask = scope.fork(() ->
                shippingService.calculateShipping(order)
            );
            
            // Wait for all tasks or fail fast on first error
            scope.join()           // Wait for all tasks
                 .throwIfFailed(); // Propagate any exceptions
            
            // All tasks completed successfully
            return new EnrichedOrder(
                order,
                customerTask.get(),
                productsTask.get(),
                shippingTask.get()
            );
        }
    }
}
```

### Advanced Virtual Thread Patterns

#### Custom Thread Factory for Observability

```java
@Configuration
public class VirtualThreadConfig {
    
    @Bean
    public TomcatProtocolHandlerCustomizer<?> protocolHandlerVirtualThreadCustomizer() {
        return protocolHandler -> {
            protocolHandler.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
        };
    }
    
    @Bean
    public AsyncTaskExecutor applicationTaskExecutor() {
        // Custom virtual thread factory with names for debugging
        var factory = Thread.ofVirtual()
            .name("app-vthread-", 0)
            .factory();
            
        return new TaskExecutorAdapter(Executors.newThreadPerTaskExecutor(factory));
    }
}
```

### ❌ DON'T: Mix Virtual Threads with Thread Pools

Virtual threads are cheap; create new ones instead of pooling.

```java
// Bad - Don't pool virtual threads
private final ExecutorService pool = Executors.newFixedThreadPool(10);

// Good - Create virtual threads on demand
private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
```

---

## 3. Modern Database Access Patterns

### ✅ DO: Use jOOQ for Complex Queries with Type Safety

jOOQ provides SQL-like DSL with compile-time safety, perfect for complex reporting queries.

```java
@Repository
@RequiredArgsConstructor
public class OrderRepository {
    private final DSLContext dsl;
    
    public List<OrderSummary> findOrderSummaries(
            LocalDate startDate, 
            LocalDate endDate,
            OrderStatus status) {
        
        return dsl.select(
                ORDERS.ID,
                ORDERS.ORDER_NUMBER,
                ORDERS.CREATED_AT,
                CUSTOMERS.NAME.as("customerName"),
                DSL.sum(ORDER_ITEMS.QUANTITY.multiply(ORDER_ITEMS.PRICE)).as("totalAmount"),
                DSL.count(ORDER_ITEMS.ID).as("itemCount")
            )
            .from(ORDERS)
            .join(CUSTOMERS).on(ORDERS.CUSTOMER_ID.eq(CUSTOMERS.ID))
            .leftJoin(ORDER_ITEMS).on(ORDERS.ID.eq(ORDER_ITEMS.ORDER_ID))
            .where(ORDERS.CREATED_AT.between(startDate.atStartOfDay(), endDate.plusDays(1).atStartOfDay()))
            .and(ORDERS.STATUS.eq(status.name()))
            .groupBy(ORDERS.ID, ORDERS.ORDER_NUMBER, ORDERS.CREATED_AT, CUSTOMERS.NAME)
            .orderBy(ORDERS.CREATED_AT.desc())
            .fetchInto(OrderSummary.class);
    }
    
    // Batch operations with jOOQ
    public void batchUpdateOrderStatus(List<Long> orderIds, OrderStatus newStatus) {
        dsl.update(ORDERS)
            .set(ORDERS.STATUS, newStatus.name())
            .set(ORDERS.UPDATED_AT, LocalDateTime.now())
            .where(ORDERS.ID.in(orderIds))
            .execute();
    }
}
```

### ✅ DO: Use Spring Data JPA with Projections for Simple Cases

For straightforward CRUD operations, Spring Data JPA with proper projections prevents N+1 queries.

```java
// Define projection interfaces or records
public record CustomerProjection(Long id, String name, String email) {}

@Repository
public interface CustomerRepository extends JpaRepository<Customer, Long> {
    
    // Use projections to fetch only needed fields
    @Query("""
        SELECT new com.enterprise.dto.CustomerProjection(c.id, c.name, c.email)
        FROM Customer c
        WHERE c.status = :status
        ORDER BY c.name
        """)
    List<CustomerProjection> findActiveCustomersProjection(@Param("status") Status status);
    
    // Entity graphs for eager loading
    @EntityGraph(attributePaths = {"orders", "orders.items"})
    Optional<Customer> findWithOrdersById(Long id);
    
    // Modifying queries with clear transaction boundaries
    @Modifying
    @Query("UPDATE Customer c SET c.lastLoginAt = :timestamp WHERE c.id = :id")
    void updateLastLogin(@Param("id") Long id, @Param("timestamp") Instant timestamp);
}
```

### Advanced JPA Patterns: Second-Level Cache with Redis

```java
@Entity
@Cacheable
@org.hibernate.annotations.Cache(usage = CacheConcurrencyStrategy.READ_WRITE)
public class Product {
    @Id
    @GeneratedValue(strategy = GenerationType.SEQUENCE)
    private Long id;
    
    // Natural ID for cache key
    @NaturalId
    @Column(unique = true, nullable = false)
    private String sku;
    
    // Other fields...
}

// Configuration for Redis as L2 Cache
@Configuration
@EnableCaching
public class CacheConfig {
    
    @Bean
    public RedisCacheManager cacheManager(RedisConnectionFactory connectionFactory) {
        var config = RedisCacheConfiguration.defaultCacheConfig()
            .entryTtl(Duration.ofMinutes(60))
            .serializeKeysWith(RedisSerializationContext.SerializationPair.fromSerializer(new StringRedisSerializer()))
            .serializeValuesWith(RedisSerializationContext.SerializationPair.fromSerializer(new GenericJackson2JsonRedisSerializer()));
            
        return RedisCacheManager.builder(connectionFactory)
            .cacheDefaults(config)
            .transactionAware()
            .build();
    }
}
```

---

## 4. Pattern Matching and Modern Java Features

### ✅ DO: Use Pattern Matching for Elegant Business Logic

Java 21's pattern matching eliminates verbose instanceof checks and visitor patterns.

```java
@Service
public class PricingService {
    
    public BigDecimal calculateDiscount(Customer customer, Order order) {
        // Pattern matching with switch expressions
        return switch (customer) {
            case PremiumCustomer(var id, var name, var tier) when tier == Tier.GOLD -> 
                order.subtotal().multiply(BigDecimal.valueOf(0.20));
                
            case PremiumCustomer(var id, var name, var tier) when tier == Tier.SILVER -> 
                order.subtotal().multiply(BigDecimal.valueOf(0.10));
                
            case CorporateCustomer(var id, var name, var contractDiscount) -> 
                order.subtotal().multiply(contractDiscount);
                
            case RegularCustomer c when order.subtotal().compareTo(BigDecimal.valueOf(100)) > 0 -> 
                BigDecimal.valueOf(5); // Flat $5 discount for orders over $100
                
            default -> BigDecimal.ZERO;
        };
    }
    
    // Pattern matching with sealed classes
    public void processPayment(Payment payment) {
        switch (payment) {
            case CreditCardPayment(var number, var cvv, var expiry) -> 
                creditCardProcessor.process(number, cvv, expiry);
                
            case BankTransferPayment(var accountNumber, var routingNumber) -> 
                bankProcessor.process(accountNumber, routingNumber);
                
            case CryptoPayment(var wallet, var currency, var amount) -> 
                cryptoProcessor.process(wallet, currency, amount);
                
            // Compiler ensures all cases are covered with sealed classes
        }
    }
}

// Sealed class hierarchy
public sealed interface Payment 
    permits CreditCardPayment, BankTransferPayment, CryptoPayment {}

public record CreditCardPayment(String number, String cvv, YearMonth expiry) implements Payment {}
public record BankTransferPayment(String accountNumber, String routingNumber) implements Payment {}
public record CryptoPayment(String wallet, String currency, BigDecimal amount) implements Payment {}
```

### ✅ DO: Use Text Blocks for Complex Queries and Templates

```java
@Repository
public class ReportRepository {
    private final JdbcTemplate jdbcTemplate;
    
    public List<RevenueReport> generateRevenueReport(YearMonth period) {
        var sql = """
            WITH monthly_revenue AS (
                SELECT 
                    DATE_TRUNC('day', o.created_at) as order_date,
                    c.segment as customer_segment,
                    SUM(oi.quantity * oi.unit_price) as daily_revenue,
                    COUNT(DISTINCT o.id) as order_count,
                    COUNT(DISTINCT o.customer_id) as unique_customers
                FROM orders o
                JOIN customers c ON o.customer_id = c.id
                JOIN order_items oi ON o.id = oi.order_id
                WHERE DATE_TRUNC('month', o.created_at) = ?::date
                  AND o.status = 'COMPLETED'
                GROUP BY DATE_TRUNC('day', o.created_at), c.segment
            )
            SELECT 
                order_date,
                customer_segment,
                daily_revenue,
                order_count,
                unique_customers,
                daily_revenue / NULLIF(order_count, 0) as avg_order_value
            FROM monthly_revenue
            ORDER BY order_date, customer_segment
            """;
            
        return jdbcTemplate.query(
            sql,
            new BeanPropertyRowMapper<>(RevenueReport.class),
            period.atDay(1)
        );
    }
}
```

---

## 5. Reactive Programming When You Actually Need It

While virtual threads eliminate the need for reactive programming in most cases, some scenarios still benefit from reactive streams.

### ✅ DO: Use Reactive Streams for Event Streaming and Backpressure

```java
@RestController
@RequestMapping("/api/v1/events")
public class EventStreamController {
    private final KafkaReceiver<String, OrderEvent> kafkaReceiver;
    
    @GetMapping(value = "/orders", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<OrderEventDto>> streamOrderEvents(
            @RequestParam(required = false) String customerId) {
        
        return kafkaReceiver
            .receive()
            .filter(record -> customerId == null || 
                             record.value().customerId().equals(customerId))
            .map(record -> record.value())
            .map(this::toDto)
            .map(dto -> ServerSentEvent.<OrderEventDto>builder()
                .id(String.valueOf(dto.eventId()))
                .event(dto.eventType())
                .data(dto)
                .build())
            .doOnCancel(() -> log.info("Client disconnected from event stream"));
    }
}

// R2DBC for reactive database access
@Repository
public interface ReactiveOrderRepository extends ReactiveCrudRepository<Order, Long> {
    
    @Query("""
        SELECT o.* FROM orders o 
        WHERE o.status = :status 
        AND o.created_at > :since
        ORDER BY o.created_at DESC
        """)
    Flux<Order> findRecentOrdersByStatus(String status, LocalDateTime since);
}
```

---

## 6. Security Patterns for Zero-Trust Architecture

### ✅ DO: Implement Defense in Depth with Spring Security

```java
@Configuration
@EnableWebSecurity
@EnableMethodSecurity
public class SecurityConfig {
    
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        return http
            .csrf(csrf -> csrf
                .csrfTokenRepository(CookieCsrfTokenRepository.withHttpOnlyFalse())
                .ignoringRequestMatchers("/api/webhooks/**")
            )
            .sessionManagement(session -> session
                .sessionCreationPolicy(SessionCreationPolicy.STATELESS)
            )
            .authorizeHttpRequests(authz -> authz
                .requestMatchers("/api/public/**").permitAll()
                .requestMatchers("/api/admin/**").hasRole("ADMIN")
                .requestMatchers("/actuator/health/**").permitAll()
                .anyRequest().authenticated()
            )
            .oauth2ResourceServer(oauth2 -> oauth2
                .jwt(jwt -> jwt
                    .jwtAuthenticationConverter(jwtAuthenticationConverter())
                )
            )
            .exceptionHandling(exceptions -> exceptions
                .authenticationEntryPoint(new BearerTokenAuthenticationEntryPoint())
                .accessDeniedHandler(new BearerTokenAccessDeniedHandler())
            )
            .build();
    }
    
    @Bean
    public JwtAuthenticationConverter jwtAuthenticationConverter() {
        var converter = new JwtAuthenticationConverter();
        converter.setJwtGrantedAuthoritiesConverter(jwt -> {
            // Extract roles from JWT claims
            Collection<String> roles = jwt.getClaim("roles");
            return roles.stream()
                .map(role -> new SimpleGrantedAuthority("ROLE_" + role))
                .collect(Collectors.toList());
        });
        return converter;
    }
}

// Method-level security
@Service
@Slf4j
public class OrderService {
    
    @PreAuthorize("hasRole('USER') and #customerId == authentication.principal.customerId")
    public Order createOrder(String customerId, CreateOrderRequest request) {
        // Method only accessible if user owns the customer account
    }
    
    @PostAuthorize("returnObject.customerId == authentication.principal.customerId")
    public Order getOrder(Long orderId) {
        // Ensures users can only see their own orders
    }
}
```

### ✅ DO: Implement Rate Limiting with Bucket4j

```java
@Component
public class RateLimitingFilter extends OncePerRequestFilter {
    private final Map<String, Bucket> buckets = new ConcurrentHashMap<>();
    
    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain) throws ServletException, IOException {
        
        String key = getClientKey(request);
        Bucket bucket = buckets.computeIfAbsent(key, k -> createNewBucket());
        
        ConsumptionProbe probe = bucket.tryConsumeAndReturnRemaining(1);
        
        if (probe.isConsumed()) {
            response.addHeader("X-Rate-Limit-Remaining", 
                String.valueOf(probe.getRemainingTokens()));
            filterChain.doFilter(request, response);
        } else {
            response.addHeader("X-Rate-Limit-Retry-After-Seconds", 
                String.valueOf(probe.getNanosToWaitForRefill() / 1_000_000_000));
            response.sendError(HttpStatus.TOO_MANY_REQUESTS.value(), 
                "Rate limit exceeded");
        }
    }
    
    private Bucket createNewBucket() {
        return Bucket.builder()
            .addLimit(Bandwidth.classic(100, Refill.intervally(100, Duration.ofMinutes(1))))
            .build();
    }
}
```

---

## 7. Testing Patterns for Confidence

### ✅ DO: Use Testcontainers for Integration Tests

```java
@SpringBootTest
@Testcontainers
@AutoConfigureMockMvc
class OrderControllerIntegrationTest {
    
    @Container
    static PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:16-alpine")
        .withDatabaseName("testdb")
        .withUsername("test")
        .withPassword("test")
        .withInitScript("schema.sql");
    
    @Container
    static KafkaContainer kafka = new KafkaContainer(DockerImageName.parse("confluentinc/cp-kafka:7.7.0"))
        .withKraft(); // Use KRaft mode (no Zookeeper)
    
    @DynamicPropertySource
    static void properties(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", postgres::getJdbcUrl);
        registry.add("spring.datasource.username", postgres::getUsername);
        registry.add("spring.datasource.password", postgres::getPassword);
        registry.add("spring.kafka.bootstrap-servers", kafka::getBootstrapServers);
    }
    
    @Test
    @WithMockUser(roles = "USER")
    void createOrder_ShouldPublishEvent() {
        // Given
        var request = new CreateOrderRequest(
            "CUST-123",
            List.of(new OrderItem("PROD-1", 2)),
            new Address("123 Main St", "City", "12345")
        );
        
        // When
        var response = given()
            .contentType(ContentType.JSON)
            .body(request)
            .when()
            .post("/api/v1/orders")
            .then()
            .statusCode(201)
            .extract()
            .as(OrderResponse.class);
        
        // Then
        await().atMost(5, SECONDS).untilAsserted(() -> {
            var records = KafkaTestUtils.getRecords(consumer);
            assertThat(records).hasSize(1);
            assertThat(records.iterator().next().value())
                .contains(response.orderId());
        });
    }
}
```

### ✅ DO: Use ArchUnit for Architecture Tests

```java
@AnalyzeClasses(packages = "com.enterprise")
class ArchitectureTest {
    
    @ArchTest
    static final ArchRule domainShouldNotDependOnInfrastructure = 
        noClasses().that().resideInAPackage("..domain..")
            .should().dependOnClassesThat()
            .resideInAPackage("..infrastructure..");
    
    @ArchTest
    static final ArchRule controllersShouldNotUseRepositories =
        noClasses().that().resideInAPackage("..adapter.in.web..")
            .should().dependOnClassesThat()
            .areAnnotatedWith(Repository.class);
    
    @ArchTest
    static final ArchRule servicesShouldBeAnnotated =
        classes().that().resideInAPackage("..application.service..")
            .should().beAnnotatedWith(Service.class);
    
    @ArchTest
    static final ArchRule interfacesShouldNotHavePublicFields =
        noClasses().that().areInterfaces()
            .should().havePublicFields();
}
```

---

## 8. Performance Optimization Patterns

### ✅ DO: Use JMH for Microbenchmarking

```java
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Benchmark)
@Fork(value = 2, jvmArgs = {"-Xms2G", "-Xmx2G", "--enable-preview"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class SerializationBenchmark {
    
    private Order order;
    private ObjectMapper jacksonMapper;
    private Gson gson;
    
    @Setup
    public void setup() {
        order = createComplexOrder();
        jacksonMapper = new ObjectMapper()
            .registerModule(new JavaTimeModule())
            .disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
        gson = new GsonBuilder()
            .registerTypeAdapter(LocalDateTime.class, new LocalDateTimeAdapter())
            .create();
    }
    
    @Benchmark
    public String jacksonSerialization() throws Exception {
        return jacksonMapper.writeValueAsString(order);
    }
    
    @Benchmark
    public String gsonSerialization() {
        return gson.toJson(order);
    }
    
    @Benchmark
    public String recordToString() {
        // Leverage record's toString() for logging scenarios
        return order.toString();
    }
}
```

### ✅ DO: Profile and Optimize Garbage Collection

```yaml
# For low-latency services (startup script)
JAVA_OPTS="
  -XX:+UseZGC
  -XX:+ZGenerational
  -XX:MaxGCPauseMillis=10
  -XX:+AlwaysPreTouch
  -XX:+UseNUMA
  -XX:+UseCompressedOops
  -XX:+UseStringDeduplication
  -Xms4g
  -Xmx4g
"

# For throughput-oriented batch processing
JAVA_OPTS="
  -XX:+UseParallelGC
  -XX:MaxGCPauseMillis=200
  -XX:+UseCompressedOops
  -Xms8g
  -Xmx8g
"
```

---

## 9. Observability and Monitoring

### ✅ DO: Implement Comprehensive Telemetry

```java
@Configuration
public class ObservabilityConfig {
    
    @Bean
    public ObservationRegistry observationRegistry() {
        ObservationRegistry registry = ObservationRegistry.create();
        
        // Add Micrometer metrics
        registry.observationConfig()
            .observationHandler(new DefaultMeterObservationHandler(meterRegistry));
            
        // Add distributed tracing
        registry.observationConfig()
            .observationHandler(new PropagatingSenderTracingObservationHandler(tracer, propagator));
            
        return registry;
    }
    
    @Bean
    @ConditionalOnProperty(value = "management.tracing.enabled", havingValue = "true")
    public Sampler sampler(@Value("${management.tracing.sampling.probability:0.1}") float probability) {
        return Sampler.probability(probability);
    }
    
    // Custom metrics
    @Component
    @RequiredArgsConstructor
    public class BusinessMetrics {
        private final MeterRegistry meterRegistry;
        
        public void recordOrderCreated(Order order) {
            meterRegistry.counter("orders.created",
                "type", order.type().name(),
                "customer_segment", order.customerSegment()
            ).increment();
            
            meterRegistry.gauge("order.value", 
                Tags.of("currency", order.currency()),
                order.totalAmount().doubleValue()
            );
        }
        
        @EventListener
        public void handleOrderCompleted(OrderCompletedEvent event) {
            var timer = Timer.start(meterRegistry);
            var duration = Duration.between(event.order().createdAt(), event.completedAt());
            
            timer.stop(Timer.builder("order.processing.duration")
                .tag("type", event.order().type().name())
                .tag("success", String.valueOf(event.isSuccessful()))
                .register(meterRegistry)
            );
        }
    }
}

// Structured logging with context
@Slf4j
@Component
public class OrderProcessor {
    
    public void processOrder(Order order) {
        try (var ignored = MDC.putCloseable("orderId", order.id());
             var ignored2 = MDC.putCloseable("customerId", order.customerId())) {
            
            log.info("Processing order started");
            
            // Process order...
            
            log.info("Order processed successfully", 
                kv("duration_ms", System.currentTimeMillis() - startTime),
                kv("items_count", order.items().size())
            );
        }
    }
}
```

---

## 10. Container Deployment & Native Images

### ✅ DO: Build Optimized Container Images

```dockerfile
# Multi-stage Dockerfile for JVM deployment
FROM bellsoft/liberica-runtime-container:jre-21-cds-slim-glibc as builder

WORKDIR /app
COPY build/libs/*.jar app.jar

# Extract layers for better caching
RUN java -Djarmode=tools -jar app.jar extract --layers --launcher

# Generate CDS archive for faster startup
RUN java -XX:ArchiveClassesAtExit=app.jsa -Dspring.context.exit=onRefresh -jar app.jar

FROM bellsoft/liberica-runtime-container:jre-21-cds-slim-glibc

# Add non-root user
RUN addgroup -S spring && adduser -S spring -G spring
USER spring:spring

WORKDIR /app

# Copy layers in order of change frequency
COPY --from=builder /app/dependencies/ ./
COPY --from=builder /app/spring-boot-loader/ ./
COPY --from=builder /app/snapshot-dependencies/ ./
COPY --from=builder /app/application/ ./
COPY --from=builder /app/app.jsa ./

# JVM flags for containers
ENV JAVA_TOOL_OPTIONS="\
  -XX:SharedArchiveFile=app.jsa \
  -XX:MaxRAMPercentage=75 \
  -XX:+UseZGC \
  -XX:+ZGenerational \
  --enable-preview"

EXPOSE 8080
ENTRYPOINT ["java", "org.springframework.boot.loader.launch.JarLauncher"]
```

### GraalVM Native Image Configuration

```java
// Native hints for reflection/proxies
@Configuration
@ImportRuntimeHints(NativeConfig.AppRuntimeHints.class)
public class NativeConfig {
    
    static class AppRuntimeHints implements RuntimeHintsRegistrar {
        @Override
        public void registerHints(RuntimeHints hints, ClassLoader classLoader) {
            // Register for reflection
            hints.reflection()
                .registerType(OrderEvent.class, MemberCategory.values())
                .registerType(TypeReference.of("com.enterprise.dto.OrderResponse"), 
                    MemberCategory.values());
            
            // Register resources
            hints.resources()
                .registerPattern("db/migration/*.sql")
                .registerPattern("templates/*.html");
            
            // Register for serialization
            hints.serialization()
                .registerType(OrderCreatedEvent.class)
                .registerType(PaymentProcessedEvent.class);
        }
    }
}
```

---

## 11. Event-Driven Architecture with Domain Events

### ✅ DO: Implement Transactional Outbox Pattern

```java
@Entity
@Table(name = "outbox_events")
public class OutboxEvent {
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;
    
    @Column(nullable = false)
    private String aggregateId;
    
    @Column(nullable = false)
    private String eventType;
    
    @Column(columnDefinition = "jsonb")
    private String payload;
    
    @Column(nullable = false)
    private Instant createdAt;
    
    @Column
    private Instant processedAt;
}

@Component
@Transactional
@RequiredArgsConstructor
public class OrderService {
    private final OrderRepository orderRepository;
    private final OutboxEventRepository outboxRepository;
    
    public Order createOrder(CreateOrderRequest request) {
        // Create order
        var order = new Order(/* ... */);
        orderRepository.save(order);
        
        // Store event in outbox (same transaction)
        var event = new OrderCreatedEvent(order);
        var outboxEvent = new OutboxEvent(
            order.getId().toString(),
            event.getClass().getSimpleName(),
            serialize(event),
            Instant.now()
        );
        outboxRepository.save(outboxEvent);
        
        return order;
    }
}

// Async processor for outbox events
@Component
@EnableScheduling
@RequiredArgsConstructor
@Slf4j
public class OutboxProcessor {
    private final OutboxEventRepository repository;
    private final KafkaTemplate<String, Object> kafkaTemplate;
    
    @Scheduled(fixedDelay = 5000)
    public void processOutboxEvents() {
        var unprocessedEvents = repository.findUnprocessedEvents(100);
        
        for (var event : unprocessedEvents) {
            try {
                kafkaTemplate.send(
                    "domain-events",
                    event.getAggregateId(),
                    event.getPayload()
                ).get(5, TimeUnit.SECONDS);
                
                event.setProcessedAt(Instant.now());
                repository.save(event);
                
            } catch (Exception e) {
                log.error("Failed to publish event: {}", event.getId(), e);
                // Will retry on next run
            }
        }
    }
}
```

---

## 12. CI/CD Pipeline with GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

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
        java: [ 21, 23 ]
    
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
          
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up JDK ${{ matrix.java }}
      uses: actions/setup-java@v4
      with:
        java-version: ${{ matrix.java }}
        distribution: 'liberica'
        cache: 'gradle'
    
    - name: Run tests
      run: ./gradlew test integrationTest --info
      env:
        TESTCONTAINERS_REUSE_ENABLE: true
    
    - name: Analyze with SonarQube
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
      run: ./gradlew sonar
    
    - name: Build native image
      if: matrix.java == 21
      run: ./gradlew nativeCompile
    
    - name: Test native image
      if: matrix.java == 21
      run: |
        ./build/native/nativeCompile/enterprise-app &
        sleep 10
        curl -f http://localhost:8080/actuator/health || exit 1
        
    - name: Build and push Docker image
      if: github.ref == 'refs/heads/main' && matrix.java == 21
      run: |
        ./gradlew bootBuildImage --publishImage
      env:
        DOCKER_USERNAME: ${{ secrets.DOCKER_USERNAME }}
        DOCKER_PASSWORD: ${{ secrets.DOCKER_PASSWORD }}
```

---

## Summary: The Modern Java Enterprise Stack

The mid-2025 Java enterprise landscape is defined by:

1. **Virtual Threads as the default** - Eliminating complex reactive code while maintaining performance
2. **Pattern matching and records** - Reducing boilerplate and increasing expressiveness
3. **Native compilation** - Sub-second startup times and reduced memory footprint
4. **Comprehensive observability** - Built-in from day one with OpenTelemetry
5. **Type-safe data access** - jOOQ for complex queries, Spring Data for simple CRUD
6. **Container-first deployment** - Optimized images with CDS and native compilation
7. **Event-driven architecture** - Transactional outbox pattern for reliability
8. **Zero-trust security** - Defense in depth with Spring Security 6

This architecture provides the foundation for building systems that scale from startup MVP to enterprise-grade platforms handling millions of requests per day.