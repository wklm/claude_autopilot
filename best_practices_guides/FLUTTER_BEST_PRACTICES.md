# The Definitive Guide to Flutter Mobile Development with Dart 3 & Impeller (Mid-2025)

This guide synthesizes modern best practices for building production-grade, cross-platform mobile applications with Flutter 3.29+, Dart 3.6+, and the Impeller renderer. It moves beyond basic tutorials to provide battle-tested architectural patterns for scalable, performant applications.

### Prerequisites & Core Configuration

Ensure your project uses **Flutter 3.29.0+**, **Dart 3.6.0+**, and targets **iOS 15+** and **Android API 24+** (Android 7.0).

Enable modern features in your `pubspec.yaml`:

```yaml
name: smartapp
description: Production-grade Flutter application
version: 1.0.0+1
publish_to: 'none'

environment:
  sdk: '>=3.6.0 <4.0.0'
  flutter: '>=3.29.0'

dependencies:
  flutter:
    sdk: flutter
  
  # Core dependencies - pin exact versions for production
  flutter_riverpod: 2.6.1
  riverpod_annotation: 2.6.1
  go_router: 15.2.4
  dio: 5.9.0
  freezed_annotation: 3.0.6
  json_annotation: 4.12.0
  drift: 2.26.0
  sqlite3_flutter_libs: ^0.5.0
  shared_preferences: 2.4.1
  flutter_secure_storage: 9.5.0
  
  # UI/UX enhancements
  flutter_native_splash: 2.5.1
  cached_network_image: 3.5.0
  shimmer: 3.2.0
  lottie: 3.3.1
  
  # Platform integration
  permission_handler: 11.5.0
  device_info_plus: 11.2.0
  package_info_plus: 9.1.0
  
dev_dependencies:
  flutter_test:
    sdk: flutter
  
  # Code generation
  build_runner: 2.4.16
  riverpod_generator: 2.6.1
  freezed: 3.0.6
  json_serializable: 6.10.0
  drift_dev: 2.26.0
  
  # Linting and analysis
  flutter_lints: 5.0.0
  custom_lint: 0.7.0
  riverpod_lint: 2.7.0
  
  # Testing
  mocktail: 1.1.0
  flutter_test_robots: 0.5.0
  integration_test:
    sdk: flutter

flutter:
  uses-material-design: true
  
  # Impeller is now default on iOS and Android in Flutter 3.29+
  # No configuration needed for Impeller - it's enabled by default
```

Configure analysis options for strict type safety:

```yaml
# analysis_options.yaml
include: package:flutter_lints/flutter.yaml

analyzer:
  language:
    strict-casts: true
    strict-inference: true
    strict-raw-types: true
  
  exclude:
    - "**/*.g.dart"
    - "**/*.freezed.dart"
    - "lib/generated/**"
  
  errors:
    invalid_annotation_target: ignore # For freezed
    
linter:
  rules:
    # Additional strict rules for production code
    always_declare_return_types: true
    always_use_package_imports: true
    avoid_dynamic_calls: true
    avoid_slow_async_io: true
    cancel_subscriptions: true
    close_sinks: true
    prefer_const_constructors: true
    prefer_const_declarations: true
    prefer_final_locals: true
    unawaited_futures: true
    use_build_context_synchronously: true
```

---

## 1. Scalable Architecture & Project Structure

A well-organized project structure is critical for maintainability as your app grows. Adopt a feature-first approach with clear separation of concerns.

### ✅ DO: Use Feature-First Architecture

This structure scales from small MVPs to large enterprise applications with multiple teams.

```
lib/
├── core/                      # Shared utilities and foundation
│   ├── constants/             # App-wide constants
│   │   ├── api_endpoints.dart
│   │   └── app_colors.dart
│   ├── errors/                # Error handling and exceptions
│   │   ├── exceptions.dart
│   │   └── failures.dart
│   ├── extensions/            # Dart extensions
│   │   ├── context_extensions.dart
│   │   └── string_extensions.dart
│   ├── network/               # HTTP client configuration
│   │   ├── api_client.dart
│   │   ├── interceptors/
│   │   └── authenticator.dart
│   ├── routing/               # Navigation configuration
│   │   ├── app_router.dart
│   │   └── route_guards.dart
│   └── utils/                 # Pure utility functions
│       ├── formatters.dart
│       └── validators.dart
│
├── features/                  # Feature modules
│   ├── auth/
│   │   ├── data/
│   │   │   ├── datasources/   # Remote/local data sources
│   │   │   ├── models/        # Data transfer objects
│   │   │   └── repositories/  # Repository implementations
│   │   ├── domain/
│   │   │   ├── entities/      # Business entities
│   │   │   ├── repositories/  # Repository contracts
│   │   │   └── usecases/      # Business logic
│   │   └── presentation/
│   │       ├── controllers/   # Riverpod providers
│   │       ├── pages/         # Route pages
│   │       └── widgets/       # Feature-specific widgets
│   │
│   └── products/
│       └── ... (same structure)
│
├── shared/                    # Shared across features
│   ├── models/                # Common data models
│   ├── providers/             # Global providers
│   ├── services/              # App-wide services
│   └── widgets/               # Reusable UI components
│       ├── buttons/
│       ├── cards/
│       └── forms/
│
└── main.dart                  # Entry point
```

### ✅ DO: Create Multiple Entry Points for Different Environments

```dart
// lib/main_development.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'app/app.dart';
import 'core/config/environment.dart';

void main() {
  Environment.init(EnvironmentType.development);
  runApp(
    ProviderScope(
      overrides: [
        // Development-specific overrides
      ],
      child: const App(),
    ),
  );
}

// lib/main_staging.dart
// lib/main_production.dart (similar structure)
```

Run with: `flutter run -t lib/main_development.dart`

---

## 2. State Management with Riverpod 2.x

**Note: Riverpod 3.0 is available in preview (3.0.0-dev.16). This guide uses stable 2.6.1, but consider migrating to 3.0 when stable for simplified APIs and better performance.**

Riverpod 2.x with code generation is the gold standard for Flutter state management, offering compile-time safety, powerful testing capabilities, and excellent performance.

### ✅ DO: Use Riverpod Generator for Type-Safe Providers

```dart
// features/products/presentation/controllers/products_controller.dart
import 'package:riverpod_annotation/riverpod_annotation.dart';
import 'package:freezed_annotation/freezed_annotation.dart';

part 'products_controller.g.dart';
part 'products_controller.freezed.dart';

@freezed
class ProductsState with _$ProductsState {
  const factory ProductsState({
    @Default([]) List<Product> products,
    @Default(false) bool isLoading,
    @Default(null) String? error,
    @Default('') String searchQuery,
    @Default(ProductSort.name) ProductSort sortBy,
  }) = _ProductsState;
}

@riverpod
class ProductsController extends _$ProductsController {
  @override
  Future<ProductsState> build() async {
    // Initial load
    return _fetchProducts();
  }

  Future<ProductsState> _fetchProducts() async {
    state = const AsyncValue.loading();
    
    try {
      final repository = ref.read(productRepositoryProvider);
      final products = await repository.getProducts();
      
      return ProductsState(products: products);
    } catch (e, stack) {
      // Riverpod will automatically wrap this in AsyncError
      throw e;
    }
  }

  Future<void> search(String query) async {
    state = AsyncValue.data(
      state.valueOrNull?.copyWith(searchQuery: query) ?? const ProductsState(),
    );
    
    // Debounce search
    await Future.delayed(const Duration(milliseconds: 300));
    
    // Check if query hasn't changed (user stopped typing)
    if (state.valueOrNull?.searchQuery == query) {
      await _performSearch(query);
    }
  }

  Future<void> _performSearch(String query) async {
    final currentState = state.valueOrNull;
    if (currentState == null) return;

    state = const AsyncValue.loading();
    
    try {
      final repository = ref.read(productRepositoryProvider);
      final products = await repository.searchProducts(query);
      
      state = AsyncValue.data(
        currentState.copyWith(products: products),
      );
    } catch (e, stack) {
      state = AsyncValue.error(e, stack);
    }
  }
}

// Auto-dispose provider that rebuilds when dependencies change
@riverpod
Future<List<Product>> filteredProducts(FilteredProductsRef ref) async {
  final productsAsync = ref.watch(productsControllerProvider);
  final filter = ref.watch(productFilterProvider);
  
  return productsAsync.when(
    data: (state) => state.products.where((p) => filter.matches(p)).toList(),
    loading: () => [],
    error: (_, __) => [],
  );
}
```

### ✅ DO: Use AsyncNotifier for Complex Async State

```dart
@riverpod
class UserProfile extends _$UserProfile {
  @override
  Future<User?> build() async {
    // Watch auth state changes
    final authState = ref.watch(authControllerProvider);
    
    return authState.when(
      authenticated: (user) => _loadProfile(user.id),
      unauthenticated: () => null,
    );
  }

  Future<User?> _loadProfile(String userId) async {
    try {
      final repository = ref.read(userRepositoryProvider);
      return await repository.getUser(userId);
    } catch (e) {
      // Log error but don't crash - return null for guest experience
      ref.read(loggerProvider).error('Failed to load profile', e);
      return null;
    }
  }

  Future<void> updateProfile(UserUpdate update) async {
    final currentUser = state.valueOrNull;
    if (currentUser == null) return;

    // Optimistic update
    state = AsyncValue.data(currentUser.copyWith(
      name: update.name ?? currentUser.name,
      email: update.email ?? currentUser.email,
    ));

    try {
      final repository = ref.read(userRepositoryProvider);
      final updatedUser = await repository.updateUser(currentUser.id, update);
      state = AsyncValue.data(updatedUser);
    } catch (e, stack) {
      // Revert on error
      state = AsyncValue.error(e, stack);
      // Re-fetch original
      ref.invalidateSelf();
    }
  }
}
```

### ❌ DON'T: Use ChangeNotifier or setState for Complex State

```dart
// Bad - Avoid ChangeNotifier for anything beyond trivial local state
class ProductsViewModel extends ChangeNotifier {
  List<Product> _products = [];
  bool _isLoading = false;
  
  Future<void> loadProducts() async {
    _isLoading = true;
    notifyListeners(); // Easy to forget
    
    try {
      _products = await api.getProducts();
    } catch (e) {
      // Error handling is manual and error-prone
    }
    
    _isLoading = false;
    notifyListeners(); // Duplicate calls everywhere
  }
}
```

---

## 3. Navigation with go_router

go_router provides declarative, type-safe routing with deep linking support and excellent web compatibility. As of 2025, go_router is in maintenance mode with the Flutter team focusing on stability rather than new features.

### ✅ DO: Define Type-Safe Routes with Extension Methods

```dart
// core/routing/app_router.dart
import 'package:go_router/go_router.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'app_router.g.dart';

enum AppRoute {
  splash,
  onboarding,
  login,
  home,
  productList,
  productDetail,
  cart,
  profile,
  settings,
}

extension AppRouteExtension on AppRoute {
  String get path {
    switch (this) {
      case AppRoute.splash:
        return '/splash';
      case AppRoute.onboarding:
        return '/onboarding';
      case AppRoute.login:
        return '/login';
      case AppRoute.home:
        return '/home';
      case AppRoute.productList:
        return '/products';
      case AppRoute.productDetail:
        return '/products/:id';
      case AppRoute.cart:
        return '/cart';
      case AppRoute.profile:
        return '/profile';
      case AppRoute.settings:
        return '/settings';
    }
  }

  String get name => toString();
}

@riverpod
GoRouter appRouter(AppRouterRef ref) {
  final authState = ref.watch(authControllerProvider);

  return GoRouter(
    initialLocation: AppRoute.splash.path,
    debugLogDiagnostics: kDebugMode,
    refreshListenable: GoRouterRefreshStream(authState.stream),
    
    redirect: (context, state) {
      final isAuthenticated = authState.valueOrNull?.isAuthenticated ?? false;
      final isAuthRoute = state.matchedLocation == AppRoute.login.path ||
                          state.matchedLocation == AppRoute.onboarding.path;

      if (!isAuthenticated && !isAuthRoute) {
        return AppRoute.login.path;
      }

      if (isAuthenticated && isAuthRoute) {
        return AppRoute.home.path;
      }

      return null;
    },

    routes: [
      GoRoute(
        path: AppRoute.splash.path,
        name: AppRoute.splash.name,
        builder: (context, state) => const SplashPage(),
      ),
      
      GoRoute(
        path: AppRoute.login.path,
        name: AppRoute.login.name,
        builder: (context, state) => const LoginPage(),
      ),

      ShellRoute(
        builder: (context, state, child) => AppShell(child: child),
        routes: [
          GoRoute(
            path: AppRoute.home.path,
            name: AppRoute.home.name,
            builder: (context, state) => const HomePage(),
          ),
          
          GoRoute(
            path: AppRoute.productList.path,
            name: AppRoute.productList.name,
            builder: (context, state) => const ProductListPage(),
            routes: [
              GoRoute(
                path: ':id',
                name: AppRoute.productDetail.name,
                builder: (context, state) {
                  final productId = state.pathParameters['id']!;
                  return ProductDetailPage(productId: productId);
                },
              ),
            ],
          ),
        ],
      ),
    ],

    errorBuilder: (context, state) => ErrorPage(error: state.error),
  );
}

// Helper class for auth state changes
class GoRouterRefreshStream extends ChangeNotifier {
  GoRouterRefreshStream(Stream<AsyncValue<AuthState>> stream) {
    _subscription = stream.listen((_) => notifyListeners());
  }

  late final StreamSubscription<AsyncValue<AuthState>> _subscription;

  @override
  void dispose() {
    _subscription.cancel();
    super.dispose();
  }
}
```

### ✅ DO: Use Type-Safe Navigation Extensions

```dart
// core/routing/navigation_extensions.dart
extension NavigationExtension on BuildContext {
  void pushNamed(AppRoute route, {Map<String, String>? params, Object? extra}) {
    goNamed(route.name, pathParameters: params ?? {}, extra: extra);
  }

  void pushReplacementNamed(AppRoute route, {Map<String, String>? params, Object? extra}) {
    goNamed(route.name, pathParameters: params ?? {}, extra: extra);
  }

  void navigateToProduct(String productId) {
    pushNamed(
      AppRoute.productDetail,
      params: {'id': productId},
    );
  }
}

// Usage in widgets
ElevatedButton(
  onPressed: () => context.navigateToProduct(product.id),
  child: const Text('View Product'),
)
```

---

## 4. Network Layer with Dio

Dio provides a robust HTTP client with interceptors, timeout handling, and excellent error management.

### ✅ DO: Create a Centralized API Client with Interceptors

```dart
// core/network/api_client.dart
import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

// Dio 5.9.0 includes improved interceptor APIs and better error handling
@riverpod
Dio apiClient(Ref ref) {
  final dio = Dio(BaseOptions(
    baseUrl: const String.fromEnvironment('API_BASE_URL'),
    connectTimeout: const Duration(seconds: 5),
    receiveTimeout: const Duration(seconds: 10),
    sendTimeout: const Duration(seconds: 10),
    contentType: 'application/json',
  ));

  // Add interceptors
  dio.interceptors.addAll([
    AuthInterceptor(ref),
    LoggingInterceptor(),
    RetryInterceptor(dio),
    ErrorInterceptor(ref),
  ]);

  return dio;
}

class AuthInterceptor extends Interceptor {
  final Ref ref;
  
  AuthInterceptor(this.ref);

  @override
  void onRequest(RequestOptions options, RequestInterceptorHandler handler) async {
    final token = await ref.read(tokenStorageProvider).getAccessToken();
    
    if (token != null) {
      options.headers['Authorization'] = 'Bearer $token';
    }
    
    handler.next(options);
  }

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) async {
    if (err.response?.statusCode == 401) {
      // Try to refresh token
      try {
        await ref.read(authControllerProvider.notifier).refreshToken();
        
        // Retry original request
        final response = await ref.read(apiClientProvider).fetch(err.requestOptions);
        handler.resolve(response);
        return;
      } catch (e) {
        // Refresh failed, logout user
        await ref.read(authControllerProvider.notifier).logout();
      }
    }
    
    handler.next(err);
  }
}

class RetryInterceptor extends Interceptor {
  final Dio dio;
  
  RetryInterceptor(this.dio);

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) async {
    final shouldRetry = err.type == DioExceptionType.connectionTimeout ||
                        err.type == DioExceptionType.receiveTimeout ||
                        (err.response?.statusCode ?? 0) >= 500;

    if (shouldRetry && (err.requestOptions.extra['retryCount'] ?? 0) < 3) {
      err.requestOptions.extra['retryCount'] = 
        (err.requestOptions.extra['retryCount'] ?? 0) + 1;
      
      // Exponential backoff
      await Future.delayed(
        Duration(seconds: err.requestOptions.extra['retryCount'] * 2),
      );
      
      try {
        final response = await dio.fetch(err.requestOptions);
        handler.resolve(response);
        return;
      } catch (e) {
        // Continue with error
      }
    }
    
    handler.next(err);
  }
}
```

### ✅ DO: Create Type-Safe API Services

```dart
// features/products/data/datasources/products_api_service.dart
import 'package:dio/dio.dart';
import 'package:retrofit/retrofit.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'products_api_service.g.dart';

@RestApi()
abstract class ProductsApiService {
  factory ProductsApiService(Dio dio) = _ProductsApiService;

  @GET('/products')
  Future<List<ProductDto>> getProducts({
    @Query('page') int page = 1,
    @Query('limit') int limit = 20,
    @Query('sort') String? sort,
  });

  @GET('/products/{id}')
  Future<ProductDto> getProduct(@Path('id') String id);

  @POST('/products')
  Future<ProductDto> createProduct(@Body() CreateProductRequest request);

  @PUT('/products/{id}')
  Future<ProductDto> updateProduct(
    @Path('id') String id,
    @Body() UpdateProductRequest request,
  );

  @DELETE('/products/{id}')
  Future<void> deleteProduct(@Path('id') String id);

  @GET('/products/search')
  Future<List<ProductDto>> searchProducts(
    @Query('q') String query,
    @Query('filters') Map<String, dynamic>? filters,
  );
}

@riverpod
ProductsApiService productsApiService(ProductsApiServiceRef ref) {
  return ProductsApiService(ref.watch(apiClientProvider));
}
```

---

## 5. Local Storage with Drift

Drift (formerly Moor) provides type-safe, reactive SQLite access with excellent performance for complex local data needs.

### ✅ DO: Define Type-Safe Database Schema

```dart
// core/database/app_database.dart
import 'package:drift/drift.dart';
import 'package:drift_flutter/drift_flutter.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'app_database.g.dart';

@DataClassName('ProductEntity')
class Products extends Table {
  IntColumn get id => integer().autoIncrement()();
  TextColumn get externalId => text().unique()();
  TextColumn get name => text().withLength(min: 1, max: 255)();
  TextColumn get description => text()();
  RealColumn get price => real()();
  TextColumn get imageUrl => text().nullable()();
  DateTimeColumn get createdAt => dateTime().withDefault(currentDateAndTime)();
  DateTimeColumn get updatedAt => dateTime().withDefault(currentDateAndTime)();
  BoolColumn get isFavorite => boolean().withDefault(const Constant(false))();
  TextColumn get metadata => text().map(const JsonTypeConverter()).nullable()();

  @override
  Set<Column> get primaryKey => {id};
}

@DataClassName('CartItemEntity')
class CartItems extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get productId => integer().references(Products, #id)();
  IntColumn get quantity => integer().withDefault(const Constant(1))();
  DateTimeColumn get addedAt => dateTime().withDefault(currentDateAndTime)();
}

// Custom type converter for JSON
class JsonTypeConverter extends TypeConverter<Map<String, dynamic>?, String?> {
  const JsonTypeConverter();

  @override
  Map<String, dynamic>? fromSql(String? fromDb) {
    if (fromDb == null) return null;
    return json.decode(fromDb) as Map<String, dynamic>;
  }

  @override
  String? toSql(Map<String, dynamic>? value) {
    if (value == null) return null;
    return json.encode(value);
  }
}

@DriftDatabase(tables: [Products, CartItems])
class AppDatabase extends _$AppDatabase {
  AppDatabase() : super(_openConnection());

  @override
  int get schemaVersion => 1;

  @override
  MigrationStrategy get migration {
    return MigrationStrategy(
      onCreate: (Migrator m) async {
        await m.createAll();
        
        // Create indexes for better performance
        await customStatement(
          'CREATE INDEX idx_products_name ON products(name)',
        );
        await customStatement(
          'CREATE INDEX idx_products_favorite ON products(is_favorite)',
        );
      },
      onUpgrade: (Migrator m, int from, int to) async {
        // Handle migrations
      },
    );
  }

  static QueryExecutor _openConnection() {
    return driftDatabase(
      name: 'app_database',
      web: DriftWebOptions(
        sqlite3Wasm: Uri.parse('sqlite3.wasm'),
        driftWorker: Uri.parse('drift_worker.js'),
      ),
    );
  }

  // Queries
  Future<List<ProductEntity>> getAllProducts() => select(products).get();
  
  Future<List<ProductEntity>> getFavoriteProducts() {
    return (select(products)..where((p) => p.isFavorite.equals(true))).get();
  }

  Stream<List<ProductEntity>> watchProducts() => select(products).watch();

  Future<void> insertProduct(ProductsCompanion product) {
    return into(products).insert(product);
  }

  Future<void> toggleFavorite(String externalId) {
    return transaction(() async {
      final product = await (select(products)
        ..where((p) => p.externalId.equals(externalId)))
        .getSingleOrNull();
      
      if (product != null) {
        await (update(products)
          ..where((p) => p.id.equals(product.id)))
          .write(ProductsCompanion(
            isFavorite: Value(!product.isFavorite),
            updatedAt: Value(DateTime.now()),
          ));
      }
    });
  }

  // Complex query with joins
  Future<List<CartItemWithProduct>> getCartItemsWithProducts() {
    final query = select(cartItems).join([
      innerJoin(products, products.id.equalsExp(cartItems.productId)),
    ]);

    return query.map((row) {
      return CartItemWithProduct(
        cartItem: row.readTable(cartItems),
        product: row.readTable(products),
      );
    }).get();
  }
}

@riverpod
AppDatabase appDatabase(AppDatabaseRef ref) {
  final db = AppDatabase();
  ref.onDispose(() => db.close());
  return db;
}
```

---

## 6. Performance Optimization with Impeller

Impeller is Flutter's new rendering engine, providing predictable performance and eliminating shader compilation jank.

### ✅ DO: Optimize for Impeller's Strengths

```dart
// Impeller excels at these patterns:

// 1. Use const constructors everywhere possible
class ProductCard extends StatelessWidget {
  const ProductCard({super.key, required this.product}); // const constructor
  
  final Product product;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Column(
        children: const [
          // Const widgets are cached by Impeller
          _ProductCardHeader(),
          _ProductCardBody(),
        ],
      ),
    );
  }
}

// 2. Leverage RepaintBoundary for complex widgets
class ComplexAnimation extends StatelessWidget {
  const ComplexAnimation({super.key});

  @override
  Widget build(BuildContext context) {
    return RepaintBoundary(
      child: CustomPaint(
        painter: ComplexPainter(),
        size: const Size(300, 300),
      ),
    );
  }
}

// 3. Use LayerLink for optimal overlay performance
class TooltipOverlay extends StatefulWidget {
  const TooltipOverlay({super.key});

  @override
  State<TooltipOverlay> createState() => _TooltipOverlayState();
}

class _TooltipOverlayState extends State<TooltipOverlay> {
  final LayerLink _layerLink = LayerLink();
  OverlayEntry? _overlayEntry;

  void _showOverlay() {
    _overlayEntry = OverlayEntry(
      builder: (context) => Positioned(
        width: 200,
        child: CompositedTransformFollower(
          link: _layerLink,
          targetAnchor: Alignment.bottomCenter,
          followerAnchor: Alignment.topCenter,
          child: Material(
            elevation: 8,
            borderRadius: BorderRadius.circular(8),
            child: const Padding(
              padding: EdgeInsets.all(8),
              child: Text('Tooltip content'),
            ),
          ),
        ),
      ),
    );
    
    Overlay.of(context).insert(_overlayEntry!);
  }

  @override
  Widget build(BuildContext context) {
    return CompositedTransformTarget(
      link: _layerLink,
      child: GestureDetector(
        onTap: _showOverlay,
        child: const Icon(Icons.info),
      ),
    );
  }
}
```

### ✅ DO: Profile and Optimize Render Performance

```dart
// Enable performance overlay in debug builds
class App extends ConsumerWidget {
  const App({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return MaterialApp(
      showPerformanceOverlay: kDebugMode && const bool.fromEnvironment('SHOW_PERFORMANCE'),
      checkerboardRasterCacheImages: kDebugMode,
      checkerboardOffscreenLayers: kDebugMode,
      // ...
    );
  }
}

// Use Flutter DevTools Timeline for detailed performance analysis
// Run with: flutter run --profile --trace-skia
```

### ❌ DON'T: Use Expensive Operations in Build Methods

```dart
// Bad - Expensive computation in build
class BadWidget extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    // This runs on every rebuild!
    final processedData = expensiveDataProcessing(rawData);
    
    return Text(processedData.toString());
  }
}

// Good - Cache expensive computations
class GoodWidget extends ConsumerWidget {
  const GoodWidget({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Computed only when dependencies change
    final processedData = ref.watch(processedDataProvider);
    
    return Text(processedData.toString());
  }
}

@riverpod
String processedData(ProcessedDataRef ref) {
  final rawData = ref.watch(rawDataProvider);
  return expensiveDataProcessing(rawData);
}
```

---

## 7. Platform-Specific Code & Method Channels

Flutter's platform channels enable seamless integration with native iOS and Android features.

### ✅ DO: Create Type-Safe Platform Channels

```dart
// core/platform/native_platform.dart
import 'package:flutter/services.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'native_platform.g.dart';

abstract class NativePlatform {
  Future<BiometricType> checkBiometricSupport();
  Future<bool> authenticateWithBiometrics(String reason);
  Future<String?> getDeviceId();
  Future<void> vibrateHaptic(HapticType type);
  Stream<double> watchBatteryLevel();
}

enum BiometricType { none, fingerprint, face, iris }
enum HapticType { selection, impact, notification }

class NativePlatformImpl implements NativePlatform {
  static const _channel = MethodChannel('com.smartapp/native');
  static const _eventChannel = EventChannel('com.smartapp/battery');

  @override
  Future<BiometricType> checkBiometricSupport() async {
    try {
      final String type = await _channel.invokeMethod('checkBiometric');
      return BiometricType.values.firstWhere(
        (e) => e.name == type,
        orElse: () => BiometricType.none,
      );
    } on PlatformException {
      return BiometricType.none;
    }
  }

  @override
  Future<bool> authenticateWithBiometrics(String reason) async {
    try {
      final bool result = await _channel.invokeMethod(
        'authenticate',
        {'reason': reason},
      );
      return result;
    } on PlatformException catch (e) {
      if (e.code == 'UserCanceled') {
        return false;
      }
      rethrow;
    }
  }

  @override
  Future<String?> getDeviceId() async {
    try {
      return await _channel.invokeMethod('getDeviceId');
    } on PlatformException {
      return null;
    }
  }

  @override
  Future<void> vibrateHaptic(HapticType type) async {
    try {
      await _channel.invokeMethod('vibrateHaptic', {'type': type.name});
    } on PlatformException {
      // Ignore - not all devices support haptics
    }
  }

  @override
  Stream<double> watchBatteryLevel() {
    return _eventChannel
        .receiveBroadcastStream()
        .map((dynamic level) => level as double);
  }
}

@riverpod
NativePlatform nativePlatform(NativePlatformRef ref) {
  return NativePlatformImpl();
}
```

### Native Implementation (iOS - Swift)

```swift
// ios/Runner/NativePlatformHandler.swift
import Flutter
import LocalAuthentication

class NativePlatformHandler: NSObject {
    private let channel: FlutterMethodChannel
    private let batteryChannel: FlutterEventChannel
    
    init(messenger: FlutterBinaryMessenger) {
        self.channel = FlutterMethodChannel(
            name: "com.smartapp/native",
            binaryMessenger: messenger
        )
        self.batteryChannel = FlutterEventChannel(
            name: "com.smartapp/battery",
            binaryMessenger: messenger
        )
        super.init()
        
        channel.setMethodCallHandler(handle)
        batteryChannel.setStreamHandler(BatteryStreamHandler())
    }
    
    private func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "checkBiometric":
            checkBiometric(result: result)
        case "authenticate":
            if let args = call.arguments as? [String: Any],
               let reason = args["reason"] as? String {
                authenticate(reason: reason, result: result)
            } else {
                result(FlutterError(code: "INVALID_ARGUMENT", message: nil, details: nil))
            }
        case "getDeviceId":
            result(UIDevice.current.identifierForVendor?.uuidString)
        case "vibrateHaptic":
            if let args = call.arguments as? [String: Any],
               let type = args["type"] as? String {
                vibrateHaptic(type: type)
                result(nil)
            }
        default:
            result(FlutterMethodNotImplemented)
        }
    }
    
    private func checkBiometric(result: @escaping FlutterResult) {
        let context = LAContext()
        var error: NSError?
        
        if context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error) {
            switch context.biometryType {
            case .faceID:
                result("face")
            case .touchID:
                result("fingerprint")
            default:
                result("none")
            }
        } else {
            result("none")
        }
    }
    
    private func authenticate(reason: String, result: @escaping FlutterResult) {
        let context = LAContext()
        
        context.evaluatePolicy(
            .deviceOwnerAuthenticationWithBiometrics,
            localizedReason: reason
        ) { success, error in
            DispatchQueue.main.async {
                if success {
                    result(true)
                } else if let error = error as? LAError {
                    switch error.code {
                    case .userCancel:
                        result(FlutterError(code: "UserCanceled", message: nil, details: nil))
                    default:
                        result(false)
                    }
                } else {
                    result(false)
                }
            }
        }
    }
    
    private func vibrateHaptic(type: String) {
        let generator: UIFeedbackGenerator
        
        switch type {
        case "selection":
            generator = UISelectionFeedbackGenerator()
        case "impact":
            generator = UIImpactFeedbackGenerator(style: .medium)
        case "notification":
            generator = UINotificationFeedbackGenerator()
        default:
            return
        }
        
        generator.prepare()
        
        if let impact = generator as? UIImpactFeedbackGenerator {
            impact.impactOccurred()
        } else if let selection = generator as? UISelectionFeedbackGenerator {
            selection.selectionChanged()
        } else if let notification = generator as? UINotificationFeedbackGenerator {
            notification.notificationOccurred(.success)
        }
    }
}

class BatteryStreamHandler: NSObject, FlutterStreamHandler {
    private var eventSink: FlutterEventSink?
    private var timer: Timer?
    
    func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        self.eventSink = events
        
        // Start monitoring battery
        UIDevice.current.isBatteryMonitoringEnabled = true
        
        timer = Timer.scheduledTimer(withTimeInterval: 60.0, repeats: true) { _ in
            events(UIDevice.current.batteryLevel)
        }
        
        // Send initial value
        events(UIDevice.current.batteryLevel)
        
        return nil
    }
    
    func onCancel(withArguments arguments: Any?) -> FlutterError? {
        timer?.invalidate()
        timer = nil
        eventSink = nil
        UIDevice.current.isBatteryMonitoringEnabled = false
        return nil
    }
}
```

---

## 8. Testing Strategy

A comprehensive testing strategy ensures reliability and prevents regressions as your app grows.

### ✅ DO: Write Widget Tests with Robot Pattern

```dart
// test/features/auth/presentation/pages/login_page_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_test_robots/flutter_test_robots.dart';
import 'package:mocktail/mocktail.dart';

import '../../../helpers/test_helpers.dart';

class LoginRobot extends Robot {
  LoginRobot(super.tester);

  Future<void> enterEmail(String email) async {
    await enterText(find.byKey(const Key('emailField')), email);
  }

  Future<void> enterPassword(String password) async {
    await enterText(find.byKey(const Key('passwordField')), password);
  }

  Future<void> tapLoginButton() async {
    await tap(find.byKey(const Key('loginButton')));
    await pumpAndSettle();
  }

  Future<void> verifyErrorMessage(String message) async {
    expect(find.text(message), findsOneWidget);
  }

  Future<void> verifyNavigatedToHome() async {
    expect(find.byType(HomePage), findsOneWidget);
  }
}

void main() {
  group('LoginPage', () {
    late MockAuthRepository authRepository;
    late MockRouter router;

    setUp(() {
      authRepository = MockAuthRepository();
      router = MockRouter();
    });

    testWidgets('successful login navigates to home', (tester) async {
      // Arrange
      when(() => authRepository.login(any(), any()))
          .thenAnswer((_) async => const AuthSuccess());

      // Act
      final robot = LoginRobot(tester);
      await tester.pumpWidget(
        TestApp(
          overrides: [
            authRepositoryProvider.overrideWithValue(authRepository),
          ],
          child: const LoginPage(),
        ),
      );

      await robot.enterEmail('test@example.com');
      await robot.enterPassword('password123');
      await robot.tapLoginButton();

      // Assert
      await robot.verifyNavigatedToHome();
      verify(() => authRepository.login('test@example.com', 'password123')).called(1);
    });

    testWidgets('invalid credentials shows error', (tester) async {
      // Arrange
      when(() => authRepository.login(any(), any()))
          .thenAnswer((_) async => const AuthFailure('Invalid credentials'));

      // Act
      final robot = LoginRobot(tester);
      await tester.pumpWidget(
        TestApp(
          overrides: [
            authRepositoryProvider.overrideWithValue(authRepository),
          ],
          child: const LoginPage(),
        ),
      );

      await robot.enterEmail('wrong@example.com');
      await robot.enterPassword('wrongpass');
      await robot.tapLoginButton();

      // Assert
      await robot.verifyErrorMessage('Invalid credentials');
    });
  });
}
```

### ✅ DO: Write Integration Tests for Critical User Flows

```dart
// integration_test/app_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:smartapp/main.dart' as app;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('end-to-end test', () {
    testWidgets('complete purchase flow', (tester) async {
      app.main();
      await tester.pumpAndSettle();

      // Login
      await tester.tap(find.text('Login'));
      await tester.pumpAndSettle();
      
      await tester.enterText(find.byKey(const Key('emailField')), 'test@example.com');
      await tester.enterText(find.byKey(const Key('passwordField')), 'password123');
      await tester.tap(find.text('Sign In'));
      await tester.pumpAndSettle();

      // Browse products
      expect(find.text('Products'), findsOneWidget);
      await tester.tap(find.text('Electronics'));
      await tester.pumpAndSettle();

      // Add to cart
      await tester.tap(find.text('iPhone 15').first);
      await tester.pumpAndSettle();
      
      await tester.tap(find.text('Add to Cart'));
      await tester.pumpAndSettle();

      // Checkout
      await tester.tap(find.byIcon(Icons.shopping_cart));
      await tester.pumpAndSettle();
      
      await tester.tap(find.text('Checkout'));
      await tester.pumpAndSettle();

      // Verify order confirmation
      expect(find.text('Order Confirmed'), findsOneWidget);
      expect(find.textContaining('Order #'), findsOneWidget);
    });
  });
}
```

---

## 9. Security Best Practices

Security should be built into your app from the ground up, not added as an afterthought.

### ✅ DO: Implement Certificate Pinning

```dart
// core/network/certificate_pinning.dart
import 'package:dio/dio.dart';
import 'package:dio_certificate_pinning/dio_certificate_pinning.dart';

class CertificatePinningInterceptor extends Interceptor {
  final List<String> allowedSHAFingerprints;
  
  CertificatePinningInterceptor({required this.allowedSHAFingerprints});

  @override
  void onRequest(RequestOptions options, RequestInterceptorHandler handler) async {
    // Only pin for production API
    if (!options.uri.host.contains('api.smartapp.com')) {
      handler.next(options);
      return;
    }

    try {
      final dio = Dio();
      dio.interceptors.add(
        CertificatePinningInterceptor(
          allowedSHAFingerprints: allowedSHAFingerprints,
          timeout: 30,
        ),
      );
      
      // Verify certificate
      await dio.head(options.uri.toString());
      handler.next(options);
    } catch (e) {
      handler.reject(
        DioException(
          requestOptions: options,
          error: 'Certificate pinning failed',
          type: DioExceptionType.cancel,
        ),
      );
    }
  }
}

// Usage in production
final dio = Dio();
if (kReleaseMode) {
  dio.interceptors.add(
    CertificatePinningInterceptor(
      allowedSHAFingerprints: [
        'SHA256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=',
        'SHA256:BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=', // Backup
      ],
    ),
  );
}
```

### ✅ DO: Secure Local Storage

```dart
// core/storage/secure_storage_service.dart
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'secure_storage_service.g.dart';

@riverpod
SecureStorageService secureStorage(SecureStorageRef ref) {
  return SecureStorageService();
}

class SecureStorageService {
  static const _storage = FlutterSecureStorage(
    aOptions: AndroidOptions(
      encryptedSharedPreferences: true,
      keyCipherAlgorithm: KeyCipherAlgorithm.RSA_ECB_PKCS1Padding,
      storageCipherAlgorithm: StorageCipherAlgorithm.AES_GCM_NoPadding,
    ),
    iOptions: IOSOptions(
      accessibility: KeychainAccessibility.unlocked_this_device_only,
      accountName: 'SmartAppSecureStorage',
    ),
  );

  // Tokens
  Future<void> saveTokens({
    required String accessToken,
    required String refreshToken,
  }) async {
    await Future.wait([
      _storage.write(key: 'access_token', value: accessToken),
      _storage.write(key: 'refresh_token', value: refreshToken),
      _storage.write(key: 'token_saved_at', value: DateTime.now().toIso8601String()),
    ]);
  }

  Future<TokenPair?> getTokens() async {
    final accessToken = await _storage.read(key: 'access_token');
    final refreshToken = await _storage.read(key: 'refresh_token');
    
    if (accessToken == null || refreshToken == null) {
      return null;
    }
    
    return TokenPair(
      accessToken: accessToken,
      refreshToken: refreshToken,
    );
  }

  Future<void> clearTokens() async {
    await _storage.delete(key: 'access_token');
    await _storage.delete(key: 'refresh_token');
    await _storage.delete(key: 'token_saved_at');
  }

  // Biometric protected values
  Future<void> saveBiometricProtectedValue(String key, String value) async {
    await _storage.write(
      key: 'biometric_$key',
      value: value,
      aOptions: const AndroidOptions(
        biometricOnly: true,
      ),
      iOptions: const IOSOptions(
        biometryOnly: true,
      ),
    );
  }

  // Encrypted preferences
  Future<void> saveEncryptedPreference(String key, String value) async {
    final encryptedValue = await _encryptValue(value);
    await _storage.write(key: 'pref_$key', value: encryptedValue);
  }

  Future<String?> getEncryptedPreference(String key) async {
    final encryptedValue = await _storage.read(key: 'pref_$key');
    if (encryptedValue == null) return null;
    
    return await _decryptValue(encryptedValue);
  }

  // Encryption helpers (simplified - use proper crypto in production)
  Future<String> _encryptValue(String value) async {
    // Implement proper encryption
    return base64Encode(utf8.encode(value));
  }

  Future<String> _decryptValue(String encryptedValue) async {
    // Implement proper decryption
    return utf8.decode(base64Decode(encryptedValue));
  }
}
```

---

## 10. Advanced Patterns

### Dynamic Feature Modules

For large apps, implement dynamic feature loading to reduce initial bundle size:

```dart
// core/features/feature_manager.dart
import 'package:flutter/services.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'feature_manager.g.dart';

@riverpod
class FeatureManager extends _$FeatureManager {
  static const _channel = MethodChannel('com.smartapp/features');

  @override
  Future<Set<String>> build() async {
    return _loadAvailableFeatures();
  }

  Future<Set<String>> _loadAvailableFeatures() async {
    try {
      final List<dynamic> features = await _channel.invokeMethod('getAvailableFeatures');
      return features.cast<String>().toSet();
    } catch (e) {
      return {};
    }
  }

  Future<void> downloadFeature(String featureName) async {
    state = const AsyncValue.loading();
    
    try {
      await _channel.invokeMethod('downloadFeature', {'name': featureName});
      
      // Refresh available features
      final features = await _loadAvailableFeatures();
      state = AsyncValue.data(features);
    } catch (e, stack) {
      state = AsyncValue.error(e, stack);
    }
  }

  bool isFeatureAvailable(String featureName) {
    return state.valueOrNull?.contains(featureName) ?? false;
  }
}

// Usage in routing
@riverpod
GoRouter appRouter(AppRouterRef ref) {
  final featureManager = ref.watch(featureManagerProvider);

  return GoRouter(
    routes: [
      // Core routes always available
      GoRoute(
        path: '/home',
        builder: (context, state) => const HomePage(),
      ),
      
      // Feature-gated routes
      if (featureManager.valueOrNull?.contains('premium') ?? false)
        GoRoute(
          path: '/premium',
          builder: (context, state) => const PremiumFeaturePage(),
        ),
    ],
  );
}
```

### Compile-Time Code Generation for API Models

Leverage `freezed` and `json_serializable` for bulletproof data models:

```dart
// features/products/domain/entities/product.dart
import 'package:freezed_annotation/freezed_annotation.dart';

part 'product.freezed.dart';
part 'product.g.dart';

@freezed
class Product with _$Product {
  const factory Product({
    required String id,
    required String name,
    required String description,
    required double price,
    required ProductCategory category,
    required List<String> imageUrls,
    @Default(0) int stockQuantity,
    @Default(false) bool isFeatured,
    @Default(null) double? discountPercentage,
    required DateTime createdAt,
    required DateTime updatedAt,
    @JsonKey(includeFromJson: false, includeToJson: false)
    @Default(false) bool isInCart,
  }) = _Product;

  factory Product.fromJson(Map<String, dynamic> json) => _$ProductFromJson(json);
}

@freezed
class ProductCategory with _$ProductCategory {
  const factory ProductCategory({
    required String id,
    required String name,
    required String slug,
    String? parentId,
    @Default([]) List<ProductCategory> subcategories,
  }) = _ProductCategory;

  factory ProductCategory.fromJson(Map<String, dynamic> json) => 
      _$ProductCategoryFromJson(json);
}

// Extension methods for business logic
extension ProductX on Product {
  double get finalPrice {
    if (discountPercentage != null && discountPercentage! > 0) {
      return price * (1 - discountPercentage! / 100);
    }
    return price;
  }

  bool get isOnSale => discountPercentage != null && discountPercentage! > 0;
  
  bool get isInStock => stockQuantity > 0;
  
  String get availability {
    if (!isInStock) return 'Out of Stock';
    if (stockQuantity < 10) return 'Low Stock';
    return 'In Stock';
  }
}
```

### Error Handling with Result Types

Implement functional error handling without exceptions:

```dart
// core/errors/result.dart
import 'package:freezed_annotation/freezed_annotation.dart';

part 'result.freezed.dart';

@freezed
class Result<T> with _$Result<T> {
  const factory Result.success(T value) = Success<T>;
  const factory Result.failure(Failure failure) = Error<T>;
}

@freezed
class Failure with _$Failure {
  const factory Failure.network({
    required String message,
    int? statusCode,
  }) = NetworkFailure;
  
  const factory Failure.cache({
    required String message,
  }) = CacheFailure;
  
  const factory Failure.validation({
    required String message,
    Map<String, List<String>>? errors,
  }) = ValidationFailure;
  
  const factory Failure.unknown({
    required String message,
    Object? error,
    StackTrace? stackTrace,
  }) = UnknownFailure;
}

// Usage in repositories
class ProductRepositoryImpl implements ProductRepository {
  @override
  Future<Result<List<Product>>> getProducts() async {
    try {
      final response = await apiService.getProducts();
      
      // Try cache first if network fails
      if (response.statusCode != 200) {
        final cached = await database.getAllProducts();
        if (cached.isNotEmpty) {
          return Result.success(cached.toDomainList());
        }
        
        return Result.failure(
          Failure.network(
            message: 'Failed to load products',
            statusCode: response.statusCode,
          ),
        );
      }
      
      final products = response.data.toDomainList();
      
      // Update cache
      await database.cacheProducts(products);
      
      return Result.success(products);
    } catch (e, stack) {
      return Result.failure(
        Failure.unknown(
          message: 'Unexpected error loading products',
          error: e,
          stackTrace: stack,
        ),
      );
    }
  }
}

// Usage in UI
class ProductListPage extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final productsAsync = ref.watch(productsControllerProvider);
    
    return productsAsync.when(
      data: (state) => state.result.when(
        success: (products) => ProductGrid(products: products),
        failure: (failure) => ErrorWidget(
          failure: failure,
          onRetry: () => ref.invalidate(productsControllerProvider),
        ),
      ),
      loading: () => const LoadingShimmer(),
      error: (e, s) => ErrorWidget.fromError(e, s),
    );
  }
}
```

### Performance Monitoring

Implement comprehensive performance tracking:

```dart
// core/monitoring/performance_monitor.dart
import 'package:flutter/scheduler.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'performance_monitor.g.dart';

@riverpod
class PerformanceMonitor extends _$PerformanceMonitor {
  Timer? _timer;
  int _frameCount = 0;
  
  @override
  PerformanceMetrics build() {
    _startMonitoring();
    ref.onDispose(_stopMonitoring);
    
    return const PerformanceMetrics();
  }

  void _startMonitoring() {
    // Monitor frame timing
    SchedulerBinding.instance.addTimingsCallback(_onFrameTimings);
    
    // Calculate FPS every second
    _timer = Timer.periodic(const Duration(seconds: 1), (_) {
      final fps = _frameCount;
      _frameCount = 0;
      
      state = state.copyWith(
        fps: fps,
        isJanky: fps < 50, // Below 50 FPS is considered janky
      );
    });
  }

  void _stopMonitoring() {
    SchedulerBinding.instance.removeTimingsCallback(_onFrameTimings);
    _timer?.cancel();
  }

  void _onFrameTimings(List<FrameTiming> timings) {
    for (final timing in timings) {
      _frameCount++;
      
      // Track slow frames
      final frameDuration = timing.totalSpan.inMilliseconds;
      if (frameDuration > 16) { // 16ms = 60fps
        state = state.copyWith(
          slowFrameCount: state.slowFrameCount + 1,
        );
      }
      
      // Update build times
      state = state.copyWith(
        lastFrameBuildTime: timing.buildDuration.inMicroseconds / 1000,
        lastFrameRasterTime: timing.rasterDuration.inMicroseconds / 1000,
      );
    }
  }

  void trackCustomMetric(String name, double value) {
    state = state.copyWith(
      customMetrics: {...state.customMetrics, name: value},
    );
  }
}

@freezed
class PerformanceMetrics with _$PerformanceMetrics {
  const factory PerformanceMetrics({
    @Default(60) int fps,
    @Default(false) bool isJanky,
    @Default(0) int slowFrameCount,
    @Default(0.0) double lastFrameBuildTime,
    @Default(0.0) double lastFrameRasterTime,
    @Default({}) Map<String, double> customMetrics,
  }) = _PerformanceMetrics;
}

// Performance overlay widget
class PerformanceOverlay extends ConsumerWidget {
  const PerformanceOverlay({super.key, required this.child});
  
  final Widget child;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    if (!kDebugMode) return child;
    
    final metrics = ref.watch(performanceMonitorProvider);
    
    return Stack(
      children: [
        child,
        Positioned(
          top: 50,
          right: 10,
          child: Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: metrics.isJanky ? Colors.red : Colors.green,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              'FPS: ${metrics.fps}',
              style: const TextStyle(color: Colors.white),
            ),
          ),
        ),
      ],
    );
  }
}
```

### CI/CD Configuration

Configure GitHub Actions for automated testing and deployment:

```yaml
# .github/workflows/flutter.yml
name: Flutter CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Flutter
      uses: subosito/flutter-action@v2
      with:
        flutter-version: '3.27.0'
        channel: 'stable'
        cache: true
    
    - name: Install dependencies
      run: flutter pub get
    
    - name: Check formatting
      run: dart format --set-exit-if-changed .
    
    - name: Run linter
      run: flutter analyze --no-fatal-warnings
    
    - name: Run tests
      run: flutter test --coverage
    
    - name: Upload coverage
      uses: codecov/codecov-action@v4
      with:
        file: coverage/lcov.info
    
    - name: Build APK
      run: flutter build apk --release --split-per-abi
    
    - name: Build iOS
      if: runner.os == 'macOS'
      run: |
        flutter build ios --release --no-codesign
        cd ios && xcodebuild -workspace Runner.xcworkspace \
          -scheme Runner -configuration Release \
          -archivePath Runner.xcarchive archive
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v4
      with:
        name: release-artifacts
        path: |
          build/app/outputs/flutter-apk/*.apk
          ios/Runner.xcarchive

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to Firebase App Distribution
      uses: wzieba/Firebase-Distribution-Github-Action@v1
      with:
        appId: ${{ secrets.FIREBASE_APP_ID }}
        serviceCredentialsFileContent: ${{ secrets.FIREBASE_CREDENTIALS }}
        groups: testers
        file: build/app/outputs/flutter-apk/app-arm64-v8a-release.apk
```

### Conclusion

This guide provides a production-ready architecture for Flutter applications using the latest features of Dart 3 and the Impeller renderer. Key takeaways:

1. **Architecture First**: A scalable feature-first architecture with clear separation of concerns
2. **Type Safety**: Leverage Dart 3's sound null safety and code generation for bulletproof apps
3. **Performance**: Optimize for Impeller with const widgets, RepaintBoundary, and proper profiling
4. **State Management**: Use Riverpod 2.x with code generation for predictable, testable state
5. **Testing**: Comprehensive testing with robot pattern and integration tests
6. **Security**: Implement defense in depth with certificate pinning and secure storage
7. **Platform Integration**: Type-safe platform channels for native features
8. **Error Handling**: Functional error handling with Result types
9. **Monitoring**: Built-in performance tracking and observability

Remember: The best architecture is one that fits your team's needs and can evolve with your application. Start with these patterns and adapt them as you learn more about your specific requirements.