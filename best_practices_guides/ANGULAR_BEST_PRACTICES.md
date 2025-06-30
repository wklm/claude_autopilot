# The Definitive Guide to Angular 20, Signals, and Modern Web Development (2025)

This guide synthesizes modern best practices for building scalable, performant, and maintainable applications with Angular 20, the stable Signals-based reactivity system, and contemporary tooling. It provides a production-grade architectural blueprint for teams building enterprise Angular applications in 2025.

### Prerequisites & Configuration

Ensure your project uses **Angular 20.0+**, **TypeScript 5.8+**, and **Node.js 20.19+ LTS, 22.12+ LTS, or 24+**. This guide assumes you're using the new **Vite-powered dev server** (default since Angular 18) and **esbuild** for production builds.

Initialize your project with the Angular CLI's modern defaults:

```bash
# Using pnpm (recommended for monorepos) or npm/yarn/bun
pnpm dlx @angular/cli@latest new my-app --style=scss --ssr --routing

# Key flags:
# --ssr: Enables server-side rendering with the new @angular/ssr package
# --routing: Sets up the router with new typed routes
# --style=scss: Uses SCSS for advanced styling (or 'css' for CSS-in-JS approaches)
```

Configure your `angular.json` for modern development:

```json
{
  "$schema": "./node_modules/@angular/cli/lib/config/schema.json",
  "version": 1,
  "newProjectRoot": "projects",
  "projects": {
    "my-app": {
      "architect": {
        "build": {
          "builder": "@angular/build:application",
          "options": {
            "outputPath": "dist/my-app",
            "index": "src/index.html",
            "browser": "src/main.ts",
            "polyfills": ["zone.js"],
            "tsConfig": "tsconfig.app.json",
            "inlineStyleLanguage": "scss",
            "optimization": {
              "scripts": true,
              "styles": {
                "minify": true,
                "inlineCritical": true
              },
              "fonts": true
            },
            "budgets": [
              {
                "type": "initial",
                "maximumWarning": "500kB",
                "maximumError": "1MB"
              }
            ]
          },
          "configurations": {
            "production": {
              "optimization": true,
              "sourceMap": false,
              "namedChunks": false,
              "extractLicenses": true,
              "serviceWorker": "ngsw-config.json",
              "subresourceIntegrity": true
            },
            "development": {
              "optimization": false,
              "extractLicenses": false,
              "sourceMap": true,
              "namedChunks": true,
              "watch": true
            }
          }
        },
        "serve": {
          "builder": "@angular/build:dev-server",
          "options": {
            "buildTarget": "my-app:build",
            "hmr": true,
            "port": 4200
          }
        }
      }
    }
  }
}
```

TypeScript configuration optimized for Angular 20:

```typescript
// tsconfig.json
{
  "compilerOptions": {
    "baseUrl": "./",
    "outDir": "./dist/out-tsc",
    "forceConsistentCasingInFileNames": true,
    "strict": true,
    "noImplicitOverride": true,
    "noPropertyAccessFromIndexSignature": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "skipLibCheck": true,
    "isolatedModules": true,
    "esModuleInterop": true,
    "sourceMap": true,
    "declaration": false,
    "experimentalDecorators": true,
    "moduleResolution": "bundler",
    "importHelpers": true,
    "target": "ES2022",
    "module": "ES2022",
    "lib": ["ES2023", "dom", "dom.iterable"],
    "paths": {
      "@/*": ["src/app/*"],
      "@core/*": ["src/app/core/*"],
      "@shared/*": ["src/app/shared/*"],
      "@features/*": ["src/app/features/*"],
      "@environments/*": ["src/environments/*"]
    }
  },
  "angularCompilerOptions": {
    "enableI18nLegacyMessageIdFormat": false,
    "strictInjectionParameters": true,
    "strictInputAccessModifiers": true,
    "strictTemplates": true,
    "extendedDiagnostics": {
      "checks": {
        "optionalChainNotNullable": "error",
        "textAttributeNotBinding": "error",
        "missingControlFlowDirective": "error"
      }
    }
  }
}
```

---

## 1. Foundational Architecture & Project Structure

Angular 20 emphasizes **standalone components** and **feature-based architecture**. The traditional NgModule-based approach is now considered legacy.

### ✅ DO: Use Feature-Based Architecture with Standalone Components

```
/src
├── app/
│   ├── core/                    # Singleton services, guards, interceptors
│   │   ├── auth/
│   │   │   ├── auth.service.ts
│   │   │   ├── auth.guard.ts
│   │   │   └── auth.interceptor.ts
│   │   ├── api/
│   │   │   ├── api.service.ts
│   │   │   └── api.interceptor.ts
│   │   └── layout/
│   │       ├── header/
│   │       └── footer/
│   ├── shared/                  # Reusable components, directives, pipes
│   │   ├── ui/                   # Pure UI components
│   │   │   ├── button/
│   │   │   ├── card/
│   │   │   └── dialog/
│   │   ├── directives/
│   │   ├── pipes/
│   │   └── utils/
│   ├── features/                 # Feature modules
│   │   ├── dashboard/
│   │   │   ├── dashboard.routes.ts
│   │   │   ├── dashboard.component.ts
│   │   │   ├── components/      # Feature-specific components
│   │   │   ├── services/        # Feature-specific services
│   │   │   └── models/          # Feature-specific interfaces
│   │   └── products/
│   │       ├── products.routes.ts
│   │       ├── list/
│   │       ├── detail/
│   │       └── services/
│   ├── app.component.ts         # Root component
│   ├── app.config.ts            # Application configuration
│   └── app.routes.ts            # Root routing configuration
├── assets/
├── environments/
├── styles/                       # Global styles and design tokens
│   ├── _variables.scss
│   ├── _mixins.scss
│   └── styles.scss
└── index.html
```

### ✅ DO: Bootstrap with Standalone Components

```typescript
// main.ts - Modern Angular bootstrap
import { bootstrapApplication } from '@angular/platform-browser';
import { appConfig } from './app/app.config';
import { AppComponent } from './app/app.component';

bootstrapApplication(AppComponent, appConfig)
  .catch(err => console.error(err));
```

```typescript
// app.config.ts - Centralized configuration
import { ApplicationConfig, provideZoneChangeDetection } from '@angular/core';
import { provideRouter, withComponentInputBinding, withViewTransitions } from '@angular/router';
import { provideClientHydration, withEventReplay, withIncrementalHydration } from '@angular/platform-browser';
import { provideHttpClient, withInterceptors, withFetch } from '@angular/common/http';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';

import { routes } from './app.routes';
import { authInterceptor } from '@core/auth/auth.interceptor';
import { apiInterceptor } from '@core/api/api.interceptor';

export const appConfig: ApplicationConfig = {
  providers: [
    // Zone.js optimization - signals reduce need for change detection
    provideZoneChangeDetection({ eventCoalescing: true }),
    
    // Router with modern features
    provideRouter(routes, 
      withComponentInputBinding(),      // Bind route params to @Input()
      withViewTransitions()             // Native view transitions API
    ),
    
    // HTTP client with fetch API (smaller bundle)
    provideHttpClient(
      withFetch(),
      withInterceptors([authInterceptor, apiInterceptor])
    ),
    
    // Async animations for better performance
    provideAnimationsAsync(),
    
    // SSR hydration with event replay and incremental hydration (new in v20)
    provideClientHydration(
      withEventReplay(),
      withIncrementalHydration()  // New: hydrate only interactive parts
    )
  ]
};
```

---

## 2. Signals: The Stable Reactivity Paradigm

Angular 20's **Signals** are now fully stable, providing fine-grained reactivity, eliminating many Zone.js performance issues and enabling better tree-shaking.

### ✅ DO: Use Signals for Reactive State

```typescript
// dashboard.component.ts
import { Component, computed, effect, signal } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  template: `
    <div class="dashboard">
      <h1>Welcome {{ user()?.name }}</h1>
      
      <!-- Signal-based state updates automatically -->
      <div class="stats">
        <p>Total Items: {{ itemCount() }}</p>
        <p>Filtered Items: {{ filteredItems().length }}</p>
      </div>
      
      <!-- Two-way binding with signals -->
      <input [(ngModel)]="searchTerm" placeholder="Search..." />
      
      @for (item of filteredItems(); track item.id) {
        <app-item-card [item]="item" />
      }
    </div>
  `
})
export class DashboardComponent {
  // Writable signals
  user = signal<User | null>(null);
  items = signal<Item[]>([]);
  searchTerm = signal('');
  
  // Computed signals (memoized automatically)
  itemCount = computed(() => this.items().length);
  
  filteredItems = computed(() => {
    const term = this.searchTerm().toLowerCase();
    return this.items().filter(item => 
      item.name.toLowerCase().includes(term)
    );
  });
  
  // Convert Observable to Signal
  user$ = this.authService.user$;
  userSignal = toSignal(this.user$, { initialValue: null });
  
  // Effects for side effects
  logEffect = effect(() => {
    console.log(`Search term changed: ${this.searchTerm()}`);
    // Automatically re-runs when searchTerm changes
  });
  
  constructor(private authService: AuthService) {
    // Load initial data
    this.loadDashboardData();
  }
  
  async loadDashboardData() {
    const data = await this.apiService.getDashboardData();
    this.items.set(data.items);
    this.user.set(data.user);
  }
}
```

### ✅ DO: Use Signal-Based Forms (Angular 20 - Beta)

```typescript
// product-form.component.ts
import { Component, signal, computed, effect } from '@angular/core';
import { FormsModule } from '@angular/forms';
// Note: Signal-based forms API in beta
import { formSignal, signalFormControl } from '@angular/forms/experimental';

interface ProductForm {
  name: string;
  price: number;
  category: string;
  inStock: boolean;
}

@Component({
  selector: 'app-product-form',
  standalone: true,
  imports: [FormsModule],
  template: `
    <form (ngSubmit)="onSubmit()">
      <input [(ngModel)]="form.name" name="name" required />
      <input [(ngModel)]="form.price" name="price" type="number" required />
      <select [(ngModel)]="form.category" name="category">
        @for (cat of categories(); track cat) {
          <option [value]="cat">{{ cat }}</option>
        }
      </select>
      
      <button [disabled]="!isValid()">
        {{ submitLabel() }}
      </button>
    </form>
  `
})
export class ProductFormComponent {
  // Form state as signals
  form = signal<ProductForm>({
    name: '',
    price: 0,
    category: '',
    inStock: true
  });
  
  categories = signal(['Electronics', 'Clothing', 'Books']);
  isSubmitting = signal(false);
  
  // Computed validation
  isValid = computed(() => {
    const f = this.form();
    return f.name.length > 0 && f.price > 0 && f.category !== '';
  });
  
  // Dynamic submit label
  submitLabel = computed(() => 
    this.isSubmitting() ? 'Saving...' : 'Save Product'
  );
  
  // Auto-save draft effect
  autoSaveEffect = effect(() => {
    const formData = this.form();
    if (this.isValid()) {
      localStorage.setItem('productDraft', JSON.stringify(formData));
    }
  });
  
  async onSubmit() {
    if (!this.isValid()) return;
    
    this.isSubmitting.set(true);
    try {
      await this.productService.create(this.form());
      this.resetForm();
    } finally {
      this.isSubmitting.set(false);
    }
  }
  
  resetForm() {
    this.form.set({
      name: '',
      price: 0,
      category: '',
      inStock: true
    });
  }
}
```

### ✅ DO: Use linkedSignal for Dependent State (Stable in v20)

```typescript
// dashboard.component.ts
import { Component, signal, computed, linkedSignal } from '@angular/core';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  template: `...`
})
export class DashboardComponent {
  selectedCategory = signal<string>('all');
  
  // linkedSignal automatically updates when selectedCategory changes
  filteredProducts = linkedSignal({
    source: this.selectedCategory,
    computation: (category) => {
      if (category === 'all') return this.allProducts();
      return this.allProducts().filter(p => p.category === category);
    }
  });
}
```

---

## 3. Modern Control Flow and Template Syntax

Angular 20 enhances the built-in control flow with additional features and performance optimizations.

### ✅ DO: Use Built-in Control Flow with New Features

```typescript
@Component({
  template: `
    <!-- Conditional rendering -->
    @if (user()) {
      <p>Welcome, {{ user().name }}!</p>
    } @else if (isLoading()) {
      <p>Loading user data...</p>
    } @else {
      <p>Please log in</p>
    }
    
    <!-- Loops with mandatory track and new 'in' operator -->
    @for (item of items(); track item.id; let i = $index, let isEven = $even) {
      <div [class.even]="isEven">
        {{ i + 1 }}. {{ item.name }}
      </div>
    } @empty {
      <p>No items found</p>
    }
    
    <!-- New: 'in' operator for object iteration -->
    @for (key in config(); track key) {
      <p>{{ key }}: {{ config()[key] }}</p>
    }
    
    <!-- Switch statements -->
    @switch (userRole()) {
      @case ('admin') {
        <app-admin-dashboard />
      }
      @case ('user') {
        <app-user-dashboard />
      }
      @default {
        <app-guest-view />
      }
    }
    
    <!-- Defer loading with incremental hydration support -->
    @defer (on viewport; prefetch on idle; hydrate on interaction) {
      <app-heavy-component />
    } @placeholder (minimum 100ms) {
      <app-skeleton-loader />
    } @loading (after 100ms; minimum 1s) {
      <app-spinner />
    } @error {
      <p>Failed to load component</p>
    }
  `
})
```

### ✅ DO: Optimize Bundle Size with Defer

```typescript
// app.routes.ts - Lazy load routes
export const routes: Routes = [
  {
    path: 'dashboard',
    loadComponent: () => import('./features/dashboard/dashboard.component')
      .then(m => m.DashboardComponent),
    canActivate: [authGuard]
  },
  {
    path: 'products',
    loadChildren: () => import('./features/products/products.routes')
      .then(m => m.PRODUCTS_ROUTES)
  }
];

// Within components, defer heavy dependencies
@Component({
  template: `
    @defer (on interaction; prefetch on idle) {
      <app-chart [data]="chartData()" />
    } @placeholder {
      <div class="chart-placeholder">Click to load chart</div>
    }
  `
})
```

---

## 4. State Management with Signal Store

Angular 20 works best with **NgRx Signal Store** or similar signal-based state management.

### ✅ DO: Use Signal Store for Complex State

```typescript
// store/product.store.ts
import { signalStore, withState, withMethods, withComputed, patchState, withHooks } from '@ngrx/signals';
import { rxMethod } from '@ngrx/signals/rxjs-interop';
import { tapResponse } from '@ngrx/operators';
import { inject } from '@angular/core';
import { pipe, switchMap, tap } from 'rxjs';

interface ProductState {
  products: Product[];
  selectedId: string | null;
  loading: boolean;
  error: string | null;
  filter: ProductFilter;
}

const initialState: ProductState = {
  products: [],
  selectedId: null,
  loading: false,
  error: null,
  filter: { category: 'all', inStock: true }
};

export const ProductStore = signalStore(
  { providedIn: 'root' },
  withState(initialState),
  
  // Computed values
  withComputed(({ products, selectedId, filter }) => ({
    selectedProduct: computed(() => 
      products().find(p => p.id === selectedId())
    ),
    
    filteredProducts: computed(() => {
      const prods = products();
      const f = filter();
      return prods.filter(p => 
        (f.category === 'all' || p.category === f.category) &&
        (!f.inStock || p.inStock)
      );
    }),
    
    totalValue: computed(() => 
      products().reduce((sum, p) => sum + p.price * p.quantity, 0)
    )
  })),
  
  // Methods for state updates
  withMethods((store, productService = inject(ProductService)) => ({
    // Synchronous updates
    selectProduct(id: string) {
      patchState(store, { selectedId: id });
    },
    
    updateFilter(filter: Partial<ProductFilter>) {
      patchState(store, state => ({
        filter: { ...state.filter, ...filter }
      }));
    },
    
    // Async operations with rxMethod
    loadProducts: rxMethod<void>(
      pipe(
        tap(() => patchState(store, { loading: true, error: null })),
        switchMap(() => 
          productService.getAll().pipe(
            tapResponse({
              next: (products) => patchState(store, { products, loading: false }),
              error: (error) => patchState(store, { 
                error: error.message, 
                loading: false 
              })
            })
          )
        )
      )
    ),
    
    // Optimistic updates
    async updateProduct(id: string, changes: Partial<Product>) {
      // Optimistic update
      const oldProducts = store.products();
      patchState(store, {
        products: oldProducts.map(p => 
          p.id === id ? { ...p, ...changes } : p
        )
      });
      
      try {
        await productService.update(id, changes);
      } catch (error) {
        // Rollback on error
        patchState(store, { products: oldProducts });
        throw error;
      }
    }
  })),
  
  // Lifecycle hooks
  withHooks({
    onInit({ loadProducts }) {
      // Load products when store initializes
      loadProducts();
    }
  })
);

// Usage in component
@Component({
  selector: 'app-product-list',
  template: `
    <div class="filters">
      <select [ngModel]="store.filter().category" 
              (ngModelChange)="store.updateFilter({ category: $event })">
        <option value="all">All Categories</option>
        <option value="electronics">Electronics</option>
        <option value="clothing">Clothing</option>
      </select>
    </div>
    
    @if (store.loading()) {
      <app-spinner />
    }
    
    @for (product of store.filteredProducts(); track product.id) {
      <app-product-card 
        [product]="product"
        [selected]="product.id === store.selectedId()"
        (click)="store.selectProduct(product.id)"
      />
    }
    
    <div class="summary">
      Total Value: {{ store.totalValue() | currency }}
    </div>
  `
})
export class ProductListComponent {
  readonly store = inject(ProductStore);
}
```

### ✅ DO: Use Entity Management with Signal Store

```typescript
// store/entity-product.store.ts
import { signalStore, withEntities, withMethods } from '@ngrx/signals';
import { addEntity, updateEntity, removeEntity, setEntities } from '@ngrx/signals/entities';

export const EntityProductStore = signalStore(
  { providedIn: 'root' },
  withEntities<Product>(),
  
  withMethods((store, productService = inject(ProductService)) => ({
    async loadProducts() {
      const products = await productService.getAll();
      patchState(store, setEntities(products));
    },
    
    async addProduct(product: Omit<Product, 'id'>) {
      const newProduct = await productService.create(product);
      patchState(store, addEntity(newProduct));
    },
    
    async updateProduct(id: string, changes: Partial<Product>) {
      // Optimistic update
      patchState(store, updateEntity({ id, changes }));
      
      try {
        await productService.update(id, changes);
      } catch (error) {
        // Could implement rollback here
        throw error;
      }
    },
    
    async deleteProduct(id: string) {
      patchState(store, removeEntity(id));
      await productService.delete(id);
    }
  }))
);
```

---

## 5. Advanced HTTP Patterns with Interceptors

### ✅ DO: Use Functional Interceptors for Cross-Cutting Concerns

```typescript
// core/api/auth.interceptor.ts
import { HttpInterceptorFn, HttpRequest } from '@angular/common/http';
import { inject } from '@angular/core';
import { AuthService } from '../auth/auth.service';
import { switchMap, take } from 'rxjs/operators';

export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const authService = inject(AuthService);
  
  // Skip auth for public endpoints
  if (req.url.includes('/public/')) {
    return next(req);
  }
  
  // Use signals for sync access
  const token = authService.authToken();
  
  if (token) {
    req = req.clone({
      setHeaders: { Authorization: `Bearer ${token}` }
    });
  }
  
  return next(req);
};

// core/api/retry.interceptor.ts
import { HttpInterceptorFn } from '@angular/common/http';
import { retry, timer } from 'rxjs';
import { catchError, switchMap } from 'rxjs/operators';

export const retryInterceptor: HttpInterceptorFn = (req, next) => {
  // Exponential backoff retry for failed requests
  return next(req).pipe(
    retry({
      count: 3,
      delay: (error, retryCount) => {
        // Skip retry for client errors (4xx)
        if (error.status >= 400 && error.status < 500) {
          throw error;
        }
        
        // Exponential backoff: 1s, 2s, 4s
        const delay = Math.pow(2, retryCount - 1) * 1000;
        console.log(`Retry attempt ${retryCount} after ${delay}ms`);
        return timer(delay);
      }
    })
  );
};

// core/api/cache.interceptor.ts
import { HttpInterceptorFn, HttpResponse } from '@angular/common/http';
import { of, tap } from 'rxjs';
import { inject } from '@angular/core';

interface CacheEntry {
  response: HttpResponse<any>;
  timestamp: number;
}

export const cacheInterceptor: HttpInterceptorFn = (req, next) => {
  // Only cache GET requests
  if (req.method !== 'GET') {
    return next(req);
  }
  
  const cache = inject(CacheService);
  const cached = cache.get(req.url);
  
  // Return cached response if fresh (< 5 minutes)
  if (cached && Date.now() - cached.timestamp < 300000) {
    return of(cached.response.clone());
  }
  
  return next(req).pipe(
    tap(event => {
      if (event instanceof HttpResponse) {
        cache.set(req.url, { response: event, timestamp: Date.now() });
      }
    })
  );
};
```

### ✅ DO: Type-Safe API Client with Generics

```typescript
// core/api/api.service.ts
import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

interface ApiOptions {
  params?: Record<string, string | number | boolean>;
  headers?: Record<string, string>;
}

interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  pageSize: number;
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  private http = inject(HttpClient);
  private baseUrl = '/api/v1';
  
  get<T>(endpoint: string, options?: ApiOptions): Observable<T> {
    return this.http.get<T>(`${this.baseUrl}${endpoint}`, {
      params: this.buildParams(options?.params),
      headers: options?.headers
    });
  }
  
  post<T>(endpoint: string, body: any, options?: ApiOptions): Observable<T> {
    return this.http.post<T>(`${this.baseUrl}${endpoint}`, body, {
      params: this.buildParams(options?.params),
      headers: options?.headers
    });
  }
  
  put<T>(endpoint: string, body: any, options?: ApiOptions): Observable<T> {
    return this.http.put<T>(`${this.baseUrl}${endpoint}`, body, {
      params: this.buildParams(options?.params),
      headers: options?.headers
    });
  }
  
  delete<T>(endpoint: string, options?: ApiOptions): Observable<T> {
    return this.http.delete<T>(`${this.baseUrl}${endpoint}`, {
      params: this.buildParams(options?.params),
      headers: options?.headers
    });
  }
  
  // Paginated requests
  getPaginated<T>(
    endpoint: string, 
    page: number = 1, 
    pageSize: number = 20,
    options?: ApiOptions
  ): Observable<PaginatedResponse<T>> {
    const params = {
      ...options?.params,
      page: page.toString(),
      pageSize: pageSize.toString()
    };
    
    return this.get<PaginatedResponse<T>>(endpoint, { ...options, params });
  }
  
  private buildParams(params?: Record<string, string | number | boolean>): HttpParams {
    let httpParams = new HttpParams();
    
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== null && value !== undefined) {
          httpParams = httpParams.set(key, value.toString());
        }
      });
    }
    
    return httpParams;
  }
}

// Feature-specific service using the generic API
@Injectable()
export class ProductService {
  private api = inject(ApiService);
  
  getAll() {
    return this.api.get<Product[]>('/products');
  }
  
  getById(id: string) {
    return this.api.get<Product>(`/products/${id}`);
  }
  
  create(product: Omit<Product, 'id'>) {
    return this.api.post<Product>('/products', product);
  }
  
  update(id: string, changes: Partial<Product>) {
    return this.api.put<Product>(`/products/${id}`, changes);
  }
  
  delete(id: string) {
    return this.api.delete<void>(`/products/${id}`);
  }
  
  search(query: string, filters?: ProductFilter) {
    return this.api.getPaginated<Product>('/products/search', 1, 20, {
      params: { q: query, ...filters }
    });
  }
}
```

---

## 6. Authentication & Authorization Patterns

### ✅ DO: Use Guards with Dependency Injection

```typescript
// core/auth/auth.guard.ts
import { inject } from '@angular/core';
import { Router, CanActivateFn, ActivatedRouteSnapshot } from '@angular/router';
import { AuthService } from './auth.service';

export const authGuard: CanActivateFn = (route: ActivatedRouteSnapshot) => {
  const authService = inject(AuthService);
  const router = inject(Router);
  
  // Check authentication state
  if (!authService.isAuthenticated()) {
    // Store attempted URL for redirecting after login
    authService.redirectUrl.set(route.url.join('/'));
    return router.createUrlTree(['/login']);
  }
  
  // Check role-based access
  const requiredRoles = route.data['roles'] as string[] | undefined;
  if (requiredRoles && !authService.hasAnyRole(requiredRoles)) {
    return router.createUrlTree(['/unauthorized']);
  }
  
  return true;
};

// Usage in routes
export const routes: Routes = [
  {
    path: 'admin',
    loadChildren: () => import('./features/admin/admin.routes'),
    canActivate: [authGuard],
    data: { roles: ['admin', 'super-admin'] }
  }
];

// core/auth/auth.service.ts
import { Injectable, computed, effect, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';

interface AuthState {
  user: User | null;
  token: string | null;
  refreshToken: string | null;
  expiresAt: Date | null;
}

@Injectable({ providedIn: 'root' })
export class AuthService {
  // Auth state as signals
  private state = signal<AuthState>({
    user: null,
    token: null,
    refreshToken: null,
    expiresAt: null
  });
  
  // Public computed signals
  user = computed(() => this.state().user);
  authToken = computed(() => this.state().token);
  isAuthenticated = computed(() => !!this.state().token && !this.isTokenExpired());
  userRoles = computed(() => this.state().user?.roles || []);
  
  redirectUrl = signal<string | null>(null);
  
  constructor(
    private http: HttpClient,
    private router: Router
  ) {
    // Load auth state from localStorage on startup
    this.loadStoredAuth();
    
    // Persist auth state changes
    effect(() => {
      const state = this.state();
      if (state.token) {
        localStorage.setItem('auth_token', state.token);
        localStorage.setItem('auth_user', JSON.stringify(state.user));
        if (state.refreshToken) {
          localStorage.setItem('refresh_token', state.refreshToken);
        }
      } else {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('auth_user');
        localStorage.removeItem('refresh_token');
      }
    });
    
    // Auto-refresh token before expiry
    effect(() => {
      const expiresAt = this.state().expiresAt;
      if (expiresAt) {
        const refreshTime = expiresAt.getTime() - Date.now() - 60000; // 1 min before expiry
        if (refreshTime > 0) {
          setTimeout(() => this.refreshToken(), refreshTime);
        }
      }
    });
  }
  
  async login(credentials: LoginCredentials) {
    try {
      const response = await this.http.post<AuthResponse>('/api/auth/login', credentials)
        .toPromise();
      
      this.handleAuthResponse(response!);
      
      // Navigate to redirect URL or dashboard
      const redirectUrl = this.redirectUrl();
      if (redirectUrl) {
        this.router.navigateByUrl(redirectUrl);
        this.redirectUrl.set(null);
      } else {
        this.router.navigate(['/dashboard']);
      }
    } catch (error) {
      throw new Error('Login failed');
    }
  }
  
  async logout() {
    // Call logout endpoint to invalidate token on server
    try {
      await this.http.post('/api/auth/logout', {}).toPromise();
    } catch {
      // Continue with local logout even if server call fails
    }
    
    this.state.set({
      user: null,
      token: null,
      refreshToken: null,
      expiresAt: null
    });
    
    this.router.navigate(['/login']);
  }
  
  async refreshToken() {
    const refreshToken = this.state().refreshToken;
    if (!refreshToken) return;
    
    try {
      const response = await this.http.post<AuthResponse>('/api/auth/refresh', {
        refreshToken
      }).toPromise();
      
      this.handleAuthResponse(response!);
    } catch (error) {
      // Refresh failed, logout user
      this.logout();
    }
  }
  
  hasRole(role: string): boolean {
    return this.userRoles().includes(role);
  }
  
  hasAnyRole(roles: string[]): boolean {
    return roles.some(role => this.hasRole(role));
  }
  
  private handleAuthResponse(response: AuthResponse) {
    const expiresAt = new Date(Date.now() + response.expiresIn * 1000);
    
    this.state.set({
      user: response.user,
      token: response.accessToken,
      refreshToken: response.refreshToken,
      expiresAt
    });
  }
  
  private loadStoredAuth() {
    const token = localStorage.getItem('auth_token');
    const userStr = localStorage.getItem('auth_user');
    const refreshToken = localStorage.getItem('refresh_token');
    
    if (token && userStr) {
      try {
        const user = JSON.parse(userStr);
        // Validate token with server on app startup
        this.validateStoredToken(token, user, refreshToken);
      } catch {
        // Invalid stored data
      }
    }
  }
  
  private async validateStoredToken(token: string, user: User, refreshToken: string | null) {
    try {
      // Validate token with server
      const response = await this.http.post<{ valid: boolean; expiresIn?: number }>(
        '/api/auth/validate',
        { token }
      ).toPromise();
      
      if (response?.valid && response.expiresIn) {
        const expiresAt = new Date(Date.now() + response.expiresIn * 1000);
        this.state.set({ user, token, refreshToken, expiresAt });
      }
    } catch {
      // Token invalid, clear storage
      this.logout();
    }
  }
  
  private isTokenExpired(): boolean {
    const expiresAt = this.state().expiresAt;
    return !expiresAt || expiresAt.getTime() <= Date.now();
  }
}
```

---

## 7. Performance Optimization Strategies

### ✅ DO: Use OnPush Change Detection with Signals

```typescript
@Component({
  selector: 'app-product-card',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush, // Essential with signals
  template: `
    <div class="card" [class.selected]="selected()">
      <h3>{{ product().name }}</h3>
      <p>{{ product().price | currency }}</p>
      <button (click)="addToCart()">Add to Cart</button>
    </div>
  `
})
export class ProductCardComponent {
  // Input as signal
  product = input.required<Product>();
  selected = input(false);
  
  // Output remains the same
  @Output() addedToCart = new EventEmitter<Product>();
  
  addToCart() {
    this.addedToCart.emit(this.product());
  }
}
```

### ✅ DO: Implement Virtual Scrolling for Large Lists

```typescript
// product-list.component.ts
import { CdkVirtualScrollViewport, ScrollingModule } from '@angular/cdk/scrolling';

@Component({
  selector: 'app-product-list',
  standalone: true,
  imports: [ScrollingModule, ProductCardComponent],
  template: `
    <cdk-virtual-scroll-viewport 
      [itemSize]="150" 
      class="product-viewport">
      
      <app-product-card
        *cdkVirtualFor="let product of products(); trackBy: trackByFn"
        [product]="product"
        (addedToCart)="onAddToCart($event)"
      />
      
    </cdk-virtual-scroll-viewport>
  `,
  styles: [`
    .product-viewport {
      height: 600px;
      width: 100%;
    }
  `]
})
export class ProductListComponent {
  products = signal<Product[]>([]);
  
  trackByFn(index: number, item: Product) {
    return item.id;
  }
  
  onAddToCart(product: Product) {
    // Handle cart addition
  }
}
```

### ✅ DO: Optimize Bundle Size with Tree-Shaking

```typescript
// Only import what you need
import { debounceTime, distinctUntilChanged, switchMap } from 'rxjs/operators';
// NOT: import * as rxjs from 'rxjs';

// Use barrel exports carefully
// shared/ui/index.ts
export { ButtonComponent } from './button/button.component';
export { CardComponent } from './card/card.component';
// Only exported components will be included in bundles

// Lazy load heavy libraries
async function loadChartingLibrary() {
  const { Chart } = await import('chart.js');
  return Chart;
}
```

### ✅ DO: Implement Image Optimization

```typescript
// image.directive.ts
import { Directive, ElementRef, Input, OnInit } from '@angular/core';

@Directive({
  selector: 'img[appOptimized]',
  standalone: true
})
export class OptimizedImageDirective implements OnInit {
  @Input() appOptimized!: string; // Image URL
  @Input() width?: number;
  @Input() height?: number;
  
  constructor(private el: ElementRef<HTMLImageElement>) {}
  
  ngOnInit() {
    const img = this.el.nativeElement;
    
    // Set loading lazy by default
    img.loading = 'lazy';
    
    // Add responsive srcset
    if (this.width && this.height) {
      img.srcset = `
        ${this.getImageUrl(this.width * 0.5, this.height * 0.5)} 0.5x,
        ${this.getImageUrl(this.width, this.height)} 1x,
        ${this.getImageUrl(this.width * 1.5, this.height * 1.5)} 1.5x,
        ${this.getImageUrl(this.width * 2, this.height * 2)} 2x
      `;
    }
    
    // Set final src
    img.src = this.getImageUrl(this.width, this.height);
  }
  
  private getImageUrl(width?: number, height?: number): string {
    // Integrate with your image CDN/optimization service
    const params = new URLSearchParams();
    if (width) params.set('w', width.toString());
    if (height) params.set('h', height.toString());
    params.set('format', 'webp');
    params.set('quality', '85');
    
    return `https://cdn.example.com/optimize?url=${encodeURIComponent(this.appOptimized)}&${params}`;
  }
}
```

---

## 8. Testing Strategies

### ✅ DO: Use Component Testing with Testing Library

```typescript
// product-card.component.spec.ts
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { render, screen, fireEvent } from '@testing-library/angular';
import { ProductCardComponent } from './product-card.component';

describe('ProductCardComponent', () => {
  const mockProduct: Product = {
    id: '1',
    name: 'Test Product',
    price: 99.99,
    inStock: true
  };
  
  it('should display product information', async () => {
    await render(ProductCardComponent, {
      inputs: { product: mockProduct }
    });
    
    expect(screen.getByText('Test Product')).toBeInTheDocument();
    expect(screen.getByText('$99.99')).toBeInTheDocument();
  });
  
  it('should emit event when add to cart clicked', async () => {
    const addedToCart = jest.fn();
    
    await render(ProductCardComponent, {
      inputs: { product: mockProduct },
      on: { addedToCart }
    });
    
    const button = screen.getByText('Add to Cart');
    fireEvent.click(button);
    
    expect(addedToCart).toHaveBeenCalledWith(mockProduct);
  });
});
```

### ✅ DO: Test Signals and Effects

```typescript
// auth.service.spec.ts
import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';
import { AuthService } from './auth.service';

describe('AuthService', () => {
  let service: AuthService;
  let httpMock: HttpTestingController;
  
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [AuthService]
    });
    
    service = TestBed.inject(AuthService);
    httpMock = TestBed.inject(HttpTestingController);
  });
  
  it('should update signals on successful login', async () => {
    const mockResponse = {
      user: { id: '1', name: 'Test User', roles: ['user'] },
      accessToken: 'token123',
      refreshToken: 'refresh123',
      expiresIn: 3600
    };
    
    const loginPromise = service.login({ email: 'test@example.com', password: 'password' });
    
    const req = httpMock.expectOne('/api/auth/login');
    req.flush(mockResponse);
    
    await loginPromise;
    
    // Test signal values
    expect(service.user()?.name).toBe('Test User');
    expect(service.isAuthenticated()).toBe(true);
    expect(service.authToken()).toBe('token123');
  });
  
  it('should persist auth state to localStorage', async () => {
    const spy = jest.spyOn(Storage.prototype, 'setItem');
    
    // Trigger login
    await service.login({ email: 'test@example.com', password: 'password' });
    
    // Verify localStorage was updated (via effect)
    expect(spy).toHaveBeenCalledWith('auth_token', expect.any(String));
    expect(spy).toHaveBeenCalledWith('auth_user', expect.any(String));
  });
});
```

### ✅ DO: E2E Testing with Playwright

```typescript
// e2e/auth.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Authentication Flow', () => {
  test('should login and redirect to dashboard', async ({ page }) => {
    await page.goto('/login');
    
    // Fill login form
    await page.fill('input[name="email"]', 'test@example.com');
    await page.fill('input[name="password"]', 'password123');
    
    // Submit form
    await page.click('button[type="submit"]');
    
    // Wait for navigation
    await page.waitForURL('/dashboard');
    
    // Verify user is logged in
    await expect(page.locator('h1')).toContainText('Welcome');
  });
  
  test('should protect authenticated routes', async ({ page }) => {
    // Try to access protected route without auth
    await page.goto('/admin');
    
    // Should redirect to login
    await expect(page).toHaveURL('/login');
  });
});

// playwright.config.ts
import { defineConfig } from '@playwright/test';

export default defineConfig({
  use: {
    baseURL: 'http://localhost:4200',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
  ],
  webServer: {
    command: 'npm run start',
    port: 4200,
    reuseExistingServer: !process.env.CI,
  },
});
```

---

## 9. Progressive Web App (PWA) Configuration

### ✅ DO: Configure Service Worker with Workbox

```typescript
// ngsw-config.json
{
  "$schema": "./node_modules/@angular/service-worker/config/schema.json",
  "index": "/index.html",
  "assetGroups": [
    {
      "name": "app",
      "installMode": "prefetch",
      "resources": {
        "files": [
          "/favicon.ico",
          "/index.html",
          "/manifest.webmanifest",
          "/*.css",
          "/*.js"
        ]
      }
    },
    {
      "name": "assets",
      "installMode": "lazy",
      "updateMode": "prefetch",
      "resources": {
        "files": [
          "/assets/**",
          "/*.(svg|cur|jpg|jpeg|png|apng|webp|avif|gif|otf|ttf|woff|woff2)"
        ]
      }
    }
  ],
  "dataGroups": [
    {
      "name": "api-freshness",
      "urls": ["/api/**"],
      "cacheConfig": {
        "strategy": "freshness",
        "maxSize": 100,
        "maxAge": "12h",
        "timeout": "10s"
      }
    },
    {
      "name": "api-cache",
      "urls": ["/api/products/**"],
      "cacheConfig": {
        "strategy": "performance",
        "maxSize": 100,
        "maxAge": "7d"
      }
    }
  ],
  "navigationUrls": [
    "/**",
    "!/**/*.*",
    "!/**/*__*",
    "!/**/*__*/**"
  ]
}

// app.component.ts - Update notifications
import { SwUpdate } from '@angular/service-worker';

@Component({
  selector: 'app-root',
  template: `
    @if (updateAvailable()) {
      <div class="update-banner">
        <p>New version available!</p>
        <button (click)="updateApp()">Update Now</button>
      </div>
    }
    
    <router-outlet />
  `
})
export class AppComponent {
  updateAvailable = signal(false);
  
  constructor(private swUpdate: SwUpdate) {
    if (this.swUpdate.isEnabled) {
      // Check for updates
      this.swUpdate.versionUpdates.subscribe(event => {
        if (event.type === 'VERSION_READY') {
          this.updateAvailable.set(true);
        }
      });
    }
  }
  
  async updateApp() {
    await this.swUpdate.activateUpdate();
    document.location.reload();
  }
}
```

---

## 10. Internationalization (i18n)

### ✅ DO: Use Angular's Built-in i18n with Signals

```typescript
// locale.service.ts
import { Injectable, signal, computed, effect } from '@angular/core';
import { loadTranslations } from '@angular/localize';

@Injectable({ providedIn: 'root' })
export class LocaleService {
  currentLocale = signal<string>('en-US');
  availableLocales = signal(['en-US', 'es-ES', 'fr-FR', 'de-DE']);
  
  translations = computed(() => {
    // Dynamically load translations based on locale
    return this.loadTranslationsForLocale(this.currentLocale());
  });
  
  constructor() {
    // Persist locale preference
    effect(() => {
      localStorage.setItem('preferred-locale', this.currentLocale());
    });
    
    // Load saved locale
    const savedLocale = localStorage.getItem('preferred-locale');
    if (savedLocale && this.availableLocales().includes(savedLocale)) {
      this.currentLocale.set(savedLocale);
    }
  }
  
  async changeLocale(locale: string) {
    if (!this.availableLocales().includes(locale)) {
      throw new Error(`Locale ${locale} not supported`);
    }
    
    // Dynamically load locale data
    const localeModule = await import(`../locales/${locale}.js`);
    loadTranslations(localeModule.translations);
    
    this.currentLocale.set(locale);
  }
  
  private async loadTranslationsForLocale(locale: string) {
    const response = await fetch(`/assets/i18n/${locale}.json`);
    return response.json();
  }
}

// Component using i18n
@Component({
  template: `
    <h1 i18n="@@welcome.title">Welcome to our app!</h1>
    
    <p i18n="User greeting|Greeting message for logged in user@@user.greeting">
      Hello, {{ userName() }}! You have {{ messageCount() }} new messages.
    </p>
    
    <select [ngModel]="locale.currentLocale()" 
            (ngModelChange)="locale.changeLocale($event)">
      @for (loc of locale.availableLocales(); track loc) {
        <option [value]="loc">{{ getLocaleName(loc) }}</option>
      }
    </select>
  `
})
export class InternationalizedComponent {
  locale = inject(LocaleService);
  userName = signal('John');
  messageCount = signal(5);
  
  getLocaleName(locale: string): string {
    const names: Record<string, string> = {
      'en-US': 'English',
      'es-ES': 'Español',
      'fr-FR': 'Français',
      'de-DE': 'Deutsch'
    };
    return names[locale] || locale;
  }
}
```

---

## 11. Server-Side Rendering (SSR) Best Practices

### ✅ DO: Optimize for SSR with Transfer State

```typescript
// app.config.server.ts
import { ApplicationConfig, TransferState } from '@angular/core';
import { provideServerRendering } from '@angular/platform-server';

export const serverConfig: ApplicationConfig = {
  providers: [
    provideServerRendering(),
    {
      provide: 'SERVER_REQUEST',
      useFactory: () => inject('REQUEST')
    }
  ]
};

// product.service.ts - Transfer state pattern
import { TransferState, makeStateKey } from '@angular/core';

const PRODUCTS_KEY = makeStateKey<Product[]>('products');

@Injectable()
export class ProductService {
  constructor(
    private http: HttpClient,
    private transferState: TransferState,
    @Inject(PLATFORM_ID) private platformId: Object
  ) {}
  
  async getProducts(): Promise<Product[]> {
    // Check if data exists in transfer state
    const cached = this.transferState.get(PRODUCTS_KEY, null);
    if (cached) {
      // Remove from transfer state after reading
      this.transferState.remove(PRODUCTS_KEY);
      return cached;
    }
    
    const products = await this.http.get<Product[]>('/api/products').toPromise();
    
    // Store in transfer state if on server
    if (isPlatformServer(this.platformId)) {
      this.transferState.set(PRODUCTS_KEY, products!);
    }
    
    return products!;
  }
}
```

### ✅ DO: Handle Platform-Specific Code

```typescript
// platform-aware.component.ts
import { isPlatformBrowser, isPlatformServer } from '@angular/common';
import { PLATFORM_ID, inject } from '@angular/core';

@Component({
  selector: 'app-chart',
  template: `
    @if (isBrowser) {
      <canvas #chartCanvas></canvas>
    } @else {
      <div class="chart-placeholder">
        <img [src]="chartPreview" alt="Chart preview" />
      </div>
    }
  `
})
export class ChartComponent implements AfterViewInit {
  @ViewChild('chartCanvas') canvas!: ElementRef<HTMLCanvasElement>;
  
  platformId = inject(PLATFORM_ID);
  isBrowser = isPlatformBrowser(this.platformId);
  chartPreview = '/assets/chart-preview.png';
  
  ngAfterViewInit() {
    if (this.isBrowser) {
      // Only load Chart.js in browser
      this.loadChart();
    }
  }
  
  private async loadChart() {
    const { Chart } = await import('chart.js/auto');
    
    new Chart(this.canvas.nativeElement, {
      type: 'line',
      data: this.chartData,
      options: this.chartOptions
    });
  }
}
```

---

## 12. Micro Frontend Architecture

### ✅ DO: Use Module Federation for Micro Frontends

```typescript
// webpack.config.js - Shell application
const { withModuleFederation } = require('@nx/angular/module-federation');

module.exports = withModuleFederation({
  name: 'shell',
  remotes: ['products', 'checkout', 'user-profile'],
  shared: {
    '@angular/core': { singleton: true, strictVersion: true },
    '@angular/common': { singleton: true, strictVersion: true },
    '@angular/router': { singleton: true, strictVersion: true },
    '@ngrx/signals': { singleton: true, strictVersion: true }
  }
});

// shell/src/app/app.routes.ts
export const routes: Routes = [
  {
    path: 'products',
    loadChildren: () => 
      loadRemoteModule({
        type: 'module',
        remoteEntry: 'http://localhost:4201/remoteEntry.js',
        exposedModule: './Module'
      }).then(m => m.ProductsModule)
  }
];

// products/webpack.config.js - Remote application
module.exports = withModuleFederation({
  name: 'products',
  exposes: {
    './Module': './src/app/products/products.module.ts',
    './ProductCard': './src/app/products/components/product-card.component.ts'
  },
  shared: {
    '@angular/core': { singleton: true, strictVersion: true },
    '@angular/common': { singleton: true, strictVersion: true }
  }
});
```

---

## 13. Advanced Debugging & DevTools

### ✅ DO: Use Angular DevTools Profiler API

```typescript
// performance.service.ts
import { Injectable, isDevMode } from '@angular/core';
import { performanceMarkFeature } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class PerformanceService {
  private marks = new Map<string, number>();
  
  startMeasure(name: string) {
    if (isDevMode()) {
      this.marks.set(name, performance.now());
      performanceMarkFeature(`angular-${name}-start`);
    }
  }
  
  endMeasure(name: string) {
    if (isDevMode() && this.marks.has(name)) {
      const start = this.marks.get(name)!;
      const duration = performance.now() - start;
      
      performanceMarkFeature(`angular-${name}-end`);
      
      console.log(`[Performance] ${name}: ${duration.toFixed(2)}ms`);
      
      // Send to analytics in production
      if (!isDevMode()) {
        this.sendToAnalytics(name, duration);
      }
      
      this.marks.delete(name);
    }
  }
  
  private sendToAnalytics(name: string, duration: number) {
    // Implementation depends on your analytics provider
  }
}

// Usage in component
export class DataTableComponent implements OnInit {
  perf = inject(PerformanceService);
  
  async ngOnInit() {
    this.perf.startMeasure('data-table-init');
    
    await this.loadData();
    this.processData();
    
    this.perf.endMeasure('data-table-init');
  }
}
```

---

## 14. Build Optimization & Deployment

### ✅ DO: Configure Build Optimization

```json
// angular.json - Advanced build configuration
{
  "configurations": {
    "production": {
      "budgets": [
        {
          "type": "initial",
          "maximumWarning": "500kb",
          "maximumError": "1mb"
        },
        {
          "type": "anyComponentStyle",
          "maximumWarning": "2kb",
          "maximumError": "4kb"
        }
      ],
      "outputHashing": "all",
      "optimization": {
        "scripts": true,
        "styles": {
          "minify": true,
          "inlineCritical": true,
          "removeSpecialComments": true
        },
        "fonts": true
      },
      "sourceMap": {
        "scripts": false,
        "styles": false,
        "hidden": true,
        "vendor": false
      },
      "vendorChunk": true,
      "commonChunk": true,
      "namedChunks": false,
      "aot": true,
      "buildOptimizer": true,
      "fileReplacements": [
        {
          "replace": "src/environments/environment.ts",
          "with": "src/environments/environment.prod.ts"
        }
      ]
    }
  }
}
```

### ✅ DO: Implement Preloading Strategies

```typescript
// app/core/preloading-strategy.ts
import { Injectable } from '@angular/core';
import { PreloadingStrategy, Route } from '@angular/router';
import { Observable, of, timer } from 'rxjs';
import { mergeMap } from 'rxjs/operators';

@Injectable({ providedIn: 'root' })
export class NetworkAwarePreloadStrategy implements PreloadingStrategy {
  preload(route: Route, load: () => Observable<any>): Observable<any> {
    // Check network connection
    const connection = (navigator as any).connection;
    
    if (connection && connection.saveData) {
      // Don't preload on data saver mode
      return of(null);
    }
    
    const shouldPreload = route.data?.['preload'] ?? false;
    
    if (!shouldPreload) {
      return of(null);
    }
    
    // Delay preloading to prioritize initial load
    const delay = route.data?.['delay'] ?? 5000;
    
    return timer(delay).pipe(
      mergeMap(() => {
        console.log(`Preloading: ${route.path}`);
        return load();
      })
    );
  }
}

// app.config.ts
export const appConfig: ApplicationConfig = {
  providers: [
    provideRouter(routes, 
      withPreloading(NetworkAwarePreloadStrategy)
    )
  ]
};

// Route configuration with preloading hints
export const routes: Routes = [
  {
    path: 'dashboard',
    loadComponent: () => import('./dashboard/dashboard.component'),
    data: { preload: true, delay: 3000 }
  },
  {
    path: 'reports',
    loadComponent: () => import('./reports/reports.component'),
    data: { preload: false } // Heavy module, don't preload
  }
];
```

---

## 15. CI/CD Pipeline Configuration

### ✅ DO: Implement Comprehensive CI/CD

```yaml
# .github/workflows/ci.yml
name: Angular CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  NODE_VERSION: '22.x'
  PNPM_VERSION: '9'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: pnpm/action-setup@v4
        with:
          version: ${{ env.PNPM_VERSION }}
      
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'pnpm'
      
      - name: Install dependencies
        run: pnpm install --frozen-lockfile
      
      - name: Lint
        run: pnpm run lint
      
      - name: Type check
        run: pnpm run type-check
      
      - name: Unit tests
        run: pnpm run test:ci
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage/lcov.info
  
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: pnpm/action-setup@v4
        with:
          version: ${{ env.PNPM_VERSION }}
      
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'pnpm'
      
      - name: Install dependencies
        run: pnpm install --frozen-lockfile
      
      - name: Install Playwright browsers
        run: pnpm exec playwright install --with-deps
      
      - name: Run E2E tests
        run: pnpm run e2e:ci
      
      - uses: actions/upload-artifact@v4
        if: failure()
        with:
          name: playwright-traces
          path: test-results/
  
  build:
    needs: [test, e2e]
    runs-on: ubuntu-latest
    strategy:
      matrix:
        environment: [staging, production]
    
    steps:
      - uses: actions/checkout@v4
      
      - uses: pnpm/action-setup@v4
        with:
          version: ${{ env.PNPM_VERSION }}
      
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'pnpm'
      
      - name: Install dependencies
        run: pnpm install --frozen-lockfile
      
      - name: Build
        run: pnpm run build:${{ matrix.environment }}
      
      - name: Build SSR
        run: pnpm run build:ssr:${{ matrix.environment }}
      
      - name: Analyze bundle
        run: pnpm run analyze
      
      - name: Upload bundle stats
        uses: actions/upload-artifact@v4
        with:
          name: bundle-stats-${{ matrix.environment }}
          path: dist/stats.json
      
      - name: Deploy to ${{ matrix.environment }}
        if: github.ref == 'refs/heads/main'
        run: |
          # Deploy logic here (e.g., to AWS, Azure, etc.)
          echo "Deploying to ${{ matrix.environment }}"
```

### ✅ DO: Implement Bundle Analysis

```json
// package.json
{
  "scripts": {
    "build:stats": "ng build --stats-json",
    "analyze": "webpack-bundle-analyzer dist/stats.json",
    "lighthouse": "lighthouse http://localhost:4200 --output=json --output-path=./lighthouse-report.json",
    "test:ci": "ng test --no-watch --no-progress --browsers=ChromeHeadless --code-coverage",
    "e2e:ci": "playwright test --config=e2e/playwright.config.ts"
  }
}
```

---

## 16. Error Handling & Monitoring

### ✅ DO: Implement Global Error Handler

```typescript
// core/error-handler.service.ts
import { ErrorHandler, Injectable, inject } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import { Router } from '@angular/router';

interface ErrorContext {
  message: string;
  stack?: string;
  userAgent: string;
  timestamp: string;
  url: string;
  userId?: string;
  metadata?: Record<string, any>;
}

@Injectable()
export class GlobalErrorHandler implements ErrorHandler {
  private router = inject(Router);
  private errorService = inject(ErrorService);
  private authService = inject(AuthService);
  
  handleError(error: Error | HttpErrorResponse): void {
    const errorContext = this.buildErrorContext(error);
    
    // Log to console in development
    if (isDevMode()) {
      console.error('Global error handler:', error);
      console.table(errorContext);
    }
    
    // Send to monitoring service
    this.errorService.logError(errorContext);
    
    // Handle specific error types
    if (error instanceof HttpErrorResponse) {
      this.handleHttpError(error);
    } else if (error instanceof TypeError) {
      this.handleTypeError(error);
    } else {
      this.handleGenericError(error);
    }
  }
  
  private buildErrorContext(error: Error | HttpErrorResponse): ErrorContext {
    return {
      message: error.message || 'Unknown error',
      stack: error.stack,
      userAgent: navigator.userAgent,
      timestamp: new Date().toISOString(),
      url: this.router.url,
      userId: this.authService.user()?.id,
      metadata: error instanceof HttpErrorResponse ? {
        status: error.status,
        statusText: error.statusText,
        url: error.url
      } : undefined
    };
  }
  
  private handleHttpError(error: HttpErrorResponse) {
    switch (error.status) {
      case 401:
        this.authService.logout();
        this.router.navigate(['/login']);
        break;
      case 403:
        this.router.navigate(['/unauthorized']);
        break;
      case 404:
        this.router.navigate(['/not-found']);
        break;
      case 500:
      case 502:
      case 503:
        this.router.navigate(['/error'], {
          queryParams: { message: 'Server error. Please try again later.' }
        });
        break;
    }
  }
  
  private handleTypeError(error: TypeError) {
    // Specific handling for TypeErrors
    if (error.message.includes('Cannot read properties of null')) {
      console.warn('Null reference error:', error);
    }
  }
  
  private handleGenericError(error: Error) {
    // Generic error handling
    console.error('Unhandled error:', error);
  }
}

// app.config.ts
export const appConfig: ApplicationConfig = {
  providers: [
    { provide: ErrorHandler, useClass: GlobalErrorHandler },
    // ... other providers
  ]
};
```

### ✅ DO: Implement Error Boundaries

```typescript
// shared/components/error-boundary.component.ts
import { Component, ErrorHandler, OnDestroy, signal } from '@angular/core';
import { Subject, takeUntil } from 'rxjs';

@Component({
  selector: 'app-error-boundary',
  standalone: true,
  template: `
    @if (hasError()) {
      <div class="error-boundary">
        <h2>Something went wrong</h2>
        <p>{{ errorMessage() }}</p>
        <button (click)="retry()">Try Again</button>
      </div>
    } @else {
      <ng-content />
    }
  `,
  styles: [`
    .error-boundary {
      padding: 2rem;
      text-align: center;
      background-color: #fee;
      border: 1px solid #fcc;
      border-radius: 4px;
    }
  `]
})
export class ErrorBoundaryComponent implements OnDestroy {
  hasError = signal(false);
  errorMessage = signal('An unexpected error occurred');
  private destroy$ = new Subject<void>();
  
  constructor(private errorHandler: ErrorHandler) {
    // Subscribe to errors
    this.setupErrorHandling();
  }
  
  ngOnDestroy() {
    this.destroy$.next();
    this.destroy$.complete();
  }
  
  retry() {
    this.hasError.set(false);
    this.errorMessage.set('');
    // Optionally reload the component
  }
  
  private setupErrorHandling() {
    // This is a simplified example
    // In practice, you'd need more sophisticated error catching
    window.addEventListener('error', (event) => {
      this.hasError.set(true);
      this.errorMessage.set(event.message);
    });
  }
}
```

---

## 17. Accessibility (a11y) Best Practices

### ✅ DO: Implement Comprehensive Accessibility

```typescript
// shared/directives/a11y.directive.ts
import { Directive, ElementRef, Input, OnInit } from '@angular/core';

@Directive({
  selector: '[appA11y]',
  standalone: true
})
export class A11yDirective implements OnInit {
  @Input() appA11y: 'button' | 'link' | 'form' | 'modal' = 'button';
  @Input() label?: string;
  @Input() description?: string;
  
  constructor(private el: ElementRef<HTMLElement>) {}
  
  ngOnInit() {
    const element = this.el.nativeElement;
    
    switch (this.appA11y) {
      case 'button':
        this.enhanceButton(element);
        break;
      case 'link':
        this.enhanceLink(element);
        break;
      case 'form':
        this.enhanceForm(element);
        break;
      case 'modal':
        this.enhanceModal(element);
        break;
    }
    
    // Add ARIA label if provided
    if (this.label) {
      element.setAttribute('aria-label', this.label);
    }
    
    // Add ARIA description if provided
    if (this.description) {
      const descId = `desc-${Math.random().toString(36).substr(2, 9)}`;
      element.setAttribute('aria-describedby', descId);
      
      const descElement = document.createElement('span');
      descElement.id = descId;
      descElement.className = 'sr-only';
      descElement.textContent = this.description;
      element.appendChild(descElement);
    }
  }
  
  private enhanceButton(element: HTMLElement) {
    if (!element.hasAttribute('type') && element.tagName === 'BUTTON') {
      element.setAttribute('type', 'button');
    }
    
    if (!element.hasAttribute('role') && element.tagName !== 'BUTTON') {
      element.setAttribute('role', 'button');
      element.setAttribute('tabindex', '0');
      
      // Add keyboard support
      element.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          element.click();
        }
      });
    }
  }
  
  private enhanceLink(element: HTMLElement) {
    if (element.tagName === 'A' && !element.hasAttribute('href')) {
      element.setAttribute('role', 'button');
      element.setAttribute('tabindex', '0');
    }
  }
  
  private enhanceForm(element: HTMLElement) {
    element.setAttribute('novalidate', 'true');
    
    // Add required indicators
    const requiredInputs = element.querySelectorAll('[required]');
    requiredInputs.forEach(input => {
      input.setAttribute('aria-required', 'true');
    });
  }
  
  private enhanceModal(element: HTMLElement) {
    element.setAttribute('role', 'dialog');
    element.setAttribute('aria-modal', 'true');
    
    // Trap focus within modal
    this.setupFocusTrap(element);
  }
  
  private setupFocusTrap(element: HTMLElement) {
    const focusableElements = element.querySelectorAll(
      'a[href], button, textarea, input[type="text"], input[type="radio"], input[type="checkbox"], select'
    );
    
    const firstFocusable = focusableElements[0] as HTMLElement;
    const lastFocusable = focusableElements[focusableElements.length - 1] as HTMLElement;
    
    element.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        if (e.shiftKey) {
          if (document.activeElement === firstFocusable) {
            lastFocusable.focus();
            e.preventDefault();
          }
        } else {
          if (document.activeElement === lastFocusable) {
            firstFocusable.focus();
            e.preventDefault();
          }
        }
      }
      
      if (e.key === 'Escape') {
        // Close modal logic
      }
    });
  }
}

// shared/services/announcer.service.ts
import { Injectable } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class AnnouncerService {
  private announcer: HTMLElement;
  
  constructor() {
    this.announcer = this.createAnnouncer();
  }
  
  announce(message: string, priority: 'polite' | 'assertive' = 'polite') {
    this.announcer.setAttribute('aria-live', priority);
    this.announcer.textContent = message;
    
    // Clear after announcement
    setTimeout(() => {
      this.announcer.textContent = '';
    }, 1000);
  }
  
  private createAnnouncer(): HTMLElement {
    const element = document.createElement('div');
    element.setAttribute('aria-live', 'polite');
    element.setAttribute('aria-atomic', 'true');
    element.className = 'sr-only';
    document.body.appendChild(element);
    return element;
  }
}
```

---

## Conclusion

This guide represents the state-of-the-art in Angular development as of mid-2025. The key themes are:

1. **Signals everywhere** - The new reactivity model simplifies state management and improves performance
2. **Standalone components** - Module-free architecture is now the default
3. **Performance first** - OnPush change detection, defer loading, and virtual scrolling
4. **Type safety** - Full TypeScript support with strict mode
5. **Modern tooling** - Vite, esbuild, and native browser features
6. **Testing** - Comprehensive testing with modern tools
7. **Accessibility** - Built-in a11y support from the start

Remember that the JavaScript ecosystem evolves rapidly. Stay current with the Angular blog, participate in the community, and always measure performance in your specific use cases.

For the latest updates and more advanced patterns, refer to:
- [Angular Official Documentation](https://angular.dev)
- [Angular RFC Repository](https://github.com/angular/angular/discussions)
- [Angular DevTools](https://angular.io/guide/devtools)

Happy coding! 🚀