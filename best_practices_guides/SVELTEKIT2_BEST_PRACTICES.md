# The Definitive Guide to SvelteKit 2.21, Bun, and Modern State Management (mid-2025)

This guide synthesizes modern best practices for building scalable, secure, and performant applications with SvelteKit 2.21, Bun runtime, and contemporary state management patterns. It provides a production-grade architectural blueprint that leverages Svelte's unique compilation advantages and Bun's blazing-fast runtime.

### Prerequisites & Configuration

Ensure your project uses **SvelteKit 2.21+**, **Svelte 5.33+**, **Bun 1.2+**, and **TypeScript 5.8+**.

SvelteKit 2.21 supports Bun as a first-class runtime. Configure modern features in your `svelte.config.js`:

```javascript
// svelte.config.js
import adapter from '@sveltejs/adapter-bun';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  
  kit: {
    adapter: adapter({
      // Bun adapter options
      precompress: true,
      dynamic_origin: true,
    }),
    
    // SvelteKit 2.21 features
    experimental: {
      modulePreload: 'smart',   // Intelligent preloading
      hmr: {
        preserveLocalState: true,
      }
    },
    
    // Enhanced CSP with nonce support
    csp: {
      mode: 'auto',
      directives: {
        'script-src': ['self', 'nonce'],
        'style-src': ['self', 'unsafe-inline'],
      }
    },
    
    // Improved bundling
    build: {
      analyzeBundle: true,
      crossOriginPreload: 'use-credentials',
    }
  },
  
  // Svelte 5 compiler options
  compilerOptions: {
    runes: true,                    // Enable new reactivity system
    modernAst: true,                // Faster parsing
    enableSourcemap: true,
    hydratable: true,
    immutable: true,                // Assume immutable data by default
    accessors: false,               // Discourage getters/setters
    stripComments: true,            // Production optimization
  }
};

export default config;
```

> **Note**: Bun's native TypeScript support means you can use `svelte.config.ts` directly without transpilation.

### What's New in Svelte 5.33+ and SvelteKit 2.21+

**Svelte 5.33 Features:**
- **Attachments**: A more flexible replacement for actions with better composition
- **Generic Snippets**: Type-safe snippets with generics support
- **Improved Class Support**: State fields can be declared inside class constructors
- **XHTML Compliance**: New `fragments` option for better CSP compliance
- **Enhanced `class` Attribute**: Now accepts objects and arrays using `clsx` syntax

**SvelteKit 2.21 Features:**
- **Transport Hook**: Encode/decode custom non-POJOs across server/client boundary
- **Bundle Strategy**: Choose between 'split', 'single', and 'inline' output options
- **Top-level Client Code**: Allowed in universal pages when SSR is disabled
- **$app/state Module**: Replaces `$app/stores` with Svelte 5 state primitives

**TypeScript 5.8+ Integration:**
- Full support for granular type checking with conditional returns
- Direct TypeScript execution with Node.js experimental flags
- Enhanced inference for complex generic patterns

---

## 1. Foundational Architecture & File Organization

SvelteKit's file-based routing demands a well-organized structure. Adopt a hybrid approach balancing feature colocation with shared resources.

### ✅ DO: Use a Scalable Project Layout

This structure leverages SvelteKit's conventions while maintaining clear separation of concerns.

```
/src
├── routes/                   # SvelteKit routing (pages, layouts, endpoints)
│   ├── (auth)/              # Route group for authentication
│   │   ├── login/
│   │   │   ├── +page.svelte
│   │   │   ├── +page.server.ts    # Server-side logic
│   │   │   └── LoginForm.svelte   # Colocated component
│   │   └── +layout.svelte          # Auth layout wrapper
│   ├── api/                        # API routes
│   │   └── users/
│   │       └── +server.ts          # REST endpoint
│   └── +layout.svelte              # Root layout
├── lib/                            # Shared application code ($lib alias)
│   ├── components/                 # Reusable components
│   │   ├── ui/                    # Design system primitives
│   │   └── features/              # Domain-specific components
│   ├── stores/                    # State management
│   │   ├── auth.svelte.ts        # Rune-based store
│   │   └── ui.svelte.ts          # UI state
│   ├── server/                    # Server-only code
│   │   ├── db/                   # Database client
│   │   └── auth.ts               # Auth utilities
│   └── utils/                     # Shared utilities
├── params/                        # Route parameter matchers
├── hooks.server.ts               # Server hooks
├── hooks.client.ts               # Client hooks
└── app.d.ts                      # TypeScript declarations
```

### ✅ DO: Leverage Route Groups for Organization

Use parentheses to create logical groups without affecting URLs:

```
routes/
├── (marketing)/          # Public pages
│   ├── +layout.svelte   # Marketing-specific layout
│   ├── +page.svelte     # Landing page
│   └── pricing/
├── (app)/               # Authenticated app
│   ├── +layout.server.ts # Auth check
│   ├── dashboard/
│   └── settings/
└── (api)/               # API routes group
    └── v1/
```

---

## 2. The Rune Revolution: Modern Reactivity

Svelte 5's runes fundamentally change how we handle reactivity. Embrace the new paradigm.

### ✅ DO: Use Runes for All Reactive State

Runes provide fine-grained reactivity with better performance and cleaner syntax.

```typescript
// Good - Modern rune-based component
<script lang="ts">
  // State rune replaces let declarations
  let count = $state(0);
  let items = $state<string[]>([]);
  
  // Derived rune replaces $: reactive statements  
  let doubled = $derived(count * 2);
  let itemCount = $derived(items.length);
  
  // Effect rune replaces onMount/afterUpdate for side effects
  $effect(() => {
    console.log(`Count changed to ${count}`);
    
    // Cleanup function (replaces onDestroy)
    return () => {
      console.log('Cleaning up effect');
    };
  });
  
  // Props with runes
  let { title = 'Default', onUpdate }: { 
    title?: string;
    onUpdate?: (val: number) => void;
  } = $props();
</script>
```

### ❌ DON'T: Mix Legacy and Rune Patterns

This creates confusion and potential bugs. Migrate components fully.

```typescript
// Bad - Mixing paradigms
<script>
  let oldCount = 0;  // Legacy reactive
  let newCount = $state(0);  // Rune reactive
  
  // This won't work as expected!
  $: doubled = oldCount * 2;
  let newDoubled = $derived(newCount * 2);
</script>
```

### Advanced Rune Patterns

#### Custom Reactive Classes
```typescript
// lib/stores/todo.svelte.ts
export class TodoStore {
  items = $state<Todo[]>([]);
  filter = $state<'all' | 'active' | 'completed'>('all');
  
  // Derived state using getters
  get filtered() {
    return $derived(
      this.filter === 'all' 
        ? this.items 
        : this.items.filter(item => 
            this.filter === 'completed' ? item.completed : !item.completed
          )
    );
  }
  
  get stats() {
    return $derived({
      total: this.items.length,
      completed: this.items.filter(i => i.completed).length,
      active: this.items.filter(i => !i.completed).length
    });
  }
  
  add(text: string) {
    this.items.push({
      id: crypto.randomUUID(),
      text,
      completed: false
    });
  }
  
  toggle(id: string) {
    const item = this.items.find(i => i.id === id);
    if (item) item.completed = !item.completed;
  }
}

// Singleton instance
export const todos = new TodoStore();
```

---

## 3. Data Loading: Embrace the Load Function

SvelteKit's load functions are the cornerstone of data fetching. Master both server and universal patterns.

### ✅ DO: Use +page.server.ts for Sensitive Data

Server load functions keep secrets safe and reduce client bundle size.

```typescript
// routes/products/[id]/+page.server.ts
import { error } from '@sveltejs/kit';
import { db } from '$lib/server/db';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ params, locals }) => {
  // Access server-only resources
  const user = locals.user;  // From hooks.server.ts
  
  const product = await db.product.findUnique({
    where: { id: params.id },
    include: { 
      reviews: {
        take: 10,
        orderBy: { createdAt: 'desc' }
      }
    }
  });
  
  if (!product) {
    throw error(404, 'Product not found');
  }
  
  // This data is serialized and sent to the client
  return {
    product,
    userCanEdit: user?.id === product.authorId
  };
};
```

### ✅ DO: Use +page.ts for Client-Compatible Data

Universal load functions run on both server and client for optimal performance.

```typescript
// routes/products/+page.ts
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ fetch, url, depends }) => {
  // Use SvelteKit's fetch for automatic request deduplication
  const page = Number(url.searchParams.get('page') ?? '1');
  
  // Custom dependency for fine-grained invalidation
  depends('products:list');
  
  const res = await fetch(`/api/products?page=${page}`);
  
  if (!res.ok) {
    throw error(res.status, 'Failed to load products');
  }
  
  return {
    products: await res.json(),
    page
  };
};
```

### Data Streaming and Deferred Loading

SvelteKit 2.8 enhances streaming capabilities:

```typescript
// +page.server.ts - Stream slow data
export const load: PageServerLoad = async ({ fetch }) => {
  // Return promises for streaming
  return {
    // Fast data loads immediately
    user: await loadUser(),
    
    // Slow data streams in progressively
    recommendations: loadRecommendations(),  // Returns Promise
    analytics: loadAnalytics(),              // Returns Promise
  };
};
```

```svelte
<!-- +page.svelte - Handle streamed data -->
<script lang="ts">
  import { page } from '$app/stores';
  
  let { data } = $props();
</script>

<h1>Welcome {data.user.name}</h1>

{#await data.recommendations}
  <LoadingSpinner />
{:then recommendations}
  <Recommendations items={recommendations} />
{:catch error}
  <ErrorMessage {error} />
{/await}
```

---

## 4. State Management: Beyond Simple Stores

Modern Svelte applications need sophisticated state management. Use rune-based patterns for maximum performance.

### ✅ DO: Create Type-Safe Store Factories

Build reusable, type-safe store patterns:

```typescript
// lib/stores/create-store.svelte.ts
import { getContext, setContext } from 'svelte';

export function createStore<T extends Record<string, any>>(
  name: string,
  initializer: () => T
) {
  return {
    init() {
      const store = $state(initializer());
      setContext(name, store);
      return store;
    },
    
    get(): T {
      const store = getContext<T>(name);
      if (!store) {
        throw new Error(`Store "${name}" not initialized`);
      }
      return store;
    }
  };
}

// lib/stores/app-store.svelte.ts
interface AppState {
  theme: 'light' | 'dark' | 'system';
  sidebarOpen: boolean;
  user: User | null;
}

export const appStore = createStore<AppState>('app', () => ({
  theme: 'system',
  sidebarOpen: true,
  user: null,
  
  // Methods can be included in the state
  toggleSidebar() {
    this.sidebarOpen = !this.sidebarOpen;
  },
  
  setUser(user: User | null) {
    this.user = user;
  }
}));
```

### ✅ DO: Use Context for Component Trees

Provide stores at the appropriate level in your component hierarchy:

```svelte
<!-- routes/+layout.svelte -->
<script lang="ts">
  import { appStore } from '$lib/stores/app-store.svelte';
  
  // Initialize store at root level
  const app = appStore.init();
  
  // Can access store data directly
  $effect(() => {
    document.documentElement.setAttribute('data-theme', app.theme);
  });
</script>

<!-- Anywhere in the component tree -->
<script lang="ts">
  import { appStore } from '$lib/stores/app-store.svelte';
  
  const app = appStore.get();
</script>

<button onclick={() => app.toggleSidebar()}>
  Toggle Sidebar
</button>
```

### Advanced State Patterns

#### Async State Management
```typescript
// lib/stores/async-state.svelte.ts
export class AsyncState<T> {
  data = $state<T | undefined>(undefined);
  error = $state<Error | null>(null);
  loading = $state(false);
  
  constructor(private fetcher: () => Promise<T>) {}
  
  async load() {
    this.loading = true;
    this.error = null;
    
    try {
      this.data = await this.fetcher();
    } catch (e) {
      this.error = e instanceof Error ? e : new Error(String(e));
    } finally {
      this.loading = false;
    }
  }
  
  get value() {
    return $derived({
      data: this.data,
      error: this.error,
      loading: this.loading,
      hasData: this.data !== undefined,
      hasError: this.error !== null
    });
  }
}

// Usage
const userProfile = new AsyncState(() => fetch('/api/profile').then(r => r.json()));
```

---

## 5. Performance Optimization with Bun

Bun's speed advantages extend beyond just package management. Leverage its runtime features.

### ✅ DO: Use Bun's Built-in APIs

Replace Node.js APIs with Bun's faster alternatives:

```typescript
// hooks.server.ts - Using Bun APIs
import { dev } from '$app/environment';

export async function handle({ event, resolve }) {
  // Bun's crypto is 3x faster than Node's
  const requestId = Bun.hash(JSON.stringify({
    url: event.url.pathname,
    time: Date.now(),
    random: Math.random()
  })).toString(16);
  
  event.locals.requestId = requestId;
  
  // Bun's file I/O is significantly faster
  if (dev) {
    const start = Bun.nanoseconds();
    const response = await resolve(event);
    const duration = (Bun.nanoseconds() - start) / 1_000_000;
    
    // Async file append without blocking
    Bun.write('logs/requests.log', 
      `${requestId} ${event.url.pathname} ${duration}ms\n`,
      { append: true }
    );
    
    return response;
  }
  
  return resolve(event);
}
```

### ✅ DO: Optimize Bundle Size with Bun

Bun's bundler provides superior tree-shaking:

```javascript
// vite.config.js
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  
  build: {
    // Bun handles minification more efficiently
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
        pure_funcs: ['console.log', 'console.info'],
        passes: 2,
      },
    },
    
    // Advanced chunking strategy
    rollupOptions: {
      output: {
        manualChunks: (id) => {
          // Separate vendor chunks
          if (id.includes('node_modules')) {
            if (id.includes('@sveltejs/kit')) return 'sveltekit';
            if (id.includes('svelte')) return 'svelte';
            return 'vendor';
          }
        },
      },
    },
  },
  
  // Bun-specific optimizations
  optimizeDeps: {
    exclude: ['@sveltejs/kit', 'svelte'],
    include: ['lodash-es', 'date-fns'],
  },
});
```

### Bun Workspace Management

For monorepo setups, Bun provides built-in workspace support:

```json
// package.json
{
  "name": "myapp",
  "private": true,
  "workspaces": [
    "apps/*",
    "packages/*"
  ],
  "scripts": {
    // Bun runs scripts in parallel by default
    "dev": "bun run --filter '*' dev",
    "build": "bun run --filter '*' build",
    "test": "bun test"
  }
}
```

---

## 6. Form Actions: The SvelteKit Way

Form actions provide progressive enhancement and eliminate the need for client-side form handling.

### ✅ DO: Use Form Actions for Mutations

This pattern works without JavaScript and provides optimal UX:

```typescript
// routes/todos/+page.server.ts
import { fail, redirect } from '@sveltejs/kit';
import { z } from 'zod';
import type { Actions, PageServerLoad } from './$types';

const TodoSchema = z.object({
  title: z.string().min(1).max(200),
  description: z.string().optional(),
});

export const load: PageServerLoad = async ({ locals }) => {
  return {
    todos: await db.todo.findMany({
      where: { userId: locals.user.id },
      orderBy: { createdAt: 'desc' },
    }),
  };
};

export const actions: Actions = {
  create: async ({ request, locals }) => {
    const formData = await request.formData();
    
    // Validate with Zod
    const result = TodoSchema.safeParse({
      title: formData.get('title'),
      description: formData.get('description'),
    });
    
    if (!result.success) {
      return fail(400, {
        errors: result.error.flatten(),
        values: Object.fromEntries(formData),
      });
    }
    
    // Create in database
    await db.todo.create({
      data: {
        ...result.data,
        userId: locals.user.id,
      },
    });
    
    // Return success (SvelteKit reloads data automatically)
    return { success: true };
  },
  
  toggle: async ({ url, locals }) => {
    const id = url.searchParams.get('id');
    if (!id) return fail(400, { message: 'Missing ID' });
    
    await db.todo.update({
      where: { id, userId: locals.user.id },
      data: { 
        completed: { set: db.raw('NOT completed') } 
      },
    });
  },
  
  delete: async ({ url, locals }) => {
    const id = url.searchParams.get('id');
    if (!id) return fail(400);
    
    await db.todo.delete({
      where: { id, userId: locals.user.id },
    });
  },
};
```

```svelte
<!-- routes/todos/+page.svelte -->
<script lang="ts">
  import { enhance } from '$app/forms';
  
  let { data, form } = $props();
  let creating = $state(false);
</script>

<!-- Progressive enhancement with enhance -->
<form 
  method="POST" 
  action="?/create"
  use:enhance={() => {
    creating = true;
    
    return async ({ update }) => {
      await update();
      creating = false;
    };
  }}
>
  <input
    name="title"
    required
    value={form?.values?.title ?? ''}
    aria-invalid={form?.errors?.fieldErrors?.title ? 'true' : undefined}
  />
  
  {#if form?.errors?.fieldErrors?.title}
    <span class="error">{form.errors.fieldErrors.title[0]}</span>
  {/if}
  
  <button disabled={creating}>
    {creating ? 'Creating...' : 'Add Todo'}
  </button>
</form>

<!-- Optimistic UI for toggle/delete -->
{#each data.todos as todo (todo.id)}
  <div class="todo" class:completed={todo.completed}>
    <form
      method="POST"
      action="?/toggle&id={todo.id}"
      use:enhance={() => {
        // Optimistic update
        todo.completed = !todo.completed;
        
        return async ({ update }) => {
          // Revert if failed
          await update({ reset: false });
        };
      }}
    >
      <button>{todo.completed ? '✓' : '○'}</button>
    </form>
    
    <span>{todo.title}</span>
    
    <form method="POST" action="?/delete&id={todo.id}" use:enhance>
      <button>Delete</button>
    </form>
  </div>
{/each}
```

---

## 7. API Design: Type-Safe Endpoints

Build robust APIs with full type safety and validation.

### ✅ DO: Create Type-Safe API Helpers

Build reusable patterns for API endpoints:

```typescript
// lib/server/api.ts
import { json, error } from '@sveltejs/kit';
import { z } from 'zod';

export function createEndpoint<TInput, TOutput>(config: {
  input?: z.ZodSchema<TInput>;
  output?: z.ZodSchema<TOutput>;
  handler: (input: TInput, event: RequestEvent) => Promise<TOutput>;
}) {
  return async (event: RequestEvent) => {
    try {
      // Validate input if schema provided
      let input: TInput;
      if (config.input) {
        const body = await event.request.json().catch(() => ({}));
        const result = config.input.safeParse(body);
        
        if (!result.success) {
          throw error(400, {
            message: 'Validation failed',
            errors: result.error.flatten(),
          });
        }
        
        input = result.data;
      } else {
        input = {} as TInput;
      }
      
      // Run handler
      const output = await config.handler(input, event);
      
      // Validate output if schema provided
      if (config.output) {
        const result = config.output.safeParse(output);
        if (!result.success) {
          console.error('Output validation failed:', result.error);
          throw error(500, 'Internal server error');
        }
      }
      
      return json(output);
    } catch (err) {
      if (err instanceof Error && 'status' in err) {
        throw err;  // Re-throw SvelteKit errors
      }
      console.error('Endpoint error:', err);
      throw error(500, 'Internal server error');
    }
  };
}

// routes/api/users/[id]/+server.ts
import { createEndpoint } from '$lib/server/api';

const UpdateUserSchema = z.object({
  name: z.string().optional(),
  email: z.string().email().optional(),
});

const UserResponseSchema = z.object({
  id: z.string(),
  name: z.string(),
  email: z.string(),
  updatedAt: z.string(),
});

export const PATCH = createEndpoint({
  input: UpdateUserSchema,
  output: UserResponseSchema,
  handler: async (input, { params, locals }) => {
    // Full type safety
    const user = await db.user.update({
      where: { id: params.id },
      data: input,
    });
    
    return {
      id: user.id,
      name: user.name,
      email: user.email,
      updatedAt: user.updatedAt.toISOString(),
    };
  },
});
```

### Rate Limiting with Bun

Implement efficient rate limiting using Bun's performance:

```typescript
// lib/server/rate-limit.ts
interface RateLimitStore {
  requests: Map<string, number[]>;
}

const store: RateLimitStore = {
  requests: new Map(),
};

export function createRateLimiter(options: {
  windowMs: number;
  max: number;
}) {
  return async (event: RequestEvent) => {
    const ip = event.getClientAddress();
    const now = Date.now();
    const windowStart = now - options.windowMs;
    
    // Get existing requests
    const requests = store.requests.get(ip) || [];
    
    // Filter out old requests
    const recentRequests = requests.filter(time => time > windowStart);
    
    if (recentRequests.length >= options.max) {
      throw error(429, {
        message: 'Too many requests',
        retryAfter: Math.ceil(options.windowMs / 1000),
      });
    }
    
    // Add current request
    recentRequests.push(now);
    store.requests.set(ip, recentRequests);
    
    // Cleanup old entries periodically
    if (Math.random() < 0.01) {
      for (const [key, times] of store.requests.entries()) {
        if (times.every(t => t < windowStart)) {
          store.requests.delete(key);
        }
      }
    }
  };
}

// Usage in endpoint
export const POST = createEndpoint({
  middleware: [
    createRateLimiter({ windowMs: 60000, max: 10 }),
  ],
  handler: async () => {
    // Your logic here
  },
});
```

---

## 8. Authentication & Security

Implement robust authentication using SvelteKit's server-side capabilities.

### ✅ DO: Use Lucia Auth v3 with Bun

Lucia provides a modern, type-safe authentication solution:

```typescript
// lib/server/auth.ts
import { Lucia } from 'lucia';
import { BunSQLiteAdapter } from '@lucia-auth/adapter-sqlite';
import { db } from './db';

export const lucia = new Lucia(
  new BunSQLiteAdapter(db, {
    user: 'users',
    session: 'sessions',
  }),
  {
    sessionCookie: {
      attributes: {
        secure: !dev,
        sameSite: 'lax',
        path: '/',
      },
    },
    getUserAttributes: (attributes) => {
      return {
        email: attributes.email,
        name: attributes.name,
        role: attributes.role,
      };
    },
  }
);

declare module 'lucia' {
  interface Register {
    Lucia: typeof lucia;
    DatabaseUserAttributes: {
      email: string;
      name: string;
      role: 'user' | 'admin';
    };
  }
}

// hooks.server.ts
export async function handle({ event, resolve }) {
  const sessionId = event.cookies.get(lucia.sessionCookieName);
  
  if (!sessionId) {
    event.locals.user = null;
    event.locals.session = null;
    return resolve(event);
  }
  
  const { session, user } = await lucia.validateSession(sessionId);
  
  if (session && session.fresh) {
    const sessionCookie = lucia.createSessionCookie(session.id);
    event.cookies.set(sessionCookie.name, sessionCookie.value, {
      path: '.',
      ...sessionCookie.attributes,
    });
  }
  
  if (!session) {
    const sessionCookie = lucia.createBlankSessionCookie();
    event.cookies.set(sessionCookie.name, sessionCookie.value, {
      path: '.',
      ...sessionCookie.attributes,
    });
  }
  
  event.locals.user = user;
  event.locals.session = session;
  
  return resolve(event);
}
```

### OAuth Integration

```typescript
// routes/auth/github/+server.ts
import { generateState } from 'arctic';
import { github } from '$lib/server/oauth';

export async function GET(event) {
  const state = generateState();
  const url = await github.createAuthorizationURL(state, {
    scopes: ['user:email'],
  });
  
  event.cookies.set('github_oauth_state', state, {
    path: '/',
    secure: !dev,
    httpOnly: true,
    maxAge: 60 * 10,
    sameSite: 'lax',
  });
  
  redirect(302, url.toString());
}

// routes/auth/github/callback/+server.ts
export async function GET(event) {
  const code = event.url.searchParams.get('code');
  const state = event.url.searchParams.get('state');
  const storedState = event.cookies.get('github_oauth_state');
  
  if (!code || !state || !storedState || state !== storedState) {
    throw error(400, 'Invalid request');
  }
  
  try {
    const tokens = await github.validateAuthorizationCode(code);
    const githubUser = await getGithubUser(tokens.accessToken);
    
    // Create or update user
    const user = await db.user.upsert({
      where: { githubId: githubUser.id },
      update: { 
        email: githubUser.email,
        avatarUrl: githubUser.avatar_url,
      },
      create: {
        githubId: githubUser.id,
        email: githubUser.email,
        name: githubUser.name,
        avatarUrl: githubUser.avatar_url,
      },
    });
    
    // Create session
    const session = await lucia.createSession(user.id, {});
    const sessionCookie = lucia.createSessionCookie(session.id);
    
    event.cookies.set(sessionCookie.name, sessionCookie.value, {
      path: '.',
      ...sessionCookie.attributes,
    });
    
    redirect(302, '/dashboard');
  } catch (e) {
    console.error('OAuth error:', e);
    throw error(500, 'Authentication failed');
  }
}
```

---

## 9. Testing Strategies

Comprehensive testing ensures reliability and maintainability.

### ✅ DO: Use Vitest with Bun

Vitest provides the best testing experience for Vite-based projects:

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';
import { sveltekit } from '@sveltejs/kit/vite';

export default defineConfig({
  plugins: [sveltekit()],
  
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./tests/setup.ts'],
    include: ['src/**/*.{test,spec}.{js,ts}'],
    coverage: {
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'tests/',
        '**/*.d.ts',
        '**/*.config.*',
        '**/mockServiceWorker.js',
      ],
    },
    // Bun test runner integration
    pool: 'threads',
    poolOptions: {
      threads: {
        singleThread: true,
      },
    },
  },
});
```

### Component Testing

```typescript
// lib/components/ui/Button.test.ts
import { render, fireEvent } from '@testing-library/svelte';
import { expect, test, vi } from 'vitest';
import Button from './Button.svelte';

test('Button renders with props', async () => {
  const onClick = vi.fn();
  const { getByRole } = render(Button, {
    props: {
      variant: 'primary',
      onclick: onClick,
    },
  });
  
  const button = getByRole('button');
  expect(button).toHaveClass('btn-primary');
  
  await fireEvent.click(button);
  expect(onClick).toHaveBeenCalledOnce();
});
```

### Integration Testing

```typescript
// tests/integration/auth.test.ts
import { test, expect } from '@playwright/test';

test.describe('Authentication flow', () => {
  test('user can sign up, log in, and log out', async ({ page }) => {
    // Sign up
    await page.goto('/signup');
    await page.fill('[name="email"]', 'test@example.com');
    await page.fill('[name="password"]', 'SecurePass123!');
    await page.click('button[type="submit"]');
    
    await expect(page).toHaveURL('/dashboard');
    await expect(page.locator('h1')).toContainText('Welcome');
    
    // Log out
    await page.click('button:has-text("Log out")');
    await expect(page).toHaveURL('/');
    
    // Log in
    await page.goto('/login');
    await page.fill('[name="email"]', 'test@example.com');
    await page.fill('[name="password"]', 'SecurePass123!');
    await page.click('button[type="submit"]');
    
    await expect(page).toHaveURL('/dashboard');
  });
});
```

### API Testing

```typescript
// routes/api/users/users.test.ts
import { test, expect } from 'vitest';
import { createRequest } from '$lib/test-utils';

test('GET /api/users returns user list', async () => {
  const response = await app.handle(
    createRequest('GET', '/api/users', {
      headers: {
        Authorization: 'Bearer test-token',
      },
    })
  );
  
  expect(response.status).toBe(200);
  const data = await response.json();
  expect(data).toHaveProperty('users');
  expect(Array.isArray(data.users)).toBe(true);
});
```

---

## 10. Advanced Component Patterns

### ✅ DO: Build Accessible Component Systems

Create reusable, accessible components:

```svelte
<!-- lib/components/ui/Modal.svelte -->
<script lang="ts" context="module">
  export interface ModalProps {
    open?: boolean;
    onClose?: () => void;
    title?: string;
    description?: string;
  }
</script>

<script lang="ts">
  import { portal } from '$lib/actions/portal';
  import { trapFocus } from '$lib/actions/trap-focus';
  import { fade, scale } from 'svelte/transition';
  
  let { 
    open = false, 
    onClose,
    title,
    description,
    children
  }: ModalProps = $props();
  
  let dialog: HTMLDialogElement;
  
  $effect(() => {
    if (open && dialog) {
      dialog.showModal();
    } else if (!open && dialog) {
      dialog.close();
    }
  });
  
  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape' && onClose) {
      onClose();
    }
  }
</script>

{#if open}
  <div 
    class="modal-backdrop" 
    transition:fade={{ duration: 200 }}
    onclick={onClose}
    aria-hidden="true"
  />
  
  <dialog
    bind:this={dialog}
    class="modal"
    transition:scale={{ duration: 200, start: 0.95 }}
    use:portal
    use:trapFocus
    onkeydown={handleKeydown}
    aria-labelledby={title ? 'modal-title' : undefined}
    aria-describedby={description ? 'modal-description' : undefined}
  >
    {#if title}
      <h2 id="modal-title" class="modal-title">{title}</h2>
    {/if}
    
    {#if description}
      <p id="modal-description" class="modal-description">{description}</p>
    {/if}
    
    <div class="modal-content">
      {@render children?.()}
    </div>
    
    <button 
      class="modal-close" 
      onclick={onClose}
      aria-label="Close dialog"
    >
      ×
    </button>
  </dialog>
{/if}

<style>
  .modal-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.5);
    z-index: 50;
  }
  
  .modal {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    max-width: 90vw;
    max-height: 90vh;
    overflow: auto;
    background: white;
    border-radius: 8px;
    padding: 2rem;
    z-index: 100;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
  }
</style>
```

### Custom Actions

```typescript
// lib/actions/intersection-observer.ts
export function intersectionObserver(
  node: HTMLElement,
  params: {
    callback: (entry: IntersectionObserverEntry) => void;
    options?: IntersectionObserverInit;
  }
) {
  const observer = new IntersectionObserver(
    (entries) => entries.forEach(params.callback),
    params.options
  );
  
  observer.observe(node);
  
  return {
    destroy() {
      observer.disconnect();
    },
  };
}

// Usage
<div
  use:intersectionObserver={{
    callback: (entry) => {
      if (entry.isIntersecting) {
        loadMore();
      }
    },
    options: { rootMargin: '100px' }
  }}
>
  Loading trigger
</div>
```

---

## 11. Deployment & Production

### ✅ DO: Optimize for Bun Runtime

Deploy with Bun for maximum performance:

```dockerfile
# Dockerfile
FROM oven/bun:1.2 as deps
WORKDIR /app
COPY package.json bun.lockb ./
RUN bun install --frozen-lockfile

FROM oven/bun:1.2 as builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN bun run build

FROM oven/bun:1.2-slim
WORKDIR /app

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

# Copy built app
COPY --from=builder --chown=nodejs:nodejs /app/build ./build
COPY --from=builder --chown=nodejs:nodejs /app/package.json ./

# Install production dependencies only
RUN bun install --production --frozen-lockfile

USER nodejs
EXPOSE 3000

ENV NODE_ENV=production
ENV PORT=3000

CMD ["bun", "run", "build/index.js"]
```

### Environment Configuration

```typescript
// app.d.ts
declare global {
  namespace App {
    interface Locals {
      user: import('lucia').User | null;
      session: import('lucia').Session | null;
    }
    
    interface PageData {
      flash?: { type: 'success' | 'error'; message: string };
    }
    
    interface Platform {
      env: {
        DB: D1Database;
        KV: KVNamespace;
        BUCKET: R2Bucket;
      };
      context: {
        waitUntil(promise: Promise<any>): void;
      };
      caches: CacheStorage & { default: Cache };
    }
  }
}

export {};

// lib/server/env.ts
import { z } from 'zod';
import { dev } from '$app/environment';

const envSchema = z.object({
  DATABASE_URL: z.string().url(),
  DATABASE_AUTH_TOKEN: z.string().min(1).optional(),
  PUBLIC_SITE_URL: z.string().url(),
  GITHUB_CLIENT_ID: z.string(),
  GITHUB_CLIENT_SECRET: z.string(),
  EMAIL_FROM: z.string().email(),
  EMAIL_SERVER: z.string().url(),
  REDIS_URL: z.string().url().optional(),
});

export const env = envSchema.parse(
  dev 
    ? process.env 
    : {
        // Production uses platform bindings
        DATABASE_URL: platform?.env?.DATABASE_URL,
        // etc...
      }
);
```

### Performance Monitoring

```typescript
// hooks.server.ts
export async function handle({ event, resolve }) {
  const start = performance.now();
  
  // Add request ID
  const requestId = crypto.randomUUID();
  event.locals.requestId = requestId;
  
  // Track metrics
  const response = await resolve(event, {
    transformPageChunk: ({ html }) => {
      // Inject performance marks
      return html.replace(
        '</head>',
        `<script>
          window.__REQUEST_ID__ = '${requestId}';
          window.__SERVER_TIMING__ = ${performance.now() - start};
        </script></head>`
      );
    },
  });
  
  // Add server timing header
  response.headers.set(
    'Server-Timing',
    `total;dur=${performance.now() - start}`
  );
  
  // Log slow requests
  const duration = performance.now() - start;
  if (duration > 1000) {
    console.warn(`Slow request: ${event.url.pathname} took ${duration}ms`);
  }
  
  return response;
}
```

---

## 12. Real-time Features with WebSockets

### Server-Sent Events (SSE) Pattern

```typescript
// routes/api/notifications/stream/+server.ts
export async function GET({ locals, request }) {
  // Verify auth
  if (!locals.user) {
    throw error(401);
  }
  
  // Create SSE stream
  const stream = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();
      
      // Send initial connection
      controller.enqueue(
        encoder.encode(`data: ${JSON.stringify({ type: 'connected' })}\n\n`)
      );
      
      // Subscribe to events
      const unsubscribe = subscribeToUserEvents(locals.user.id, (event) => {
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify(event)}\n\n`)
        );
      });
      
      // Heartbeat to keep connection alive
      const heartbeat = setInterval(() => {
        controller.enqueue(encoder.encode(': heartbeat\n\n'));
      }, 30000);
      
      // Cleanup on close
      request.signal.addEventListener('abort', () => {
        unsubscribe();
        clearInterval(heartbeat);
        controller.close();
      });
    },
  });
  
  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}

// Client usage
const events = new EventSource('/api/notifications/stream');

events.onmessage = (e) => {
  const data = JSON.parse(e.data);
  handleNotification(data);
};
```

### WebSocket with Bun

```typescript
// server/websocket.ts
import type { ServerWebSocket } from 'bun';

interface WSData {
  userId: string;
  room: string;
}

const rooms = new Map<string, Set<ServerWebSocket<WSData>>>();

export const websocket = {
  message(ws: ServerWebSocket<WSData>, message: string) {
    const data = JSON.parse(message);
    
    switch (data.type) {
      case 'join':
        ws.data.room = data.room;
        
        if (!rooms.has(data.room)) {
          rooms.set(data.room, new Set());
        }
        rooms.get(data.room)!.add(ws);
        
        // Broadcast join event
        broadcast(data.room, {
          type: 'user-joined',
          userId: ws.data.userId,
        }, ws);
        break;
        
      case 'message':
        broadcast(ws.data.room, {
          type: 'message',
          userId: ws.data.userId,
          content: data.content,
          timestamp: new Date().toISOString(),
        });
        break;
    }
  },
  
  open(ws: ServerWebSocket<WSData>) {
    ws.send(JSON.stringify({ type: 'connected' }));
  },
  
  close(ws: ServerWebSocket<WSData>) {
    if (ws.data.room && rooms.has(ws.data.room)) {
      rooms.get(ws.data.room)!.delete(ws);
      
      broadcast(ws.data.room, {
        type: 'user-left',
        userId: ws.data.userId,
      });
    }
  },
};

function broadcast(
  room: string, 
  message: any, 
  exclude?: ServerWebSocket<WSData>
) {
  const sockets = rooms.get(room);
  if (!sockets) return;
  
  const data = JSON.stringify(message);
  for (const socket of sockets) {
    if (socket !== exclude && socket.readyState === 1) {
      socket.send(data);
    }
  }
}
```

---

## 13. Error Handling & Observability

### Comprehensive Error Tracking

```typescript
// hooks.server.ts
import { Toucan } from 'toucan-js';

export async function handleError({ error, event, status, message }) {
  const errorId = crypto.randomUUID();
  
  // Initialize Sentry client
  const sentry = new Toucan({
    dsn: env.SENTRY_DSN,
    context: event,
    request: event.request,
  });
  
  // Capture with context
  sentry.withScope((scope) => {
    scope.setTag('errorId', errorId);
    scope.setContext('sveltekit', {
      status,
      message,
      route: event.route.id,
      params: event.params,
    });
    
    if (event.locals.user) {
      scope.setUser({
        id: event.locals.user.id,
        email: event.locals.user.email,
      });
    }
    
    sentry.captureException(error);
  });
  
  // Log to console in dev
  if (dev) {
    console.error(`Error ${errorId}:`, error);
  }
  
  // Return user-friendly error
  return {
    message: status >= 500 
      ? `Internal error (${errorId})` 
      : message,
  };
}

// Custom error page
// routes/+error.svelte
<script>
  import { page } from '$app/stores';
  
  let { status = 500, message = 'Unknown error' } = $page.error || {};
</script>

<div class="error-page">
  <h1>{status}</h1>
  <p>{message}</p>
  
  {#if status === 404}
    <a href="/">Go home</a>
  {:else}
    <button onclick={() => location.reload()}>Try again</button>
  {/if}
</div>
```

### Structured Logging

```typescript
// lib/server/logger.ts
import winston from 'winston';

export const logger = winston.createLogger({
  level: dev ? 'debug' : 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { 
    service: 'sveltekit-app',
    version: process.env.npm_package_version,
  },
  transports: [
    new winston.transports.Console({
      format: dev ? winston.format.simple() : undefined,
    }),
  ],
});

// Request logging middleware
export function logRequest(event: RequestEvent) {
  const start = Date.now();
  
  event.locals.log = logger.child({
    requestId: event.locals.requestId,
    method: event.request.method,
    path: event.url.pathname,
    userId: event.locals.user?.id,
  });
  
  // Log after response
  event.locals.log.info('request', {
    duration: Date.now() - start,
    status: event.locals.status,
  });
}
```

---

## 14. Performance Patterns

### Image Optimization

```svelte
<!-- lib/components/Image.svelte -->
<script lang="ts">
  interface ImageProps {
    src: string;
    alt: string;
    width?: number;
    height?: number;
    lazy?: boolean;
    sizes?: string;
  }
  
  let { 
    src, 
    alt, 
    width, 
    height, 
    lazy = true,
    sizes = '100vw',
    ...rest 
  }: ImageProps = $props();
  
  // Generate srcset for responsive images
  const widths = [640, 768, 1024, 1280, 1536];
  const srcset = widths
    .filter(w => !width || w <= width)
    .map(w => `${src}?w=${w} ${w}w`)
    .join(', ');
</script>

<img
  {src}
  {alt}
  {srcset}
  {sizes}
  {width}
  {height}
  loading={lazy ? 'lazy' : 'eager'}
  decoding="async"
  {...rest}
/>
```

### Resource Hints

```svelte
<!-- app.html -->
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%sveltekit.assets%/favicon.png" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    
    <!-- Preconnect to external domains -->
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="dns-prefetch" href="https://api.example.com" />
    
    <!-- Resource hints injected by SvelteKit -->
    %sveltekit.head%
  </head>
  <body data-sveltekit-preload-data="hover">
    <div style="display: contents">%sveltekit.body%</div>
  </body>
</html>
```

### Code Splitting

```typescript
// Lazy load heavy components
const HeavyChart = $state<typeof import('$lib/components/Chart.svelte')>();

async function loadChart() {
  const module = await import('$lib/components/Chart.svelte');
  HeavyChart = module.default;
}

// In template
{#if showChart}
  {#if HeavyChart}
    <HeavyChart data={chartData} />
  {:else}
    <button onclick={loadChart}>Load Chart</button>
  {/if}
{/if}
```

---

## 15. Advanced Bun Features

### SQLite with Bun

```typescript
// lib/server/db.ts
import { Database } from 'bun:sqlite';
import { drizzle } from 'drizzle-orm/bun-sqlite';
import * as schema from './schema';

// Bun's SQLite is significantly faster than better-sqlite3
const sqlite = new Database('app.db', { 
  strict: true,
  create: true,
});

// Enable WAL mode for better concurrency
sqlite.exec('PRAGMA journal_mode = WAL');
sqlite.exec('PRAGMA synchronous = NORMAL');

export const db = drizzle(sqlite, { schema });

// Use Bun's native prepared statements
export const queries = {
  getUserById: sqlite.prepare(
    'SELECT * FROM users WHERE id = ?'
  ),
  
  updateUserLastSeen: sqlite.prepare(
    'UPDATE users SET last_seen = CURRENT_TIMESTAMP WHERE id = ?'
  ),
};
```

### Native Crypto Performance

```typescript
// Use Bun's faster crypto APIs
export async function hashPassword(password: string): Promise<string> {
  // Bun.password is 10x faster than bcrypt
  return await Bun.password.hash(password, {
    algorithm: 'argon2id',
    memoryCost: 65536,
    timeCost: 3,
  });
}

export async function verifyPassword(
  password: string, 
  hash: string
): Promise<boolean> {
  return await Bun.password.verify(password, hash);
}

// Fast UUID generation
export function generateId(): string {
  // Bun's crypto is faster than crypto.randomUUID()
  return Bun.hash(
    crypto.getRandomValues(new Uint8Array(16))
  ).toString('hex');
}
```

### Shell Scripting

```typescript
// scripts/deploy.ts
#!/usr/bin/env bun

import { $ } from 'bun';

// Bun's shell is async by default
await $`bun test`;
await $`bun run build`;

// Deploy to production
const result = await $`flyctl deploy`.quiet();

if (result.exitCode === 0) {
  console.log('✅ Deployed successfully');
} else {
  console.error('❌ Deployment failed');
  process.exit(1);
}
```

---

## Conclusion

This guide represents the state-of-the-art for building Svelte applications with Bun in mid-2025. The ecosystem continues to evolve rapidly, but these patterns provide a solid foundation for building fast, maintainable, and scalable web applications.

Key takeaways:
- Embrace Svelte 5's rune system for cleaner, more performant reactivity
- Leverage Bun's speed advantages throughout your stack
- Use SvelteKit's server-first approach for optimal performance
- Build with progressive enhancement in mind
- Type everything for maintainability at scale

For updates and community discussion, join the [Svelte Discord](https://discord.gg/svelte) and follow [@sveltejs](https://twitter.com/sveltejs) for the latest developments.