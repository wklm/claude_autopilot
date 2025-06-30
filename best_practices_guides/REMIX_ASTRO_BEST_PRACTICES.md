# The Definitive Guide to Remix 2 & Astro 4 for Content-Heavy Sites (2025)

This guide synthesizes modern best practices for building scalable, performant, and SEO-optimized content sites using Remix 2 and Astro 4. It provides a production-grade architectural blueprint for content-heavy applications in mid-2025.

### Prerequisites & Technology Selection
Ensure your project uses **Remix 2.13+**, **Astro 4.18+**, **Node.js 22.11+**, and **TypeScript 5.6+**.

## The Remix vs Astro Decision Matrix

Choose based on your content site's primary characteristics:

| Use Case | Recommended Stack | Why |
| :--- | :--- | :--- |
| **Marketing Sites with CMS** | Astro + Content Collections | Superior build performance, optimal static generation |
| **Documentation Sites** | Astro + Starlight | Purpose-built for docs with built-in search, i18n |
| **News/Magazine Sites** | Remix + Astro Islands | Dynamic content with static shell, real-time updates |
| **E-commerce Content** | Remix (primary) + Astro (blog) | Dynamic product pages, static content sections |
| **Corporate Sites** | Astro with Remix API routes | Mostly static with dynamic contact forms |
| **Multi-tenant Platforms** | Remix with Vite | Dynamic routing, user-specific content |

### The Hybrid Approach: When to Use Both

```typescript
// astro.config.mjs - Integrate Remix for dynamic sections
import { defineConfig } from 'astro/config';
import remix from '@astrojs/remix';
import node from '@astrojs/node';

export default defineConfig({
  output: 'hybrid', // Static by default, SSR where needed
  adapter: node({ mode: 'middleware' }),
  integrations: [
    remix({
      // Mount Remix app at specific routes
      include: ['/app/*', '/api/*'],
    }),
  ],
  prefetch: {
    defaultStrategy: 'viewport',
    prefetchAll: true, // Prefetch all links in viewport
  },
});
```

---

## 1. Foundational Architecture for Content Sites

A well-structured content architecture separates static content, dynamic features, and shared components.

### ✅ DO: Use a Content-First Directory Structure

```
/src
├── content/              # Astro Content Collections
│   ├── blog/            # Markdown/MDX posts
│   │   ├── _schemas.ts  # Zod schemas for frontmatter
│   │   └── 2025-01-15-best-practices.mdx
│   ├── docs/            # Documentation pages
│   ├── authors/         # Author profiles (referenced)
│   └── config.ts        # Collection definitions
├── pages/               # Astro pages (static routes)
│   ├── blog/
│   │   ├── [...slug].astro    # Dynamic blog routes
│   │   └── index.astro         # Blog listing
│   └── index.astro      # Homepage
├── app/                 # Remix app (dynamic features)
│   ├── routes/          # Remix routes
│   │   ├── _index.tsx   # App dashboard
│   │   └── api.$.tsx    # Catch-all API routes
│   └── root.tsx         # Remix root
├── components/          # Shared UI components
│   ├── astro/          # Astro-specific components
│   ├── react/          # React components (used by both)
│   └── islands/        # Interactive islands
├── layouts/            # Page layouts
├── lib/                # Shared utilities
│   ├── content/        # Content helpers
│   ├── search/         # Search implementation
│   └── analytics/      # Analytics setup
└── middleware/         # Edge middleware (auth, redirects)
```

### Content Collection Schema Definition (Astro 4)

```typescript
// src/content/config.ts
import { defineCollection, z, reference } from 'astro:content';
import { glob } from 'astro/loaders'; // New in Astro 4.14

const blogCollection = defineCollection({
  // Use the new glob loader for better performance
  loader: glob({ 
    pattern: "**/*.{md,mdx}",
    base: "./src/content/blog",
  }),
  schema: z.object({
    title: z.string().max(60), // SEO optimal length
    description: z.string().min(120).max(160), // Meta description
    publishDate: z.coerce.date(),
    updateDate: z.coerce.date().optional(),
    author: reference('authors'), // Reference author collection
    category: z.enum(['tutorial', 'news', 'guide', 'update']),
    tags: z.array(z.string()).default([]),
    image: z.object({
      src: z.string(),
      alt: z.string(),
      caption: z.string().optional(),
    }).optional(),
    draft: z.boolean().default(false),
    featured: z.boolean().default(false),
    // SEO fields
    seo: z.object({
      metaTitle: z.string().optional(),
      metaDescription: z.string().optional(),
      canonical: z.string().url().optional(),
      noindex: z.boolean().default(false),
    }).optional(),
  }),
});

const authorsCollection = defineCollection({
  type: 'data', // JSON/YAML data files
  schema: z.object({
    name: z.string(),
    bio: z.string(),
    avatar: z.string(),
    social: z.object({
      twitter: z.string().optional(),
      github: z.string().optional(),
      linkedin: z.string().optional(),
    }),
  }),
});

export const collections = {
  blog: blogCollection,
  authors: authorsCollection,
};
```

---

## 2. Content Rendering & Performance Optimization

### ✅ DO: Implement Smart Rendering Strategies

```typescript
// src/pages/blog/[...slug].astro
---
import { getCollection, getEntry, render } from 'astro:content';
import { Image } from 'astro:assets';
import BaseLayout from '@/layouts/BaseLayout.astro';
import { generateTableOfContents } from '@/lib/content/toc';

export async function getStaticPaths() {
  const posts = await getCollection('blog', ({ data }) => {
    // Filter out drafts in production
    return import.meta.env.PROD ? !data.draft : true;
  });
  
  return posts.map((post) => ({
    params: { slug: post.id },
    props: { post },
  }));
}

const { post } = Astro.props;
const { Content, headings, remarkPluginFrontmatter } = await render(post);

// Generate TOC from headings
const toc = generateTableOfContents(headings);

// Calculate reading time
const readingTime = remarkPluginFrontmatter.readingTime;
---

<BaseLayout 
  title={post.data.seo?.metaTitle || post.data.title}
  description={post.data.seo?.metaDescription || post.data.description}
>
  <article class="prose lg:prose-xl mx-auto">
    {post.data.image && (
      <Image
        src={post.data.image.src}
        alt={post.data.image.alt}
        width={1200}
        height={630}
        loading="eager"
        format="avif"
        quality={85}
        class="w-full rounded-lg"
      />
    )}
    
    <header>
      <h1>{post.data.title}</h1>
      <div class="metadata">
        <time datetime={post.data.publishDate.toISOString()}>
          {post.data.publishDate.toLocaleDateString()}
        </time>
        <span>{readingTime}</span>
      </div>
    </header>
    
    <aside class="toc">
      <nav>{toc}</nav>
    </aside>
    
    <Content components={{
      // Override MDX components for consistent styling
      img: Image,
      a: LinkWithPrefetch,
      code: SyntaxHighlighter,
    }} />
  </article>
</BaseLayout>
```

### ✅ DO: Implement View Transitions for SPA-like Experience

```astro
---
// src/layouts/BaseLayout.astro
import { ViewTransitions } from 'astro:transitions';
---
<!DOCTYPE html>
<html lang="en">
  <head>
    <ViewTransitions />
    <script>
      // Persist interactive elements during transitions
      document.addEventListener('astro:before-swap', (e) => {
        const oldSearch = e.from.querySelector('#search-modal');
        const newSearch = e.to.querySelector('#search-modal');
        if (oldSearch && newSearch) {
          newSearch.replaceWith(oldSearch);
        }
      });
    </script>
  </head>
</html>
```

### Advanced Image Optimization

```typescript
// src/components/OptimizedImage.astro
---
import { getImage } from 'astro:assets';

interface Props {
  src: string;
  alt: string;
  sizes?: string;
  loading?: 'eager' | 'lazy';
  fetchpriority?: 'high' | 'low' | 'auto';
}

const { src, alt, sizes = '100vw', loading = 'lazy', fetchpriority = 'auto' } = Astro.props;

// Generate responsive images
const avif = await getImage({ src, format: 'avif', width: 1920 });
const webp = await getImage({ src, format: 'webp', width: 1920 });
const jpeg = await getImage({ src, format: 'jpeg', width: 1920 });

// Generate srcset for different sizes
const widths = [320, 640, 768, 1024, 1280, 1920];
const avifSrcset = await Promise.all(
  widths.map(async (w) => {
    const img = await getImage({ src, format: 'avif', width: w });
    return `${img.src} ${w}w`;
  })
);
---

<picture>
  <source
    type="image/avif"
    srcset={avifSrcset.join(', ')}
    sizes={sizes}
  />
  <source
    type="image/webp"
    srcset={webp.srcset}
    sizes={sizes}
  />
  <img
    src={jpeg.src}
    alt={alt}
    loading={loading}
    fetchpriority={fetchpriority}
    decoding="async"
    width={jpeg.attributes.width}
    height={jpeg.attributes.height}
  />
</picture>
```

---

## 3. Remix Integration for Dynamic Features

### ✅ DO: Use Remix for Interactive Content Features

```typescript
// app/routes/search.tsx - Full-text search with Remix
import { json, type LoaderFunctionArgs } from '@remix-run/node';
import { useFetcher } from '@remix-run/react';
import { searchContent } from '@/lib/search/client.server';

export async function loader({ request }: LoaderFunctionArgs) {
  const url = new URL(request.url);
  const query = url.searchParams.get('q');
  
  if (!query) return json({ results: [] });
  
  // Use Typesense/Meilisearch for fast search
  const results = await searchContent(query, {
    limit: 20,
    typoTolerance: true,
    highlightFields: ['title', 'content'],
  });
  
  return json({ results }, {
    headers: {
      'Cache-Control': 'public, max-age=300, stale-while-revalidate=600',
    },
  });
}

export function SearchModal() {
  const fetcher = useFetcher<typeof loader>();
  const [query, setQuery] = useState('');
  
  // Debounced search
  useEffect(() => {
    const timer = setTimeout(() => {
      if (query.length > 2) {
        fetcher.load(`/search?q=${encodeURIComponent(query)}`);
      }
    }, 300);
    
    return () => clearTimeout(timer);
  }, [query]);
  
  return (
    <dialog id="search-modal" className="search-modal">
      <form method="dialog">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search content..."
          autoFocus
        />
        
        {fetcher.state === 'loading' && <LoadingSpinner />}
        
        {fetcher.data?.results && (
          <SearchResults results={fetcher.data.results} />
        )}
      </form>
    </dialog>
  );
}
```

### ✅ DO: Implement Progressive Enhancement

```typescript
// app/routes/newsletter.tsx - Works without JavaScript
import { json, type ActionFunctionArgs } from '@remix-run/node';
import { Form, useActionData, useNavigation } from '@remix-run/react';
import { subscribeToNewsletter } from '@/lib/newsletter.server';

export async function action({ request }: ActionFunctionArgs) {
  const formData = await request.formData();
  const email = formData.get('email');
  
  try {
    await subscribeToNewsletter(email as string);
    return json({ success: true, message: 'Successfully subscribed!' });
  } catch (error) {
    return json({ success: false, message: 'Invalid email address' }, { status: 400 });
  }
}

export default function NewsletterForm() {
  const actionData = useActionData<typeof action>();
  const navigation = useNavigation();
  const isSubmitting = navigation.state === 'submitting';
  
  return (
    <Form method="post" className="newsletter-form">
      <input
        type="email"
        name="email"
        required
        placeholder="your@email.com"
        aria-label="Email address"
      />
      
      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? 'Subscribing...' : 'Subscribe'}
      </button>
      
      {actionData?.message && (
        <p className={actionData.success ? 'success' : 'error'}>
          {actionData.message}
        </p>
      )}
    </Form>
  );
}
```

---

## 4. Content Search Implementation

### ✅ DO: Build a Hybrid Search System

```typescript
// src/lib/search/build-index.ts - Build search index at compile time
import { getCollection } from 'astro:content';
import { create, insert } from '@orama/orama';
import { persistToFile } from '@orama/plugin-data-persistence';

export async function buildSearchIndex() {
  // Create Orama schema
  const db = await create({
    schema: {
      id: 'string',
      title: 'string',
      content: 'string',
      description: 'string',
      category: 'string',
      tags: 'string[]',
      publishDate: 'number',
      url: 'string',
    },
    components: {
      tokenizer: {
        stemming: true,
        stopWords: ['the', 'a', 'an'],
      },
    },
  });
  
  // Index all content
  const posts = await getCollection('blog');
  const documents = posts.map(post => ({
    id: post.id,
    title: post.data.title,
    content: post.body, // Raw content
    description: post.data.description,
    category: post.data.category,
    tags: post.data.tags,
    publishDate: post.data.publishDate.getTime(),
    url: `/blog/${post.slug}`,
  }));
  
  await insert(db, documents);
  
  // Persist for client-side search
  await persistToFile(db, './public/search-index.json');
  
  return db;
}

// Client-side search component
// src/components/islands/SearchIsland.tsx
import { create, load, search } from '@orama/orama';
import { useState, useEffect } from 'react';

export default function SearchIsland() {
  const [db, setDb] = useState(null);
  const [results, setResults] = useState([]);
  
  useEffect(() => {
    // Load pre-built index
    fetch('/search-index.json')
      .then(res => res.json())
      .then(async (data) => {
        const database = await create({ /* schema */ });
        await load(database, data);
        setDb(database);
      });
  }, []);
  
  const handleSearch = async (query: string) => {
    if (!db || query.length < 3) return;
    
    const searchResults = await search(db, {
      term: query,
      properties: ['title', 'content', 'description'],
      boost: {
        title: 2,
        description: 1.5,
      },
      limit: 10,
    });
    
    setResults(searchResults.hits);
  };
  
  return (
    <div className="search-widget">
      <input
        type="search"
        onChange={(e) => handleSearch(e.target.value)}
        placeholder="Search articles..."
      />
      
      {results.length > 0 && (
        <div className="search-results">
          {results.map(hit => (
            <a key={hit.id} href={hit.document.url}>
              <h3>{hit.document.title}</h3>
              <p>{hit.document.description}</p>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
```

---

## 5. Content Management & Editorial Workflow

### ✅ DO: Implement a Git-Based CMS Integration

```typescript
// astro.config.mjs - Integrate with Tina CMS or Decap CMS
import { defineConfig } from 'astro/config';
import tina from 'astro-tina-cms';

export default defineConfig({
  integrations: [
    tina({
      clientId: process.env.TINA_CLIENT_ID,
      token: process.env.TINA_TOKEN,
      branch: process.env.VERCEL_GIT_COMMIT_REF || 'main',
      schema: {
        collections: [
          {
            name: 'blog',
            label: 'Blog Posts',
            path: 'src/content/blog',
            format: 'mdx',
            fields: [
              {
                name: 'title',
                label: 'Title',
                type: 'string',
                required: true,
              },
              {
                name: 'publishDate',
                label: 'Publish Date',
                type: 'datetime',
                required: true,
              },
              {
                name: 'author',
                label: 'Author',
                type: 'reference',
                collections: ['authors'],
              },
              {
                name: 'body',
                label: 'Content',
                type: 'rich-text',
                isBody: true,
                templates: [
                  // Custom MDX components
                  {
                    name: 'Callout',
                    label: 'Callout Box',
                    fields: [
                      {
                        name: 'type',
                        label: 'Type',
                        type: 'string',
                        options: ['info', 'warning', 'tip', 'danger'],
                      },
                      {
                        name: 'content',
                        label: 'Content',
                        type: 'string',
                        ui: { component: 'textarea' },
                      },
                    ],
                  },
                  {
                    name: 'CodeBlock',
                    label: 'Code Block',
                    fields: [
                      {
                        name: 'language',
                        label: 'Language',
                        type: 'string',
                      },
                      {
                        name: 'code',
                        label: 'Code',
                        type: 'string',
                        ui: { component: 'textarea' },
                      },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      },
    }),
  ],
});
```

### ✅ DO: Implement Content Versioning and Preview

```typescript
// app/routes/api.preview.tsx - Preview unpublished content
import { json, type LoaderFunctionArgs } from '@remix-run/node';
import { createPreviewHandler } from '@/lib/preview.server';

export async function loader({ request }: LoaderFunctionArgs) {
  const url = new URL(request.url);
  const token = url.searchParams.get('token');
  const slug = url.searchParams.get('slug');
  
  // Validate preview token
  if (!isValidPreviewToken(token)) {
    throw new Response('Invalid preview token', { status: 401 });
  }
  
  // Fetch content from CMS/Git
  const content = await fetchPreviewContent(slug);
  
  return json(content, {
    headers: {
      'Cache-Control': 'no-store', // Never cache preview content
      'X-Robots-Tag': 'noindex',
    },
  });
}
```

---

## 6. Multi-Language Content Architecture

### ✅ DO: Implement Proper i18n for Content Sites

```typescript
// src/i18n/config.ts
export const languages = {
  en: 'English',
  es: 'Español',
  fr: 'Français',
  de: 'Deutsch',
  ja: '日本語',
} as const;

export const defaultLang = 'en';

export type Lang = keyof typeof languages;

// src/content/config.ts - Locale-aware collections
const blogCollection = defineCollection({
  loader: glob({ 
    pattern: "**/*.{md,mdx}",
    base: "./src/content/blog",
  }),
  schema: z.object({
    title: z.string(),
    lang: z.enum(['en', 'es', 'fr', 'de', 'ja']).default('en'),
    translationKey: z.string(), // Links translations together
    // ... other fields
  }),
});

// src/pages/[lang]/blog/[...slug].astro
export async function getStaticPaths() {
  const posts = await getCollection('blog');
  
  return posts.map((post) => ({
    params: { 
      lang: post.data.lang,
      slug: post.id.replace(`${post.data.lang}/`, ''),
    },
    props: { post },
  }));
}

// Get translated versions
const translations = await getCollection('blog', ({ data }) => {
  return data.translationKey === post.data.translationKey && 
         data.lang !== post.data.lang;
});
```

### Content Negotiation Middleware

```typescript
// src/middleware/index.ts
import type { MiddlewareHandler } from 'astro';
import { getBestMatchingLanguage } from '@/lib/i18n/negotiation';

export const onRequest: MiddlewareHandler = async (context, next) => {
  const { pathname } = context.url;
  
  // Skip API routes and assets
  if (pathname.startsWith('/api') || pathname.includes('.')) {
    return next();
  }
  
  // Check if language is in URL
  const langMatch = pathname.match(/^\/([a-z]{2})\//);
  if (langMatch) {
    context.locals.lang = langMatch[1];
    return next();
  }
  
  // Auto-detect language
  const acceptLanguage = context.request.headers.get('accept-language');
  const bestLang = getBestMatchingLanguage(acceptLanguage);
  
  // Redirect to localized version
  return context.redirect(`/${bestLang}${pathname}`, 302);
};
```

---

## 7. Performance & Core Web Vitals Optimization

### ✅ DO: Optimize for Perfect Lighthouse Scores

```typescript
// astro.config.mjs - Performance configuration
export default defineConfig({
  build: {
    // Inline critical CSS
    inlineStylesheets: 'auto',
    // Split vendored code
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'utils': ['date-fns', 'clsx'],
        },
      },
    },
  },
  vite: {
    build: {
      // Use Brotli compression
      cssMinify: 'lightningcss',
      modulePreload: {
        polyfill: false, // Modern browsers only
      },
    },
    css: {
      transformer: 'lightningcss',
      lightningcss: {
        targets: {
          chrome: 95, // 2025 baseline
        },
      },
    },
  },
});

// src/components/CriticalCSS.astro
---
// Extract critical CSS for above-the-fold content
const criticalStyles = `
  /* Reset and base styles */
  *, *::before, *::after { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, sans-serif; }
  
  /* Layout critical styles */
  .header { height: 60px; background: var(--header-bg); }
  .hero { min-height: 50vh; }
  
  /* Font loading optimization */
  @font-face {
    font-family: 'Inter';
    src: url('/fonts/inter-var.woff2') format('woff2');
    font-display: swap;
    unicode-range: U+0000-00FF;
  }
`;
---

<style is:inline set:html={criticalStyles}></style>
```

### ✅ DO: Implement Resource Hints

```astro
---
// src/layouts/BaseLayout.astro
import { getEntry } from 'astro:content';

// Preconnect to external domains
const externalDomains = [
  'https://fonts.googleapis.com',
  'https://cdn.example.com',
];

// Prefetch next likely navigation
const currentPost = await getEntry('blog', Astro.params.slug);
const relatedPosts = await getRelatedPosts(currentPost);
---

<head>
  <!-- DNS prefetch for external resources -->
  {externalDomains.map(domain => (
    <link rel="dns-prefetch" href={domain} />
    <link rel="preconnect" href={domain} crossorigin />
  ))}
  
  <!-- Prefetch likely next navigations -->
  {relatedPosts.slice(0, 3).map(post => (
    <link rel="prefetch" href={`/blog/${post.slug}`} />
  ))}
  
  <!-- Preload critical resources -->
  <link rel="preload" href="/fonts/inter-var.woff2" as="font" type="font/woff2" crossorigin />
  
  <!-- Speculation Rules API (Chrome 121+) -->
  <script type="speculationrules">
  {
    "prefetch": [{
      "source": "list",
      "urls": ["/blog", "/about", "/contact"]
    }],
    "prerender": [{
      "source": "list",
      "urls": ["/"]
    }]
  }
  </script>
</head>
```

---

## 8. Edge Deployment & Caching Strategy

### ✅ DO: Deploy to Edge with Smart Caching

```typescript
// astro.config.mjs - Cloudflare Pages deployment
import { defineConfig } from 'astro/config';
import cloudflare from '@astrojs/cloudflare';

export default defineConfig({
  output: 'hybrid',
  adapter: cloudflare({
    mode: 'directory',
    functionPerRoute: true, // Split functions for better cold starts
    routes: {
      strategy: 'include',
      include: ['/api/*', '/search/*'], // Only these need edge functions
    },
  }),
});

// src/pages/api/revalidate.ts - ISR-like revalidation
export const prerender = false;

export async function POST({ request }: APIContext) {
  const { secret, tag } = await request.json();
  
  if (secret !== import.meta.env.REVALIDATION_SECRET) {
    return new Response('Unauthorized', { status: 401 });
  }
  
  // Purge Cloudflare cache by tag
  await fetch(`https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/purge_cache`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${CF_API_TOKEN}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      tags: [tag],
    }),
  });
  
  return new Response('Revalidated', { status: 200 });
}
```

### Cache Headers Strategy

```typescript
// src/middleware/cache.ts
export const cacheRules = {
  // Static assets - 1 year
  '/assets/*': 'public, max-age=31536000, immutable',
  
  // Images - 1 month with revalidation
  '/images/*': 'public, max-age=2592000, stale-while-revalidate=86400',
  
  // Blog posts - 1 hour with background refresh
  '/blog/*': 'public, max-age=3600, stale-while-revalidate=86400',
  
  // Homepage - 5 minutes
  '/': 'public, max-age=300, stale-while-revalidate=600',
  
  // API routes - no cache
  '/api/*': 'no-store',
  
  // Search - cache for 5 minutes
  '/search': 'public, max-age=300',
} as const;

export function getCacheHeaders(pathname: string): string {
  for (const [pattern, headers] of Object.entries(cacheRules)) {
    if (matchesPattern(pathname, pattern)) {
      return headers;
    }
  }
  return 'public, max-age=60'; // Default
}
```

---

## 9. SEO & Structured Data

### ✅ DO: Implement Comprehensive SEO

```astro
---
// src/components/SEO.astro
export interface Props {
  title: string;
  description: string;
  image?: string;
  article?: {
    publishedTime: string;
    modifiedTime?: string;
    author: string;
    section: string;
    tags: string[];
  };
  noindex?: boolean;
  canonical?: string;
}

const {
  title,
  description,
  image = '/og-default.jpg',
  article,
  noindex = false,
  canonical = Astro.url.href,
} = Astro.props;

const ogImage = new URL(image, Astro.site).href;

// Generate structured data
const structuredData = article ? {
  '@context': 'https://schema.org',
  '@type': 'Article',
  headline: title,
  description: description,
  image: ogImage,
  datePublished: article.publishedTime,
  dateModified: article.modifiedTime || article.publishedTime,
  author: {
    '@type': 'Person',
    name: article.author,
  },
  publisher: {
    '@type': 'Organization',
    name: 'Your Site',
    logo: {
      '@type': 'ImageObject',
      url: new URL('/logo.png', Astro.site).href,
    },
  },
  mainEntityOfPage: {
    '@type': 'WebPage',
    '@id': canonical,
  },
} : {
  '@context': 'https://schema.org',
  '@type': 'WebSite',
  name: 'Your Site',
  url: Astro.site?.href,
};
---

<title>{title}</title>
<meta name="description" content={description} />
<link rel="canonical" href={canonical} />

<!-- Open Graph -->
<meta property="og:type" content={article ? 'article' : 'website'} />
<meta property="og:title" content={title} />
<meta property="og:description" content={description} />
<meta property="og:image" content={ogImage} />
<meta property="og:url" content={canonical} />

<!-- Twitter -->
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content={title} />
<meta name="twitter:description" content={description} />
<meta name="twitter:image" content={ogImage} />

{article && (
  <>
    <meta property="article:published_time" content={article.publishedTime} />
    <meta property="article:modified_time" content={article.modifiedTime} />
    <meta property="article:author" content={article.author} />
    <meta property="article:section" content={article.section} />
    {article.tags.map(tag => (
      <meta property="article:tag" content={tag} />
    ))}
  </>
)}

{noindex && <meta name="robots" content="noindex, nofollow" />}

<!-- Structured Data -->
<script type="application/ld+json" set:html={JSON.stringify(structuredData)} />
```

### Dynamic Sitemap Generation

```typescript
// src/pages/sitemap.xml.ts
import { getCollection } from 'astro:content';
import { SitemapStream, streamToPromise } from 'sitemap';
import { Readable } from 'stream';

export async function GET(context: APIContext) {
  const site = context.site;
  const posts = await getCollection('blog', ({ data }) => !data.draft);
  
  const links = [
    { url: '/', changefreq: 'daily', priority: 1.0 },
    { url: '/blog', changefreq: 'daily', priority: 0.8 },
    { url: '/about', changefreq: 'monthly', priority: 0.5 },
    // Dynamic content
    ...posts.map(post => ({
      url: `/blog/${post.slug}`,
      lastmod: post.data.updateDate || post.data.publishDate,
      changefreq: 'weekly' as const,
      priority: 0.7,
      img: post.data.image ? [{
        url: new URL(post.data.image.src, site).href,
        title: post.data.image.alt,
      }] : undefined,
    })),
  ];
  
  const stream = new SitemapStream({ hostname: site?.href });
  const xml = await streamToPromise(Readable.from(links).pipe(stream));
  
  return new Response(xml, {
    headers: {
      'Content-Type': 'application/xml',
      'Cache-Control': 'public, max-age=3600',
    },
  });
}
```

---

## 10. Analytics & Performance Monitoring

### ✅ DO: Implement Privacy-First Analytics

```astro
---
// src/components/Analytics.astro
// Using Fathom/Plausible for privacy-first analytics
const analyticsEndpoint = import.meta.env.PUBLIC_ANALYTICS_ENDPOINT;
const siteId = import.meta.env.PUBLIC_ANALYTICS_SITE_ID;
---

{import.meta.env.PROD && (
  <script define:vars={{ analyticsEndpoint, siteId }}>
    // Minimal analytics script
    (function() {
      const queue = window.fa = window.fa || [];
      
      if (!queue.initialize) {
        queue.methods = ['track', 'set', 'ecommerce'];
        queue.factory = function(method) {
          return function() {
            const args = Array.prototype.slice.call(arguments);
            args.unshift(method);
            queue.push(args);
            return queue;
          };
        };
        
        for (let i = 0; i < queue.methods.length; i++) {
          const method = queue.methods[i];
          queue[method] = queue.factory(method);
        }
        
        queue.load = function() {
          const script = document.createElement('script');
          script.async = true;
          script.src = analyticsEndpoint;
          script.dataset.site = siteId;
          document.head.appendChild(script);
        };
        
        queue.initialize = true;
        queue.load();
      }
      
      // Track page views with View Transitions
      document.addEventListener('astro:page-load', () => {
        if (window.fa) {
          window.fa.track();
        }
      });
      
      // Track Web Vitals
      if ('PerformanceObserver' in window) {
        try {
          const observer = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
              window.fa.track('web-vitals', {
                metric: entry.name,
                value: Math.round(entry.value),
                rating: entry.rating || 'none',
              });
            }
          });
          
          observer.observe({ entryTypes: ['web-vital'] });
        } catch (e) {
          // Silently fail for unsupported browsers
        }
      }
    })();
  </script>
)}
```

### Real User Monitoring (RUM)

```typescript
// src/components/RUM.astro
---
const rumEndpoint = import.meta.env.PUBLIC_RUM_ENDPOINT;
---

<script define:vars={{ rumEndpoint }}>
  // Collect performance metrics
  function collectMetrics() {
    const navigation = performance.getEntriesByType('navigation')[0];
    const paint = performance.getEntriesByType('paint');
    
    const metrics = {
      // Navigation timing
      dns: navigation.domainLookupEnd - navigation.domainLookupStart,
      tcp: navigation.connectEnd - navigation.connectStart,
      ttfb: navigation.responseStart - navigation.requestStart,
      
      // Paint timing
      fcp: paint.find(p => p.name === 'first-contentful-paint')?.startTime,
      lcp: 0, // Will be updated by observer
      
      // Resource timing
      resources: performance.getEntriesByType('resource').map(r => ({
        name: r.name,
        duration: r.duration,
        size: r.transferSize,
        type: r.initiatorType,
      })).filter(r => r.duration > 50), // Only slow resources
      
      // Device info
      connection: navigator.connection?.effectiveType,
      deviceMemory: navigator.deviceMemory,
      url: window.location.pathname,
      referrer: document.referrer,
    };
    
    return metrics;
  }
  
  // Send metrics after page load
  if ('requestIdleCallback' in window) {
    requestIdleCallback(() => {
      const metrics = collectMetrics();
      
      // Use sendBeacon for reliability
      if ('sendBeacon' in navigator) {
        navigator.sendBeacon(rumEndpoint, JSON.stringify(metrics));
      }
    }, { timeout: 5000 });
  }
</script>
```

---

## 11. Content Security & Protection

### ✅ DO: Implement Content Security Policy

```typescript
// src/middleware/security.ts
import type { MiddlewareHandler } from 'astro';

export const securityHeaders: MiddlewareHandler = async (context, next) => {
  const response = await next();
  
  // Content Security Policy
  const csp = [
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: https:",
    "font-src 'self' data:",
    "connect-src 'self' https://analytics.example.com",
    "frame-ancestors 'none'",
    "base-uri 'self'",
    "form-action 'self'",
  ].join('; ');
  
  response.headers.set('Content-Security-Policy', csp);
  response.headers.set('X-Frame-Options', 'DENY');
  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
  response.headers.set('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
  
  return response;
};
```

### Content Theft Protection

```typescript
// src/components/CopyProtection.astro
---
// Disable right-click and text selection on premium content
const isPremiumContent = Astro.props.premium || false;
---

{isPremiumContent && (
  <script>
    // Disable right-click
    document.addEventListener('contextmenu', (e) => {
      if (e.target.closest('.premium-content')) {
        e.preventDefault();
        return false;
      }
    });
    
    // Disable text selection
    document.addEventListener('selectstart', (e) => {
      if (e.target.closest('.premium-content')) {
        e.preventDefault();
        return false;
      }
    });
    
    // Watermark copied text
    document.addEventListener('copy', (e) => {
      const selection = window.getSelection().toString();
      if (selection.length > 50) {
        e.clipboardData.setData('text/plain', 
          `${selection}\n\nSource: ${window.location.href}\n© ${new Date().getFullYear()} Your Site`
        );
        e.preventDefault();
      }
    });
  </script>
)}
```

---

## 12. Testing & Quality Assurance

### ✅ DO: Implement Comprehensive Testing

```typescript
// tests/content.test.ts - Content validation tests
import { expect, test } from '@playwright/test';
import { getCollection } from 'astro:content';

test.describe('Content Integrity', () => {
  test('all blog posts have valid frontmatter', async () => {
    const posts = await getCollection('blog');
    
    for (const post of posts) {
      expect(post.data.title).toBeTruthy();
      expect(post.data.title.length).toBeLessThanOrEqual(60);
      expect(post.data.description.length).toBeBetween(120, 160);
      expect(post.data.publishDate).toBeInstanceOf(Date);
      
      // Check image optimization
      if (post.data.image) {
        expect(post.data.image.src).toMatch(/\.(jpg|jpeg|png|webp|avif)$/);
        expect(post.data.image.alt).toBeTruthy();
      }
    }
  });
  
  test('no broken internal links', async ({ page }) => {
    const posts = await getCollection('blog');
    const brokenLinks = [];
    
    for (const post of posts) {
      await page.goto(`/blog/${post.slug}`);
      
      const links = await page.$$eval('a[href^="/"]', links => 
        links.map(link => link.href)
      );
      
      for (const link of links) {
        const response = await page.request.get(link);
        if (!response.ok()) {
          brokenLinks.push({ post: post.slug, link, status: response.status() });
        }
      }
    }
    
    expect(brokenLinks).toHaveLength(0);
  });
});

// Visual regression testing
test.describe('Visual Regression', () => {
  test('homepage snapshot', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    await expect(page).toHaveScreenshot('homepage.png', {
      fullPage: true,
      animations: 'disabled',
    });
  });
  
  test('blog post layout', async ({ page }) => {
    await page.goto('/blog/sample-post');
    
    // Wait for content to load
    await page.waitForSelector('.prose');
    
    await expect(page.locator('.prose')).toHaveScreenshot('blog-post.png');
  });
});
```

### Performance Budget Testing

```typescript
// tests/performance.test.ts
import { test, expect } from '@playwright/test';

test.describe('Performance Budget', () => {
  test('homepage meets performance budget', async ({ page }) => {
    await page.goto('/');
    
    const metrics = await page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0];
      const fcp = performance.getEntriesByName('first-contentful-paint')[0];
      
      return {
        ttfb: navigation.responseStart - navigation.requestStart,
        fcp: fcp?.startTime || 0,
        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
        jsHeapSize: performance.memory?.usedJSHeapSize || 0,
      };
    });
    
    expect(metrics.ttfb).toBeLessThan(200); // TTFB < 200ms
    expect(metrics.fcp).toBeLessThan(1500); // FCP < 1.5s
    expect(metrics.domContentLoaded).toBeLessThan(2000); // DCL < 2s
    expect(metrics.jsHeapSize).toBeLessThan(50 * 1024 * 1024); // < 50MB
  });
  
  test('blog post bundle size', async ({ page }) => {
    const response = await page.goto('/blog/sample-post');
    const resources = await page.evaluate(() => 
      performance.getEntriesByType('resource').map(r => ({
        name: r.name,
        size: r.transferSize,
        type: r.initiatorType,
      }))
    );
    
    const jsSize = resources
      .filter(r => r.type === 'script')
      .reduce((sum, r) => sum + r.size, 0);
    
    const cssSize = resources
      .filter(r => r.type === 'link' && r.name.includes('.css'))
      .reduce((sum, r) => sum + r.size, 0);
    
    expect(jsSize).toBeLessThan(200 * 1024); // JS < 200KB
    expect(cssSize).toBeLessThan(50 * 1024); // CSS < 50KB
  });
});
```

---

## 13. Migration Strategies

### ✅ DO: Plan for Content Migration

```typescript
// scripts/migrate-content.ts
import { readdir, readFile, writeFile } from 'fs/promises';
import { parse } from 'yaml';
import { remark } from 'remark';
import { visit } from 'unist-util-visit';
import { basename } from 'path';

interface LegacyPost {
  title: string;
  date: string;
  categories: string[];
  content: string;
}

async function migrateLegacyContent() {
  const legacyPosts = await readdir('./legacy-content');
  
  for (const file of legacyPosts) {
    const content = await readFile(`./legacy-content/${file}`, 'utf-8');
    const { data, content: body } = parseFrontmatter(content);
    
    // Transform legacy frontmatter
    const newFrontmatter = {
      title: data.title,
      description: extractDescription(body),
      publishDate: new Date(data.date).toISOString(),
      updateDate: data.modified ? new Date(data.modified).toISOString() : undefined,
      author: data.author || 'legacy-author',
      category: mapLegacyCategory(data.categories?.[0]),
      tags: data.tags || data.categories || [],
      redirectFrom: [`/${data.slug}`, `/${data.id}.html`], // Old URLs
      draft: false,
    };
    
    // Transform content
    const transformedBody = await transformContent(body);
    
    // Write new content file
    const slug = generateSlug(data.title, data.date);
    const newContent = `---
${stringify(newFrontmatter)}
---

${transformedBody}`;
    
    await writeFile(
      `./src/content/blog/${slug}.mdx`,
      newContent,
      'utf-8'
    );
  }
}

async function transformContent(content: string) {
  const processor = remark()
    .use(transformWordPressShortcodes)
    .use(transformImagePaths)
    .use(addImageCaptions);
  
  const result = await processor.process(content);
  return result.toString();
}

// Transform WordPress-style shortcodes to MDX components
function transformWordPressShortcodes() {
  return (tree: any) => {
    visit(tree, 'text', (node, index, parent) => {
      const shortcodeRegex = /\[(\w+)([^\]]*)\](.*?)\[\/\1\]/g;
      const matches = [...node.value.matchAll(shortcodeRegex)];
      
      if (matches.length > 0) {
        const newNodes = [];
        let lastIndex = 0;
        
        for (const match of matches) {
          // Add text before shortcode
          if (match.index > lastIndex) {
            newNodes.push({
              type: 'text',
              value: node.value.slice(lastIndex, match.index),
            });
          }
          
          // Transform shortcode to MDX component
          const [, name, attrs, content] = match;
          newNodes.push({
            type: 'mdxJsxFlowElement',
            name: mapShortcodeToComponent(name),
            attributes: parseShortcodeAttributes(attrs),
            children: [{ type: 'text', value: content }],
          });
          
          lastIndex = match.index + match[0].length;
        }
        
        // Add remaining text
        if (lastIndex < node.value.length) {
          newNodes.push({
            type: 'text',
            value: node.value.slice(lastIndex),
          });
        }
        
        parent.children.splice(index, 1, ...newNodes);
      }
    });
  };
}
```

### URL Redirect Management

```typescript
// src/middleware/redirects.ts
import type { MiddlewareHandler } from 'astro';
import redirectsData from '@/data/redirects.json';

// Build redirect map at startup
const redirectMap = new Map(
  redirectsData.map(r => [r.from, { to: r.to, status: r.status || 301 }])
);

export const handleRedirects: MiddlewareHandler = async (context, next) => {
  const { pathname } = context.url;
  
  // Check for exact match
  const redirect = redirectMap.get(pathname);
  if (redirect) {
    return context.redirect(redirect.to, redirect.status);
  }
  
  // Check for pattern matches (e.g., /blog/2023/01/post -> /blog/post)
  for (const [pattern, redirect] of redirectMap.entries()) {
    if (pattern.includes('*')) {
      const regex = new RegExp(pattern.replace('*', '(.*)'));
      const match = pathname.match(regex);
      if (match) {
        const to = redirect.to.replace('$1', match[1]);
        return context.redirect(to, redirect.status);
      }
    }
  }
  
  return next();
};
```

---

## 14. CI/CD & Deployment Pipeline

### ✅ DO: Implement Comprehensive CI/CD

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Type check
        run: npm run typecheck
      
      - name: Lint
        run: npm run lint
      
      - name: Unit tests
        run: npm run test:unit
      
      - name: Build
        run: npm run build
        env:
          PUBLIC_SITE_URL: ${{ vars.SITE_URL }}
      
      - name: E2E tests
        run: npm run test:e2e
      
      - name: Lighthouse CI
        run: |
          npm install -g @lhci/cli
          lhci autorun
        env:
          LHCI_GITHUB_APP_TOKEN: ${{ secrets.LHCI_GITHUB_APP_TOKEN }}

  deploy-preview:
    needs: test
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to Vercel
        run: |
          npm i -g vercel
          vercel pull --yes --environment=preview --token=${{ secrets.VERCEL_TOKEN }}
          vercel build --token=${{ secrets.VERCEL_TOKEN }}
          vercel deploy --prebuilt --token=${{ secrets.VERCEL_TOKEN }}

  deploy-production:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to Cloudflare Pages
        env:
          CLOUDFLARE_API_TOKEN: ${{ secrets.CF_API_TOKEN }}
          CLOUDFLARE_ACCOUNT_ID: ${{ secrets.CF_ACCOUNT_ID }}
        run: |
          npm run build
          npx wrangler pages deploy dist \
            --project-name=my-site \
            --branch=main
```

### Automated Content Validation

```typescript
// scripts/validate-content.ts
import { glob } from 'glob';
import { readFile } from 'fs/promises';
import { parse } from 'yaml';
import { validateSchema } from '@/content/schemas';

async function validateAllContent() {
  const files = await glob('src/content/**/*.{md,mdx}');
  const errors = [];
  
  for (const file of files) {
    try {
      const content = await readFile(file, 'utf-8');
      const { data } = parseFrontmatter(content);
      
      // Validate against schema
      const result = validateSchema(file, data);
      if (!result.success) {
        errors.push({ file, errors: result.errors });
      }
      
      // Check for common issues
      if (data.title?.length > 60) {
        errors.push({ file, error: 'Title too long for SEO' });
      }
      
      if (!data.description || data.description.length < 120) {
        errors.push({ file, error: 'Description too short' });
      }
      
      // Validate images exist
      if (data.image?.src) {
        const imagePath = resolve('./public', data.image.src);
        if (!existsSync(imagePath)) {
          errors.push({ file, error: `Image not found: ${data.image.src}` });
        }
      }
    } catch (e) {
      errors.push({ file, error: e.message });
    }
  }
  
  if (errors.length > 0) {
    console.error('Content validation failed:');
    errors.forEach(e => console.error(`- ${e.file}: ${e.error}`));
    process.exit(1);
  }
  
  console.log(`✓ Validated ${files.length} content files`);
}
```

---

## 15. Monitoring & Observability

### ✅ DO: Implement Comprehensive Monitoring

```typescript
// src/lib/monitoring/sentry.ts
import * as Sentry from '@sentry/astro';

export function initSentry() {
  Sentry.init({
    dsn: import.meta.env.SENTRY_DSN,
    environment: import.meta.env.MODE,
    
    integrations: [
      // Capture console errors
      Sentry.captureConsoleIntegration({
        levels: ['error', 'warn'],
      }),
      
      // Track performance
      Sentry.browserTracingIntegration({
        tracingOrigins: ['localhost', 'yoursite.com'],
        routingInstrumentation: Sentry.reactRouterV6Instrumentation(
          React.useEffect,
          useLocation,
          useNavigationType,
          createRoutesFromChildren,
          matchRoutes
        ),
      }),
      
      // Session replay for errors
      Sentry.replayIntegration({
        maskAllText: true,
        blockAllMedia: true,
        sampleRate: 0.1,
        errorSampleRate: 1.0,
      }),
    ],
    
    // Performance monitoring
    tracesSampleRate: import.meta.env.PROD ? 0.1 : 1.0,
    
    // Release tracking
    release: import.meta.env.VITE_RELEASE_VERSION,
    
    // Filter out known issues
    beforeSend(event, hint) {
      // Ignore browser extension errors
      if (event.exception?.values?.[0]?.value?.includes('extension://')) {
        return null;
      }
      
      // Ignore network errors for analytics
      if (event.request?.url?.includes('analytics')) {
        return null;
      }
      
      return event;
    },
  });
}

// Error boundary for React components
export function ErrorBoundary({ children }: { children: React.ReactNode }) {
  return (
    <Sentry.ErrorBoundary
      fallback={({ error, resetError }) => (
        <div className="error-boundary">
          <h2>Something went wrong</h2>
          <details>
            <summary>Error details</summary>
            <pre>{error.message}</pre>
          </details>
          <button onClick={resetError}>Try again</button>
        </div>
      )}
      showDialog
    >
      {children}
    </Sentry.ErrorBoundary>
  );
}
```

### Custom Metrics Collection

```typescript
// src/lib/monitoring/metrics.ts
interface Metric {
  name: string;
  value: number;
  tags?: Record<string, string>;
  timestamp?: number;
}

class MetricsCollector {
  private buffer: Metric[] = [];
  private endpoint: string;
  
  constructor(endpoint: string) {
    this.endpoint = endpoint;
    
    // Flush metrics every 10 seconds
    setInterval(() => this.flush(), 10000);
    
    // Flush on page unload
    if (typeof window !== 'undefined') {
      window.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden') {
          this.flush();
        }
      });
    }
  }
  
  track(name: string, value: number, tags?: Record<string, string>) {
    this.buffer.push({
      name,
      value,
      tags,
      timestamp: Date.now(),
    });
    
    // Flush if buffer is getting large
    if (this.buffer.length > 100) {
      this.flush();
    }
  }
  
  private async flush() {
    if (this.buffer.length === 0) return;
    
    const metrics = [...this.buffer];
    this.buffer = [];
    
    try {
      if ('sendBeacon' in navigator) {
        navigator.sendBeacon(this.endpoint, JSON.stringify(metrics));
      } else {
        await fetch(this.endpoint, {
          method: 'POST',
          body: JSON.stringify(metrics),
          keepalive: true,
        });
      }
    } catch (error) {
      // Re-add metrics to buffer on failure
      this.buffer.unshift(...metrics);
    }
  }
}

export const metrics = new MetricsCollector('/api/metrics');

// Usage examples
metrics.track('page_view', 1, { page: '/blog' });
metrics.track('search_performed', 1, { query_length: query.length });
metrics.track('content_engagement', scrollDepth, { 
  content_type: 'blog',
  read_time: timeOnPage,
});
```

---

## Conclusion

This guide provides a comprehensive blueprint for building modern content-heavy sites with Remix 2 and Astro 4 in 2025. The key principles are:

1. **Choose the right tool**: Astro for static content, Remix for dynamic features
2. **Optimize aggressively**: Every millisecond matters for content sites
3. **Cache strategically**: Use edge caching and smart invalidation
4. **Monitor everything**: You can't improve what you don't measure
5. **Test comprehensively**: Content integrity is as important as code quality

Remember that the best architecture is one that serves your specific content needs while maintaining excellent performance and developer experience. Start with Astro's static generation for most content, add Remix for dynamic features, and continuously optimize based on real user metrics.

For updates and community discussion, join the Astro Discord (#content-sites) and Remix Discord (#performance) channels.