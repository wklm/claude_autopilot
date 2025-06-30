# The Definitive Guide to React Native with Fabric Architecture (Mid-2025)

This guide synthesizes modern best practices for building performant, secure, and maintainable cross-platform mobile applications with React Native's Fabric architecture. It moves beyond basic tutorials to provide production-grade architectural patterns for apps that truly feel native.

### Prerequisites & Configuration
Ensure your project uses **React Native 0.80+**, **React 19.1+**, **TypeScript 5.8+**, and **Node.js 22+**. The Fabric architecture is now stable and should be the default for all new projects.

Initialize with the latest template and enable modern features:

```bash
# Create a new project with TypeScript template
npx @react-native-community/cli@latest init MyApp --template react-native-template-typescript

# Enable Fabric and TurboModules
cd MyApp
npx react-native-fabric enable
```

Configure your `react-native.config.js` for optimal performance:

```javascript
// react-native.config.js
module.exports = {
  project: {
    ios: {
      automaticPodsInstallation: true,
    },
    android: {
      enableHermes: true, // Hermes is now default and highly optimized
      enableFabric: true,
      enableTurboModules: true,
    },
  },
  // New architecture flags
  experimental: {
    reactCompiler: true, // Enable React Compiler for auto-optimization
    asyncBatching: true, // Better performance for async operations
    concurrentRoot: true, // React 19 concurrent features
  },
};
```

### React Native 0.80+ Enhancements

React Native 0.80 brings significant improvements that enhance both developer experience and app performance:

**JavaScript Deep Imports Deprecated**: Direct imports from subpaths are now formally deprecated. Update your imports:
```typescript
// ❌ Old way - Deep import
import {Alert} from 'react-native/Libraries/Alert/Alert';

// ✅ New way - Root import
import {Alert} from 'react-native';
```

**Strict TypeScript API**: A new opt-in TypeScript API provides better type safety and is generated directly from source code:
```json
// tsconfig.json
{
  "compilerOptions": {
    "types": ["react-native/types/strict"]
  }
}
```

**Improved Metro Performance**: Metro 0.82 delivers up to 3x faster cold starts with deferred hashing, especially beneficial for large projects and monorepos.

**Android Startup Optimization**: Uncompressed JavaScript bundles reduce time-to-interactive by up to 400ms (12% improvement) on mid-range devices. This is enabled by default but can be configured:
```gradle
// app/build.gradle
react {
  enableBundleCompression = false // Default: faster startup, larger storage
}
```

**iOS Prebuilt Dependencies**: Experimental support for prebuilt React Native dependencies reduces initial iOS build times by ~12%:
```bash
# Enable prebuilt dependencies
RCT_USE_RN_DEP=1 bundle exec pod install
```

---

## 1. Project Structure & Architecture

A well-organized structure is crucial for maintaining large React Native applications across platforms. Use a feature-based architecture with clear separation of concerns.

### ✅ DO: Use a Scalable Feature-Based Structure

```
/src
├── app/                    # Navigation and app-level setup
│   ├── navigation/         # Navigation configuration
│   │   ├── RootNavigator.tsx
│   │   ├── AuthNavigator.tsx
│   │   └── MainNavigator.tsx
│   ├── providers/          # Global providers (theme, auth, etc.)
│   └── App.tsx            # Root component
├── features/              # Feature-based modules
│   ├── auth/              # Authentication feature
│   │   ├── screens/       # Screen components
│   │   ├── components/    # Feature-specific components
│   │   ├── hooks/         # Feature-specific hooks
│   │   ├── services/      # API calls and business logic
│   │   ├── stores/        # Feature state (Zustand/Redux)
│   │   └── types.ts       # TypeScript definitions
│   └── dashboard/         # Dashboard feature
├── components/            # Shared UI components
│   ├── ui/                # Atomic design system components
│   │   ├── Button/
│   │   ├── Input/
│   │   └── Card/
│   └── layout/            # Layout components
├── native/                # Platform-specific code
│   ├── modules/           # Native modules (TurboModules)
│   │   ├── BiometricAuth/
│   │   └── SecureStorage/
│   └── components/        # Native UI components
├── services/              # Global services
│   ├── api/               # API client and configuration
│   ├── storage/           # Encrypted storage abstraction
│   └── analytics/         # Analytics abstraction
├── hooks/                 # Global custom hooks
├── utils/                 # Utility functions
├── types/                 # Global TypeScript types
└── constants/             # App-wide constants
```

### ✅ DO: Separate Platform-Specific Code Properly

Use platform-specific file extensions for cleaner code organization:

```typescript
// components/ui/Button/Button.tsx - Shared logic
import { ButtonProps } from './types';
import { useButtonLogic } from './useButtonLogic';

export { default as Button } from './Button.native';

// components/ui/Button/Button.ios.tsx - iOS specific
import React from 'react';
import { Pressable, Text } from 'react-native';
import Haptics from 'react-native-haptic-feedback';

export default function Button({ onPress, children, haptic = true }: ButtonProps) {
  const handlePress = () => {
    if (haptic) {
      Haptics.trigger('impactLight', {
        enableVibrateFallback: true,
        ignoreAndroidSystemSettings: false,
      });
    }
    onPress();
  };

  return (
    <Pressable 
      onPress={handlePress}
      style={({ pressed }) => ({
        opacity: pressed ? 0.7 : 1,
        // iOS-specific shadow
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
      })}
    >
      <Text>{children}</Text>
    </Pressable>
  );
}

// components/ui/Button/Button.android.tsx - Android specific
import React from 'react';
import { Pressable, Text } from 'react-native';

export default function Button({ onPress, children }: ButtonProps) {
  return (
    <Pressable 
      onPress={onPress}
      android_ripple={{ color: 'rgba(0, 0, 0, 0.1)' }}
      style={{
        elevation: 4, // Android shadow
      }}
    >
      <Text>{children}</Text>
    </Pressable>
  );
}
```

---

## 2. Fabric Architecture: Leveraging the New Renderer

The Fabric architecture eliminates the bridge, providing synchronous access to the UI thread and enabling truly responsive interfaces.

### ✅ DO: Use Synchronous Measure and Layout Operations

With Fabric, you can now perform synchronous measurements without performance penalties:

```typescript
// components/AutoHeightTextInput.tsx
import React, { useRef, useState } from 'react';
import { TextInput, View, LayoutChangeEvent } from 'react-native';

export function AutoHeightTextInput() {
  const inputRef = useRef<TextInput>(null);
  const [height, setHeight] = useState(40);

  const handleContentSizeChange = (event: any) => {
    // Fabric allows synchronous measure without bridge delays
    inputRef.current?.measure((x, y, width, height) => {
      setHeight(Math.min(height, 120)); // Cap at 120px
    });
  };

  return (
    <View style={{ minHeight: height }}>
      <TextInput
        ref={inputRef}
        multiline
        onContentSizeChange={handleContentSizeChange}
        style={{ flex: 1 }}
      />
    </View>
  );
}
```

### ✅ DO: Leverage Concurrent Features for Better UX

React 19's concurrent features work seamlessly with Fabric for responsive interactions:

```typescript
// features/search/components/SearchResults.tsx
import React, { useState, useTransition, useDeferredValue } from 'react';
import { FlatList, TextInput, ActivityIndicator } from 'react-native';

export function SearchResults() {
  const [query, setQuery] = useState('');
  const [isPending, startTransition] = useTransition();
  const deferredQuery = useDeferredValue(query);

  const handleSearch = (text: string) => {
    setQuery(text);
    
    // Non-urgent update that won't block typing
    startTransition(() => {
      // Expensive search operation
      performSearch(text);
    });
  };

  return (
    <>
      <TextInput
        value={query}
        onChangeText={handleSearch}
        placeholder="Search..."
      />
      {isPending && <ActivityIndicator />}
      <FlatList
        data={searchResults}
        // List updates are deferred for smooth typing
        extraData={deferredQuery}
        renderItem={({ item }) => <SearchResultItem item={item} />}
      />
    </>
  );
}
```

---

## 3. TurboModules: Type-Safe Native Integration

TurboModules provide lazy-loading, type-safe native module access with significantly better performance than the old architecture.

### ✅ DO: Create Type-Safe TurboModules

**1. Define the TypeScript Interface**

```typescript
// native/modules/BiometricAuth/NativeBiometricAuth.ts
import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

export interface Spec extends TurboModule {
  isSupported(): Promise<boolean>;
  authenticate(reason: string): Promise<boolean>;
  getBiometryType(): Promise<'FaceID' | 'TouchID' | 'Fingerprint' | 'Face' | 'None'>;
}

export default TurboModuleRegistry.getEnforcing<Spec>('BiometricAuth');
```

**2. Implement Native Code (iOS)**

```objective-c
// ios/BiometricAuth.mm
#import "BiometricAuth.h"
#import <LocalAuthentication/LocalAuthentication.h>

@implementation BiometricAuth

RCT_EXPORT_MODULE()

- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params
{
  return std::make_shared<facebook::react::NativeBiometricAuthSpecJSI>(params);
}

- (void)authenticate:(NSString *)reason
               resolve:(RCTPromiseResolveBlock)resolve
               reject:(RCTPromiseRejectBlock)reject
{
  dispatch_async(dispatch_get_main_queue(), ^{
    LAContext *context = [[LAContext alloc] init];
    NSError *error = nil;
    
    if ([context canEvaluatePolicy:LAPolicyDeviceOwnerAuthenticationWithBiometrics error:&error]) {
      [context evaluatePolicy:LAPolicyDeviceOwnerAuthenticationWithBiometrics
              localizedReason:reason
                        reply:^(BOOL success, NSError *error) {
        if (success) {
          resolve(@YES);
        } else {
          reject(@"auth_failed", @"Authentication failed", error);
        }
      }];
    } else {
      reject(@"not_available", @"Biometric authentication not available", error);
    }
  });
}

@end
```

**3. Create a Clean JavaScript API**

```typescript
// services/biometric/BiometricService.ts
import NativeBiometricAuth from '@/native/modules/BiometricAuth/NativeBiometricAuth';
import { Platform } from 'react-native';

class BiometricService {
  private static instance: BiometricService;

  static getInstance(): BiometricService {
    if (!BiometricService.instance) {
      BiometricService.instance = new BiometricService();
    }
    return BiometricService.instance;
  }

  async authenticate(reason?: string): Promise<boolean> {
    try {
      const defaultReason = Platform.select({
        ios: 'Authenticate to access your account',
        android: 'Confirm your identity',
        default: 'Please authenticate',
      });

      return await NativeBiometricAuth.authenticate(reason || defaultReason);
    } catch (error) {
      console.error('Biometric authentication failed:', error);
      return false;
    }
  }

  async checkAvailability(): Promise<{
    isAvailable: boolean;
    biometryType: string;
  }> {
    const isSupported = await NativeBiometricAuth.isSupported();
    const biometryType = isSupported 
      ? await NativeBiometricAuth.getBiometryType() 
      : 'None';

    return {
      isAvailable: isSupported,
      biometryType,
    };
  }
}

export const biometricService = BiometricService.getInstance();
```

---

## 4. State Management: Modern Patterns for React Native

State management in React Native requires careful consideration of performance, persistence, and platform differences.

### ✅ DO: Use Zustand with MMKV for Persistent State

MMKV provides the fastest persistent storage for React Native, perfect for Zustand persistence:

```typescript
// stores/createStore.ts
import { create } from 'zustand';
import { createJSONStorage, persist, StateStorage } from 'zustand/middleware';
import { MMKV } from 'react-native-mmkv';

const storage = new MMKV({
  id: 'app-storage',
  encryptionKey: 'your-encryption-key', // Use react-native-keychain for this
});

const mmkvStorage: StateStorage = {
  getItem: (name) => {
    const value = storage.getString(name);
    return value ?? null;
  },
  setItem: (name, value) => {
    storage.set(name, value);
  },
  removeItem: (name) => {
    storage.delete(name);
  },
};

// stores/authStore.ts
interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  login: (credentials: LoginCredentials) => Promise<void>;
  logout: () => void;
  refreshToken: () => Promise<void>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      isAuthenticated: false,

      login: async (credentials) => {
        try {
          const { user, token, refreshToken } = await authService.login(credentials);
          
          // Store tokens securely
          await secureStorage.setItem('accessToken', token);
          await secureStorage.setItem('refreshToken', refreshToken);
          
          set({ user, isAuthenticated: true });
        } catch (error) {
          throw new AuthError('Login failed', error);
        }
      },

      logout: () => {
        secureStorage.removeItem('accessToken');
        secureStorage.removeItem('refreshToken');
        set({ user: null, isAuthenticated: false });
      },

      refreshToken: async () => {
        const refreshToken = await secureStorage.getItem('refreshToken');
        if (!refreshToken) {
          get().logout();
          return;
        }

        try {
          const { token } = await authService.refresh(refreshToken);
          await secureStorage.setItem('accessToken', token);
        } catch {
          get().logout();
        }
      },
    }),
    {
      name: 'auth-storage',
      storage: createJSONStorage(() => mmkvStorage),
      partialize: (state) => ({ user: state.user }), // Only persist user data
    }
  )
);
```

### ✅ DO: Implement Optimistic Updates for Better UX

Mobile users expect instant feedback. Implement optimistic updates with rollback:

```typescript
// features/posts/stores/postsStore.ts
interface PostsState {
  posts: Post[];
  createPost: (content: string) => Promise<void>;
  likePost: (postId: string) => Promise<void>;
}

export const usePostsStore = create<PostsState>((set, get) => ({
  posts: [],

  createPost: async (content: string) => {
    // Generate temporary ID
    const tempId = `temp_${Date.now()}`;
    const optimisticPost: Post = {
      id: tempId,
      content,
      author: getCurrentUser(),
      createdAt: new Date().toISOString(),
      isOptimistic: true,
    };

    // Optimistically add the post
    set((state) => ({
      posts: [optimisticPost, ...state.posts],
    }));

    try {
      // Make API call
      const createdPost = await api.posts.create({ content });
      
      // Replace optimistic post with real one
      set((state) => ({
        posts: state.posts.map((post) =>
          post.id === tempId ? createdPost : post
        ),
      }));
    } catch (error) {
      // Rollback on failure
      set((state) => ({
        posts: state.posts.filter((post) => post.id !== tempId),
      }));
      
      // Show error notification
      notificationService.error('Failed to create post');
      throw error;
    }
  },

  likePost: async (postId: string) => {
    // Store previous state for rollback
    const previousPosts = get().posts;
    
    // Optimistic update
    set((state) => ({
      posts: state.posts.map((post) =>
        post.id === postId
          ? { ...post, liked: true, likeCount: post.likeCount + 1 }
          : post
      ),
    }));

    try {
      await api.posts.like(postId);
    } catch (error) {
      // Rollback
      set({ posts: previousPosts });
      notificationService.error('Failed to like post');
    }
  },
}));
```

---

## 5. Navigation: Type-Safe and Performant

Use React Navigation 7 with proper TypeScript configuration for type-safe navigation.

### ✅ DO: Define Typed Navigation

```typescript
// types/navigation.ts
import { NavigatorScreenParams } from '@react-navigation/native';

export type RootStackParamList = {
  Auth: NavigatorScreenParams<AuthStackParamList>;
  Main: NavigatorScreenParams<MainTabParamList>;
  Modal: { type: 'info' | 'error'; message: string };
};

export type AuthStackParamList = {
  Login: undefined;
  Register: undefined;
  ForgotPassword: { email?: string };
};

export type MainTabParamList = {
  Home: undefined;
  Profile: { userId: string };
  Settings: undefined;
};

// Type helpers
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { CompositeScreenProps } from '@react-navigation/native';
import { BottomTabScreenProps } from '@react-navigation/bottom-tabs';

export type RootStackScreenProps<T extends keyof RootStackParamList> =
  NativeStackScreenProps<RootStackParamList, T>;

export type AuthStackScreenProps<T extends keyof AuthStackParamList> =
  CompositeScreenProps
    NativeStackScreenProps<AuthStackParamList, T>,
    RootStackScreenProps<keyof RootStackParamList>
  >;

export type MainTabScreenProps<T extends keyof MainTabParamList> =
  CompositeScreenProps
    BottomTabScreenProps<MainTabParamList, T>,
    RootStackScreenProps<keyof RootStackParamList>
  >;

// Global type augmentation
declare global {
  namespace ReactNavigation {
    interface RootParamList extends RootStackParamList {}
  }
}
```

### ✅ DO: Implement Deep Linking with Validation

```typescript
// app/navigation/linking.ts
import { LinkingOptions } from '@react-navigation/native';
import { Linking } from 'react-native';
import branch from 'react-native-branch';

const linking: LinkingOptions<RootStackParamList> = {
  prefixes: [
    'myapp://',
    'https://myapp.com',
    'https://link.myapp.com', // Branch.io links
  ],
  
  async getInitialURL() {
    // Check if app was opened from a deep link
    const url = await Linking.getInitialURL();
    if (url) return url;

    // Check for Branch deferred deep link
    const branchParams = await branch.getLatestReferringParams();
    if (branchParams?.['+url']) {
      return branchParams['+url'];
    }

    return null;
  },

  subscribe(listener) {
    // Listen to incoming links from deep linking
    const linkingSubscription = Linking.addEventListener('url', ({ url }) => {
      listener(url);
    });

    // Listen to Branch links
    const branchSubscription = branch.subscribe(({ error, params }) => {
      if (!error && params?.['+url']) {
        listener(params['+url']);
      }
    });

    return () => {
      linkingSubscription.remove();
      branchSubscription();
    };
  },

  config: {
    screens: {
      Main: {
        screens: {
          Profile: {
            path: 'user/:userId',
            parse: {
              userId: (userId: string) => {
                // Validate userId format
                if (!/^[0-9a-fA-F]{24}$/.test(userId)) {
                  throw new Error('Invalid user ID format');
                }
                return userId;
              },
            },
          },
        },
      },
      Auth: {
        screens: {
          ForgotPassword: {
            path: 'reset-password',
            parse: {
              email: (email: string) => decodeURIComponent(email),
            },
          },
        },
      },
    },
  },
};
```

---

## 6. Performance Optimization: 60 FPS on All Devices

Mobile performance is critical. Users expect smooth 60 FPS animations and instant responses.

### ✅ DO: Use the React Compiler for Automatic Optimization

With React 19's compiler enabled, you get automatic memoization without manual `memo`, `useMemo`, or `useCallback`:

```typescript
// Before: Manual optimization needed
const ExpensiveList = memo(({ data, onItemPress }: Props) => {
  const processedData = useMemo(() => processData(data), [data]);
  const handlePress = useCallback((id: string) => onItemPress(id), [onItemPress]);
  
  return <FlatList data={processedData} />;
});

// After: Compiler handles optimization
function ExpensiveList({ data, onItemPress }: Props) {
  const processedData = processData(data); // Automatically memoized
  const handlePress = (id: string) => onItemPress(id); // Automatically stable
  
  return <FlatList data={processedData} />;
}
```

### ✅ DO: Implement Virtualization Properly

```typescript
// components/OptimizedList.tsx
import React, { useCallback } from 'react';
import { FlatList, View, Text, ListRenderItem } from 'react-native';
import { FlashList } from '@shopify/flash-list';

interface OptimizedListProps<T> {
  data: T[];
  estimatedItemSize?: number;
}

export function OptimizedList<T extends { id: string }>({ 
  data, 
  estimatedItemSize = 50 
}: OptimizedListProps<T>) {
  // Use FlashList for better performance
  const renderItem: ListRenderItem<T> = useCallback(({ item }) => (
    <View style={{ height: estimatedItemSize }}>
      <Text>{item.id}</Text>
    </View>
  ), [estimatedItemSize]);

  const keyExtractor = useCallback((item: T) => item.id, []);
  
  return (
    <FlashList
      data={data}
      renderItem={renderItem}
      keyExtractor={keyExtractor}
      estimatedItemSize={estimatedItemSize}
      // Performance optimizations
      removeClippedSubviews={true}
      maxToRenderPerBatch={10}
      updateCellsBatchingPeriod={50}
      windowSize={10}
      initialNumToRender={10}
      // Prevent content jumps
      maintainVisibleContentPosition={{
        minIndexForVisible: 0,
      }}
      // Draw distance for smoother scrolling
      drawDistance={250}
    />
  );
}
```

### ✅ DO: Optimize Images with Intelligent Loading

```typescript
// components/OptimizedImage.tsx
import React, { useState, useEffect } from 'react';
import { View, Image, ActivityIndicator } from 'react-native';
import FastImage from 'react-native-fast-image';
import { useNetInfo } from '@react-native-community/netinfo';

interface OptimizedImageProps {
  source: { uri: string };
  style?: any;
  placeholder?: string;
}

export function OptimizedImage({ source, style, placeholder }: OptimizedImageProps) {
  const netInfo = useNetInfo();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  // Determine quality based on connection
  const getImageQuality = () => {
    switch (netInfo.type) {
      case 'cellular':
        return netInfo.details?.cellularGeneration === '2g' ? 'low' : 'medium';
      case 'wifi':
        return 'high';
      default:
        return 'medium';
    }
  };

  const imageUri = `${source.uri}?quality=${getImageQuality()}`;

  return (
    <View style={style}>
      <FastImage
        style={{ flex: 1 }}
        source={{
          uri: imageUri,
          priority: FastImage.priority.normal,
          cache: FastImage.cacheControl.immutable,
        }}
        resizeMode={FastImage.resizeMode.cover}
        onLoadStart={() => setLoading(true)}
        onLoadEnd={() => setLoading(false)}
        onError={() => {
          setError(true);
          setLoading(false);
        }}
      />
      
      {loading && (
        <View style={{ position: 'absolute', inset: 0, justifyContent: 'center', alignItems: 'center' }}>
          <ActivityIndicator />
        </View>
      )}
      
      {error && placeholder && (
        <Image source={{ uri: placeholder }} style={{ flex: 1 }} />
      )}
    </View>
  );
}
```

---

## 7. Animations: Smooth 60 FPS with Reanimated 3

Reanimated 3 with Fabric provides butter-smooth animations that run on the UI thread.

### ✅ DO: Use Gesture-Driven Animations

```typescript
// components/SwipeableCard.tsx
import React from 'react';
import { View, Text } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  interpolate,
  runOnJS,
  useAnimatedGestureHandler,
  withTiming,
} from 'react-native-reanimated';
import { PanGestureHandler, PanGestureHandlerGestureEvent } from 'react-native-gesture-handler';

interface SwipeableCardProps {
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  children: React.ReactNode;
}

const SWIPE_THRESHOLD = 120;

export function SwipeableCard({ onSwipeLeft, onSwipeRight, children }: SwipeableCardProps) {
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);

  const gestureHandler = useAnimatedGestureHandler<PanGestureHandlerGestureEvent>({
    onActive: (event) => {
      translateX.value = event.translationX;
      translateY.value = event.translationY;
    },
    onEnd: () => {
      const shouldDismissRight = translateX.value > SWIPE_THRESHOLD;
      const shouldDismissLeft = translateX.value < -SWIPE_THRESHOLD;

      if (shouldDismissRight) {
        translateX.value = withTiming(500);
        if (onSwipeRight) runOnJS(onSwipeRight)();
      } else if (shouldDismissLeft) {
        translateX.value = withTiming(-500);
        if (onSwipeLeft) runOnJS(onSwipeLeft)();
      } else {
        translateX.value = withSpring(0);
        translateY.value = withSpring(0);
      }
    },
  });

  const animatedStyle = useAnimatedStyle(() => {
    const rotate = interpolate(
      translateX.value,
      [-200, 0, 200],
      [-15, 0, 15]
    );

    return {
      transform: [
        { translateX: translateX.value },
        { translateY: translateY.value },
        { rotate: `${rotate}deg` },
      ],
    };
  });

  const likeOpacity = useAnimatedStyle(() => ({
    opacity: interpolate(translateX.value, [0, 100], [0, 1]),
  }));

  const nopeOpacity = useAnimatedStyle(() => ({
    opacity: interpolate(translateX.value, [-100, 0], [1, 0]),
  }));

  return (
    <PanGestureHandler onGestureEvent={gestureHandler}>
      <Animated.View style={[{ position: 'relative' }, animatedStyle]}>
        {children}
        
        <Animated.View
          style={[
            { position: 'absolute', top: 20, right: 20 },
            likeOpacity,
          ]}
        >
          <Text style={{ color: 'green', fontSize: 24, fontWeight: 'bold' }}>
            LIKE
          </Text>
        </Animated.View>
        
        <Animated.View
          style={[
            { position: 'absolute', top: 20, left: 20 },
            nopeOpacity,
          ]}
        >
          <Text style={{ color: 'red', fontSize: 24, fontWeight: 'bold' }}>
            NOPE
          </Text>
        </Animated.View>
      </Animated.View>
    </PanGestureHandler>
  );
}
```

---

## 8. Network Layer: Type-Safe API Integration

Build a robust, type-safe API layer with proper error handling and offline support.

### ✅ DO: Create a Centralized API Client

```typescript
// services/api/client.ts
import axios, { AxiosInstance, AxiosError } from 'axios';
import NetInfo from '@react-native-community/netinfo';
import { secureStorage } from '@/services/storage';
import { queryClient } from '@/services/query';

class ApiClient {
  private client: AxiosInstance;
  private refreshPromise: Promise<string> | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: Config.API_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      async (config) => {
        // Check network connectivity
        const netInfo = await NetInfo.fetch();
        if (!netInfo.isConnected) {
          throw new NetworkError('No internet connection');
        }

        // Add auth token
        const token = await secureStorage.getItem('accessToken');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }

        // Add request ID for tracking
        config.headers['X-Request-ID'] = generateRequestId();

        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as any;

        // Handle 401 - Token refresh
        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;

          try {
            const newToken = await this.refreshToken();
            originalRequest.headers.Authorization = `Bearer ${newToken}`;
            return this.client(originalRequest);
          } catch {
            // Refresh failed, logout user
            await this.logout();
            throw new AuthError('Session expired');
          }
        }

        // Transform error for better handling
        throw this.transformError(error);
      }
    );
  }

  private async refreshToken(): Promise<string> {
    // Prevent multiple refresh calls
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    this.refreshPromise = (async () => {
      try {
        const refreshToken = await secureStorage.getItem('refreshToken');
        if (!refreshToken) throw new Error('No refresh token');

        const response = await this.client.post('/auth/refresh', {
          refreshToken,
        });

        const { accessToken, refreshToken: newRefreshToken } = response.data;
        
        await secureStorage.setItem('accessToken', accessToken);
        await secureStorage.setItem('refreshToken', newRefreshToken);
        
        return accessToken;
      } finally {
        this.refreshPromise = null;
      }
    })();

    return this.refreshPromise;
  }

  private transformError(error: AxiosError): AppError {
    if (!error.response) {
      return new NetworkError('Network request failed');
    }

    const { status, data } = error.response;

    switch (status) {
      case 400:
        return new ValidationError(data.message || 'Invalid request', data.errors);
      case 401:
        return new AuthError('Unauthorized');
      case 403:
        return new PermissionError('Forbidden');
      case 404:
        return new NotFoundError(data.message || 'Resource not found');
      case 429:
        return new RateLimitError('Too many requests', data.retryAfter);
      case 500:
        return new ServerError('Internal server error');
      default:
        return new AppError(data.message || 'An error occurred', status);
    }
  }

  // Typed API methods
  async get<T>(url: string, config?: any): Promise<T> {
    const response = await this.client.get<T>(url, config);
    return response.data;
  }

  async post<T>(url: string, data?: any, config?: any): Promise<T> {
    const response = await this.client.post<T>(url, data, config);
    return response.data;
  }

  async put<T>(url: string, data?: any, config?: any): Promise<T> {
    const response = await this.client.put<T>(url, data, config);
    return response.data;
  }

  async delete<T>(url: string, config?: any): Promise<T> {
    const response = await this.client.delete<T>(url, config);
    return response.data;
  }
}

export const apiClient = new ApiClient();
```

### ✅ DO: Implement Offline Support with Queue

```typescript
// services/offline/OfflineQueue.ts
import { MMKV } from 'react-native-mmkv';
import NetInfo from '@react-native-community/netinfo';
import BackgroundFetch from 'react-native-background-fetch';

interface QueuedRequest {
  id: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  url: string;
  data?: any;
  timestamp: number;
  retryCount: number;
}

class OfflineQueue {
  private storage = new MMKV({ id: 'offline-queue' });
  private processing = false;

  constructor() {
    this.setupBackgroundSync();
    this.setupNetworkListener();
  }

  private setupBackgroundSync() {
    BackgroundFetch.configure(
      {
        minimumFetchInterval: 15, // minutes
        enableHeadless: true,
        startOnBoot: true,
      },
      async (taskId) => {
        await this.processQueue();
        BackgroundFetch.finish(taskId);
      },
      (taskId) => {
        BackgroundFetch.finish(taskId);
      }
    );
  }

  private setupNetworkListener() {
    NetInfo.addEventListener((state) => {
      if (state.isConnected && !this.processing) {
        this.processQueue();
      }
    });
  }

  async add(request: Omit<QueuedRequest, 'id' | 'timestamp' | 'retryCount'>) {
    const id = generateId();
    const queuedRequest: QueuedRequest = {
      ...request,
      id,
      timestamp: Date.now(),
      retryCount: 0,
    };

    const queue = this.getQueue();
    queue.push(queuedRequest);
    this.storage.set('queue', JSON.stringify(queue));

    // Try to process immediately if online
    const netInfo = await NetInfo.fetch();
    if (netInfo.isConnected) {
      this.processQueue();
    }
  }

  private getQueue(): QueuedRequest[] {
    const queueString = this.storage.getString('queue');
    return queueString ? JSON.parse(queueString) : [];
  }

  private async processQueue() {
    if (this.processing) return;
    this.processing = true;

    try {
      const queue = this.getQueue();
      const pendingRequests = queue.filter(
        (req) => req.retryCount < 3 // Max 3 retries
      );

      for (const request of pendingRequests) {
        try {
          await this.processRequest(request);
          // Remove successful request from queue
          this.removeFromQueue(request.id);
        } catch (error) {
          // Increment retry count
          this.updateRequest(request.id, {
            retryCount: request.retryCount + 1,
          });
        }
      }
    } finally {
      this.processing = false;
    }
  }

  private async processRequest(request: QueuedRequest) {
    switch (request.method) {
      case 'POST':
        await apiClient.post(request.url, request.data);
        break;
      case 'PUT':
        await apiClient.put(request.url, request.data);
        break;
      case 'DELETE':
        await apiClient.delete(request.url);
        break;
      default:
        throw new Error(`Unsupported method: ${request.method}`);
    }
  }

  private removeFromQueue(id: string) {
    const queue = this.getQueue().filter((req) => req.id !== id);
    this.storage.set('queue', JSON.stringify(queue));
  }

  private updateRequest(id: string, updates: Partial<QueuedRequest>) {
    const queue = this.getQueue().map((req) =>
      req.id === id ? { ...req, ...updates } : req
    );
    this.storage.set('queue', JSON.stringify(queue));
  }
}

export const offlineQueue = new OfflineQueue();
```

---

## 9. Security: Mobile-First Security Patterns

Mobile apps require special security considerations due to their distributed nature.

### ✅ DO: Implement Certificate Pinning

```typescript
// services/security/CertificatePinning.ts
import { NativeModules, Platform } from 'react-native';
import RNFetchBlob from 'rn-fetch-blob';

class CertificatePinning {
  private pins: Map<string, string[]> = new Map();

  constructor() {
    this.configurePins();
  }

  private configurePins() {
    // Pin your API certificates
    this.pins.set('api.myapp.com', [
      'sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=',
      'sha256/BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=', // Backup pin
    ]);
  }

  async fetch(url: string, options: any = {}) {
    const hostname = new URL(url).hostname;
    const pins = this.pins.get(hostname);

    if (!pins || pins.length === 0) {
      throw new Error(`No certificate pins configured for ${hostname}`);
    }

    if (Platform.OS === 'ios') {
      return RNFetchBlob.config({
        trusty: true,
        certificates: pins,
      }).fetch(options.method || 'GET', url, options.headers, options.body);
    } else {
      // Android implementation
      return RNFetchBlob.config({
        trusty: true,
        wifiOnly: false,
      }).fetch(options.method || 'GET', url, options.headers, options.body)
        .then(async (response) => {
          // Verify certificate on Android
          const cert = response.info().headers['Public-Key-Pins'];
          if (!pins.includes(cert)) {
            throw new Error('Certificate verification failed');
          }
          return response;
        });
    }
  }
}

export const certificatePinning = new CertificatePinning();
```

### ✅ DO: Secure Storage Implementation

```typescript
// services/storage/SecureStorage.ts
import * as Keychain from 'react-native-keychain';
import { NativeModules, Platform } from 'react-native';
import CryptoJS from 'crypto-js';

const { AESCrypt } = NativeModules;

class SecureStorage {
  private encryptionKey: string | null = null;

  async initialize() {
    // Generate or retrieve encryption key
    const credentials = await Keychain.getInternetCredentials('encryption_key');
    
    if (credentials) {
      this.encryptionKey = credentials.password;
    } else {
      this.encryptionKey = this.generateKey();
      await Keychain.setInternetCredentials(
        'encryption_key',
        'app',
        this.encryptionKey
      );
    }
  }

  private generateKey(): string {
    return CryptoJS.lib.WordArray.random(256 / 8).toString();
  }

  async setItem(key: string, value: string, options?: { accessible?: string }) {
    if (!this.encryptionKey) {
      await this.initialize();
    }

    // Encrypt the value
    const encrypted = CryptoJS.AES.encrypt(value, this.encryptionKey!).toString();

    // Store in Keychain for sensitive data
    await Keychain.setInternetCredentials(
      key,
      'encrypted',
      encrypted,
      {
        accessible: options?.accessible || Keychain.ACCESSIBLE.WHEN_UNLOCKED,
        ...(Platform.OS === 'ios' && {
          accessGroup: 'group.com.myapp.shared',
        }),
      }
    );
  }

  async getItem(key: string): Promise<string | null> {
    if (!this.encryptionKey) {
      await this.initialize();
    }

    try {
      const credentials = await Keychain.getInternetCredentials(key);
      if (!credentials) return null;

      // Decrypt the value
      const decrypted = CryptoJS.AES.decrypt(
        credentials.password,
        this.encryptionKey!
      );
      
      return decrypted.toString(CryptoJS.enc.Utf8);
    } catch {
      return null;
    }
  }

  async removeItem(key: string) {
    await Keychain.resetInternetCredentials(key);
  }

  async clear() {
    // Clear all app keychain items
    if (Platform.OS === 'ios') {
      await Keychain.resetGenericPassword();
    }
    // On Android, we need to track and clear individual items
  }

  // Biometric-protected storage
  async setSecureItem(key: string, value: string) {
    await Keychain.setInternetCredentials(
      key,
      'secure',
      value,
      {
        accessible: Keychain.ACCESSIBLE.WHEN_UNLOCKED,
        authenticatePrompt: 'Authenticate to access secure data',
        authenticationPromptTitle: 'Authentication Required',
      }
    );
  }
}

export const secureStorage = new SecureStorage();
```

---

## 10. Testing: Comprehensive Testing Strategy

Implement a multi-layer testing approach covering unit, integration, and E2E tests.

### ✅ DO: Write Component Tests with React Native Testing Library

```typescript
// __tests__/components/LoginForm.test.tsx
import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import { LoginForm } from '@/features/auth/components/LoginForm';
import { authService } from '@/services/auth';

jest.mock('@/services/auth');

describe('LoginForm', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should display validation errors for invalid input', async () => {
    const { getByPlaceholderText, getByText, queryByText } = render(
      <LoginForm onSuccess={jest.fn()} />
    );

    const submitButton = getByText('Login');
    
    // Submit without filling form
    fireEvent.press(submitButton);

    await waitFor(() => {
      expect(getByText('Email is required')).toBeTruthy();
      expect(getByText('Password is required')).toBeTruthy();
    });

    // Fill invalid email
    fireEvent.changeText(getByPlaceholderText('Email'), 'invalid-email');
    fireEvent.press(submitButton);

    await waitFor(() => {
      expect(getByText('Invalid email format')).toBeTruthy();
      expect(queryByText('Email is required')).toBeNull();
    });
  });

  it('should handle successful login', async () => {
    const onSuccess = jest.fn();
    const mockUser = { id: '1', email: 'test@example.com' };
    
    (authService.login as jest.Mock).mockResolvedValueOnce({
      user: mockUser,
      token: 'mock-token',
    });

    const { getByPlaceholderText, getByText } = render(
      <LoginForm onSuccess={onSuccess} />
    );

    fireEvent.changeText(getByPlaceholderText('Email'), 'test@example.com');
    fireEvent.changeText(getByPlaceholderText('Password'), 'password123');
    fireEvent.press(getByText('Login'));

    await waitFor(() => {
      expect(authService.login).toHaveBeenCalledWith({
        email: 'test@example.com',
        password: 'password123',
      });
      expect(onSuccess).toHaveBeenCalledWith(mockUser);
    });
  });

  it('should handle login failure', async () => {
    (authService.login as jest.Mock).mockRejectedValueOnce(
      new Error('Invalid credentials')
    );

    const { getByPlaceholderText, getByText } = render(
      <LoginForm onSuccess={jest.fn()} />
    );

    fireEvent.changeText(getByPlaceholderText('Email'), 'test@example.com');
    fireEvent.changeText(getByPlaceholderText('Password'), 'wrongpassword');
    fireEvent.press(getByText('Login'));

    await waitFor(() => {
      expect(getByText('Invalid credentials')).toBeTruthy();
    });
  });
});
```

### ✅ DO: Integration Tests for Native Modules

```typescript
// __tests__/integration/BiometricAuth.test.ts
import { NativeModules } from 'react-native';
import { biometricService } from '@/services/biometric';

// Mock the native module
NativeModules.BiometricAuth = {
  isSupported: jest.fn(),
  authenticate: jest.fn(),
  getBiometryType: jest.fn(),
};

describe('BiometricAuth Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should handle unsupported devices gracefully', async () => {
    NativeModules.BiometricAuth.isSupported.mockResolvedValueOnce(false);

    const result = await biometricService.checkAvailability();

    expect(result).toEqual({
      isAvailable: false,
      biometryType: 'None',
    });
  });

  it('should authenticate successfully on supported devices', async () => {
    NativeModules.BiometricAuth.isSupported.mockResolvedValueOnce(true);
    NativeModules.BiometricAuth.authenticate.mockResolvedValueOnce(true);
    NativeModules.BiometricAuth.getBiometryType.mockResolvedValueOnce('FaceID');

    const authenticated = await biometricService.authenticate('Test authentication');

    expect(authenticated).toBe(true);
    expect(NativeModules.BiometricAuth.authenticate).toHaveBeenCalledWith(
      'Test authentication'
    );
  });
});
```

### ✅ DO: E2E Tests with Maestro

```yaml
# e2e/flows/login-flow.yaml
appId: com.myapp
name: Login Flow
---
- launchApp:
    clearState: true

# Check initial state
- assertVisible: "Welcome to MyApp"
- assertVisible: "Login"

# Navigate to login
- tapOn: "Login"

# Test validation
- tapOn: "Sign In"
- assertVisible: "Email is required"
- assertVisible: "Password is required"

# Fill form with invalid email
- tapOn:
    id: "email-input"
- inputText: "invalid-email"
- tapOn: "Sign In"
- assertVisible: "Invalid email format"

# Complete valid login
- clearText
- inputText: "test@example.com"
- tapOn:
    id: "password-input"
- inputText: "Test123!"
- tapOn: "Sign In"

# Verify successful login
- waitForAnimationToEnd
- assertVisible: "Welcome back!"
- assertNotVisible: "Login"

# Test biometric prompt (if available)
- tapOn: "Enable Biometric Login"
- assertVisible: "Authenticate to enable biometric login"
```

---

## 11. CI/CD: Automated Deployment Pipeline

Set up a robust CI/CD pipeline for both iOS and Android.

### ✅ DO: GitHub Actions for Automated Builds

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  NODE_VERSION: '22'
  RUBY_VERSION: '3.3'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run tests
        run: |
          npm run type-check
          npm run lint
          npm run test:unit
          npm run test:integration
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage/lcov.info

  build-android:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Java
        uses: actions/setup-java@v4
        with:
          distribution: 'zulu'
          java-version: '17'
      
      - name: Setup Android SDK
        uses: android-actions/setup-android@v3
      
      - name: Cache Gradle
        uses: actions/cache@v4
        with:
          path: |
            ~/.gradle/caches
            ~/.gradle/wrapper
          key: ${{ runner.os }}-gradle-${{ hashFiles('**/*.gradle*', '**/gradle-wrapper.properties') }}
      
      - name: Decode keystore
        env:
          ANDROID_KEYSTORE_BASE64: ${{ secrets.ANDROID_KEYSTORE_BASE64 }}
        run: |
          echo "$ANDROID_KEYSTORE_BASE64" | base64 -d > android/app/release.keystore
      
      - name: Build Android release
        env:
          ANDROID_KEYSTORE_PASSWORD: ${{ secrets.ANDROID_KEYSTORE_PASSWORD }}
          ANDROID_KEY_ALIAS: ${{ secrets.ANDROID_KEY_ALIAS }}
          ANDROID_KEY_PASSWORD: ${{ secrets.ANDROID_KEY_PASSWORD }}
        run: |
          cd android
          ./gradlew assembleRelease
      
      - name: Upload APK
        uses: actions/upload-artifact@v4
        with:
          name: app-release.apk
          path: android/app/build/outputs/apk/release/app-release.apk

  build-ios:
    needs: test
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Ruby
        uses: ruby/setup-ruby@v1
        with:
          ruby-version: ${{ env.RUBY_VERSION }}
          bundler-cache: true
      
      - name: Install CocoaPods
        run: |
          cd ios
          pod install
      
      - name: Setup certificates
        uses: apple-actions/import-codesign-certs@v2
        with:
          p12-file-base64: ${{ secrets.IOS_P12_BASE64 }}
          p12-password: ${{ secrets.IOS_P12_PASSWORD }}
      
      - name: Build iOS
        run: |
          cd ios
          xcodebuild -workspace MyApp.xcworkspace \
            -scheme MyApp \
            -configuration Release \
            -sdk iphoneos \
            -derivedDataPath build \
            -archivePath build/MyApp.xcarchive \
            archive
      
      - name: Export IPA
        run: |
          cd ios
          xcodebuild -exportArchive \
            -archivePath build/MyApp.xcarchive \
            -exportPath build \
            -exportOptionsPlist ExportOptions.plist
      
      - name: Upload IPA
        uses: actions/upload-artifact@v4
        with:
          name: MyApp.ipa
          path: ios/build/MyApp.ipa

  deploy:
    needs: [build-android, build-ios]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Download artifacts
        uses: actions/download-artifact@v4
      
      - name: Deploy to App Store Connect
        run: |
          xcrun altool --upload-app \
            --type ios \
            --file MyApp.ipa \
            --username ${{ secrets.APPLE_ID }} \
            --password ${{ secrets.APPLE_APP_PASSWORD }}
      
      - name: Deploy to Google Play
        uses: r0adkll/upload-google-play@v1
        with:
          serviceAccountJsonPlainText: ${{ secrets.GOOGLE_PLAY_SERVICE_ACCOUNT }}
          packageName: com.myapp
          releaseFiles: app-release.apk
          track: internal
```

---

## 12. Monitoring & Analytics

Implement comprehensive monitoring to catch issues before users report them.

### ✅ DO: Implement Crash Reporting with Sentry

```typescript
// services/monitoring/Sentry.ts
import * as Sentry from '@sentry/react-native';
import { Platform } from 'react-native';
import Config from 'react-native-config';

export function initializeSentry() {
  Sentry.init({
    dsn: Config.SENTRY_DSN,
    environment: Config.ENVIRONMENT,
    
    // Performance Monitoring
    tracesSampleRate: Config.ENVIRONMENT === 'production' ? 0.1 : 1.0,
    
    // Release Health
    enableAutoSessionTracking: true,
    sessionTrackingIntervalMillis: 30000,
    
    // Breadcrumbs
    maxBreadcrumbs: 50,
    
    // Integrations
    integrations: [
      new Sentry.ReactNativeTracing({
        routingInstrumentation,
        tracingOrigins: ['localhost', /^https:\/\/api\.myapp\.com\/api/],
        
        // Trace specific operations
        shouldCreateSpanForRequest: (url) => {
          return !url.includes('analytics');
        },
      }),
    ],
    
    // Before send hook
    beforeSend: (event, hint) => {
      // Filter out known issues
      if (event.exception?.values?.[0]?.value?.includes('Network request failed')) {
        return null;
      }
      
      // Add user context
      event.user = {
        id: getUserId(),
        email: getUserEmail(),
      };
      
      // Add custom context
      event.contexts = {
        ...event.contexts,
        app: {
          build_number: getBuildNumber(),
          version: getVersion(),
        },
        device: {
          battery_level: getBatteryLevel(),
          memory_usage: getMemoryUsage(),
          storage_free: getFreeStorage(),
        },
      };
      
      return event;
    },
    
    // Attachments
    attachScreenshot: true,
    attachViewHierarchy: true,
  });
}

// Custom error boundary
export class ErrorBoundary extends React.Component
  { children: React.ReactNode; fallback?: React.ComponentType<any> },
  { hasError: boolean; error?: Error }
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    Sentry.withScope((scope) => {
      scope.setExtras(errorInfo);
      scope.setLevel('error');
      Sentry.captureException(error);
    });
  }

  render() {
    if (this.state.hasError) {
      const Fallback = this.props.fallback || DefaultErrorFallback;
      return <Fallback error={this.state.error} resetError={() => this.setState({ hasError: false })} />;
    }

    return this.props.children;
  }
}
```

### ✅ DO: Performance Monitoring

```typescript
// hooks/usePerformanceMonitoring.ts
import { useEffect, useRef } from 'react';
import * as Sentry from '@sentry/react-native';
import analytics from '@react-native-firebase/analytics';
import performance from '@react-native-firebase/perf';

export function usePerformanceMonitoring(screenName: string) {
  const mountTime = useRef(Date.now());
  const trace = useRef<any>(null);

  useEffect(() => {
    // Start performance trace
    (async () => {
      trace.current = await performance().startTrace(`screen_${screenName}`);
      trace.current.putAttribute('screen_name', screenName);
    })();

    // Track screen view
    analytics().logScreenView({
      screen_name: screenName,
      screen_class: screenName,
    });

    // Sentry transaction
    const transaction = Sentry.startTransaction({
      name: `Screen: ${screenName}`,
      op: 'navigation',
    });
    
    Sentry.getCurrentHub().configureScope((scope) => scope.setSpan(transaction));

    return () => {
      // Calculate time to interactive
      const tti = Date.now() - mountTime.current;
      
      // Log to Firebase
      analytics().logEvent('screen_tti', {
        screen_name: screenName,
        time_to_interactive: tti,
      });

      // Stop trace
      if (trace.current) {
        trace.current.putMetric('time_to_interactive', tti);
        trace.current.stop();
      }

      // Finish Sentry transaction
      transaction.finish();
    };
  }, [screenName]);
}

// Usage in screens
export function HomeScreen() {
  usePerformanceMonitoring('Home');
  
  return <View>{/* Your content */}</View>;
}
```

---

## 13. Advanced Patterns

### React Native + Web Code Sharing

```typescript
// Create platform-agnostic components
// components/ui/Text/Text.tsx
export interface TextProps {
  variant?: 'h1' | 'h2' | 'body' | 'caption';
  color?: string;
  children: React.ReactNode;
}

// components/ui/Text/Text.native.tsx
import { Text as RNText } from 'react-native';

export function Text({ variant = 'body', color, children }: TextProps) {
  const styles = getTextStyles(variant, color);
  return <RNText style={styles}>{children}</RNText>;
}

// components/ui/Text/Text.web.tsx
export function Text({ variant = 'body', color, children }: TextProps) {
  const Component = variantToTag[variant];
  const styles = getTextStyles(variant, color);
  return <Component style={styles}>{children}</Component>;
}
```

### Custom Dev Menu Items

```typescript
// development/DevMenu.ts
import { DevSettings } from 'react-native';

if (__DEV__) {
  // Add custom dev menu items
  DevSettings.addMenuItem('Clear AsyncStorage', () => {
    AsyncStorage.clear();
    DevSettings.reload();
  });

  DevSettings.addMenuItem('Show Performance Overlay', () => {
    // Toggle custom performance overlay
    performanceOverlay.toggle();
  });

  DevSettings.addMenuItem('Crash Test', () => {
    throw new Error('Test crash for Sentry');
  });

  DevSettings.addMenuItem('Switch Environment', () => {
    // Show environment switcher
    showEnvironmentSwitcher();
  });
}
```

This comprehensive guide provides a solid foundation for building production-ready React Native applications in 2025. The patterns and practices outlined here will help you create apps that are performant, maintainable, and truly feel native on both iOS and Android platforms.