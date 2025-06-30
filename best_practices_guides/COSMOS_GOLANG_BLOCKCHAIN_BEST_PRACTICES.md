# The Definitive Guide to Cosmos SDK, Ignite CLI, and CosmJS (2025)

This guide synthesizes modern best practices for building scalable, secure, and performant blockchain applications with Cosmos SDK 0.52+, Ignite CLI, CosmJS, and React Native wallets. It provides a production-grade architectural blueprint for the entire stack.

### Prerequisites & Configuration
Ensure your project uses **Cosmos SDK 0.52.0+**, **Go 1.23+**, **Ignite CLI 29.0+**, and **CosmJS 0.33+**.

Create your initial project configuration:

```yaml
# config.yml (Ignite CLI configuration)
version: 2
build:
  main: cmd/myappchaind
  binary: myappchaind
  ldflags:
    - -X github.com/cosmos/cosmos-sdk/version.Name=myappchain
    - -X github.com/cosmos/cosmos-sdk/version.AppName=myappchaind
    - -X github.com/cosmos/cosmos-sdk/version.Version={{ .Version }}
    - -X github.com/cosmos/cosmos-sdk/version.Commit={{ .Commit }}
accounts:
  - name: alice
    coins: ["10000000stake", "100000000utoken"]
  - name: bob
    coins: ["5000000stake", "50000000utoken"]
client:
  typescript:
    path: ts-client
    useSDKTypes: true
  composables:
    path: vue/src/composables
validators:
  - name: alice
    bonded: 5000000stake
genesis:
  chain_id: myappchain-1
  app_state:
    staking:
      params:
        unbonding_time: "86400s" # 1 day for testnet
    gov:
      params:
        voting_period: "172800s" # 2 days
        min_deposit:
          - denom: stake
            amount: "10000000"
```

---

## 1. Foundational Architecture & Module Organization

A well-architected Cosmos SDK application separates concerns across modules, follows consistent patterns, and maintains clear boundaries between state machine logic and external interfaces.

### ✅ DO: Use a Modular Architecture with Clear Separation

```
/x                          # Custom modules directory
├── tokenomics/            # Example custom module
│   ├── keeper/           # Core state machine logic
│   │   ├── keeper.go     # Keeper struct and constructor
│   │   ├── msg_server.go # Message handlers (Tx)
│   │   ├── query.go      # Query handlers
│   │   ├── hooks.go      # Module hooks implementation
│   │   └── invariants.go # Invariant checks
│   ├── types/            # Types, interfaces, and generated code
│   │   ├── keys.go       # Store keys and prefixes
│   │   ├── errors.go     # Module-specific errors
│   │   ├── events.go     # Event types and constructors
│   │   ├── expected_keepers.go # External keeper interfaces
│   │   └── *.pb.go       # Generated protobuf code
│   ├── client/           # Client-side helpers
│   │   └── cli/          # CLI commands
│   ├── simulation/       # Simulation operations
│   ├── genesis.go        # Genesis state handling
│   └── module.go         # AppModule implementation
├── shared/               # Shared utilities across modules
│   ├── ante/            # Custom ante handlers
│   ├── decorators/      # Custom decorators
│   └── types/           # Common types
/app                     # Application wiring
├── app.go              # Main application setup
├── encoding.go         # Custom encoding config
├── export.go           # State export logic
└── upgrades/           # Upgrade handlers
    └── v2/
        └── upgrade.go
/cmd                    # Entry points
├── myappchaind/       # Node binary
└── myappchaincli/     # Optional separate CLI
/scripts               # Build and deployment scripts
/proto                 # Protobuf definitions
└── myappchain/
    └── tokenomics/
        └── v1/
            ├── tx.proto      # Transaction messages
            ├── query.proto   # Query services
            ├── genesis.proto # Genesis state
            └── events.proto  # Event definitions
```

### ✅ DO: Define Clear Module Boundaries

Each module should be self-contained with minimal dependencies. Use **expected keeper interfaces** to define contracts between modules.

```go
// x/tokenomics/types/expected_keepers.go
package types

import (
    sdk "github.com/cosmos/cosmos-sdk/types"
    authtypes "github.com/cosmos/cosmos-sdk/x/auth/types"
)

// AccountKeeper defines the expected account keeper interface
type AccountKeeper interface {
    GetAccount(ctx sdk.Context, addr sdk.AccAddress) authtypes.AccountI
    SetAccount(ctx sdk.Context, acc authtypes.AccountI)
    NewAccountWithAddress(ctx sdk.Context, addr sdk.AccAddress) authtypes.AccountI
}

// BankKeeper defines the expected bank keeper interface
type BankKeeper interface {
    SendCoins(ctx sdk.Context, fromAddr, toAddr sdk.AccAddress, amt sdk.Coins) error
    MintCoins(ctx sdk.Context, moduleName string, amt sdk.Coins) error
    BurnCoins(ctx sdk.Context, moduleName string, amt sdk.Coins) error
    SendCoinsFromModuleToAccount(ctx sdk.Context, senderModule string, recipientAddr sdk.AccAddress, amt sdk.Coins) error
    GetBalance(ctx sdk.Context, addr sdk.AccAddress, denom string) sdk.Coin
}

// StakingKeeper defines the expected staking keeper interface
type StakingKeeper interface {
    BondDenom(ctx sdk.Context) string
    GetValidator(ctx sdk.Context, addr sdk.ValAddress) (validator stakingtypes.Validator, found bool)
}
```

---

## 2. State Machine Design: Keepers and Store Management

The keeper pattern is fundamental to Cosmos SDK. It encapsulates all state access and provides a clean API for your module's business logic.

### ✅ DO: Design Type-Safe Store Keys with Prefixes

```go
// x/tokenomics/types/keys.go
package types

import "cosmossdk.io/collections"

const (
    ModuleName = "tokenomics"
    StoreKey   = ModuleName
    RouterKey  = ModuleName
)

var (
    // Prefix bytes for different stores
    RewardPoolKey       = collections.NewPrefix(0)
    UserRewardsKey      = collections.NewPrefix(1)
    EpochInfoKey        = collections.NewPrefix(2)
    RewardScheduleKey   = collections.NewPrefix(3)
    
    // Use structured prefixes for complex keys
    UserEpochRewardsPrefix = []byte{0x10} // user -> epoch -> rewards
)

// GetUserEpochRewardsKey returns the store key for user rewards in a specific epoch
func GetUserEpochRewardsKey(user sdk.AccAddress, epoch uint64) []byte {
    return append(append(UserEpochRewardsPrefix, user.Bytes()...), sdk.Uint64ToBigEndian(epoch)...)
}
```

### ✅ DO: Use Collections API for Type-Safe Storage (SDK 0.52+)

The Collections API provides compile-time type safety and reduces boilerplate.

```go
// x/tokenomics/keeper/keeper.go
package keeper

import (
    "context"
    "cosmossdk.io/collections"
    "cosmossdk.io/core/store"
    sdk "github.com/cosmos/cosmos-sdk/types"
)

type Keeper struct {
    cdc codec.BinaryCodec
    storeService store.KVStoreService
    authority string

    // External keepers
    accountKeeper types.AccountKeeper
    bankKeeper    types.BankKeeper
    
    // Collections for type-safe storage
    Schema          collections.Schema
    RewardPools     collections.Map[string, types.RewardPool]
    UserRewards     collections.Map[collections.Pair[sdk.AccAddress, string], sdk.DecCoin]
    EpochInfo       collections.Item[types.EpochInfo]
    RewardSchedules collections.Map[uint64, types.RewardSchedule]
    
    // Indexes for efficient queries
    RewardsByUser   collections.KeySet[collections.Pair[sdk.AccAddress, string]]
}

// NewKeeper creates a new keeper instance
func NewKeeper(
    cdc codec.BinaryCodec,
    storeService store.KVStoreService,
    authority string,
    ak types.AccountKeeper,
    bk types.BankKeeper,
) Keeper {
    sb := collections.NewSchemaBuilder(storeService)
    
    k := Keeper{
        cdc:           cdc,
        storeService: storeService,
        authority:    authority,
        accountKeeper: ak,
        bankKeeper:   bk,
        
        // Initialize collections
        RewardPools: collections.NewMap(
            sb, types.RewardPoolKey, "reward_pools",
            collections.StringKey, codec.CollValue[types.RewardPool](cdc),
        ),
        UserRewards: collections.NewMap(
            sb, types.UserRewardsKey, "user_rewards",
            collections.PairKeyCodec(sdk.AccAddressKey, collections.StringKey),
            sdk.DecCoinValue,
        ),
        EpochInfo: collections.NewItem(
            sb, types.EpochInfoKey, "epoch_info",
            codec.CollValue[types.EpochInfo](cdc),
        ),
        RewardSchedules: collections.NewMap(
            sb, types.RewardScheduleKey, "reward_schedules",
            collections.Uint64Key, codec.CollValue[types.RewardSchedule](cdc),
        ),
        RewardsByUser: collections.NewKeySet(
            sb, collections.NewPrefix(4), "rewards_by_user",
            collections.PairKeyCodec(sdk.AccAddressKey, collections.StringKey),
        ),
    }
    
    schema, err := sb.Build()
    if err != nil {
        panic(err)
    }
    k.Schema = schema
    
    return k
}
```

### ❌ DON'T: Use Raw KVStore Operations

This old pattern is error-prone and lacks type safety.

```go
// Bad - Don't do this
func (k Keeper) SetRewardPool(ctx sdk.Context, pool types.RewardPool) {
    store := ctx.KVStore(k.storeKey)
    bz := k.cdc.MustMarshal(&pool)
    store.Set([]byte(pool.Id), bz)
}

func (k Keeper) GetRewardPool(ctx sdk.Context, id string) (types.RewardPool, bool) {
    store := ctx.KVStore(k.storeKey)
    bz := store.Get([]byte(id))
    if bz == nil {
        return types.RewardPool{}, false
    }
    var pool types.RewardPool
    k.cdc.MustUnmarshal(bz, &pool)
    return pool, true
}
```

---

## 3. Message Handlers and Transaction Processing

Message handlers are where your business logic lives. They must be deterministic, validate all inputs, and emit appropriate events.

### ✅ DO: Implement Comprehensive Input Validation

```go
// x/tokenomics/keeper/msg_server.go
package keeper

import (
    "context"
    errorsmod "cosmossdk.io/errors"
    sdk "github.com/cosmos/cosmos-sdk/types"
    govtypes "github.com/cosmos/cosmos-sdk/x/gov/types"
)

type msgServer struct {
    Keeper
}

// NewMsgServerImpl returns an implementation of the MsgServer interface
func NewMsgServerImpl(keeper Keeper) types.MsgServer {
    return &msgServer{Keeper: keeper}
}

// ClaimRewards handles reward claiming with comprehensive validation
func (k msgServer) ClaimRewards(goCtx context.Context, msg *types.MsgClaimRewards) (*types.MsgClaimRewardsResponse, error) {
    ctx := sdk.UnwrapSDKContext(goCtx)
    
    // Validate sender address
    sender, err := sdk.AccAddressFromBech32(msg.Sender)
    if err != nil {
        return nil, errorsmod.Wrapf(types.ErrInvalidAddress, "invalid sender address: %s", err)
    }
    
    // Check if rewards exist
    rewards, err := k.GetPendingRewards(ctx, sender)
    if err != nil {
        return nil, errorsmod.Wrap(types.ErrRewardNotFound, err.Error())
    }
    
    if rewards.IsZero() {
        return nil, types.ErrNoRewardsToClaim
    }
    
    // Apply any vesting or lock periods
    claimableRewards := k.calculateClaimableRewards(ctx, sender, rewards)
    if claimableRewards.IsZero() {
        return nil, types.ErrRewardsLocked
    }
    
    // Transfer rewards from module to user
    if err := k.bankKeeper.SendCoinsFromModuleToAccount(
        ctx, types.ModuleName, sender, claimableRewards,
    ); err != nil {
        return nil, errorsmod.Wrap(types.ErrRewardTransfer, err.Error())
    }
    
    // Update state
    if err := k.markRewardsClaimed(ctx, sender, claimableRewards); err != nil {
        return nil, err
    }
    
    // Emit events for indexing
    ctx.EventManager().EmitTypedEvent(&types.EventRewardsClaimed{
        Recipient: msg.Sender,
        Amount:    claimableRewards,
        ClaimedAt: ctx.BlockTime(),
    })
    
    // Log for monitoring
    k.Logger(ctx).Info("rewards claimed",
        "recipient", msg.Sender,
        "amount", claimableRewards.String(),
        "height", ctx.BlockHeight(),
    )
    
    return &types.MsgClaimRewardsResponse{
        ClaimedAmount: claimableRewards,
    }, nil
}

// UpdateRewardParams handles parameter updates (governance only)
func (k msgServer) UpdateRewardParams(goCtx context.Context, msg *types.MsgUpdateRewardParams) (*types.MsgUpdateRewardParamsResponse, error) {
    ctx := sdk.UnwrapSDKContext(goCtx)
    
    // Only governance can update params
    if k.authority != msg.Authority {
        return nil, errorsmod.Wrapf(
            govtypes.ErrInvalidSigner,
            "invalid authority; expected %s, got %s",
            k.authority, msg.Authority,
        )
    }
    
    // Validate new params
    if err := msg.Params.Validate(); err != nil {
        return nil, errorsmod.Wrap(types.ErrInvalidParams, err.Error())
    }
    
    // Store params
    if err := k.Params.Set(ctx, msg.Params); err != nil {
        return nil, err
    }
    
    ctx.EventManager().EmitTypedEvent(&types.EventParamsUpdated{
        OldParams: k.GetParams(ctx),
        NewParams: msg.Params,
        UpdatedAt: ctx.BlockTime(),
    })
    
    return &types.MsgUpdateRewardParamsResponse{}, nil
}
```

### ✅ DO: Use Proper Error Handling with Custom Errors

```go
// x/tokenomics/types/errors.go
package types

import (
    errorsmod "cosmossdk.io/errors"
)

var (
    ErrInvalidAddress    = errorsmod.Register(ModuleName, 2, "invalid address")
    ErrRewardNotFound    = errorsmod.Register(ModuleName, 3, "reward not found")
    ErrNoRewardsToClaim  = errorsmod.Register(ModuleName, 4, "no rewards to claim")
    ErrRewardsLocked     = errorsmod.Register(ModuleName, 5, "rewards are locked")
    ErrRewardTransfer    = errorsmod.Register(ModuleName, 6, "failed to transfer rewards")
    ErrInvalidParams     = errorsmod.Register(ModuleName, 7, "invalid parameters")
    ErrUnauthorized      = errorsmod.Register(ModuleName, 8, "unauthorized")
    ErrInsufficientFunds = errorsmod.Register(ModuleName, 9, "insufficient funds")
)
```

---

## 4. Query Handlers and gRPC Services

Queries must be efficient, paginated for large datasets, and never modify state.

### ✅ DO: Implement Efficient Paginated Queries

```go
// x/tokenomics/keeper/query.go
package keeper

import (
    "context"
    
    "cosmossdk.io/store/prefix"
    sdk "github.com/cosmos/cosmos-sdk/types"
    "github.com/cosmos/cosmos-sdk/types/query"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

var _ types.QueryServer = Keeper{}

// UserRewards returns paginated rewards for a user
func (k Keeper) UserRewards(goCtx context.Context, req *types.QueryUserRewardsRequest) (*types.QueryUserRewardsResponse, error) {
    if req == nil {
        return nil, status.Error(codes.InvalidArgument, "invalid request")
    }
    
    ctx := sdk.UnwrapSDKContext(goCtx)
    
    // Validate address
    addr, err := sdk.AccAddressFromBech32(req.Address)
    if err != nil {
        return nil, status.Error(codes.InvalidArgument, "invalid address")
    }
    
    var rewards []types.UserReward
    
    // Use collections for efficient iteration
    err = query.Paginate(
        k.UserRewards,
        req.Pagination,
        func(key collections.Pair[sdk.AccAddress, string], value sdk.DecCoin) error {
            if key.K1().Equals(addr) {
                rewards = append(rewards, types.UserReward{
                    Denom:  key.K2(),
                    Amount: value,
                })
            }
            return nil
        },
    )
    
    if err != nil {
        return nil, status.Error(codes.Internal, err.Error())
    }
    
    return &types.QueryUserRewardsResponse{
        Rewards: rewards,
        Pagination: pageRes,
    }, nil
}

// RewardPools returns all active reward pools with filters
func (k Keeper) RewardPools(goCtx context.Context, req *types.QueryRewardPoolsRequest) (*types.QueryRewardPoolsResponse, error) {
    if req == nil {
        return nil, status.Error(codes.InvalidArgument, "invalid request")
    }
    
    ctx := sdk.UnwrapSDKContext(goCtx)
    
    // Build filters
    filters := make([]func(*types.RewardPool) bool, 0)
    
    if req.Status != types.PoolStatus_POOL_STATUS_UNSPECIFIED {
        filters = append(filters, func(p *types.RewardPool) bool {
            return p.Status == req.Status
        })
    }
    
    if req.Denom != "" {
        filters = append(filters, func(p *types.RewardPool) bool {
            return p.RewardDenom == req.Denom
        })
    }
    
    var pools []types.RewardPool
    
    // Iterate with pagination
    pageRes, err := query.CollectionFilteredPaginate(
        ctx,
        k.RewardPools,
        req.Pagination,
        func(_ string, pool types.RewardPool) (bool, error) {
            // Apply all filters
            for _, filter := range filters {
                if !filter(&pool) {
                    return false, nil
                }
            }
            return true, nil
        },
        func(_ string, pool types.RewardPool) (types.RewardPool, error) {
            return pool, nil
        },
    )
    
    if err != nil {
        return nil, status.Error(codes.Internal, err.Error())
    }
    
    return &types.QueryRewardPoolsResponse{
        Pools:      pools,
        Pagination: pageRes,
    }, nil
}
```

### ✅ DO: Add Query Caching for Expensive Operations

```go
// x/tokenomics/keeper/cache.go
package keeper

import (
    "time"
    "sync"
    
    sdk "github.com/cosmos/cosmos-sdk/types"
)

type QueryCache struct {
    mu              sync.RWMutex
    totalValueLocked *CachedValue[sdk.Coins]
    activeUserCount  *CachedValue[uint64]
}

type CachedValue[T any] struct {
    value      T
    lastUpdate time.Time
    ttl        time.Duration
}

func (c *CachedValue[T]) Get(ctx sdk.Context, fetcher func() (T, error)) (T, error) {
    now := ctx.BlockTime()
    
    // Check if cache is valid
    if !c.lastUpdate.IsZero() && now.Sub(c.lastUpdate) < c.ttl {
        return c.value, nil
    }
    
    // Fetch new value
    value, err := fetcher()
    if err != nil {
        return c.value, err // Return stale value on error
    }
    
    c.value = value
    c.lastUpdate = now
    return value, nil
}

// GetTotalValueLocked returns cached TVL
func (k Keeper) GetTotalValueLocked(ctx sdk.Context) (sdk.Coins, error) {
    k.queryCache.mu.RLock()
    defer k.queryCache.mu.RUnlock()
    
    return k.queryCache.totalValueLocked.Get(ctx, func() (sdk.Coins, error) {
        return k.calculateTotalValueLocked(ctx)
    })
}
```

---

## 5. Testing Strategies: Unit and Integration Tests

Testing is critical for blockchain applications. Every line of code that handles value must be thoroughly tested.

### ✅ DO: Write Comprehensive Keeper Tests

```go
// x/tokenomics/keeper/keeper_test.go
package keeper_test

import (
    "testing"
    "time"
    
    "github.com/stretchr/testify/suite"
    tmproto "github.com/cometbft/cometbft/proto/tendermint/types"
    
    "github.com/cosmos/cosmos-sdk/testutil"
    sdk "github.com/cosmos/cosmos-sdk/types"
    moduletestutil "github.com/cosmos/cosmos-sdk/types/module/testutil"
    
    "github.com/myorg/myappchain/x/tokenomics"
    "github.com/myorg/myappchain/x/tokenomics/keeper"
    "github.com/myorg/myappchain/x/tokenomics/types"
)

type KeeperTestSuite struct {
    suite.Suite
    
    ctx           sdk.Context
    keeper        keeper.Keeper
    bankKeeper    *mockBankKeeper
    stakingKeeper *mockStakingKeeper
    msgServer     types.MsgServer
    queryServer   types.QueryServer
    
    // Test addresses
    authority sdk.AccAddress
    alice     sdk.AccAddress
    bob       sdk.AccAddress
    validator sdk.ValAddress
}

func TestKeeperTestSuite(t *testing.T) {
    suite.Run(t, new(KeeperTestSuite))
}

func (s *KeeperTestSuite) SetupTest() {
    key := sdk.NewKVStoreKey(types.StoreKey)
    testCtx := testutil.DefaultContextWithDB(s.T(), key, sdk.NewTransientStoreKey("transient_test"))
    ctx := testCtx.Ctx.WithBlockHeader(tmproto.Header{Time: time.Now()})
    encCfg := moduletestutil.MakeTestEncodingConfig(tokenomics.AppModuleBasic{})
    
    // Setup mock keepers
    s.bankKeeper = &mockBankKeeper{
        balances: make(map[string]sdk.Coins),
    }
    s.stakingKeeper = &mockStakingKeeper{
        bondDenom: "stake",
    }
    
    // Create keeper
    s.authority = sdk.AccAddress("authority")
    s.keeper = keeper.NewKeeper(
        encCfg.Codec,
        runtime.NewKVStoreService(key),
        s.authority.String(),
        s.bankKeeper,
        s.stakingKeeper,
    )
    
    s.ctx = ctx
    s.msgServer = keeper.NewMsgServerImpl(s.keeper)
    s.queryServer = s.keeper
    
    // Setup test addresses
    s.alice = sdk.AccAddress("alice")
    s.bob = sdk.AccAddress("bob")
    s.validator = sdk.ValAddress("validator")
    
    // Initialize module account
    s.bankKeeper.balances[types.ModuleName] = sdk.NewCoins(
        sdk.NewCoin("stake", sdk.NewInt(1000000000)),
        sdk.NewCoin("utoken", sdk.NewInt(1000000000)),
    )
}

func (s *KeeperTestSuite) TestClaimRewards_Success() {
    // Setup: Create a reward for alice
    err := s.keeper.UserRewards.Set(
        s.ctx,
        collections.Join(s.alice, "stake"),
        sdk.NewDecCoinFromCoin(sdk.NewCoin("stake", sdk.NewInt(1000))),
    )
    s.Require().NoError(err)
    
    // Execute claim
    msg := &types.MsgClaimRewards{
        Sender: s.alice.String(),
    }
    res, err := s.msgServer.ClaimRewards(s.ctx, msg)
    
    // Assert success
    s.Require().NoError(err)
    s.Require().NotNil(res)
    s.Require().Equal(
        sdk.NewCoins(sdk.NewCoin("stake", sdk.NewInt(1000))),
        res.ClaimedAmount,
    )
    
    // Verify state changes
    reward, err := s.keeper.UserRewards.Get(s.ctx, collections.Join(s.alice, "stake"))
    s.Require().Error(err) // Should be deleted after claim
    
    // Verify balance transfer
    s.Require().Equal(
        sdk.NewCoins(sdk.NewCoin("stake", sdk.NewInt(1000))),
        s.bankKeeper.balances[s.alice.String()],
    )
    
    // Verify events
    events := s.ctx.EventManager().Events()
    s.Require().Len(events, 1)
    
    claimEvent, err := sdk.ParseTypedEvent(events[0])
    s.Require().NoError(err)
    
    typedEvent, ok := claimEvent.(*types.EventRewardsClaimed)
    s.Require().True(ok)
    s.Require().Equal(s.alice.String(), typedEvent.Recipient)
}

func (s *KeeperTestSuite) TestClaimRewards_NoRewards() {
    // Execute claim with no rewards
    msg := &types.MsgClaimRewards{
        Sender: s.alice.String(),
    }
    _, err := s.msgServer.ClaimRewards(s.ctx, msg)
    
    // Should fail with specific error
    s.Require().Error(err)
    s.Require().ErrorIs(err, types.ErrNoRewardsToClaim)
}

// Mock implementations
type mockBankKeeper struct {
    balances map[string]sdk.Coins
}

func (m *mockBankKeeper) SendCoinsFromModuleToAccount(
    ctx sdk.Context,
    senderModule string,
    recipientAddr sdk.AccAddress,
    amt sdk.Coins,
) error {
    moduleBalance := m.balances[senderModule]
    if !moduleBalance.IsAllGTE(amt) {
        return types.ErrInsufficientFunds
    }
    
    m.balances[senderModule] = moduleBalance.Sub(amt...)
    m.balances[recipientAddr.String()] = m.balances[recipientAddr.String()].Add(amt...)
    return nil
}
```

### ✅ DO: Write Integration Tests with Real State

```go
// x/tokenomics/keeper/integration_test.go
package keeper_test

import (
    "testing"
    
    dbm "github.com/cometbft/cometbft-db"
    "github.com/cometbft/cometbft/libs/log"
    simtestutil "github.com/cosmos/cosmos-sdk/testutil/sims"
    
    "github.com/myorg/myappchain/app"
)

func TestRewardDistribution_Integration(t *testing.T) {
    // Create a real app instance
    db := dbm.NewMemDB()
    appInstance := app.New(
        log.NewNopLogger(),
        db,
        nil,
        true,
        simtestutil.NewAppOptionsWithFlagHome(t.TempDir()),
    )
    
    ctx := appInstance.NewContext(true)
    
    // Setup initial state
    validators := appInstance.StakingKeeper.GetAllValidators(ctx)
    require.Len(t, validators, 1)
    
    // Create reward pool
    pool := types.RewardPool{
        Id:           "staking-rewards",
        RewardDenom:  "utoken",
        TotalRewards: sdk.NewCoin("utoken", sdk.NewInt(1000000)),
        StartTime:    ctx.BlockTime(),
        EndTime:      ctx.BlockTime().Add(time.Hour * 24 * 30), // 30 days
        Status:       types.PoolStatus_ACTIVE,
    }
    
    err := appInstance.TokenomicsKeeper.RewardPools.Set(ctx, pool.Id, pool)
    require.NoError(t, err)
    
    // Simulate multiple blocks
    for i := 0; i < 100; i++ {
        ctx = ctx.WithBlockHeight(ctx.BlockHeight() + 1)
        ctx = ctx.WithBlockTime(ctx.BlockTime().Add(time.Minute))
        
        // Run end blocker
        err := appInstance.TokenomicsKeeper.EndBlock(ctx)
        require.NoError(t, err)
    }
    
    // Verify rewards were distributed
    delegator := sdk.AccAddress("delegator")
    rewards, err := appInstance.TokenomicsKeeper.GetPendingRewards(ctx, delegator)
    require.NoError(t, err)
    require.False(t, rewards.IsZero())
}
```

---

## 6. Client Integration with CosmJS

CosmJS provides TypeScript/JavaScript clients for interacting with your chain. Always generate types from your protobuf definitions.

### ✅ DO: Generate TypeScript Types from Proto

```bash
# Install dependencies
npm install -g @cosmology/telescope

# Generate types
telescope generate \
  --protoDirs ./proto \
  --outPath ./ts-client/src/codegen \
  --config telescope.config.json
```

```json
// telescope.config.json
{
  "protoDirs": ["./proto"],
  "outPath": "./ts-client/src/codegen",
  "options": {
    "removeUnusedImports": true,
    "tsDisable": {
      "files": ["cosmos/authz/**/*", "cosmos/gov/**/*"],
      "disableAll": false
    },
    "bundle": {
      "enabled": true
    },
    "prototypes": {
      "includePackageVar": false,
      "strictNullCheckForPrototypeMethods": true,
      "paginationDefaultFromPartial": false,
      "addTypeUrlToObjects": true,
      "addAminoTypeToObjects": true,
      "typingsFormat": {
        "duration": "duration",
        "timestamp": "date",
        "useExact": true,
        "useDeepPartial": true
      }
    },
    "aminoEncoding": {
      "enabled": true,
      "useRecursivePartial": true
    },
    "lcdClients": {
      "enabled": true
    },
    "rpcClients": {
      "enabled": true,
      "camelCase": true,
      "useConnectComet": true
    }
  }
}
```

### ✅ DO: Create a Type-Safe Client Wrapper

```typescript
// ts-client/src/client.ts
import { Registry, DirectSecp256k1HdWallet } from '@cosmjs/proto-signing';
import { SigningStargateClient, StargateClient, defaultRegistryTypes } from '@cosmjs/stargate';
import { Tendermint34Client } from '@cosmjs/tendermint-rpc';
import { GasPrice } from '@cosmjs/stargate';

// Import generated types
import { 
  myappchainTokenomicsV1MsgClaimRewards,
  myappchainTokenomicsV1QueryUserRewardsRequest,
  QueryClient as TokenomicsQueryClient,
  MsgClient as TokenomicsMsgClient,
  registry as tokenomicsRegistry
} from './codegen/myappchain/tokenomics/v1/tx';

export class MyAppChainClient {
  private queryClient: StargateClient | null = null;
  private signingClient: SigningStargateClient | null = null;
  private tokenomicsQuery: TokenomicsQueryClient | null = null;
  
  constructor(
    private rpcEndpoint: string,
    private wallet?: DirectSecp256k1HdWallet
  ) {}
  
  async connect(): Promise<void> {
    // Create Tendermint client with websocket for real-time updates
    const tmClient = await Tendermint34Client.connect(this.rpcEndpoint);
    
    // Create query client
    this.queryClient = await StargateClient.create(tmClient);
    
    // Setup module-specific query clients
    this.tokenomicsQuery = new TokenomicsQueryClient(tmClient);
    
    // Create signing client if wallet provided
    if (this.wallet) {
      // Combine default types with custom types
      const registry = new Registry([
        ...defaultRegistryTypes,
        ...tokenomicsRegistry,
      ]);
      
      const gasPrice = GasPrice.fromString('0.025utoken');
      
      this.signingClient = await SigningStargateClient.createWithSigner(
        tmClient,
        this.wallet,
        {
          registry,
          gasPrice,
          broadcastTimeoutMs: 60000,
          broadcastPollIntervalMs: 500,
        }
      );
    }
  }
  
  // Query methods with proper typing
  async getUserRewards(address: string, pagination?: PageRequest): Promise<UserReward[]> {
    if (!this.tokenomicsQuery) {
      throw new Error('Client not connected');
    }
    
    const response = await this.tokenomicsQuery.userRewards({
      address,
      pagination,
    });
    
    return response.rewards;
  }
  
  async getRewardPools(
    status?: PoolStatus,
    denom?: string,
    pagination?: PageRequest
  ): Promise<RewardPool[]> {
    if (!this.tokenomicsQuery) {
      throw new Error('Client not connected');
    }
    
    const response = await this.tokenomicsQuery.rewardPools({
      status,
      denom,
      pagination,
    });
    
    return response.pools;
  }
  
  // Transaction methods
  async claimRewards(sender: string): Promise<DeliverTxResponse> {
    if (!this.signingClient) {
      throw new Error('Signing client not initialized');
    }
    
    const msg: MsgClaimRewards = {
      sender,
    };
    
    const fee = await this.estimateFee(sender, [msg]);
    
    return this.signingClient.signAndBroadcast(
      sender,
      [msg],
      fee,
      'Claiming rewards'
    );
  }
  
  // Helper method for fee estimation
  private async estimateFee(
    address: string,
    messages: readonly EncodeObject[]
  ): Promise<StdFee> {
    if (!this.signingClient) {
      throw new Error('Signing client not initialized');
    }
    
    const gasEstimate = await this.signingClient.simulate(address, messages, '');
    const gasLimit = Math.ceil(gasEstimate * 1.3); // 30% buffer
    
    return {
      amount: [{ denom: 'utoken', amount: String(gasLimit * 0.025) }],
      gas: String(gasLimit),
    };
  }
  
  // Subscription methods for real-time updates
  subscribeToRewardEvents(
    callback: (event: RewardClaimedEvent) => void
  ): () => void {
    if (!this.queryClient) {
      throw new Error('Client not connected');
    }
    
    const subscription = this.queryClient.subscribeTx(
      { 'message.module': 'tokenomics' },
      (tx) => {
        // Parse events from transaction
        const events = tx.events.filter(e => e.type === 'reward_claimed');
        
        events.forEach(event => {
          const recipient = event.attributes.find(a => a.key === 'recipient')?.value;
          const amount = event.attributes.find(a => a.key === 'amount')?.value;
          
          if (recipient && amount) {
            callback({ recipient, amount });
          }
        });
      }
    );
    
    return () => subscription.unsubscribe();
  }
}

// Usage example
async function main() {
  // Create wallet from mnemonic
  const wallet = await DirectSecp256k1HdWallet.fromMnemonic(
    'your mnemonic here',
    { prefix: 'myapp' }
  );
  
  // Initialize client
  const client = new MyAppChainClient('http://localhost:26657', wallet);
  await client.connect();
  
  // Query rewards
  const [account] = await wallet.getAccounts();
  const rewards = await client.getUserRewards(account.address);
  console.log('Pending rewards:', rewards);
  
  // Claim rewards
  if (rewards.length > 0) {
    const result = await client.claimRewards(account.address);
    console.log('Transaction hash:', result.transactionHash);
  }
  
  // Subscribe to events
  const unsubscribe = client.subscribeToRewardEvents((event) => {
    console.log('Reward claimed:', event);
  });
  
  // Clean up when done
  // unsubscribe();
}
```

---

## 7. React Native Wallet Integration

Building a mobile wallet requires careful key management, secure storage, and efficient state synchronization.

### ✅ DO: Use React Native Keychain for Secure Storage

```typescript
// src/services/WalletService.ts
import * as Keychain from 'react-native-keychain';
import { DirectSecp256k1HdWallet } from '@cosmjs/proto-signing';
import AsyncStorage from '@react-native-async-storage/async-storage';
import CryptoJS from 'crypto-js';
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

interface WalletState {
  isUnlocked: boolean;
  accounts: AccountInfo[];
  activeAccountIndex: number;
  
  // Actions
  createWallet: (mnemonic: string, password: string) => Promise<void>;
  unlockWallet: (password: string) => Promise<boolean>;
  lockWallet: () => void;
  switchAccount: (index: number) => void;
  getActiveAccount: () => AccountInfo | null;
}

interface AccountInfo {
  address: string;
  pubkey: string;
  name: string;
  hdPath: string;
}

// Secure storage service
class SecureStorage {
  private static KEYCHAIN_SERVICE = 'MyAppChainWallet';
  
  static async storeMnemonic(mnemonic: string, password: string): Promise<void> {
    // Encrypt mnemonic with password
    const encrypted = CryptoJS.AES.encrypt(mnemonic, password).toString();
    
    // Store encrypted mnemonic in keychain
    await Keychain.setInternetCredentials(
      this.KEYCHAIN_SERVICE,
      'mnemonic',
      encrypted,
      {
        accessible: Keychain.ACCESSIBLE.WHEN_UNLOCKED_THIS_DEVICE_ONLY,
        authenticatePrompt: 'Authenticate to access wallet',
        authenticationPrompt: {
          title: 'Authentication Required',
          subtitle: 'Access your wallet',
          cancel: 'Cancel',
        },
      }
    );
  }
  
  static async retrieveMnemonic(password: string): Promise<string | null> {
    try {
      const credentials = await Keychain.getInternetCredentials(this.KEYCHAIN_SERVICE);
      
      if (!credentials) {
        return null;
      }
      
      // Decrypt mnemonic
      const decrypted = CryptoJS.AES.decrypt(credentials.password, password);
      const mnemonic = decrypted.toString(CryptoJS.enc.Utf8);
      
      // Validate decryption
      if (!mnemonic || mnemonic.split(' ').length < 12) {
        throw new Error('Invalid password');
      }
      
      return mnemonic;
    } catch (error) {
      console.error('Failed to retrieve mnemonic:', error);
      return null;
    }
  }
  
  static async hasWallet(): Promise<boolean> {
    const credentials = await Keychain.getInternetCredentials(this.KEYCHAIN_SERVICE);
    return !!credentials;
  }
  
  static async deleteWallet(): Promise<void> {
    await Keychain.resetInternetCredentials(this.KEYCHAIN_SERVICE);
  }
}

// Zustand store for wallet state
export const useWalletStore = create<WalletState>()(
  persist(
    (set, get) => ({
      isUnlocked: false,
      accounts: [],
      activeAccountIndex: 0,
      
      createWallet: async (mnemonic: string, password: string) => {
        // Validate mnemonic
        const words = mnemonic.split(' ');
        if (words.length !== 12 && words.length !== 24) {
          throw new Error('Invalid mnemonic length');
        }
        
        // Store securely
        await SecureStorage.storeMnemonic(mnemonic, password);
        
        // Generate initial accounts
        const wallet = await DirectSecp256k1HdWallet.fromMnemonic(mnemonic, {
          prefix: 'myapp',
          hdPaths: [
            // Generate multiple accounts
            "m/44'/118'/0'/0/0",
            "m/44'/118'/0'/0/1",
            "m/44'/118'/0'/0/2",
          ],
        });
        
        const accounts = await wallet.getAccounts();
        const accountInfos: AccountInfo[] = accounts.map((acc, index) => ({
          address: acc.address,
          pubkey: Buffer.from(acc.pubkey).toString('base64'),
          name: `Account ${index + 1}`,
          hdPath: `m/44'/118'/0'/0/${index}`,
        }));
        
        set({
          accounts: accountInfos,
          isUnlocked: true,
          activeAccountIndex: 0,
        });
      },
      
      unlockWallet: async (password: string) => {
        const mnemonic = await SecureStorage.retrieveMnemonic(password);
        
        if (!mnemonic) {
          return false;
        }
        
        // Wallet is valid, mark as unlocked
        set({ isUnlocked: true });
        return true;
      },
      
      lockWallet: () => {
        set({ isUnlocked: false });
      },
      
      switchAccount: (index: number) => {
        const { accounts } = get();
        if (index >= 0 && index < accounts.length) {
          set({ activeAccountIndex: index });
        }
      },
      
      getActiveAccount: () => {
        const { accounts, activeAccountIndex } = get();
        return accounts[activeAccountIndex] || null;
      },
    }),
    {
      name: 'wallet-storage',
      storage: createJSONStorage(() => AsyncStorage),
      partialize: (state) => ({
        // Only persist non-sensitive data
        accounts: state.accounts,
        activeAccountIndex: state.activeAccountIndex,
      }),
    }
  )
);

// Hook for getting signing client
export function useSigningClient() {
  const [client, setClient] = useState<SigningStargateClient | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  
  const connect = useCallback(async (password: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const mnemonic = await SecureStorage.retrieveMnemonic(password);
      if (!mnemonic) {
        throw new Error('Failed to unlock wallet');
      }
      
      const wallet = await DirectSecp256k1HdWallet.fromMnemonic(mnemonic, {
        prefix: 'myapp',
      });
      
      const client = await MyAppChainClient.createSigningClient(
        'https://rpc.myappchain.com',
        wallet
      );
      
      setClient(client);
    } catch (err) {
      setError(err as Error);
    } finally {
      setLoading(false);
    }
  }, []);
  
  return { client, connect, loading, error };
}
```

### ✅ DO: Implement Biometric Authentication

```typescript
// src/services/BiometricService.ts
import TouchID from 'react-native-touch-id';
import * as Keychain from 'react-native-keychain';

export class BiometricService {
  static async isSupported(): Promise<boolean> {
    try {
      const biometryType = await TouchID.isSupported();
      return !!biometryType;
    } catch {
      return false;
    }
  }
  
  static async authenticate(reason: string): Promise<boolean> {
    const optionalConfigObject = {
      title: 'Authentication Required',
      imageColor: '#e00606',
      imageErrorColor: '#ff0000',
      sensorDescription: 'Touch sensor',
      sensorErrorDescription: 'Failed',
      cancelText: 'Cancel',
      fallbackLabel: 'Show Passcode',
      unifiedErrors: false,
      passcodeFallback: true,
    };
    
    try {
      await TouchID.authenticate(reason, optionalConfigObject);
      return true;
    } catch (error) {
      console.error('Biometric authentication failed:', error);
      return false;
    }
  }
  
  static async enableBiometricUnlock(password: string): Promise<boolean> {
    try {
      // First authenticate with biometrics
      const authenticated = await this.authenticate('Enable biometric unlock');
      if (!authenticated) {
        return false;
      }
      
      // Store password with biometric protection
      await Keychain.setInternetCredentials(
        'MyAppChainWallet-Biometric',
        'password',
        password,
        {
          accessible: Keychain.ACCESSIBLE.WHEN_UNLOCKED_THIS_DEVICE_ONLY,
          authenticatePrompt: 'Authenticate to unlock wallet',
          accessControl: Keychain.ACCESS_CONTROL.BIOMETRY_CURRENT_SET,
        }
      );
      
      return true;
    } catch (error) {
      console.error('Failed to enable biometric unlock:', error);
      return false;
    }
  }
  
  static async unlockWithBiometrics(): Promise<string | null> {
    try {
      const credentials = await Keychain.getInternetCredentials(
        'MyAppChainWallet-Biometric',
        {
          authenticationPrompt: {
            title: 'Unlock Wallet',
            subtitle: 'Use biometrics to access your wallet',
          },
        }
      );
      
      return credentials?.password || null;
    } catch (error) {
      console.error('Biometric unlock failed:', error);
      return null;
    }
  }
}
```

### ✅ DO: Implement Efficient State Synchronization

```typescript
// src/hooks/useChainSync.ts
import { useEffect, useRef, useCallback } from 'react';
import NetInfo from '@react-native-community/netinfo';
import BackgroundFetch from 'react-native-background-fetch';
import PushNotification from 'react-native-push-notification';
import { useQueryClient } from '@tanstack/react-query';

export function useChainSync() {
  const queryClient = useQueryClient();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  
  // Setup WebSocket connection for real-time updates
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }
    
    const ws = new WebSocket('wss://rpc.myappchain.com/websocket');
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      
      // Subscribe to account events
      const account = useWalletStore.getState().getActiveAccount();
      if (account) {
        ws.send(JSON.stringify({
          jsonrpc: '2.0',
          method: 'subscribe',
          params: {
            query: `tm.event='Tx' AND transfer.recipient='${account.address}'`,
          },
          id: 1,
        }));
      }
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      // Handle new transactions
      if (data.result?.data?.type === 'tendermint/event/Tx') {
        const txData = data.result.data.value;
        
        // Invalidate relevant queries
        queryClient.invalidateQueries(['balance']);
        queryClient.invalidateQueries(['transactions']);
        
        // Show notification
        PushNotification.localNotification({
          title: 'Transaction Received',
          message: `New transaction in your wallet`,
          playSound: true,
          soundName: 'default',
        });
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
      console.log('WebSocket closed, reconnecting...');
      reconnectTimeoutRef.current = setTimeout(connectWebSocket, 5000);
    };
    
    wsRef.current = ws;
  }, [queryClient]);
  
  // Setup background sync
  const setupBackgroundSync = useCallback(async () => {
    // Configure background fetch
    await BackgroundFetch.configure(
      {
        minimumFetchInterval: 15, // 15 minutes
        forceAlarmManager: false,
        stopOnTerminate: false,
        enableHeadless: true,
        startOnBoot: true,
      },
      async (taskId) => {
        console.log('[BackgroundFetch] taskId:', taskId);
        
        try {
          // Sync account data
          const account = useWalletStore.getState().getActiveAccount();
          if (account) {
            await queryClient.fetchQuery({
              queryKey: ['balance', account.address],
              queryFn: () => fetchBalance(account.address),
            });
            
            await queryClient.fetchQuery({
              queryKey: ['rewards', account.address],
              queryFn: () => fetchRewards(account.address),
            });
          }
          
          BackgroundFetch.finish(taskId);
        } catch (error) {
          console.error('[BackgroundFetch] Error:', error);
          BackgroundFetch.finish(taskId);
        }
      },
      (taskId) => {
        console.log('[BackgroundFetch] TIMEOUT taskId:', taskId);
        BackgroundFetch.finish(taskId);
      }
    );
  }, [queryClient]);
  
  // Monitor network connectivity
  useEffect(() => {
    const unsubscribe = NetInfo.addEventListener((state) => {
      if (state.isConnected && state.isInternetReachable) {
        connectWebSocket();
      } else {
        wsRef.current?.close();
      }
    });
    
    return () => {
      unsubscribe();
      wsRef.current?.close();
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [connectWebSocket]);
  
  // Setup background sync on mount
  useEffect(() => {
    setupBackgroundSync();
  }, [setupBackgroundSync]);
  
  return {
    isConnected: wsRef.current?.readyState === WebSocket.OPEN,
  };
}
```

---

## 8. Performance Optimization

Performance is critical for blockchain applications. Every operation costs gas, and inefficient code can make your chain unusable.

### ✅ DO: Optimize Store Access Patterns

```go
// x/tokenomics/keeper/rewards.go
package keeper

import (
    "cosmossdk.io/store/prefix"
    sdk "github.com/cosmos/cosmos-sdk/types"
)

// ❌ BAD: Multiple store reads in a loop
func (k Keeper) CalculateTotalRewardsBad(ctx sdk.Context) sdk.Coins {
    total := sdk.NewCoins()
    
    // This iterates over ALL users
    k.UserRewards.Walk(ctx, nil, func(key collections.Pair[sdk.AccAddress, string], value sdk.DecCoin) (bool, error) {
        total = total.Add(sdk.NewCoin(key.K2(), value.Amount.TruncateInt()))
        return false, nil
    })
    
    return total
}

// ✅ GOOD: Use aggregated values
func (k Keeper) CalculateTotalRewardsGood(ctx sdk.Context) sdk.Coins {
    // Maintain aggregated values that are updated on each change
    totalRewards, err := k.TotalRewards.Get(ctx)
    if err != nil {
        return sdk.NewCoins()
    }
    return totalRewards
}

// ✅ GOOD: Batch operations
func (k Keeper) DistributeRewards(ctx sdk.Context, recipients []RewardRecipient) error {
    // Validate all inputs first
    for _, recipient := range recipients {
        if err := recipient.Validate(); err != nil {
            return err
        }
    }
    
    // Check total amount in one operation
    totalAmount := sdk.NewCoins()
    for _, recipient := range recipients {
        totalAmount = totalAmount.Add(recipient.Amount...)
    }
    
    moduleBalance := k.bankKeeper.GetBalance(ctx, k.GetModuleAddress(), totalAmount[0].Denom)
    if moduleBalance.IsLT(totalAmount[0]) {
        return types.ErrInsufficientFunds
    }
    
    // Batch all state changes
    for _, recipient := range recipients {
        addr, _ := sdk.AccAddressFromBech32(recipient.Address)
        
        // Update user rewards
        for _, coin := range recipient.Amount {
            key := collections.Join(addr, coin.Denom)
            existing, _ := k.UserRewards.Get(ctx, key)
            newAmount := existing.Add(sdk.NewDecCoinFromCoin(coin))
            
            if err := k.UserRewards.Set(ctx, key, newAmount); err != nil {
                return err
            }
        }
    }
    
    // Single event for the batch
    ctx.EventManager().EmitTypedEvent(&types.EventBatchRewardsDistributed{
        Count:       uint32(len(recipients)),
        TotalAmount: totalAmount,
        Timestamp:   ctx.BlockTime(),
    })
    
    return nil
}
```

### ✅ DO: Use Caching for Expensive Computations

```go
// x/tokenomics/keeper/cache.go
package keeper

import (
    "sync"
    "time"
    
    sdk "github.com/cosmos/cosmos-sdk/types"
    "github.com/patrickmn/go-cache"
)

type KeeperCache struct {
    blockCache *cache.Cache
    mu         sync.RWMutex
}

func NewKeeperCache() *KeeperCache {
    return &KeeperCache{
        blockCache: cache.New(5*time.Second, 10*time.Second),
    }
}

// CachedAPY calculates APY with caching per block
func (k Keeper) CachedAPY(ctx sdk.Context, poolID string) (sdk.Dec, error) {
    cacheKey := fmt.Sprintf("apy:%s:%d", poolID, ctx.BlockHeight())
    
    // Check cache first
    if cached, found := k.cache.blockCache.Get(cacheKey); found {
        return cached.(sdk.Dec), nil
    }
    
    // Calculate expensive APY
    apy, err := k.calculateAPY(ctx, poolID)
    if err != nil {
        return sdk.Dec{}, err
    }
    
    // Cache for this block
    k.cache.blockCache.Set(cacheKey, apy, cache.DefaultExpiration)
    
    return apy, nil
}
```

### ✅ DO: Optimize Gas Usage

```go
// x/tokenomics/types/msgs.go
package types

// Implement efficient ValidateBasic to catch errors early
func (msg *MsgClaimRewards) ValidateBasic() error {
    // These checks run before the transaction enters the mempool
    // They don't cost gas if they fail
    
    _, err := sdk.AccAddressFromBech32(msg.Sender)
    if err != nil {
        return errorsmod.Wrapf(ErrInvalidAddress, "invalid sender address: %s", err)
    }
    
    // Add more validation
    if msg.Sender == "" {
        return errorsmod.Wrap(ErrInvalidAddress, "sender cannot be empty")
    }
    
    return nil
}

// Use efficient data structures in messages
type MsgBatchClaim struct {
    Sender string   `protobuf:"bytes,1,opt,name=sender,proto3" json:"sender,omitempty"`
    // Use repeated fields for batch operations
    Claims []Claim  `protobuf:"bytes,2,rep,name=claims,proto3" json:"claims"`
}

// This is more efficient than multiple individual messages
type Claim struct {
    PoolId string    `protobuf:"bytes,1,opt,name=pool_id,json=poolId,proto3" json:"pool_id,omitempty"`
    Amount sdk.Coins `protobuf:"bytes,2,rep,name=amount,proto3" json:"amount"`
}
```

---

## 9. Security Best Practices

Security is paramount in blockchain development. A single vulnerability can lead to loss of funds.

### ✅ DO: Implement Proper Access Control

```go
// x/tokenomics/keeper/access_control.go
package keeper

import (
    sdk "github.com/cosmos/cosmos-sdk/types"
    govtypes "github.com/cosmos/cosmos-sdk/x/gov/types"
)

// Role-based access control
type Role uint32

const (
    RoleUser Role = iota
    RoleAdmin
    RoleOperator
    RolePauser
)

// CheckAuthorization verifies permissions for sensitive operations
func (k Keeper) CheckAuthorization(ctx sdk.Context, address string, requiredRole Role) error {
    // Check if governance
    if k.authority == address {
        return nil // Governance can do anything
    }
    
    addr, err := sdk.AccAddressFromBech32(address)
    if err != nil {
        return errorsmod.Wrap(ErrInvalidAddress, err.Error())
    }
    
    // Get user role
    userRole, found := k.UserRoles.Get(ctx, addr)
    if !found {
        userRole = RoleUser // Default role
    }
    
    // Check permission hierarchy
    if userRole < requiredRole {
        return errorsmod.Wrapf(
            ErrUnauthorized,
            "insufficient permissions: required %v, got %v",
            requiredRole,
            userRole,
        )
    }
    
    return nil
}

// PausePool can only be called by pauser role or governance
func (k msgServer) PausePool(goCtx context.Context, msg *types.MsgPausePool) (*types.MsgPausePoolResponse, error) {
    ctx := sdk.UnwrapSDKContext(goCtx)
    
    // Check authorization
    if err := k.CheckAuthorization(ctx, msg.Authority, RolePauser); err != nil {
        return nil, err
    }
    
    // Validate pool exists
    pool, found := k.RewardPools.Get(ctx, msg.PoolId)
    if !found {
        return nil, errorsmod.Wrap(types.ErrPoolNotFound, msg.PoolId)
    }
    
    // Update status
    pool.Status = types.PoolStatus_PAUSED
    pool.PausedAt = ctx.BlockTime()
    pool.PausedBy = msg.Authority
    
    if err := k.RewardPools.Set(ctx, msg.PoolId, pool); err != nil {
        return nil, err
    }
    
    // Emit event
    ctx.EventManager().EmitTypedEvent(&types.EventPoolPaused{
        PoolId:   msg.PoolId,
        PausedBy: msg.Authority,
        PausedAt: ctx.BlockTime(),
    })
    
    return &types.MsgPausePoolResponse{}, nil
}
```

### ✅ DO: Validate All External Inputs

```go
// x/tokenomics/keeper/validation.go
package keeper

import (
    "fmt"
    "math/big"
    
    sdk "github.com/cosmos/cosmos-sdk/types"
)

// Comprehensive validation for reward distribution
func (k Keeper) ValidateRewardDistribution(
    ctx sdk.Context,
    poolID string,
    amount sdk.Coins,
    recipients []string,
) error {
    // Check pool exists and is active
    pool, found := k.RewardPools.Get(ctx, poolID)
    if !found {
        return errorsmod.Wrap(types.ErrPoolNotFound, poolID)
    }
    
    if pool.Status != types.PoolStatus_ACTIVE {
        return errorsmod.Wrapf(
            types.ErrPoolNotActive,
            "pool %s status is %s",
            poolID,
            pool.Status,
        )
    }
    
    // Validate amount
    if amount.IsZero() {
        return errorsmod.Wrap(types.ErrInvalidAmount, "amount cannot be zero")
    }
    
    if !amount.IsValid() {
        return errorsmod.Wrap(types.ErrInvalidAmount, "invalid coin denomination")
    }
    
    // Check for overflow
    for _, coin := range amount {
        if coin.Amount.GT(types.MaxRewardAmount) {
            return errorsmod.Wrapf(
                types.ErrAmountTooLarge,
                "amount %s exceeds maximum %s",
                coin.Amount,
                types.MaxRewardAmount,
            )
        }
    }
    
    // Validate recipients
    if len(recipients) == 0 {
        return errorsmod.Wrap(types.ErrNoRecipients, "at least one recipient required")
    }
    
    if len(recipients) > types.MaxBatchSize {
        return errorsmod.Wrapf(
            types.ErrBatchTooLarge,
            "batch size %d exceeds maximum %d",
            len(recipients),
            types.MaxBatchSize,
        )
    }
    
    // Check for duplicates and validate addresses
    seen := make(map[string]bool)
    for _, recipient := range recipients {
        if seen[recipient] {
            return errorsmod.Wrapf(
                types.ErrDuplicateRecipient,
                "duplicate recipient: %s",
                recipient,
            )
        }
        seen[recipient] = true
        
        if _, err := sdk.AccAddressFromBech32(recipient); err != nil {
            return errorsmod.Wrapf(
                types.ErrInvalidAddress,
                "invalid recipient address %s: %v",
                recipient,
                err,
            )
        }
    }
    
    // Check module has sufficient balance
    for _, coin := range amount {
        moduleBalance := k.bankKeeper.GetBalance(
            ctx,
            k.GetModuleAddress(),
            coin.Denom,
        )
        
        totalRequired := coin.Amount.Mul(sdk.NewInt(int64(len(recipients))))
        if moduleBalance.Amount.LT(totalRequired) {
            return errorsmod.Wrapf(
                types.ErrInsufficientModuleBalance,
                "module balance %s insufficient for distribution of %s to %d recipients",
                moduleBalance,
                coin,
                len(recipients),
            )
        }
    }
    
    return nil
}
```

### ✅ DO: Implement Invariant Checks

```go
// x/tokenomics/keeper/invariants.go
package keeper

import (
    "fmt"
    
    sdk "github.com/cosmos/cosmos-sdk/types"
)

// RegisterInvariants registers all module invariants
func RegisterInvariants(ir sdk.InvariantRegistry, k Keeper) {
    ir.RegisterRoute(types.ModuleName, "module-balance", ModuleBalanceInvariant(k))
    ir.RegisterRoute(types.ModuleName, "total-rewards", TotalRewardsInvariant(k))
    ir.RegisterRoute(types.ModuleName, "pool-consistency", PoolConsistencyInvariant(k))
}

// ModuleBalanceInvariant checks that module balance >= sum of all pending rewards
func ModuleBalanceInvariant(k Keeper) sdk.Invariant {
    return func(ctx sdk.Context) (string, bool) {
        var (
            broken bool
            msg    string
        )
        
        // Calculate total pending rewards
        totalPending := sdk.NewCoins()
        
        err := k.UserRewards.Walk(ctx, nil, func(key collections.Pair[sdk.AccAddress, string], value sdk.DecCoin) (bool, error) {
            coin := sdk.NewCoin(key.K2(), value.Amount.TruncateInt())
            totalPending = totalPending.Add(coin)
            return false, nil
        })
        
        if err != nil {
            return fmt.Sprintf("failed to iterate user rewards: %v", err), true
        }
        
        // Get module balance
        moduleAddr := k.GetModuleAddress()
        moduleBalance := k.bankKeeper.GetAllBalances(ctx, moduleAddr)
        
        // Check invariant
        if !moduleBalance.IsAllGTE(totalPending) {
            broken = true
            msg = fmt.Sprintf(
                "module balance %s < total pending rewards %s",
                moduleBalance,
                totalPending,
            )
        }
        
        return msg, broken
    }
}

// TotalRewardsInvariant ensures sum of user rewards equals tracked total
func TotalRewardsInvariant(k Keeper) sdk.Invariant {
    return func(ctx sdk.Context) (string, bool) {
        // Get tracked total
        trackedTotal, err := k.TotalRewards.Get(ctx)
        if err != nil {
            // If not set, should be zero
            trackedTotal = sdk.NewCoins()
        }
        
        // Calculate actual total
        actualTotal := sdk.NewCoins()
        
        err = k.UserRewards.Walk(ctx, nil, func(key collections.Pair[sdk.AccAddress, string], value sdk.DecCoin) (bool, error) {
            coin := sdk.NewCoin(key.K2(), value.Amount.TruncateInt())
            actualTotal = actualTotal.Add(coin)
            return false, nil
        })
        
        if err != nil {
            return fmt.Sprintf("failed to calculate total: %v", err), true
        }
        
        // Compare
        if !trackedTotal.IsEqual(actualTotal) {
            return fmt.Sprintf(
                "tracked total %s != actual total %s",
                trackedTotal,
                actualTotal,
            ), true
        }
        
        return "", false
    }
}
```

---

## 10. Production Deployment

Deploying a Cosmos chain requires careful planning and monitoring.

### ✅ DO: Use Cosmovisor for Automatic Upgrades

```bash
#!/bin/bash
# setup-cosmovisor.sh

# Install cosmovisor
go install cosmossdk.io/tools/cosmovisor/cmd/cosmovisor@latest

# Set up directory structure
export DAEMON_NAME=myappchaind
export DAEMON_HOME=$HOME/.myappchain

mkdir -p $DAEMON_HOME/cosmovisor/genesis/bin
mkdir -p $DAEMON_HOME/cosmovisor/upgrades

# Copy initial binary
cp $(which myappchaind) $DAEMON_HOME/cosmovisor/genesis/bin/

# Create service file
sudo tee /etc/systemd/system/myappchain.service > /dev/null <<EOF
[Unit]
Description=MyAppChain Node
After=network-online.target

[Service]
Type=simple
User=$USER
ExecStart=$(which cosmovisor) run start
Restart=always
RestartSec=3
Environment="DAEMON_HOME=$HOME/.myappchain"
Environment="DAEMON_NAME=myappchaind"
Environment="UNSAFE_SKIP_BACKUP=false"
Environment="DAEMON_RESTART_AFTER_UPGRADE=true"
Environment="DAEMON_ALLOW_DOWNLOAD_BINARIES=false"
Environment="DAEMON_LOG_BUFFER_SIZE=512"

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$HOME/.myappchain

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable myappchain
sudo systemctl start myappchain
```

### ✅ DO: Configure State Sync for Fast Node Bootstrap

```toml
# config.toml
[statesync]
enable = true
rpc_servers = "https://rpc1.myappchain.com:443,https://rpc2.myappchain.com:443"
trust_height = 1000000
trust_hash = "E9F2F65B69B32B2A0E77C3DD94E6CF8A0F8F2A92CD8F8A6C09F876543210ABCD"
trust_period = "168h0m0s"

# Snapshot configuration
[snapshot]
interval = 1000
keep_recent = 2
```

### ✅ DO: Implement Comprehensive Monitoring

```go
// x/tokenomics/module.go
package tokenomics

import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

var (
    rewardsClaimedTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "myappchain_rewards_claimed_total",
            Help: "Total number of reward claims",
        },
        []string{"denom"},
    )
    
    rewardsClaimedAmount = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "myappchain_rewards_claimed_amount",
            Help: "Total amount of rewards claimed",
        },
        []string{"denom"},
    )
    
    activeRewardPools = promauto.NewGauge(
        prometheus.GaugeOpts{
            Name: "myappchain_active_reward_pools",
            Help: "Number of active reward pools",
        },
    )
    
    pendingRewardsTotal = promauto.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "myappchain_pending_rewards_total",
            Help: "Total pending rewards by denom",
        },
        []string{"denom"},
    )
)

// UpdateMetrics updates Prometheus metrics
func (k Keeper) UpdateMetrics(ctx sdk.Context) {
    // Count active pools
    activeCount := 0
    k.RewardPools.Walk(ctx, nil, func(_ string, pool types.RewardPool) (bool, error) {
        if pool.Status == types.PoolStatus_ACTIVE {
            activeCount++
        }
        return false, nil
    })
    activeRewardPools.Set(float64(activeCount))
    
    // Calculate pending rewards by denom
    pendingByDenom := make(map[string]sdk.Int)
    k.UserRewards.Walk(ctx, nil, func(_ collections.Pair[sdk.AccAddress, string], value sdk.DecCoin) (bool, error) {
        current := pendingByDenom[value.Denom]
        pendingByDenom[value.Denom] = current.Add(value.Amount.TruncateInt())
        return false, nil
    })
    
    for denom, amount := range pendingByDenom {
        pendingRewardsTotal.WithLabelValues(denom).Set(float64(amount.Int64()))
    }
}

// EndBlock updates metrics
func (am AppModule) EndBlock(ctx sdk.Context) {
    am.keeper.UpdateMetrics(ctx)
}
```

### ✅ DO: Configure Proper Logging

```go
// x/tokenomics/keeper/logger.go
package keeper

import (
    "cosmossdk.io/log"
    sdk "github.com/cosmos/cosmos-sdk/types"
)

// LogRewardEvent logs structured reward events
func (k Keeper) LogRewardEvent(ctx sdk.Context, event string, attrs ...interface{}) {
    logger := k.Logger(ctx).With(
        "module", types.ModuleName,
        "height", ctx.BlockHeight(),
        "time", ctx.BlockTime(),
    )
    
    // Add attributes in pairs
    for i := 0; i < len(attrs)-1; i += 2 {
        logger = logger.With(attrs[i], attrs[i+1])
    }
    
    logger.Info(event)
}

// Usage in handler
func (k msgServer) ClaimRewards(goCtx context.Context, msg *types.MsgClaimRewards) (*types.MsgClaimRewardsResponse, error) {
    ctx := sdk.UnwrapSDKContext(goCtx)
    
    // ... claim logic ...
    
    k.LogRewardEvent(ctx, "rewards_claimed",
        "recipient", msg.Sender,
        "amount", claimedAmount.String(),
        "tx_hash", ctx.TxBytes(),
    )
    
    return response, nil
}
```

---

## 11. Advanced Patterns

### ✅ DO: Implement Module Hooks for Cross-Module Communication

```go
// x/tokenomics/types/hooks.go
package types

import (
    sdk "github.com/cosmos/cosmos-sdk/types"
)

// TokenomicsHooks defines hooks for other modules to interact
type TokenomicsHooks interface {
    AfterValidatorBonded(ctx sdk.Context, consAddr sdk.ConsAddress, validator sdk.ValAddress) error
    AfterValidatorBeginUnbonding(ctx sdk.Context, consAddr sdk.ConsAddress, validator sdk.ValAddress) error
    AfterDelegationModified(ctx sdk.Context, delAddr sdk.AccAddress, valAddr sdk.ValAddress) error
}

// Multi-hook pattern for multiple implementations
type MultiTokenomicsHooks []TokenomicsHooks

func NewMultiTokenomicsHooks(hooks ...TokenomicsHooks) MultiTokenomicsHooks {
    return hooks
}

func (h MultiTokenomicsHooks) AfterValidatorBonded(ctx sdk.Context, consAddr sdk.ConsAddress, validator sdk.ValAddress) error {
    for i := range h {
        if err := h[i].AfterValidatorBonded(ctx, consAddr, validator); err != nil {
            return err
        }
    }
    return nil
}

// Implement in keeper
func (k Keeper) AfterValidatorBonded(ctx sdk.Context, consAddr sdk.ConsAddress, validator sdk.ValAddress) error {
    // Update validator rewards
    k.LogRewardEvent(ctx, "validator_bonded",
        "validator", validator.String(),
        "consensus_addr", consAddr.String(),
    )
    
    // Initialize validator reward pool
    return k.InitializeValidatorRewards(ctx, validator)
}
```

### ✅ DO: Implement Upgrade Handlers

```go
// app/upgrades/v2/upgrade.go
package v2

import (
    storetypes "cosmossdk.io/store/types"
    "github.com/cosmos/cosmos-sdk/types/module"
    upgradetypes "github.com/cosmos/cosmos-sdk/x/upgrade/types"
    
    "github.com/myorg/myappchain/x/tokenomics"
)

const UpgradeName = "v2"

func CreateUpgradeHandler(
    mm *module.Manager,
    configurator module.Configurator,
    keepers *keepers.AppKeepers,
) upgradetypes.UpgradeHandler {
    return func(ctx sdk.Context, plan upgradetypes.Plan, fromVM module.VersionMap) (module.VersionMap, error) {
        logger := ctx.Logger().With("upgrade", UpgradeName)
        logger.Info("running module migrations...")
        
        // Add new store keys if needed
        // This happens automatically with configurator
        
        // Run module migrations
        vm, err := mm.RunMigrations(ctx, configurator, fromVM)
        if err != nil {
            return nil, err
        }
        
        // Custom state migrations
        if err := migrateTokenomicsState(ctx, keepers.TokenomicsKeeper); err != nil {
            return nil, err
        }
        
        logger.Info("upgrade complete")
        return vm, nil
    }
}

// Custom migration logic
func migrateTokenomicsState(ctx sdk.Context, k tokenomicskeeper.Keeper) error {
    // Example: Migrate old reward format to new format
    oldRewards := make(map[string]sdk.Coins)
    
    // Read old state
    store := ctx.KVStore(k.storeKey)
    iterator := sdk.KVStorePrefixIterator(store, []byte("old_rewards/"))
    defer iterator.Close()
    
    for ; iterator.Valid(); iterator.Next() {
        var oldReward OldRewardFormat
        k.cdc.MustUnmarshal(iterator.Value(), &oldReward)
        
        // Convert to new format
        newReward := types.UserReward{
            Address: oldReward.Address,
            Denom:   oldReward.Denom,
            Amount:  sdk.NewDecFromInt(oldReward.Amount),
        }
        
        // Store in new location using collections
        addr, _ := sdk.AccAddressFromBech32(newReward.Address)
        k.UserRewards.Set(ctx, collections.Join(addr, newReward.Denom), newReward.Amount)
        
        // Delete old entry
        store.Delete(iterator.Key())
    }
    
    return nil
}

// Register in app.go
func (app *App) RegisterUpgradeHandlers() {
    app.UpgradeKeeper.SetUpgradeHandler(
        v2.UpgradeName,
        v2.CreateUpgradeHandler(app.mm, app.configurator, &app.AppKeepers),
    )
    
    // For testnet, add store during upgrade
    upgradeInfo, err := app.UpgradeKeeper.ReadUpgradeInfoFromDisk()
    if err != nil {
        panic(err)
    }
    
    if upgradeInfo.Name == v2.UpgradeName && !app.UpgradeKeeper.IsSkipHeight(upgradeInfo.Height) {
        storeUpgrades := storetypes.StoreUpgrades{
            Added: []string{newmoduletypes.StoreKey},
        }
        
        app.SetStoreLoader(upgradetypes.UpgradeStoreLoader(upgradeInfo.Height, &storeUpgrades))
    }
}
```

---

## 12. Testing Infrastructure

### ✅ DO: Use Test Fixtures and Builders

```go
// testutil/fixtures.go
package testutil

import (
    sdk "github.com/cosmos/cosmos-sdk/types"
    "github.com/myorg/myappchain/x/tokenomics/types"
)

// TestDataBuilder provides fluent API for test data
type TestDataBuilder struct {
    pools      []types.RewardPool
    rewards    map[string]sdk.Coins
    validators []TestValidator
}

func NewTestDataBuilder() *TestDataBuilder {
    return &TestDataBuilder{
        rewards: make(map[string]sdk.Coins),
    }
}

func (b *TestDataBuilder) WithRewardPool(id, denom string, amount int64, duration time.Duration) *TestDataBuilder {
    pool := types.RewardPool{
        Id:           id,
        RewardDenom:  denom,
        TotalRewards: sdk.NewCoin(denom, sdk.NewInt(amount)),
        StartTime:    time.Now(),
        EndTime:      time.Now().Add(duration),
        Status:       types.PoolStatus_ACTIVE,
    }
    b.pools = append(b.pools, pool)
    return b
}

func (b *TestDataBuilder) WithUserReward(address string, coins ...sdk.Coin) *TestDataBuilder {
    b.rewards[address] = sdk.NewCoins(coins...)
    return b
}

func (b *TestDataBuilder) WithValidator(moniker string, tokens int64) *TestDataBuilder {
    b.validators = append(b.validators, TestValidator{
        Moniker: moniker,
        Tokens:  sdk.NewInt(tokens),
    })
    return b
}

func (b *TestDataBuilder) Build(ctx sdk.Context, k keeper.Keeper) error {
    // Create pools
    for _, pool := range b.pools {
        if err := k.RewardPools.Set(ctx, pool.Id, pool); err != nil {
            return err
        }
    }
    
    // Set rewards
    for address, coins := range b.rewards {
        addr, _ := sdk.AccAddressFromBech32(address)
        for _, coin := range coins {
            key := collections.Join(addr, coin.Denom)
            if err := k.UserRewards.Set(ctx, key, sdk.NewDecCoinFromCoin(coin)); err != nil {
                return err
            }
        }
    }
    
    return nil
}

// Usage in tests
func TestComplexScenario(t *testing.T) {
    // Setup
    ctx, k := setupKeeper(t)
    
    // Build test data fluently
    err := NewTestDataBuilder().
        WithRewardPool("staking-rewards", "utoken", 1000000, 30*24*time.Hour).
        WithRewardPool("lp-rewards", "utoken", 500000, 7*24*time.Hour).
        WithUserReward("cosmos1...", sdk.NewCoin("utoken", sdk.NewInt(1000))).
        WithUserReward("cosmos2...", sdk.NewCoin("utoken", sdk.NewInt(2000))).
        WithValidator("validator1", 1000000).
        Build(ctx, k)
    
    require.NoError(t, err)
    
    // Test logic...
}
```

### ✅ DO: Implement Simulation Tests

```go
// x/tokenomics/simulation/operations.go
package simulation

import (
    "math/rand"
    
    "github.com/cosmos/cosmos-sdk/baseapp"
    sdk "github.com/cosmos/cosmos-sdk/types"
    simtypes "github.com/cosmos/cosmos-sdk/types/simulation"
    "github.com/cosmos/cosmos-sdk/x/simulation"
    
    "github.com/myorg/myappchain/x/tokenomics/keeper"
    "github.com/myorg/myappchain/x/tokenomics/types"
)

// Simulation operation weights
const (
    OpWeightMsgClaimRewards = "op_weight_msg_claim_rewards"
    OpWeightMsgCreatePool   = "op_weight_msg_create_pool"
    
    DefaultWeightMsgClaimRewards = 100
    DefaultWeightMsgCreatePool   = 20
)

// WeightedOperations returns all the operations from the module with their respective weights
func WeightedOperations(
    appParams simtypes.AppParams,
    cdc codec.JSONCodec,
    ak types.AccountKeeper,
    bk types.BankKeeper,
    k keeper.Keeper,
) simulation.WeightedOperations {
    var (
        weightMsgClaimRewards int
        weightMsgCreatePool   int
    )
    
    appParams.GetOrGenerate(cdc, OpWeightMsgClaimRewards, &weightMsgClaimRewards, nil,
        func(_ *rand.Rand) {
            weightMsgClaimRewards = DefaultWeightMsgClaimRewards
        },
    )
    
    appParams.GetOrGenerate(cdc, OpWeightMsgCreatePool, &weightMsgCreatePool, nil,
        func(_ *rand.Rand) {
            weightMsgCreatePool = DefaultWeightMsgCreatePool
        },
    )
    
    return simulation.WeightedOperations{
        simulation.NewWeightedOperation(
            weightMsgClaimRewards,
            SimulateMsgClaimRewards(ak, bk, k),
        ),
        simulation.NewWeightedOperation(
            weightMsgCreatePool,
            SimulateMsgCreatePool(ak, bk, k),
        ),
    }
}

// SimulateMsgClaimRewards generates a MsgClaimRewards with random values
func SimulateMsgClaimRewards(ak types.AccountKeeper, bk types.BankKeeper, k keeper.Keeper) simtypes.Operation {
    return func(
        r *rand.Rand, app *baseapp.BaseApp, ctx sdk.Context,
        accs []simtypes.Account, chainID string,
    ) (simtypes.OperationMsg, []simtypes.FutureOperation, error) {
        // Select random account with rewards
        var simAccount simtypes.Account
        var hasRewards bool
        
        // Try to find an account with rewards
        for _, acc := range accs {
            rewards, err := k.GetPendingRewards(ctx, acc.Address)
            if err == nil && !rewards.IsZero() {
                simAccount = acc
                hasRewards = true
                break
            }
        }
        
        // If no account has rewards, skip
        if !hasRewards {
            return simtypes.NoOpMsg(types.ModuleName, types.TypeMsgClaimRewards, "no accounts with rewards"), nil, nil
        }
        
        msg := &types.MsgClaimRewards{
            Sender: simAccount.Address.String(),
        }
        
        // Generate tx
        txCtx := simulation.OperationInput{
            R:             r,
            App:           app,
            TxGen:         app.TxConfig(),
            Cdc:           nil,
            Msg:           msg,
            Context:       ctx,
            SimAccount:    simAccount,
            AccountKeeper: ak,
            Bankkeeper:    bk,
            ModuleName:    types.ModuleName,
        }
        
        return simulation.GenAndDeliverTxWithRandFees(txCtx)
    }
}
```

---

## 13. CLI and Integration

### ✅ DO: Provide Comprehensive CLI Commands

```go
// x/tokenomics/client/cli/tx.go
package cli

import (
    "fmt"
    "strings"
    
    "github.com/spf13/cobra"
    "github.com/cosmos/cosmos-sdk/client"
    "github.com/cosmos/cosmos-sdk/client/flags"
    "github.com/cosmos/cosmos-sdk/client/tx"
    sdk "github.com/cosmos/cosmos-sdk/types"
    
    "github.com/myorg/myappchain/x/tokenomics/types"
)

// GetTxCmd returns the transaction commands for this module
func GetTxCmd() *cobra.Command {
    cmd := &cobra.Command{
        Use:                        types.ModuleName,
        Short:                      fmt.Sprintf("%s transactions subcommands", types.ModuleName),
        DisableFlagParsing:         true,
        SuggestionsMinimumDistance: 2,
        RunE:                       client.ValidateCmd,
    }
    
    cmd.AddCommand(
        CmdClaimRewards(),
        CmdCreateRewardPool(),
        CmdBatchClaim(),
    )
    
    return cmd
}

// CmdClaimRewards returns a CLI command for claiming rewards
func CmdClaimRewards() *cobra.Command {
    cmd := &cobra.Command{
        Use:   "claim-rewards [pool-id]",
        Short: "Claim pending rewards from a specific pool or all pools",
        Long: strings.TrimSpace(
            fmt.Sprintf(`Claim your pending rewards from the tokenomics module.

Examples:
$ %s tx %s claim-rewards --from mykey
$ %s tx %s claim-rewards staking-rewards --from mykey
$ %s tx %s claim-rewards --pool-ids=pool1,pool2,pool3 --from mykey
`,
                version.AppName, types.ModuleName,
                version.AppName, types.ModuleName,
                version.AppName, types.ModuleName,
            ),
        ),
        Args: cobra.MaximumNArgs(1),
        RunE: func(cmd *cobra.Command, args []string) error {
            clientCtx, err := client.GetClientTxContext(cmd)
            if err != nil {
                return err
            }
            
            // Get pool IDs from args or flags
            var poolIDs []string
            if len(args) > 0 {
                poolIDs = []string{args[0]}
            } else {
                poolIDsStr, _ := cmd.Flags().GetString("pool-ids")
                if poolIDsStr != "" {
                    poolIDs = strings.Split(poolIDsStr, ",")
                }
            }
            
            msg := &types.MsgClaimRewards{
                Sender:  clientCtx.GetFromAddress().String(),
                PoolIds: poolIDs,
            }
            
            return tx.GenerateOrBroadcastTxCLI(clientCtx, cmd.Flags(), msg)
        },
    }
    
    flags.AddTxFlagsToCmd(cmd)
    cmd.Flags().String("pool-ids", "", "Comma-separated list of pool IDs to claim from")
    
    return cmd
}

// CmdBatchClaim for validators to claim on behalf of delegators
func CmdBatchClaim() *cobra.Command {
    cmd := &cobra.Command{
        Use:   "batch-claim [delegators-file]",
        Short: "Claim rewards for multiple delegators (validator only)",
        Long: strings.TrimSpace(
            fmt.Sprintf(`Claim rewards on behalf of multiple delegators.
This command is restricted to validators to help their delegators claim rewards efficiently.

The delegators file should be a JSON file with the following format:
[
  {
    "address": "cosmos1...",
    "pool_ids": ["pool1", "pool2"]
  },
  {
    "address": "cosmos2...",
    "pool_ids": ["pool1"]
  }
]

Example:
$ %s tx %s batch-claim delegators.json --from validator-key --fees 1000utoken
`,
                version.AppName, types.ModuleName,
            ),
        ),
        Args: cobra.ExactArgs(1),
        RunE: func(cmd *cobra.Command, args []string) error {
            clientCtx, err := client.GetClientTxContext(cmd)
            if err != nil {
                return err
            }
            
            // Read delegators file
            delegators, err := ParseDelegatorsFile(args[0])
            if err != nil {
                return fmt.Errorf("failed to parse delegators file: %w", err)
            }
            
            // Validate batch size
            maxBatchSize, _ := cmd.Flags().GetUint32("max-batch-size")
            if maxBatchSize == 0 {
                maxBatchSize = 100
            }
            
            if len(delegators) > int(maxBatchSize) {
                return fmt.Errorf("batch size %d exceeds maximum %d", len(delegators), maxBatchSize)
            }
            
            msg := &types.MsgBatchClaimRewards{
                Validator:  clientCtx.GetFromAddress().String(),
                Claims:     delegators,
            }
            
            return tx.GenerateOrBroadcastTxCLI(clientCtx, cmd.Flags(), msg)
        },
    }
    
    flags.AddTxFlagsToCmd(cmd)
    cmd.Flags().Uint32("max-batch-size", 100, "Maximum number of claims in a single batch")
    
    return cmd
}
```

### ✅ DO: Add Rich Query Commands

```go
// x/tokenomics/client/cli/query.go
package cli

import (
    "context"
    "fmt"
    
    "github.com/spf13/cobra"
    "github.com/cosmos/cosmos-sdk/client"
    "github.com/cosmos/cosmos-sdk/client/flags"
    
    "github.com/myorg/myappchain/x/tokenomics/types"
)

// GetQueryCmd returns the cli query commands for this module
func GetQueryCmd(queryRoute string) *cobra.Command {
    cmd := &cobra.Command{
        Use:                        types.ModuleName,
        Short:                      fmt.Sprintf("Querying commands for the %s module", types.ModuleName),
        DisableFlagParsing:         true,
        SuggestionsMinimumDistance: 2,
        RunE:                       client.ValidateCmd,
    }
    
    cmd.AddCommand(
        CmdQueryRewards(),
        CmdQueryPools(),
        CmdQueryAPY(),
        CmdQueryClaimHistory(),
    )
    
    return cmd
}

// CmdQueryRewards queries user rewards with filtering
func CmdQueryRewards() *cobra.Command {
    cmd := &cobra.Command{
        Use:   "rewards [address]",
        Short: "Query pending rewards for an address",
        Long: strings.TrimSpace(
            fmt.Sprintf(`Query all pending rewards for a specific address with optional filtering.

Examples:
$ %s query %s rewards cosmos1...
$ %s query %s rewards cosmos1... --denom utoken
$ %s query %s rewards cosmos1... --pool-id staking-rewards
$ %s query %s rewards cosmos1... --min-amount 1000
`,
                version.AppName, types.ModuleName,
                version.AppName, types.ModuleName,
                version.AppName, types.ModuleName,
                version.AppName, types.ModuleName,
            ),
        ),
        Args: cobra.ExactArgs(1),
        RunE: func(cmd *cobra.Command, args []string) error {
            clientCtx, err := client.GetClientQueryContext(cmd)
            if err != nil {
                return err
            }
            
            queryClient := types.NewQueryClient(clientCtx)
            
            // Build request with filters
            req := &types.QueryUserRewardsRequest{
                Address: args[0],
            }
            
            // Add optional filters
            denom, _ := cmd.Flags().GetString("denom")
            if denom != "" {
                req.Denom = denom
            }
            
            poolID, _ := cmd.Flags().GetString("pool-id")
            if poolID != "" {
                req.PoolId = poolID
            }
            
            pageReq, err := client.ReadPageRequest(cmd.Flags())
            if err != nil {
                return err
            }
            req.Pagination = pageReq
            
            res, err := queryClient.UserRewards(context.Background(), req)
            if err != nil {
                return err
            }
            
            // Custom formatting for better readability
            if outputFormat, _ := cmd.Flags().GetString(cli.OutputFlag); outputFormat == "text" {
                return printRewardsTable(res.Rewards)
            }
            
            return clientCtx.PrintProto(res)
        },
    }
    
    flags.AddQueryFlagsToCmd(cmd)
    cmd.Flags().String("denom", "", "Filter by reward denomination")
    cmd.Flags().String("pool-id", "", "Filter by specific pool")
    cmd.Flags().String("min-amount", "", "Minimum reward amount to display")
    
    return cmd
}

// CmdQueryAPY shows current APY for all pools
func CmdQueryAPY() *cobra.Command {
    cmd := &cobra.Command{
        Use:   "apy",
        Short: "Query current APY for all reward pools",
        Long: strings.TrimSpace(
            fmt.Sprintf(`Query the current Annual Percentage Yield (APY) for all active reward pools.

Examples:
$ %s query %s apy
$ %s query %s apy --pool-id staking-rewards
$ %s query %s apy --min-apy 10
`,
                version.AppName, types.ModuleName,
                version.AppName, types.ModuleName,
                version.AppName, types.ModuleName,
            ),
        ),
        RunE: func(cmd *cobra.Command, args []string) error {
            clientCtx, err := client.GetClientQueryContext(cmd)
            if err != nil {
                return err
            }
            
            queryClient := types.NewQueryClient(clientCtx)
            
            // Query APY data
            res, err := queryClient.PoolAPYs(context.Background(), &types.QueryPoolAPYsRequest{
                PoolId: poolID, // empty for all pools
            })
            if err != nil {
                return err
            }
            
            // Filter by minimum APY if specified
            minAPY, _ := cmd.Flags().GetFloat64("min-apy")
            
            // Custom table output
            if outputFormat, _ := cmd.Flags().GetString(cli.OutputFlag); outputFormat == "text" {
                fmt.Println("Pool Rewards APY")
                fmt.Println("=====================================")
                for _, apy := range res.Apys {
                    if apy.Apy >= minAPY {
                        fmt.Printf("%-20s %8.2f%% %s\n", 
                            apy.PoolId, 
                            apy.Apy,
                            getAPYTrend(apy.Change7d),
                        )
                    }
                }
                return nil
            }
            
            return clientCtx.PrintProto(res)
        },
    }
    
    flags.AddQueryFlagsToCmd(cmd)
    cmd.Flags().String("pool-id", "", "Query APY for specific pool")
    cmd.Flags().Float64("min-apy", 0, "Show only pools with APY above this value")
    
    return cmd
}
```

---

## 14. Observability and Debugging

### ✅ DO: Add Comprehensive Event Logging

```go
// x/tokenomics/keeper/events.go
package keeper

import (
    sdk "github.com/cosmos/cosmos-sdk/types"
    "github.com/myorg/myappchain/x/tokenomics/types"
)

// EmitRewardClaimedEvent emits a detailed event for reward claims
func (k Keeper) EmitRewardClaimedEvent(
    ctx sdk.Context,
    recipient sdk.AccAddress,
    amount sdk.Coins,
    poolIDs []string,
) {
    // Emit typed event (for clients using gRPC streaming)
    ctx.EventManager().EmitTypedEvent(&types.EventRewardsClaimed{
        Recipient:   recipient.String(),
        Amount:      amount,
        PoolIds:     poolIDs,
        ClaimedAt:   ctx.BlockTime(),
        BlockHeight: ctx.BlockHeight(),
    })
    
    // Also emit legacy string events for backward compatibility
    ctx.EventManager().EmitEvents(sdk.Events{
        sdk.NewEvent(
            types.EventTypeRewardClaim,
            sdk.NewAttribute(types.AttributeKeyRecipient, recipient.String()),
            sdk.NewAttribute(types.AttributeKeyAmount, amount.String()),
            sdk.NewAttribute(types.AttributeKeyPoolIDs, strings.Join(poolIDs, ",")),
            sdk.NewAttribute(types.AttributeKeyTimestamp, ctx.BlockTime().Format(time.RFC3339)),
            sdk.NewAttribute(types.AttributeKeyHeight, fmt.Sprintf("%d", ctx.BlockHeight())),
        ),
    })
}

// EmitPoolCreatedEvent emits event for new pool creation
func (k Keeper) EmitPoolCreatedEvent(ctx sdk.Context, pool types.RewardPool) {
    ctx.EventManager().EmitTypedEvent(&types.EventPoolCreated{
        PoolId:       pool.Id,
        RewardDenom:  pool.RewardDenom,
        TotalRewards: pool.TotalRewards,
        StartTime:    pool.StartTime,
        EndTime:      pool.EndTime,
        Creator:      pool.Creator,
    })
}
```

### ✅ DO: Add Debug Endpoints for Development

```go
// x/tokenomics/keeper/debug.go
// Only include in debug builds
//go:build debug

package keeper

import (
    "encoding/json"
    "fmt"
    
    sdk "github.com/cosmos/cosmos-sdk/types"
)

// DebugState exports the entire module state for debugging
func (k Keeper) DebugState(ctx sdk.Context) (json.RawMessage, error) {
    state := struct {
        Pools        []types.RewardPool          `json:"pools"`
        UserRewards  map[string]sdk.Coins        `json:"user_rewards"`
        TotalRewards sdk.Coins                   `json:"total_rewards"`
        Params       types.Params                `json:"params"`
        BlockHeight  int64                       `json:"block_height"`
        BlockTime    string                      `json:"block_time"`
    }{
        Pools:        []types.RewardPool{},
        UserRewards:  make(map[string]sdk.Coins),
        BlockHeight:  ctx.BlockHeight(),
        BlockTime:    ctx.BlockTime().Format(time.RFC3339),
    }
    
    // Export all pools
    k.RewardPools.Walk(ctx, nil, func(id string, pool types.RewardPool) (bool, error) {
        state.Pools = append(state.Pools, pool)
        return false, nil
    })
    
    // Export all user rewards
    k.UserRewards.Walk(ctx, nil, func(key collections.Pair[sdk.AccAddress, string], value sdk.DecCoin) (bool, error) {
        addr := key.K1().String()
        coin := sdk.NewCoin(key.K2(), value.Amount.TruncateInt())
        state.UserRewards[addr] = state.UserRewards[addr].Add(coin)
        return false, nil
    })
    
    // Get total rewards
    state.TotalRewards, _ = k.TotalRewards.Get(ctx)
    state.Params = k.GetParams(ctx)
    
    return json.MarshalIndent(state, "", "  ")
}

// DebugInvariantCheck runs all invariants and returns detailed results
func (k Keeper) DebugInvariantCheck(ctx sdk.Context) map[string]string {
    results := make(map[string]string)
    
    invariants := []struct {
        name string
        fn   sdk.Invariant
    }{
        {"module-balance", ModuleBalanceInvariant(k)},
        {"total-rewards", TotalRewardsInvariant(k)},
        {"pool-consistency", PoolConsistencyInvariant(k)},
        {"reward-schedule", RewardScheduleInvariant(k)},
    }
    
    for _, inv := range invariants {
        msg, broken := inv.fn(ctx)
        if broken {
            results[inv.name] = fmt.Sprintf("BROKEN: %s", msg)
        } else {
            results[inv.name] = "OK"
        }
    }
    
    return results
}
```

### ✅ DO: Implement Detailed Telemetry

```go
// x/tokenomics/telemetry/telemetry.go
package telemetry

import (
    "time"
    
    "github.com/cosmos/cosmos-sdk/telemetry"
    sdk "github.com/cosmos/cosmos-sdk/types"
)

// MeasureRewardClaim tracks reward claim metrics
func MeasureRewardClaim(start time.Time, denom string, amount sdk.Int) {
    defer telemetry.ModuleMeasureSince(types.ModuleName, start, "claim_rewards_duration")
    
    telemetry.IncrCounterWithLabels(
        []string{"tokenomics", "claims", "count"},
        1,
        []metrics.Label{
            telemetry.NewLabel("denom", denom),
        },
    )
    
    if amount.IsInt64() {
        telemetry.SetGaugeWithLabels(
            []string{"tokenomics", "claims", "amount"},
            float32(amount.Int64()),
            []metrics.Label{
                telemetry.NewLabel("denom", denom),
            },
        )
    }
}

// MeasurePoolDistribution tracks distribution performance
func MeasurePoolDistribution(start time.Time, poolID string, recipientCount int, totalAmount sdk.Coins) {
    duration := time.Since(start)
    
    telemetry.SetGaugeWithLabels(
        []string{"tokenomics", "distribution", "duration_ms"},
        float32(duration.Milliseconds()),
        []metrics.Label{
            telemetry.NewLabel("pool_id", poolID),
        },
    )
    
    telemetry.SetGaugeWithLabels(
        []string{"tokenomics", "distribution", "recipients"},
        float32(recipientCount),
        []metrics.Label{
            telemetry.NewLabel("pool_id", poolID),
        },
    )
    
    // Track distribution rate
    if duration.Seconds() > 0 {
        rate := float32(recipientCount) / float32(duration.Seconds())
        telemetry.SetGaugeWithLabels(
            []string{"tokenomics", "distribution", "rate_per_second"},
            rate,
            []metrics.Label{
                telemetry.NewLabel("pool_id", poolID),
            },
        )
    }
}
```

---

## 15. Complete Example: Building a Staking Rewards Module

Let's tie everything together with a complete example of a staking rewards module that integrates with the Cosmos SDK staking module.

### Module Definition

```go
// x/tokenomics/module.go
package tokenomics

import (
    "context"
    "encoding/json"
    "fmt"
    
    "github.com/grpc-ecosystem/grpc-gateway/runtime"
    "github.com/spf13/cobra"
    
    "cosmossdk.io/core/appmodule"
    "github.com/cosmos/cosmos-sdk/client"
    "github.com/cosmos/cosmos-sdk/codec"
    cdctypes "github.com/cosmos/cosmos-sdk/codec/types"
    sdk "github.com/cosmos/cosmos-sdk/types"
    "github.com/cosmos/cosmos-sdk/types/module"
    
    "github.com/myorg/myappchain/x/tokenomics/client/cli"
    "github.com/myorg/myappchain/x/tokenomics/keeper"
    "github.com/myorg/myappchain/x/tokenomics/types"
)

var (
    _ module.AppModuleBasic      = AppModuleBasic{}
    _ appmodule.AppModule        = AppModule{}
    _ appmodule.HasBeginBlocker  = AppModule{}
    _ appmodule.HasEndBlocker    = AppModule{}
)

// AppModuleBasic defines the basic application module used by the tokenomics module
type AppModuleBasic struct {
    cdc codec.Codec
}

// Name returns the module's name
func (AppModuleBasic) Name() string {
    return types.ModuleName
}

// RegisterLegacyAminoCodec registers the module's types on the given LegacyAmino codec
func (AppModuleBasic) RegisterLegacyAminoCodec(cdc *codec.LegacyAmino) {
    types.RegisterLegacyAminoCodec(cdc)
}

// RegisterInterfaces registers the module interface types
func (b AppModuleBasic) RegisterInterfaces(registry cdctypes.InterfaceRegistry) {
    types.RegisterInterfaces(registry)
}

// DefaultGenesis returns default genesis state as raw bytes for the module
func (AppModuleBasic) DefaultGenesis(cdc codec.JSONCodec) json.RawMessage {
    return cdc.MustMarshalJSON(types.DefaultGenesis())
}

// ValidateGenesis performs genesis state validation for the module
func (AppModuleBasic) ValidateGenesis(cdc codec.JSONCodec, config client.TxEncodingConfig, bz json.RawMessage) error {
    var genState types.GenesisState
    if err := cdc.UnmarshalJSON(bz, &genState); err != nil {
        return fmt.Errorf("failed to unmarshal %s genesis state: %w", types.ModuleName, err)
    }
    return genState.Validate()
}

// RegisterGRPCGatewayRoutes registers the gRPC Gateway routes for the module
func (AppModuleBasic) RegisterGRPCGatewayRoutes(clientCtx client.Context, mux *runtime.ServeMux) {
    types.RegisterQueryHandlerClient(context.Background(), mux, types.NewQueryClient(clientCtx))
}

// GetTxCmd returns the root tx command for the module
func (AppModuleBasic) GetTxCmd() *cobra.Command {
    return cli.GetTxCmd()
}

// GetQueryCmd returns no root query command for the module
func (AppModuleBasic) GetQueryCmd() *cobra.Command {
    return cli.GetQueryCmd(types.StoreKey)
}

// AppModule implements an application module for the tokenomics module
type AppModule struct {
    AppModuleBasic
    
    keeper        keeper.Keeper
    accountKeeper types.AccountKeeper
    bankKeeper    types.BankKeeper
    stakingKeeper types.StakingKeeper
}

// NewAppModule creates a new AppModule object
func NewAppModule(
    cdc codec.Codec,
    keeper keeper.Keeper,
    accountKeeper types.AccountKeeper,
    bankKeeper types.BankKeeper,
    stakingKeeper types.StakingKeeper,
) AppModule {
    return AppModule{
        AppModuleBasic: AppModuleBasic{cdc: cdc},
        keeper:         keeper,
        accountKeeper:  accountKeeper,
        bankKeeper:     bankKeeper,
        stakingKeeper:  stakingKeeper,
    }
}

// RegisterServices registers module services
func (am AppModule) RegisterServices(cfg module.Configurator) {
    types.RegisterMsgServer(cfg.MsgServer(), keeper.NewMsgServerImpl(am.keeper))
    types.RegisterQueryServer(cfg.QueryServer(), am.keeper)
}

// RegisterInvariants registers the module invariants
func (am AppModule) RegisterInvariants(ir sdk.InvariantRegistry) {
    keeper.RegisterInvariants(ir, am.keeper)
}

// InitGenesis performs genesis initialization for the module
func (am AppModule) InitGenesis(ctx sdk.Context, cdc codec.JSONCodec, gs json.RawMessage) {
    var genState types.GenesisState
    cdc.MustUnmarshalJSON(gs, &genState)
    InitGenesis(ctx, am.keeper, genState)
}

// ExportGenesis returns the exported genesis state as raw bytes for the module
func (am AppModule) ExportGenesis(ctx sdk.Context, cdc codec.JSONCodec) json.RawMessage {
    gs := ExportGenesis(ctx, am.keeper)
    return cdc.MustMarshalJSON(gs)
}

// ConsensusVersion implements AppModule/ConsensusVersion
func (AppModule) ConsensusVersion() uint64 { return 1 }

// BeginBlock performs begin block functionality
func (am AppModule) BeginBlock(ctx context.Context) error {
    sdkCtx := sdk.UnwrapSDKContext(ctx)
    return am.keeper.BeginBlock(sdkCtx)
}

// EndBlock performs end block functionality
func (am AppModule) EndBlock(ctx context.Context) error {
    sdkCtx := sdk.UnwrapSDKContext(ctx)
    return am.keeper.EndBlock(sdkCtx)
}
```

### Complete Integration Example

```go
// x/tokenomics/keeper/distribution.go
package keeper

import (
    "fmt"
    
    sdk "github.com/cosmos/cosmos-sdk/types"
    stakingtypes "github.com/cosmos/cosmos-sdk/x/staking/types"
)

// BeginBlock processes reward distribution at the beginning of each block
func (k Keeper) BeginBlock(ctx sdk.Context) error {
    // Update epoch if needed
    if err := k.updateEpoch(ctx); err != nil {
        return err
    }
    
    return nil
}

// EndBlock processes reward distribution at the end of each block
func (k Keeper) EndBlock(ctx sdk.Context) error {
    // Process all active pools
    var errors []error
    
    err := k.RewardPools.Walk(ctx, nil, func(poolID string, pool types.RewardPool) (bool, error) {
        if pool.Status != types.PoolStatus_ACTIVE {
            return false, nil
        }
        
        // Check if pool has expired
        if ctx.BlockTime().After(pool.EndTime) {
            pool.Status = types.PoolStatus_COMPLETED
            if err := k.RewardPools.Set(ctx, poolID, pool); err != nil {
                errors = append(errors, err)
            }
            
            k.LogRewardEvent(ctx, "pool_completed",
                "pool_id", poolID,
                "total_distributed", pool.DistributedAmount.String(),
            )
            return false, nil
        }
        
        // Distribute rewards for this pool
        if err := k.distributePoolRewards(ctx, pool); err != nil {
            errors = append(errors, fmt.Errorf("failed to distribute pool %s: %w", poolID, err))
        }
        
        return false, nil
    })
    
    if err != nil {
        return err
    }
    
    // Update metrics
    k.UpdateMetrics(ctx)
    
    // Return first error if any
    if len(errors) > 0 {
        return errors[0]
    }
    
    return nil
}

// distributePoolRewards distributes rewards for a single pool
func (k Keeper) distributePoolRewards(ctx sdk.Context, pool types.RewardPool) error {
    start := time.Now()
    defer func() {
        telemetry.MeasurePoolDistribution(start, pool.Id, 0, sdk.NewCoins())
    }()
    
    // Calculate distribution for this block
    blocksPerDay := sdk.NewDec(86400 / 5) // Assuming 5 second blocks
    dailyRewards := pool.TotalRewards.Amount.Quo(sdk.NewDec(pool.DurationDays))
    blockRewards := dailyRewards.Quo(blocksPerDay)
    
    if blockRewards.IsZero() {
        return nil
    }
    
    // Get total staked amount
    totalStaked := k.stakingKeeper.TotalBondedTokens(ctx)
    if totalStaked.IsZero() {
        return nil
    }
    
    // Distribute proportionally to all stakers
    distributed := sdk.ZeroInt()
    recipientCount := 0
    
    k.stakingKeeper.IterateDelegations(ctx, func(_ int64, delegation stakingtypes.Delegation) bool {
        // Calculate delegator's share
        delegatorShare := sdk.NewDecFromInt(delegation.Shares.TruncateInt()).
            Quo(sdk.NewDecFromInt(totalStaked))
        
        delegatorReward := blockRewards.Mul(delegatorShare)
        if delegatorReward.IsZero() {
            return false
        }
        
        // Add to pending rewards
        rewardCoin := sdk.NewDecCoinFromDec(pool.RewardDenom, delegatorReward)
        if err := k.addUserReward(ctx, delegation.DelegatorAddress, rewardCoin); err != nil {
            k.Logger(ctx).Error("failed to add user reward",
                "error", err,
                "delegator", delegation.DelegatorAddress,
                "amount", rewardCoin.String(),
            )
            return false
        }
        
        distributed = distributed.Add(delegatorReward.TruncateInt())
        recipientCount++
        
        return false
    })
    
    // Update pool distributed amount
    pool.DistributedAmount = pool.DistributedAmount.Add(
        sdk.NewCoin(pool.RewardDenom, distributed),
    )
    
    if err := k.RewardPools.Set(ctx, pool.Id, pool); err != nil {
        return err
    }
    
    // Emit distribution event
    ctx.EventManager().EmitTypedEvent(&types.EventRewardsDistributed{
        PoolId:         pool.Id,
        Amount:         sdk.NewCoin(pool.RewardDenom, distributed),
        RecipientCount: uint32(recipientCount),
        BlockHeight:    ctx.BlockHeight(),
    })
    
    return nil
}

// addUserReward adds rewards to a user's pending balance
func (k Keeper) addUserReward(ctx sdk.Context, delegator string, reward sdk.DecCoin) error {
    addr, err := sdk.AccAddressFromBech32(delegator)
    if err != nil {
        return err
    }
    
    key := collections.Join(addr, reward.Denom)
    
    // Get existing rewards
    existing, err := k.UserRewards.Get(ctx, key)
    if err != nil && !errors.Is(err, collections.ErrNotFound) {
        return err
    }
    
    // Add new rewards
    newAmount := existing.Add(reward)
    if err := k.UserRewards.Set(ctx, key, newAmount); err != nil {
        return err
    }
    
    // Update index
    if err := k.RewardsByUser.Set(ctx, key); err != nil {
        return err
    }
    
    // Update total rewards tracking
    total, _ := k.TotalRewards.Get(ctx)
    total = total.Add(sdk.NewCoin(reward.Denom, reward.Amount.TruncateInt()))
    if err := k.TotalRewards.Set(ctx, total); err != nil {
        return err
    }
    
    return nil
}
```

### React Native Integration

```typescript
// src/screens/StakingRewardsScreen.tsx
import React, { useCallback, useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useWalletConnect } from '@walletconnect/react-native-dapp';
import LottieView from 'lottie-react-native';

import { MyAppChainClient } from '../services/MyAppChainClient';
import { useWalletStore } from '../stores/WalletStore';
import { formatAmount, formatAPY } from '../utils/formatters';
import { RewardCard } from '../components/RewardCard';

export function StakingRewardsScreen() {
  const { address } = useWalletStore();
  const queryClient = useQueryClient();
  const [refreshing, setRefreshing] = useState(false);
  
  // Query hooks
  const { data: rewards, isLoading: rewardsLoading } = useQuery({
    queryKey: ['rewards', address],
    queryFn: async () => {
      const client = await MyAppChainClient.create();
      return client.getUserRewards(address!);
    },
    enabled: !!address,
    refetchInterval: 30000, // Refresh every 30 seconds
  });
  
  const { data: pools } = useQuery({
    queryKey: ['reward-pools'],
    queryFn: async () => {
      const client = await MyAppChainClient.create();
      return client.getRewardPools();
    },
  });
  
  const { data: apys } = useQuery({
    queryKey: ['pool-apys'],
    queryFn: async () => {
      const client = await MyAppChainClient.create();
      return client.getPoolAPYs();
    },
    refetchInterval: 60000, // Refresh every minute
  });
  
```typescript
  // Claim mutation
  const claimMutation = useMutation({
    mutationFn: async (poolIds?: string[]) => {
      const client = await MyAppChainClient.create();
      const password = await BiometricService.unlockWithBiometrics();
      if (!password) {
        throw new Error('Authentication failed');
      }
      
      await client.unlockWallet(password);
      return client.claimRewards(address!, poolIds);
    },
    onSuccess: (result) => {
      // Invalidate queries to refresh data
      queryClient.invalidateQueries(['rewards', address]);
      queryClient.invalidateQueries(['balance', address]);
      
      // Show success notification
      showNotification({
        type: 'success',
        title: 'Rewards Claimed!',
        message: `Successfully claimed ${formatAmount(result.amount)}`,
      });
    },
    onError: (error) => {
      showNotification({
        type: 'error',
        title: 'Claim Failed',
        message: error.message,
      });
    },
  });
  
  // Pull to refresh
  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await queryClient.invalidateQueries(['rewards', address]);
    await queryClient.invalidateQueries(['pool-apys']);
    setRefreshing(false);
  }, [address, queryClient]);
  
  // Calculate total claimable
  const totalClaimable = rewards?.reduce(
    (sum, reward) => sum.add(reward.amount),
    new BigNumber(0)
  ) || new BigNumber(0);
  
  if (rewardsLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#6366f1" />
        <Text style={styles.loadingText}>Loading rewards...</Text>
      </View>
    );
  }
  
  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      {/* Total Rewards Summary */}
      <View style={styles.summaryCard}>
        <Text style={styles.summaryLabel}>Total Claimable Rewards</Text>
        <Text style={styles.summaryAmount}>
          {formatAmount(totalClaimable, 'utoken')}
        </Text>
        
        {totalClaimable.gt(0) && (
          <TouchableOpacity
            style={styles.claimAllButton}
            onPress={() => claimMutation.mutate()}
            disabled={claimMutation.isPending}
          >
            {claimMutation.isPending ? (
              <ActivityIndicator color="white" />
            ) : (
              <>
                <LottieView
                  source={require('../assets/animations/claim.json')}
                  style={styles.claimIcon}
                  autoPlay
                  loop
                />
                <Text style={styles.claimAllText}>Claim All Rewards</Text>
              </>
            )}
          </TouchableOpacity>
        )}
      </View>
      
      {/* Active Pools */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Active Reward Pools</Text>
        {pools?.map((pool) => {
          const poolApy = apys?.find(a => a.poolId === pool.id);
          const poolRewards = rewards?.filter(r => r.poolId === pool.id);
          
          return (
            <RewardCard
              key={pool.id}
              pool={pool}
              apy={poolApy?.apy}
              rewards={poolRewards}
              onClaim={() => claimMutation.mutate([pool.id])}
              isClaimPending={claimMutation.isPending}
            />
          );
        })}
      </View>
      
      {/* Reward History */}
      <TouchableOpacity
        style={styles.historyButton}
        onPress={() => navigation.navigate('RewardHistory')}
      >
        <Text style={styles.historyButtonText}>View Claim History</Text>
        <Ionicons name="chevron-forward" size={20} color="#6366f1" />
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafb',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#6b7280',
  },
  summaryCard: {
    margin: 16,
    padding: 24,
    backgroundColor: 'white',
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  summaryLabel: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 8,
  },
  summaryAmount: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#111827',
    marginBottom: 20,
  },
  claimAllButton: {
    flexDirection: 'row',
    backgroundColor: '#6366f1',
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  claimIcon: {
    width: 24,
    height: 24,
    marginRight: 8,
  },
  claimAllText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  section: {
    paddingHorizontal: 16,
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#111827',
    marginBottom: 16,
  },
  historyButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginHorizontal: 16,
    marginBottom: 32,
    padding: 16,
    backgroundColor: 'white',
    borderRadius: 12,
  },
  historyButtonText: {
    fontSize: 16,
    color: '#6366f1',
    fontWeight: '500',
  },
});

// RewardCard Component
export function RewardCard({ pool, apy, rewards, onClaim, isClaimPending }) {
  const totalRewards = rewards?.reduce(
    (sum, r) => sum.add(r.amount),
    new BigNumber(0)
  ) || new BigNumber(0);
  
  const progress = pool.distributed.div(pool.total).times(100);
  
  return (
    <View style={cardStyles.container}>
      <View style={cardStyles.header}>
        <View>
          <Text style={cardStyles.poolName}>{pool.name}</Text>
          <Text style={cardStyles.poolDenom}>{pool.rewardDenom}</Text>
        </View>
        {apy && (
          <View style={cardStyles.apyBadge}>
            <Text style={cardStyles.apyText}>{formatAPY(apy)}% APY</Text>
          </View>
        )}
      </View>
      
      <View style={cardStyles.stats}>
        <View style={cardStyles.stat}>
          <Text style={cardStyles.statLabel}>Your Rewards</Text>
          <Text style={cardStyles.statValue}>
            {formatAmount(totalRewards, pool.rewardDenom)}
          </Text>
        </View>
        <View style={cardStyles.stat}>
          <Text style={cardStyles.statLabel}>Pool Progress</Text>
          <Text style={cardStyles.statValue}>{progress.toFixed(1)}%</Text>
        </View>
      </View>
      
      <View style={cardStyles.progressBar}>
        <View
          style={[
            cardStyles.progressFill,
            { width: `${Math.min(progress.toNumber(), 100)}%` },
          ]}
        />
      </View>
      
      {totalRewards.gt(0) && (
        <TouchableOpacity
          style={cardStyles.claimButton}
          onPress={onClaim}
          disabled={isClaimPending}
        >
          <Text style={cardStyles.claimButtonText}>
            Claim {formatAmount(totalRewards, pool.rewardDenom)}
          </Text>
        </TouchableOpacity>
      )}
    </View>
  );
}

const cardStyles = StyleSheet.create({
  container: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 20,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.03,
    shadowRadius: 4,
    elevation: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 16,
  },
  poolName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#111827',
  },
  poolDenom: {
    fontSize: 14,
    color: '#6b7280',
    marginTop: 2,
  },
  apyBadge: {
    backgroundColor: '#10b981',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 16,
  },
  apyText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  stats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  stat: {
    flex: 1,
  },
  statLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 4,
  },
  statValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#111827',
  },
  progressBar: {
    height: 6,
    backgroundColor: '#e5e7eb',
    borderRadius: 3,
    marginBottom: 16,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#6366f1',
    borderRadius: 3,
  },
  claimButton: {
    backgroundColor: '#f3f4f6',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  claimButtonText: {
    color: '#6366f1',
    fontSize: 14,
    fontWeight: '600',
  },
});
```

---

## 16. Performance Optimization Best Practices

### ✅ DO: Implement Efficient Query Patterns

```go
// x/tokenomics/keeper/query_efficient.go
package keeper

import (
    "context"
    
    "cosmossdk.io/store/prefix"
    sdk "github.com/cosmos/cosmos-sdk/types"
    "github.com/cosmos/cosmos-sdk/types/query"
)

// GetUserRewardsSummary returns aggregated rewards with minimal iterations
func (k Keeper) GetUserRewardsSummary(ctx sdk.Context, address sdk.AccAddress) (*types.RewardsSummary, error) {
    summary := &types.RewardsSummary{
        Address:      address.String(),
        TotalRewards: sdk.NewCoins(),
        ByPool:       make(map[string]sdk.Coins),
        LastClaimed:  nil,
    }
    
    // Use prefix iterator for efficient scanning
    prefix := collections.Join(address, "")
    
    err := k.UserRewards.Walk(ctx, collections.NewPrefixedPairRange[sdk.AccAddress, string](address), 
        func(key collections.Pair[sdk.AccAddress, string], value sdk.DecCoin) (bool, error) {
            coin := sdk.NewCoin(key.K2(), value.Amount.TruncateInt())
            summary.TotalRewards = summary.TotalRewards.Add(coin)
            
            // Group by pool if metadata exists
            poolID := k.getPoolIDForReward(ctx, key)
            if poolID != "" {
                summary.ByPool[poolID] = summary.ByPool[poolID].Add(coin)
            }
            
            return false, nil
        })
    
    if err != nil {
        return nil, err
    }
    
    // Get last claim time from events (cached)
    summary.LastClaimed = k.getLastClaimTime(ctx, address)
    
    return summary, nil
}

// BatchGetRewards efficiently fetches rewards for multiple addresses
func (k Keeper) BatchGetRewards(ctx sdk.Context, addresses []sdk.AccAddress) (map[string]sdk.Coins, error) {
    results := make(map[string]sdk.Coins, len(addresses))
    
    // Pre-allocate map
    for _, addr := range addresses {
        results[addr.String()] = sdk.NewCoins()
    }
    
    // Single iteration through all rewards
    err := k.UserRewards.Walk(ctx, nil, func(key collections.Pair[sdk.AccAddress, string], value sdk.DecCoin) (bool, error) {
        addrStr := key.K1().String()
        if _, exists := results[addrStr]; exists {
            coin := sdk.NewCoin(key.K2(), value.Amount.TruncateInt())
            results[addrStr] = results[addrStr].Add(coin)
        }
        return false, nil
    })
    
    return results, err
}
```

### ✅ DO: Implement Batch Processing

```go
// x/tokenomics/keeper/batch_processor.go
package keeper

import (
    "sync"
    
    sdk "github.com/cosmos/cosmos-sdk/types"
)

// BatchProcessor handles efficient batch operations
type BatchProcessor struct {
    keeper     Keeper
    batchSize  int
    bufferPool sync.Pool
}

func NewBatchProcessor(k Keeper) *BatchProcessor {
    return &BatchProcessor{
        keeper:    k,
        batchSize: 1000, // Optimal batch size based on testing
        bufferPool: sync.Pool{
            New: func() interface{} {
                return make([]types.RewardUpdate, 0, 1000)
            },
        },
    }
}

// ProcessRewardUpdates efficiently processes large numbers of reward updates
func (bp *BatchProcessor) ProcessRewardUpdates(ctx sdk.Context, updates []types.RewardUpdate) error {
    // Get buffer from pool
    buffer := bp.bufferPool.Get().([]types.RewardUpdate)
    defer func() {
        buffer = buffer[:0] // Reset slice
        bp.bufferPool.Put(buffer)
    }()
    
    // Process in batches to avoid memory spikes
    for i := 0; i < len(updates); i += bp.batchSize {
        end := i + bp.batchSize
        if end > len(updates) {
            end = len(updates)
        }
        
        batch := updates[i:end]
        
        // Pre-validate batch
        for _, update := range batch {
            if err := update.Validate(); err != nil {
                return errorsmod.Wrapf(types.ErrInvalidUpdate, 
                    "invalid update at index %d: %v", i, err)
            }
        }
        
        // Bulk write to store
        if err := bp.writeBatch(ctx, batch); err != nil {
            return err
        }
        
        // Emit batch event instead of individual events
        ctx.EventManager().EmitTypedEvent(&types.EventBatchUpdate{
            UpdateCount: uint32(len(batch)),
            BatchIndex:  uint32(i / bp.batchSize),
        })
    }
    
    return nil
}

func (bp *BatchProcessor) writeBatch(ctx sdk.Context, updates []types.RewardUpdate) error {
    // Group updates by user for efficient writes
    userUpdates := make(map[string][]types.RewardUpdate)
    
    for _, update := range updates {
        userUpdates[update.Address] = append(userUpdates[update.Address], update)
    }
    
    // Write each user's updates together
    for address, userBatch := range userUpdates {
        addr, _ := sdk.AccAddressFromBech32(address)
        
        // Aggregate updates for same denom
        aggregated := make(map[string]sdk.Dec)
        for _, update := range userBatch {
            current := aggregated[update.Denom]
            aggregated[update.Denom] = current.Add(update.Amount)
        }
        
        // Single write per user per denom
        for denom, amount := range aggregated {
            key := collections.Join(addr, denom)
            
            existing, _ := bp.keeper.UserRewards.Get(ctx, key)
            newAmount := existing.Add(sdk.NewDecCoinFromDec(denom, amount))
            
            if err := bp.keeper.UserRewards.Set(ctx, key, newAmount); err != nil {
                return err
            }
        }
    }
    
    return nil
}
```

### ✅ DO: Optimize Gas Usage with Smart Batching

```go
// x/tokenomics/types/msgs_batch.go
package types

import (
    sdk "github.com/cosmos/cosmos-sdk/types"
)

// MsgBatchOperations allows multiple operations in a single transaction
type MsgBatchOperations struct {
    Sender     string      `protobuf:"bytes,1,opt,name=sender,proto3" json:"sender,omitempty"`
    Operations []Operation `protobuf:"bytes,2,rep,name=operations,proto3" json:"operations"`
}

type Operation struct {
    Type string     `protobuf:"bytes,1,opt,name=type,proto3" json:"type,omitempty"`
    Data *anypb.Any `protobuf:"bytes,2,opt,name=data,proto3" json:"data,omitempty"`
}

// Gas costs for different operations
const (
    GasPerClaim        = uint64(50_000)
    GasPerPoolCreation = uint64(100_000)
    GasPerDelegation   = uint64(80_000)
    GasBaseBatch       = uint64(20_000)
)

// GetSignBytes implements sdk.Msg
func (msg *MsgBatchOperations) GetSignBytes() []byte {
    bz := ModuleCdc.MustMarshalJSON(msg)
    return sdk.MustSortJSON(bz)
}

// ValidateBasic implements sdk.Msg with gas estimation
func (msg *MsgBatchOperations) ValidateBasic() error {
    if _, err := sdk.AccAddressFromBech32(msg.Sender); err != nil {
        return errorsmod.Wrap(sdkerrors.ErrInvalidAddress, err.Error())
    }
    
    if len(msg.Operations) == 0 {
        return errorsmod.Wrap(sdkerrors.ErrInvalidRequest, "no operations provided")
    }
    
    if len(msg.Operations) > MaxBatchOperations {
        return errorsmod.Wrapf(sdkerrors.ErrInvalidRequest, 
            "too many operations: %d > %d", len(msg.Operations), MaxBatchOperations)
    }
    
    // Validate each operation
    totalGas := GasBaseBatch
    for i, op := range msg.Operations {
        switch op.Type {
        case "claim":
            totalGas += GasPerClaim
        case "create_pool":
            totalGas += GasPerPoolCreation
        case "delegate_rewards":
            totalGas += GasPerDelegation
        default:
            return errorsmod.Wrapf(sdkerrors.ErrUnknownRequest, 
                "unknown operation type at index %d: %s", i, op.Type)
        }
    }
    
    // Check if total gas would exceed block limit
    if totalGas > MaxTransactionGas {
        return errorsmod.Wrapf(sdkerrors.ErrOutOfGas,
            "batch would consume %d gas, exceeding limit of %d",
            totalGas, MaxTransactionGas)
    }
    
    return nil
}
```

---

## 17. Common Pitfalls and How to Avoid Them

### ❌ DON'T: Store Redundant Data

```go
// Bad - Storing calculated values that can be derived
type RewardPool struct {
    TotalStaked      sdk.Int // Bad: Can be queried from staking module
    NumberOfStakers  uint64  // Bad: Can be calculated
    AverageStake     sdk.Dec // Bad: Derived value
}

// Good - Store only essential data
type RewardPool struct {
    Id           string
    RewardDenom  string
    TotalRewards sdk.Coin
    StartTime    time.Time
    EndTime      time.Time
}
```

### ❌ DON'T: Use Unbounded Iterations

```go
// Bad - Can cause chain to halt with too many items
func (k Keeper) GetAllRewards(ctx sdk.Context) []types.UserReward {
    var allRewards []types.UserReward
    
    // This could iterate over millions of entries!
    k.UserRewards.Walk(ctx, nil, func(key collections.Pair[sdk.AccAddress, string], value sdk.DecCoin) (bool, error) {
        allRewards = append(allRewards, types.UserReward{
            Address: key.K1().String(),
            Denom:   key.K2(),
            Amount:  value,
        })
        return false, nil
    })
    
    return allRewards
}

// Good - Use pagination
func (k Keeper) GetRewardsPaginated(ctx sdk.Context, pagination *query.PageRequest) ([]types.UserReward, *query.PageResponse, error) {
    var rewards []types.UserReward
    
    pageRes, err := query.CollectionPaginate(
        ctx,
        k.UserRewards,
        pagination,
        func(key collections.Pair[sdk.AccAddress, string], value sdk.DecCoin) (types.UserReward, error) {
            return types.UserReward{
                Address: key.K1().String(),
                Denom:   key.K2(),
                Amount:  value,
            }, nil
        },
    )
    
    return rewards, pageRes, err
}
```

### ❌ DON'T: Ignore Error Handling

```go
// Bad - Swallowing errors
func (k Keeper) ClaimRewards(ctx sdk.Context, address string) {
    rewards, _ := k.GetPendingRewards(ctx, address) // Ignoring error!
    k.bankKeeper.SendCoinsFromModuleToAccount(ctx, types.ModuleName, address, rewards) // Not checking error!
}

// Good - Proper error handling with context
func (k Keeper) ClaimRewards(ctx sdk.Context, address string) error {
    addr, err := sdk.AccAddressFromBech32(address)
    if err != nil {
        return errorsmod.Wrapf(types.ErrInvalidAddress, "invalid address %s: %v", address, err)
    }
    
    rewards, err := k.GetPendingRewards(ctx, addr)
    if err != nil {
        return errorsmod.Wrap(types.ErrRewardNotFound, err.Error())
    }
    
    if rewards.IsZero() {
        return types.ErrNoRewardsToClaim
    }
    
    if err := k.bankKeeper.SendCoinsFromModuleToAccount(ctx, types.ModuleName, addr, rewards); err != nil {
        return errorsmod.Wrap(types.ErrRewardTransfer, err.Error())
    }
    
    return nil
}
```

---

## 18. Debugging and Development Tools

### ✅ DO: Use Ignite CLI for Rapid Development

```bash
# Generate a new module with all boilerplate
ignite scaffold module tokenomics \
  --dep account,bank,staking \
  --params minStake:uint,rewardRate:dec

# Generate messages
ignite scaffold message createPool \
  rewardDenom:string totalRewards:coin duration:uint \
  --module tokenomics \
  --signer authority

# Generate queries with pagination
ignite scaffold query userRewards address:string \
  --response rewards:UserReward:array \
  --module tokenomics \
  --paginated

# Generate types
ignite scaffold type RewardPool \
  id:string rewardDenom:string totalRewards:coin \
  startTime:int endTime:int status:uint \
  --module tokenomics

# Start the chain with hot-reload
ignite chain serve --verbose

# Run with custom config
ignite chain serve -c config-testnet.yml --reset-once
```

### ✅ DO: Create Development Helper Scripts

```bash
#!/bin/bash
# scripts/dev-setup.sh

set -e

echo "🚀 Setting up development environment..."

# Build and install the binary
make install

# Initialize chain
myappchaind init test-node --chain-id myappchain-local

# Add test accounts
echo "test test test test test test test test test test test junk" | \
  myappchaind keys add alice --recover --keyring-backend test

echo "body fish patrol neutral another defy market prepare ankle soccer monster day" | \
  myappchaind keys add bob --recover --keyring-backend test

# Add genesis accounts
myappchaind genesis add-genesis-account alice 10000000000stake,1000000000utoken --keyring-backend test
myappchaind genesis add-genesis-account bob 5000000000stake,500000000utoken --keyring-backend test

# Create validator
myappchaind genesis gentx alice 5000000000stake \
  --chain-id myappchain-local \
  --keyring-backend test \
  --moniker "test-validator"

# Collect gentxs
myappchaind genesis collect-gentxs

# Update genesis params for faster development
jq '.app_state.gov.params.voting_period = "30s"' ~/.myappchain/config/genesis.json > temp.json && mv temp.json ~/.myappchain/config/genesis.json
jq '.app_state.staking.params.unbonding_time = "30s"' ~/.myappchain/config/genesis.json > temp.json && mv temp.json ~/.myappchain/config/genesis.json

# Configure for development
sed -i'' -e 's/minimum-gas-prices = ""/minimum-gas-prices = "0stake"/' ~/.myappchain/config/app.toml
sed -i'' -e 's/enable = false/enable = true/' ~/.myappchain/config/app.toml
sed -i'' -e 's/swagger = false/swagger = true/' ~/.myappchain/config/app.toml

echo "✅ Development chain configured!"
echo "Run 'myappchaind start' to start the chain"
```

### ✅ DO: Implement a Debug CLI

```go
// x/tokenomics/client/cli/debug.go
//go:build debug

package cli

import (
    "encoding/json"
    "fmt"
    
    "github.com/spf13/cobra"
    "github.com/cosmos/cosmos-sdk/client"
)

// GetDebugCmd returns debug commands for development
func GetDebugCmd() *cobra.Command {
    cmd := &cobra.Command{
        Use:   "debug",
        Short: "Debug commands for development",
        Long:  "⚠️  WARNING: These commands are for development only!",
    }
    
    cmd.AddCommand(
        CmdDebugState(),
        CmdDebugInvariants(),
        CmdDebugSimulateRewards(),
    )
    
    return cmd
}

func CmdDebugState() *cobra.Command {
    return &cobra.Command{
        Use:   "state",
        Short: "Export entire module state",
        RunE: func(cmd *cobra.Command, args []string) error {
            clientCtx, err := client.GetClientQueryContext(cmd)
            if err != nil {
                return err
            }
            
            queryClient := types.NewQueryClient(clientCtx)
            
            // Query debug state
            res, err := queryClient.DebugState(cmd.Context(), &types.QueryDebugStateRequest{})
            if err != nil {
                return err
            }
            
            // Pretty print
            data, err := json.MarshalIndent(res, "", "  ")
            if err != nil {
                return err
            }
            
            fmt.Println(string(data))
            return nil
        },
    }
}
```

---

## 19. Production Readiness Checklist

Before deploying to production, ensure all items are checked:

### Security
- [ ] All keeper methods validate inputs
- [ ] Access control implemented for admin functions  
- [ ] No unbounded iterations in any code path
- [ ] Invariants registered and tested
- [ ] Upgrade handlers tested on testnet
- [ ] Security audit completed

### Performance
- [ ] Store keys use efficient prefixes
- [ ] Queries are paginated
- [ ] Batch operations implemented where appropriate
- [ ] Gas costs are predictable and documented
- [ ] Load testing completed with expected transaction volume

### Monitoring
- [ ] Prometheus metrics exposed
- [ ] Events emitted for all important operations
- [ ] Structured logging implemented
- [ ] Alerts configured for invariant violations
- [ ] Dashboards created for key metrics

### Documentation
- [ ] API documentation generated
- [ ] Integration guide written
- [ ] Upgrade instructions documented
- [ ] Runbook for common operations
- [ ] Architecture decision records maintained

### Testing
- [ ] Unit test coverage > 80%
- [ ] Integration tests for all workflows
- [ ] Simulation tests pass 10M+ blocks
- [ ] Upgrade tested on testnet
- [ ] Disaster recovery procedures tested

---

## 20. Conclusion

This guide has covered the essential patterns and best practices for building production-grade Cosmos SDK applications in 2025. The key takeaways are:

1. **Architecture Matters**: A well-organized module structure with clear boundaries makes development and maintenance easier.

2. **Type Safety**: Use the Collections API and generated types to catch errors at compile time.

3. **Security First**: Validate all inputs, implement proper access control, and test invariants.

4. **Performance**: Design for scale from the beginning with efficient store usage and batch operations.

5. **Testing**: Comprehensive testing including unit, integration, and simulation tests is non-negotiable.

6. **Monitoring**: Build observability into your application from day one.

The Cosmos ecosystem continues to evolve rapidly. Stay connected with the community, follow the latest SDK releases, and always be learning. 

For more resources:
- [Cosmos SDK Documentation](https://docs.cosmos.network)
- [Ignite CLI](https://ignite.com)
- [CosmJS](https://cosmos.github.io/cosmjs)
- [Cosmos Discord](https://discord.gg/cosmosnetwork)

Happy building! 🚀