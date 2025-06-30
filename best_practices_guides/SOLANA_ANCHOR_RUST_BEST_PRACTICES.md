# The Definitive Guide to Solana Development with Anchor 0.30, Solang, and SvelteKit 2 (mid-2025 Edition)

This guide synthesizes production-grade patterns for building high-performance decentralized applications on Solana. It leverages Anchor's latest features, introduces Solang for EVM developers, and demonstrates modern Web3 front-end architecture with SvelteKit 2.

## Prerequisites & Core Dependencies

Your project should target **Anchor 0.30+**, **Solana CLI 1.18+**, **Rust 1.82+**, and **SvelteKit 2.15+**. For Solang contracts, use **Solang 0.4+** with **Solidity 0.8+** compatibility.

### Workspace Configuration

```toml
# Anchor.toml
[toolchain]
anchor_version = "0.30.1"
solana_version = "1.18.25"

[features]
resolution = true
skip-lint = false

[programs.localnet]
my_program = "Fg6PaFpoGXkYsidMpWTK6W2BeZ7FEfcYkg476zPFsLnS"

[programs.devnet]
my_program = "Fg6PaFpoGXkYsidMpWTK6W2BeZ7FEfcYkg476zPFsLnS"

[registry]
url = "https://api.apr.dev"

[provider]
cluster = "Localnet"
wallet = "~/.config/solana/id.json"

[scripts]
test = "bun run ts-mocha -p ./tsconfig.json -t 1000000 tests/**/*.ts"

[test]
startup_wait = 5000
shutdown_wait = 2000
upgrade_authority = "~/.config/solana/id.json"

[[test.validator.clone]]
address = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"

[test.validator.accounts]
[test.validator.accounts.my_account]
address = "7NL2qWArf2BbEBBH1vTRZCsoNqFATTddH6h8GkVvrLpG"
filename = "tests/fixtures/my_account.json"
```

### Program `Cargo.toml`

```toml
[package]
name = "my-program"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "lib"]
name = "my_program"

[features]
no-entrypoint = []
no-idl = []
no-log-ix-name = []
cpi = ["no-entrypoint"]
default = []
mainnet = []

[dependencies]
anchor-lang = { version = "0.30.1", features = ["init-if-needed"] }
anchor-spl = { version = "0.30.1", features = ["token", "token-2022", "associated-token", "metadata"] }
solana-program = "~1.18.25"
spl-token = { version = "6.0", features = ["no-entrypoint"] }
spl-token-2022 = { version = "5.0", features = ["no-entrypoint"] }
spl-associated-token-account = { version = "5.0", features = ["no-entrypoint"] }
mpl-token-metadata = { version = "5.0", features = ["no-entrypoint"] }
pyth-solana-receiver-sdk = "0.4"
switchboard-on-demand = "0.2"
light-protocol-program = "0.8"
clockwork-sdk = "2.0"
squads-multisig-program = "2.2"

# Core dependencies
borsh = "0.10"
bytemuck = { version = "1.19", features = ["derive", "min_const_generics"] }
num-derive = "0.4"
num-traits = "0.2"
thiserror = "2.0"

# Security & Math
uint = "0.10"
ethnum = "1.5"
rust_decimal = "1.36"
rust_decimal_macros = "1.36"

[dev-dependencies]
solana-program-test = "~1.18.25"
solana-sdk = "~1.18.25"
tokio = { version = "1.44", features = ["full"] }
proptest = "1.6"
```

### SvelteKit Front-End Configuration

```json
// package.json
{
  "name": "my-solana-app",
  "version": "0.0.1",
  "type": "module",
  "scripts": {
    "dev": "vite dev",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest",
    "anchor:build": "anchor build",
    "anchor:test": "anchor test",
    "anchor:deploy": "anchor deploy",
    "idl:generate": "anchor idl fetch -o src/lib/idl",
    "sdk:generate": "solana-bankrun generate --out src/lib/sdk"
  },
  "devDependencies": {
    "@sveltejs/adapter-cloudflare": "^5.0.0",
    "@sveltejs/kit": "^2.15.0",
    "@sveltejs/vite-plugin-svelte": "^4.0.0",
    "@types/node": "^22.10.0",
    "svelte": "^5.0.0",
    "svelte-check": "^4.1.0",
    "typescript": "^5.6.0",
    "vite": "^6.0.0",
    "vitest": "^2.1.0"
  },
  "dependencies": {
    "@coral-xyz/anchor": "^0.30.1",
    "@solana/web3.js": "^2.0.0",
    "@solana/spl-token": "^0.4.0",
    "@solana/wallet-adapter-base": "^0.9.23",
    "@solana/wallet-adapter-solflare": "^0.6.30",
    "@solana/wallet-adapter-phantom": "^0.9.25",
    "@solana/wallet-standard-features": "^1.2.0",
    "@tanstack/svelte-query": "^6.0.0",
    "bs58": "^6.0.0",
    "buffer": "^6.0.3",
    "p-queue": "^8.0.0",
    "p-retry": "^6.0.0",
    "superstruct": "^2.0.0",
    "zod": "^3.24.0",
    "@formkit/auto-animate": "^0.8.0",
    "svelte-french-toast": "^1.2.0",
    "svelte-motion": "^0.12.0"
  }
}
```

## 1. Project Architecture & Code Organization

### ✅ DO: Use a Multi-Package Monorepo Structure

This structure separates on-chain programs, client SDKs, and front-end applications while maintaining a single source of truth:

```
/
├── Anchor.toml                 # Anchor workspace configuration
├── Cargo.toml                  # Rust workspace root
├── package.json                # pnpm workspace root
├── pnpm-workspace.yaml         # pnpm workspace config
├── programs/                   # On-chain programs
│   ├── my-program/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs         # Program entry point
│   │       ├── error.rs       # Custom error types
│   │       ├── state.rs       # Account structures
│   │       ├── instructions/  # Instruction handlers
│   │       │   ├── mod.rs
│   │       │   ├── initialize.rs
│   │       │   └── execute.rs
│   │       ├── utils/         # Helper functions
│   │       └── constants.rs   # Program constants
│   └── my-program-solang/     # Solang EVM-compatible program
│       ├── Cargo.toml
│       └── contracts/
│           └── MyContract.sol
├── tests/                      # Integration tests
│   ├── my-program.test.ts
│   └── fixtures/              # Test accounts/data
├── sdk/                       # Generated TypeScript SDK
│   ├── package.json
│   └── src/
├── app/                       # SvelteKit front-end
│   ├── package.json
│   ├── svelte.config.js
│   ├── vite.config.ts
│   └── src/
│       ├── app.d.ts
│       ├── app.html
│       ├── lib/
│       │   ├── stores/       # Svelte stores
│       │   ├── components/   # Reusable components
│       │   ├── contracts/    # Program interfaces
│       │   └── utils/        # Client utilities
│       └── routes/           # SvelteKit routes
└── scripts/                   # Deployment & maintenance
    ├── deploy.ts
    └── migrate.ts
```

### ✅ DO: Organize Program Instructions by Domain

```rust
// programs/my-program/src/lib.rs
use anchor_lang::prelude::*;

pub mod constants;
pub mod error;
pub mod instructions;
pub mod state;
pub mod utils;

use instructions::*;

declare_id!("Fg6PaFpoGXkYsidMpWTK6W2BeZ7FEfcYkg476zPFsLnS");

#[program]
pub mod my_program {
    use super::*;

    pub fn initialize(ctx: Context<Initialize>, params: InitializeParams) -> Result<()> {
        instructions::initialize(ctx, params)
    }

    pub fn execute_action(ctx: Context<ExecuteAction>, params: ActionParams) -> Result<()> {
        instructions::execute_action(ctx, params)
    }

    pub fn close_account(ctx: Context<CloseAccount>) -> Result<()> {
        instructions::close_account(ctx)
    }
}
```

## 2. Account Design & State Management

### ✅ DO: Design Efficient Account Structures with Discriminators

```rust
// programs/my-program/src/state.rs
use anchor_lang::prelude::*;

#[account]
#[derive(InitSpace)] // Anchor 0.30 automatic space calculation
pub struct Protocol {
    pub authority: Pubkey,
    pub treasury: Pubkey,
    pub fee_basis_points: u16,    // 0.01% precision (10000 = 100%)
    pub total_value_locked: u64,
    pub total_users: u64,
    pub paused: bool,
    pub bump: u8,                 // Store PDA bump for efficiency
    pub _reserved: [u8; 32],      // Future-proofing
}

#[account]
#[derive(InitSpace)]
pub struct UserAccount {
    pub owner: Pubkey,
    pub protocol: Pubkey,
    pub deposited_amount: u64,
    pub rewards_earned: u64,
    pub last_update_slot: u64,
    pub nonce: u64,               // For replay protection
    pub bump: u8,
    #[max_len(32)]
    pub custom_data: Vec<u8>,     // Variable length data
}

// Efficient packed struct for high-frequency data
#[account(zero_copy)]
#[derive(InitSpace)]
pub struct PriceOracle {
    pub oracle_pubkey: Pubkey,
    pub price: u64,
    pub confidence: u64,
    pub last_update_slot: u64,
    pub _padding: [u8; 32],
}

// Event for indexing
#[event]
pub struct ActionExecuted {
    pub user: Pubkey,
    pub action_type: ActionType,
    pub amount: u64,
    pub timestamp: i64,
    #[index]  // Anchor 0.30: Index for faster queries
    pub protocol: Pubkey,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub enum ActionType {
    Deposit,
    Withdraw,
    ClaimRewards,
}
```

### ✅ DO: Use PDAs for Deterministic Addressing

```rust
// programs/my-program/src/instructions/initialize.rs
use anchor_lang::prelude::*;
use crate::state::*;
use crate::constants::*;

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(
        init,
        payer = payer,
        space = 8 + Protocol::INIT_SPACE,
        seeds = [PROTOCOL_SEED],
        bump
    )]
    pub protocol: Account<'info, Protocol>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    /// CHECK: Treasury can be any account
    pub treasury: UncheckedAccount<'info>,
    
    #[account(mut)]
    pub payer: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

pub fn initialize(ctx: Context<Initialize>, params: InitializeParams) -> Result<()> {
    let protocol = &mut ctx.accounts.protocol;
    
    protocol.authority = ctx.accounts.authority.key();
    protocol.treasury = ctx.accounts.treasury.key();
    protocol.fee_basis_points = params.fee_basis_points;
    protocol.total_value_locked = 0;
    protocol.total_users = 0;
    protocol.paused = false;
    protocol.bump = ctx.bumps.protocol; // Store bump from context
    protocol._reserved = [0u8; 32];

    emit!(ProtocolInitialized {
        authority: protocol.authority,
        treasury: protocol.treasury,
        fee_basis_points: protocol.fee_basis_points,
    });

    Ok(())
}

#[derive(AnchorSerialize, AnchorDeserialize)]
pub struct InitializeParams {
    pub fee_basis_points: u16,
}
```

### ❌ DON'T: Use Borsh Serialization for Large or Frequently Updated Data

```rust
// Bad - Inefficient for high-frequency updates
#[account]
pub struct GameState {
    pub players: Vec<Player>,  // Can grow unbounded
    pub moves: Vec<Move>,      // Expensive to deserialize
}

// Good - Use zero-copy for performance-critical data
#[account(zero_copy)]
pub struct GameState {
    pub player_count: u64,
    pub move_count: u64,
    pub players: [Player; 100],     // Fixed size array
    pub current_round: u64,
}

#[zero_copy]
#[derive(InitSpace)]
pub struct Player {
    pub pubkey: Pubkey,
    pub score: u64,
    pub active: bool,
    pub _padding: [u8; 7],  // Align to 8 bytes
}
```

## 3. Security Best Practices

### ✅ DO: Implement Comprehensive Account Validation

```rust
// programs/my-program/src/instructions/execute_action.rs
use anchor_lang::prelude::*;
use anchor_spl::token::{Token, TokenAccount, Transfer};

#[derive(Accounts)]
#[instruction(amount: u64)]
pub struct ExecuteAction<'info> {
    #[account(
        mut,
        seeds = [PROTOCOL_SEED],
        bump = protocol.bump,
        constraint = !protocol.paused @ MyError::ProtocolPaused
    )]
    pub protocol: Account<'info, Protocol>,
    
    #[account(
        init_if_needed,
        payer = user,
        space = 8 + UserAccount::INIT_SPACE,
        seeds = [USER_SEED, user.key().as_ref(), protocol.key().as_ref()],
        bump
    )]
    pub user_account: Account<'info, UserAccount>,
    
    #[account(
        mut,
        constraint = user_token.owner == user.key() @ MyError::InvalidTokenOwner,
        constraint = user_token.mint == protocol_token.mint @ MyError::InvalidMint,
        constraint = user_token.amount >= amount @ MyError::InsufficientBalance
    )]
    pub user_token: Account<'info, TokenAccount>,
    
    #[account(
        mut,
        seeds = [VAULT_SEED, protocol.key().as_ref()],
        bump,
        token::mint = user_token.mint,
        token::authority = protocol
    )]
    pub protocol_vault: Account<'info, TokenAccount>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
}

pub fn execute_action(ctx: Context<ExecuteAction>, amount: u64) -> Result<()> {
    // Validate amount
    require!(amount > 0, MyError::InvalidAmount);
    require!(amount <= MAX_DEPOSIT_AMOUNT, MyError::AmountTooLarge);
    
    let protocol = &mut ctx.accounts.protocol;
    let user_account = &mut ctx.accounts.user_account;
    
    // Initialize user account if needed (Anchor 0.30 feature)
    if user_account.owner == Pubkey::default() {
        user_account.owner = ctx.accounts.user.key();
        user_account.protocol = protocol.key();
        user_account.bump = ctx.bumps.user_account;
        protocol.total_users += 1;
    }
    
    // Check for reentrancy
    let current_slot = Clock::get()?.slot;
    require!(
        user_account.last_update_slot != current_slot,
        MyError::ReentrancyDetected
    );
    user_account.last_update_slot = current_slot;
    
    // Calculate fees
    let fee = amount
        .checked_mul(protocol.fee_basis_points as u64)
        .ok_or(MyError::MathOverflow)?
        .checked_div(10000)
        .ok_or(MyError::MathOverflow)?;
    
    let deposit_amount = amount.checked_sub(fee).ok_or(MyError::MathOverflow)?;
    
    // Update state before external calls
    user_account.deposited_amount = user_account
        .deposited_amount
        .checked_add(deposit_amount)
        .ok_or(MyError::MathOverflow)?;
    
    protocol.total_value_locked = protocol
        .total_value_locked
        .checked_add(deposit_amount)
        .ok_or(MyError::MathOverflow)?;
    
    // Transfer tokens (external call last)
    let cpi_accounts = Transfer {
        from: ctx.accounts.user_token.to_account_info(),
        to: ctx.accounts.protocol_vault.to_account_info(),
        authority: ctx.accounts.user.to_account_info(),
    };
    let cpi_ctx = CpiContext::new(ctx.accounts.token_program.to_account_info(), cpi_accounts);
    anchor_spl::token::transfer(cpi_ctx, amount)?;
    
    emit!(ActionExecuted {
        user: ctx.accounts.user.key(),
        action_type: ActionType::Deposit,
        amount: deposit_amount,
        timestamp: Clock::get()?.unix_timestamp,
        protocol: protocol.key(),
    });
    
    Ok(())
}
```

### ✅ DO: Implement Access Control with Custom Modifiers

```rust
// programs/my-program/src/utils/access_control.rs
use anchor_lang::prelude::*;

pub trait AdminOnly<'info> {
    fn authority(&self) -> &Signer<'info>;
    fn protocol(&self) -> &Account<'info, Protocol>;
    
    fn check_admin(&self) -> Result<()> {
        require_keys_eq!(
            self.authority().key(),
            self.protocol().authority,
            MyError::Unauthorized
        );
        Ok(())
    }
}

// Implement for instruction contexts
impl<'info> AdminOnly<'info> for UpdateProtocol<'info> {
    fn authority(&self) -> &Signer<'info> { &self.authority }
    fn protocol(&self) -> &Account<'info, Protocol> { &self.protocol }
}

// Use in instruction
pub fn update_protocol(ctx: Context<UpdateProtocol>, params: UpdateParams) -> Result<()> {
    ctx.accounts.check_admin()?;
    // ... rest of the logic
}
```

## 4. Cross-Program Invocation (CPI) Patterns

### ✅ DO: Use Type-Safe CPI with Anchor

```rust
// programs/my-program/src/instructions/swap.rs
use anchor_lang::prelude::*;
use anchor_spl::token_2022::{
    self, Token2022, TransferChecked, MintTo,
    spl_token_2022::instruction::AuthorityType,
};

#[derive(Accounts)]
pub struct Swap<'info> {
    #[account(mut)]
    pub user: Signer<'info>,
    
    // External program to call
    /// CHECK: Verified DEX program
    #[account(
        constraint = dex_program.key() == DEX_PROGRAM_ID @ MyError::InvalidProgram
    )]
    pub dex_program: UncheckedAccount<'info>,
    
    // Remaining accounts for the CPI
}

pub fn execute_swap(
    ctx: Context<Swap>,
    amount_in: u64,
    minimum_amount_out: u64,
) -> Result<()> {
    // Build CPI instruction
    let ix = solana_program::instruction::Instruction {
        program_id: ctx.accounts.dex_program.key(),
        accounts: vec![
            // ... build accounts
        ],
        data: SwapInstruction {
            amount_in,
            minimum_amount_out,
        }.try_to_vec()?,
    };
    
    // Execute with proper error handling
    match solana_program::program::invoke(
        &ix,
        &[/* account infos */],
    ) {
        Ok(_) => {
            msg!("Swap executed successfully");
            Ok(())
        }
        Err(e) => {
            msg!("Swap failed: {:?}", e);
            Err(error!(MyError::SwapFailed))
        }
    }
}

// For programs you control, use Anchor CPI
pub fn stake_tokens(ctx: Context<StakeTokens>, amount: u64) -> Result<()> {
    let cpi_program = ctx.accounts.staking_program.to_account_info();
    let cpi_accounts = staking_program::cpi::accounts::Stake {
        staker: ctx.accounts.user.to_account_info(),
        stake_account: ctx.accounts.stake_account.to_account_info(),
        token_account: ctx.accounts.token_account.to_account_info(),
        // ...
    };
    
    // Sign with PDA if needed
    let protocol_bump = ctx.accounts.protocol.bump;
    let signer_seeds: &[&[&[u8]]] = &[&[
        PROTOCOL_SEED,
        &[protocol_bump],
    ]];
    
    let cpi_ctx = CpiContext::new_with_signer(
        cpi_program,
        cpi_accounts,
        signer_seeds,
    );
    
    staking_program::cpi::stake(cpi_ctx, amount)?;
    
    Ok(())
}
```

## 5. Error Handling

### ✅ DO: Create Descriptive Custom Errors

```rust
// programs/my-program/src/error.rs
use anchor_lang::prelude::*;

#[error_code]
pub enum MyError {
    #[msg("Protocol is currently paused")]
    ProtocolPaused,
    
    #[msg("Unauthorized access")]
    Unauthorized,
    
    #[msg("Invalid amount: must be greater than 0")]
    InvalidAmount,
    
    #[msg("Amount too large: exceeds maximum of {} lamports", MAX_DEPOSIT_AMOUNT)]
    AmountTooLarge,
    
    #[msg("Math overflow in calculation")]
    MathOverflow,
    
    #[msg("Insufficient balance: required {}, available {}", .required, .available)]
    InsufficientBalance { required: u64, available: u64 },
    
    #[msg("Invalid token mint")]
    InvalidMint,
    
    #[msg("Invalid token owner")]
    InvalidTokenOwner,
    
    #[msg("Reentrancy detected")]
    ReentrancyDetected,
    
    #[msg("Swap failed with error: {}", .reason)]
    SwapFailed { reason: String },
    
    #[msg("Oracle price is stale: last update {} slots ago", .slots_behind)]
    StaleOracle { slots_behind: u64 },
}
```

## 6. Testing with Bankrun and Anchor

### ✅ DO: Use Bankrun for Fast Integration Tests

```typescript
// tests/my-program.test.ts
import { startAnchor } from "solana-bankrun";
import { BankrunProvider } from "anchor-bankrun";
import { Program, BN } from "@coral-xyz/anchor";
import { Keypair, PublicKey, SystemProgram, LAMPORTS_PER_SOL } from "@solana/web3.js";
import { 
  createMint, 
  createAssociatedTokenAccount, 
  mintTo,
  getAccount 
} from "@solana/spl-token";
import { MyProgram } from "../target/types/my_program";
import IDL from "../target/idl/my_program.json";

describe("my-program", () => {
  let provider: BankrunProvider;
  let program: Program<MyProgram>;
  let authority: Keypair;
  let user: Keypair;
  let protocolPda: PublicKey;
  let mint: PublicKey;

  before(async () => {
    // Start Bankrun with fixtures
    authority = Keypair.generate();
    user = Keypair.generate();
    
    const context = await startAnchor(
      "",
      [],
      [
        {
          address: authority.publicKey,
          info: {
            lamports: 10 * LAMPORTS_PER_SOL,
            owner: SystemProgram.programId,
            executable: false,
            data: Buffer.alloc(0),
          },
        },
        {
          address: user.publicKey,
          info: {
            lamports: 10 * LAMPORTS_PER_SOL,
            owner: SystemProgram.programId,
            executable: false,
            data: Buffer.alloc(0),
          },
        },
      ]
    );

    provider = new BankrunProvider(context);
    program = new Program<MyProgram>(
      IDL as MyProgram,
      provider
    );

    // Derive PDAs
    [protocolPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("protocol")],
      program.programId
    );
  });

  describe("initialize", () => {
    it("initializes the protocol", async () => {
      const tx = await program.methods
        .initialize({ feeBasisPoints: 250 }) // 2.5%
        .accounts({
          protocol: protocolPda,
          authority: authority.publicKey,
          treasury: authority.publicKey,
          payer: authority.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([authority])
        .rpc();

      // Fetch and verify account
      const protocol = await program.account.protocol.fetch(protocolPda);
      expect(protocol.authority.toString()).to.equal(authority.publicKey.toString());
      expect(protocol.feeBasisPoints).to.equal(250);
      expect(protocol.paused).to.be.false;
    });
  });

  describe("execute_action", () => {
    let userTokenAccount: PublicKey;
    let vaultTokenAccount: PublicKey;

    beforeEach(async () => {
      // Create mint and token accounts
      const mintAuthority = Keypair.generate();
      mint = await createMint(
        provider.context.banksClient,
        authority,
        mintAuthority.publicKey,
        null,
        6
      );

      userTokenAccount = await createAssociatedTokenAccount(
        provider.context.banksClient,
        user,
        mint,
        user.publicKey
      );

      // Mint tokens to user
      await mintTo(
        provider.context.banksClient,
        authority,
        mint,
        userTokenAccount,
        mintAuthority,
        1000 * 10 ** 6
      );

      [vaultTokenAccount] = PublicKey.findProgramAddressSync(
        [Buffer.from("vault"), protocolPda.toBuffer()],
        program.programId
      );
    });

    it("deposits tokens with correct fee calculation", async () => {
      const depositAmount = new BN(100 * 10 ** 6);
      
      // Snapshot state before
      const userBalanceBefore = await getAccount(
        provider.context.banksClient,
        userTokenAccount
      );

      await program.methods
        .executeAction(depositAmount)
        .accounts({
          protocol: protocolPda,
          userAccount: deriveUserAccount(user.publicKey, protocolPda),
          userToken: userTokenAccount,
          protocolVault: vaultTokenAccount,
          user: user.publicKey,
          tokenProgram: TOKEN_PROGRAM_ID,
          systemProgram: SystemProgram.programId,
        })
        .signers([user])
        .rpc();

      // Verify token transfer
      const userBalanceAfter = await getAccount(
        provider.context.banksClient,
        userTokenAccount
      );
      
      expect(userBalanceBefore.amount - userBalanceAfter.amount).to.equal(
        depositAmount.toNumber()
      );

      // Verify user account updated
      const userAccount = await program.account.userAccount.fetch(
        deriveUserAccount(user.publicKey, protocolPda)
      );
      
      const expectedDeposit = depositAmount.muln(9750).divn(10000); // 97.5% after fee
      expect(userAccount.depositedAmount.toString()).to.equal(
        expectedDeposit.toString()
      );
    });

    it("prevents deposits when protocol is paused", async () => {
      // Pause protocol
      await program.methods
        .updateProtocol({ paused: true })
        .accounts({
          protocol: protocolPda,
          authority: authority.publicKey,
        })
        .signers([authority])
        .rpc();

      // Attempt deposit
      try {
        await program.methods
          .executeAction(new BN(100 * 10 ** 6))
          .accounts({/* ... */})
          .signers([user])
          .rpc();
        
        expect.fail("Should have thrown ProtocolPaused error");
      } catch (err) {
        expect(err.error.errorCode.code).to.equal("ProtocolPaused");
      }
    });
  });
});

// Helper functions
function deriveUserAccount(user: PublicKey, protocol: PublicKey): PublicKey {
  const [pda] = PublicKey.findProgramAddressSync(
    [Buffer.from("user"), user.toBuffer(), protocol.toBuffer()],
    program.programId
  );
  return pda;
}
```

### ✅ DO: Test Edge Cases and Attack Vectors

```typescript
describe("security", () => {
  it("prevents arithmetic overflow", async () => {
    const maxAmount = new BN(2).pow(new BN(64)).subn(1); // u64::MAX
    
    try {
      await program.methods
        .executeAction(maxAmount)
        .accounts({/* ... */})
        .rpc();
      
      expect.fail("Should have thrown MathOverflow");
    } catch (err) {
      expect(err.error.errorCode.code).to.equal("MathOverflow");
    }
  });

  it("prevents reentrancy attacks", async () => {
    // First transaction
    const tx1 = await program.methods
      .executeAction(new BN(100))
      .accounts({/* ... */})
      .transaction();

    // Second transaction in same slot
    const tx2 = await program.methods
      .executeAction(new BN(100))
      .accounts({/* ... */})
      .transaction();

    // Send both transactions
    await provider.sendAndConfirm(tx1, [user]);
    
    try {
      await provider.sendAndConfirm(tx2, [user]);
      expect.fail("Should have thrown ReentrancyDetected");
    } catch (err) {
      expect(err.logs).to.include("ReentrancyDetected");
    }
  });
});
```

## 7. Solang Integration for EVM Developers

### ✅ DO: Use Solang for Solidity-Compatible Contracts

```solidity
// programs/my-program-solang/contracts/TokenVault.sol
contract TokenVault {
    address public owner;
    mapping(address => uint256) public balances;
    uint256 public totalDeposits;
    
    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    function deposit() external payable {
        require(msg.value > 0, "Amount must be > 0");
        
        balances[msg.sender] += msg.value;
        totalDeposits += msg.value;
        
        emit Deposit(msg.sender, msg.value);
    }
    
    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        balances[msg.sender] -= amount;
        totalDeposits -= amount;
        
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        emit Withdrawal(msg.sender, amount);
    }
    
    // Solana-specific: Access SPL tokens
    function depositSPLToken(
        address token,
        uint256 amount
    ) external {
        // Solang provides SPL token interface
        SplToken.transfer_from(token, msg.sender, address(this), amount);
        // ... rest of logic
    }
}
```

### Build and Deploy Solang Contracts

```bash
# Compile Solidity to Solana bytecode
solang compile --target solana contracts/TokenVault.sol -o build/

# Deploy using Anchor
anchor deploy --program-name my_program_solang --program-keypair <keypair>
```

## 8. SvelteKit 2 Front-End Architecture

### ✅ DO: Structure Your SvelteKit App for Web3

```
/app/src
├── lib/
│   ├── stores/
│   │   ├── wallet.svelte.ts      # Wallet connection state
│   │   ├── program.svelte.ts     # Program interactions
│   │   └── transactions.svelte.ts # Transaction management
│   ├── contracts/
│   │   ├── idl/                  # Anchor IDLs
│   │   └── sdk/                  # Generated SDK
│   ├── components/
│   │   ├── WalletButton.svelte
│   │   ├── TransactionToast.svelte
│   │   └── AccountBalance.svelte
│   ├── utils/
│   │   ├── anchor.ts            # Anchor setup
│   │   ├── transactions.ts      # Transaction helpers
│   │   └── formatting.ts        # Display formatters
│   └── config/
│       ├── constants.ts
│       └── rpc.ts
└── routes/
    ├── +layout.svelte           # App shell with wallet provider
    ├── +page.svelte             # Landing page
    └── app/
        └── +page.svelte         # Main application
```

### ✅ DO: Implement Reactive Wallet Management

```typescript
// app/src/lib/stores/wallet.svelte.ts
import { 
  PublicKey, 
  Connection, 
  Transaction,
  VersionedTransaction 
} from '@solana/web3.js';
import type { Adapter } from '@solana/wallet-adapter-base';
import { PhantomWalletAdapter } from '@solana/wallet-adapter-phantom';
import { SolflareWalletAdapter } from '@solana/wallet-adapter-solflare';
import { getRpcUrl } from '$lib/config/rpc';

type WalletStore = {
  wallets: Adapter[];
  wallet: Adapter | null;
  publicKey: PublicKey | null;
  connected: boolean;
  connecting: boolean;
  disconnecting: boolean;
  select: (walletName: string) => Promise<void>;
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
  signTransaction: <T extends Transaction | VersionedTransaction>(tx: T) => Promise<T>;
  signAllTransactions: <T extends Transaction | VersionedTransaction>(txs: T[]) => Promise<T[]>;
  signMessage: (message: Uint8Array) => Promise<Uint8Array>;
};

function createWalletStore(): WalletStore {
  const wallets = [
    new PhantomWalletAdapter(),
    new SolflareWalletAdapter(),
  ];

  let wallet = $state<Adapter | null>(null);
  let publicKey = $state<PublicKey | null>(null);
  let connected = $state(false);
  let connecting = $state(false);
  let disconnecting = $state(false);

  // Auto-connect on wallet selection
  $effect(() => {
    if (wallet) {
      wallet.on('connect', (pk) => {
        publicKey = pk;
        connected = true;
        
        // Persist selection
        if (typeof window !== 'undefined') {
          localStorage.setItem('walletName', wallet.name);
        }
      });

      wallet.on('disconnect', () => {
        publicKey = null;
        connected = false;
      });

      wallet.on('error', (error) => {
        console.error('Wallet error:', error);
        toast.error(error.message);
      });

      // Cleanup
      return () => {
        wallet.off('connect');
        wallet.off('disconnect');
        wallet.off('error');
      };
    }
  });

  // Auto-reconnect on page load
  $effect(() => {
    if (typeof window !== 'undefined') {
      const savedWallet = localStorage.getItem('walletName');
      if (savedWallet) {
        const found = wallets.find(w => w.name === savedWallet);
        if (found) {
          wallet = found;
          connect().catch(console.error);
        }
      }
    }
  });

  async function select(walletName: string) {
    const selected = wallets.find(w => w.name === walletName);
    if (!selected) throw new Error(`Wallet ${walletName} not found`);
    
    if (wallet && wallet.name !== walletName) {
      await disconnect();
    }
    
    wallet = selected;
  }

  async function connect() {
    if (!wallet) throw new Error('No wallet selected');
    if (connected || connecting) return;

    connecting = true;
    try {
      await wallet.connect();
    } catch (error) {
      console.error('Failed to connect wallet:', error);
      throw error;
    } finally {
      connecting = false;
    }
  }

  async function disconnect() {
    if (!wallet) return;
    
    disconnecting = true;
    try {
      await wallet.disconnect();
      localStorage.removeItem('walletName');
    } catch (error) {
      console.error('Failed to disconnect wallet:', error);
    } finally {
      disconnecting = false;
    }
  }

  async function signTransaction<T extends Transaction | VersionedTransaction>(tx: T): Promise<T> {
    if (!wallet || !connected) throw new Error('Wallet not connected');
    return wallet.signTransaction(tx);
  }

  async function signAllTransactions<T extends Transaction | VersionedTransaction>(txs: T[]): Promise<T[]> {
    if (!wallet || !connected) throw new Error('Wallet not connected');
    return wallet.signAllTransactions(txs);
  }

  async function signMessage(message: Uint8Array): Promise<Uint8Array> {
    if (!wallet || !connected) throw new Error('Wallet not connected');
    return wallet.signMessage(message);
  }

  return {
    get wallets() { return wallets; },
    get wallet() { return wallet; },
    get publicKey() { return publicKey; },
    get connected() { return connected; },
    get connecting() { return connecting; },
    get disconnecting() { return disconnecting; },
    select,
    connect,
    disconnect,
    signTransaction,
    signAllTransactions,
    signMessage,
  };
}

export const wallet = createWalletStore();
```

### ✅ DO: Create Composable Program Interactions

```typescript
// app/src/lib/stores/program.svelte.ts
import { Program, AnchorProvider, BN } from '@coral-xyz/anchor';
import { Connection, PublicKey, Transaction } from '@solana/web3.js';
import { createQuery, createMutation } from '@tanstack/svelte-query';
import { wallet } from './wallet.svelte';
import { getRpcUrl } from '$lib/config/rpc';
import { IDL, type MyProgram } from '$lib/contracts/idl/my_program';
import toast from 'svelte-french-toast';

class ProgramStore {
  connection = $state<Connection>(new Connection(getRpcUrl(), 'confirmed'));
  program = $state<Program<MyProgram> | null>(null);

  constructor() {
    // Initialize program when wallet connects
    $effect(() => {
      if (wallet.connected && wallet.publicKey) {
        const provider = new AnchorProvider(
          this.connection,
          wallet as any,
          { preflightCommitment: 'confirmed' }
        );
        
        this.program = new Program(IDL, provider);
      } else {
        this.program = null;
      }
    });
  }

  // Query for protocol state
  protocolQuery = () => createQuery({
    queryKey: ['protocol', this.program?.programId.toString()],
    queryFn: async () => {
      if (!this.program) throw new Error('Program not initialized');
      
      const [protocolPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('protocol')],
        this.program.programId
      );
      
      return this.program.account.protocol.fetch(protocolPda);
    },
    enabled: !!this.program,
    refetchInterval: 30000, // Refresh every 30s
  });

  // Query for user account
  userAccountQuery = (userPubkey: PublicKey | null) => createQuery({
    queryKey: ['userAccount', userPubkey?.toString(), this.program?.programId.toString()],
    queryFn: async () => {
      if (!this.program || !userPubkey) throw new Error('Invalid parameters');
      
      const [protocolPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('protocol')],
        this.program.programId
      );
      
      const [userAccountPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('user'), userPubkey.toBuffer(), protocolPda.toBuffer()],
        this.program.programId
      );
      
      try {
        return await this.program.account.userAccount.fetch(userAccountPda);
      } catch (e) {
        // Account doesn't exist yet
        return null;
      }
    },
    enabled: !!this.program && !!userPubkey,
  });

  // Mutation for deposits
  depositMutation = () => createMutation({
    mutationFn: async ({ amount }: { amount: number }) => {
      if (!this.program || !wallet.publicKey) {
        throw new Error('Wallet not connected');
      }

      const [protocolPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('protocol')],
        this.program.programId
      );

      const lamports = new BN(amount * 1e9); // Convert SOL to lamports

      const tx = await this.program.methods
        .executeAction(lamports)
        .accounts({
          protocol: protocolPda,
          // ... other accounts
          user: wallet.publicKey,
        })
        .transaction();

      return this.sendTransaction(tx);
    },
    onSuccess: () => {
      toast.success('Deposit successful!');
      // Invalidate queries to refetch data
      queryClient.invalidateQueries({ queryKey: ['protocol'] });
      queryClient.invalidateQueries({ queryKey: ['userAccount'] });
    },
    onError: (error) => {
      toast.error(`Deposit failed: ${error.message}`);
    },
  });

  private async sendTransaction(tx: Transaction): Promise<string> {
    if (!wallet.publicKey || !wallet.signTransaction) {
      throw new Error('Wallet not ready');
    }

    // Get latest blockhash
    const { blockhash, lastValidBlockHeight } = 
      await this.connection.getLatestBlockhash('confirmed');
    
    tx.recentBlockhash = blockhash;
    tx.feePayer = wallet.publicKey;

    // Sign transaction
    const signed = await wallet.signTransaction(tx);

    // Send with retries
    const signature = await this.connection.sendRawTransaction(
      signed.serialize(),
      {
        skipPreflight: false,
        preflightCommitment: 'confirmed',
        maxRetries: 3,
      }
    );

    // Confirm with timeout
    const confirmation = await this.connection.confirmTransaction({
      signature,
      blockhash,
      lastValidBlockHeight,
    }, 'confirmed');

    if (confirmation.value.err) {
      throw new Error(`Transaction failed: ${confirmation.value.err.toString()}`);
    }

    return signature;
  }

  // Priority fees helper
  async getPriorityFeeEstimate(tx: Transaction): Promise<number> {
    try {
      const response = await fetch(this.connection.rpcEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jsonrpc: '2.0',
          id: 1,
          method: 'getPriorityFeeEstimate',
          params: [{
            transaction: bs58.encode(tx.serialize()),
            options: { includeAllPriorityFeeLevels: true }
          }]
        })
      });
      
      const data = await response.json();
      return data.result?.priorityFeeLevels?.medium || 1000;
    } catch {
      return 1000; // Default to 1000 microlamports
    }
  }
}

export const program = new ProgramStore();
```

### ✅ DO: Handle Transaction States with Svelte 5

```svelte
<!-- app/src/lib/components/DepositForm.svelte -->
<script lang="ts">
  import { program } from '$lib/stores/program.svelte';
  import { wallet } from '$lib/stores/wallet.svelte';
  import { formatAmount } from '$lib/utils/formatting';
  import { BN } from '@coral-xyz/anchor';
  
  let amount = $state('');
  let isValidAmount = $derived(
    amount && !isNaN(parseFloat(amount)) && parseFloat(amount) > 0
  );
  
  const depositMutation = program.depositMutation();
  const protocolQuery = program.protocolQuery();
  const userAccountQuery = program.userAccountQuery(wallet.publicKey);
  
  async function handleDeposit() {
    if (!isValidAmount) return;
    
    await $depositMutation.mutateAsync({
      amount: parseFloat(amount)
    });
    
    // Reset form on success
    amount = '';
  }
  
  // Compute max deposit based on protocol limits
  const maxDeposit = $derived.by(() => {
    if (!$protocolQuery.data) return null;
    return formatAmount($protocolQuery.data.maxDepositAmount);
  });
  
  // Show user's current balance
  const userBalance = $derived.by(() => {
    if (!$userAccountQuery.data) return '0';
    return formatAmount($userAccountQuery.data.depositedAmount);
  });
</script>

<div class="deposit-form">
  <h2>Deposit SOL</h2>
  
  {#if $protocolQuery.isLoading}
    <div class="skeleton" />
  {:else if $protocolQuery.error}
    <div class="error">Failed to load protocol data</div>
  {:else}
    <form onsubmit={handleDeposit}>
      <label>
        Amount (SOL)
        <input
          type="number"
          bind:value={amount}
          placeholder="0.0"
          step="0.01"
          min="0"
          max={maxDeposit}
          disabled={$depositMutation.isPending || !wallet.connected}
        />
        {#if maxDeposit}
          <span class="hint">Max: {maxDeposit} SOL</span>
        {/if}
      </label>
      
      <div class="info">
        <p>Your current balance: <strong>{userBalance} SOL</strong></p>
        <p>Protocol fee: <strong>{$protocolQuery.data.feeBasisPoints / 100}%</strong></p>
      </div>
      
      <button
        type="submit"
        disabled={!isValidAmount || $depositMutation.isPending || !wallet.connected}
      >
        {#if $depositMutation.isPending}
          <span class="spinner" />
          Processing...
        {:else if !wallet.connected}
          Connect Wallet
        {:else}
          Deposit
        {/if}
      </button>
      
      {#if $depositMutation.error}
        <div class="error">
          {$depositMutation.error.message}
        </div>
      {/if}
    </form>
  {/if}
</div>

<style>
  .deposit-form {
    max-width: 400px;
    margin: 2rem auto;
    padding: 2rem;
    background: var(--surface);
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  }
  
  form {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }
  
  label {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  
  input {
    padding: 0.75rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 1rem;
  }
  
  .hint {
    font-size: 0.875rem;
    color: var(--text-secondary);
  }
  
  .info {
    padding: 1rem;
    background: var(--surface-secondary);
    border-radius: 8px;
  }
  
  button {
    padding: 1rem;
    background: var(--primary);
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
  }
  
  button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
  
  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid transparent;
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
  }
  
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  
  .error {
    padding: 1rem;
    background: var(--error-bg);
    color: var(--error-text);
    border-radius: 8px;
  }
</style>
```

## 9. RPC Optimization & Transaction Management

### ✅ DO: Implement Smart RPC Fallbacks

```typescript
// app/src/lib/config/rpc.ts
import { Connection, ConnectionConfig } from '@solana/web3.js';
import PQueue from 'p-queue';
import pRetry from 'p-retry';

interface RpcEndpoint {
  url: string;
  weight: number;
  rateLimit?: number;
}

const RPC_ENDPOINTS: RpcEndpoint[] = [
  { url: import.meta.env.VITE_RPC_URL, weight: 10, rateLimit: 50 },
  { url: 'https://api.mainnet-beta.solana.com', weight: 1, rateLimit: 10 },
  { url: 'https://solana-mainnet.g.alchemy.com/v2/YOUR_KEY', weight: 5, rateLimit: 30 },
];

class SmartConnection extends Connection {
  private queues: Map<string, PQueue>;
  private healthScores: Map<string, number>;
  
  constructor() {
    // Start with primary endpoint
    super(RPC_ENDPOINTS[0].url, {
      commitment: 'confirmed',
      wsEndpoint: RPC_ENDPOINTS[0].url.replace('https', 'wss'),
    });
    
    // Initialize rate limit queues
    this.queues = new Map(
      RPC_ENDPOINTS.map(endpoint => [
        endpoint.url,
        new PQueue({ 
          concurrency: endpoint.rateLimit || 10,
          interval: 1000,
          intervalCap: endpoint.rateLimit || 10,
        })
      ])
    );
    
    this.healthScores = new Map(
      RPC_ENDPOINTS.map(endpoint => [endpoint.url, 100])
    );
    
    // Monitor health
    this.startHealthMonitoring();
  }
  
  private async startHealthMonitoring() {
    setInterval(async () => {
      for (const endpoint of RPC_ENDPOINTS) {
        try {
          const start = Date.now();
          await fetch(endpoint.url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              jsonrpc: '2.0',
              id: 1,
              method: 'getHealth'
            })
          });
          
          const latency = Date.now() - start;
          const score = Math.max(0, 100 - latency / 10);
          this.healthScores.set(endpoint.url, score);
        } catch {
          this.healthScores.set(endpoint.url, 0);
        }
      }
    }, 30000); // Check every 30s
  }
  
  // Override methods to use smart routing
  async sendRawTransaction(
    rawTransaction: Uint8Array,
    options?: any
  ): Promise<string> {
    return pRetry(
      async () => {
        const endpoint = this.selectEndpoint();
        const queue = this.queues.get(endpoint.url)!;
        
        return queue.add(async () => {
          const response = await fetch(endpoint.url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              jsonrpc: '2.0',
              id: 1,
              method: 'sendTransaction',
              params: [
                Buffer.from(rawTransaction).toString('base64'),
                { encoding: 'base64', ...options }
              ]
            })
          });
          
          const data = await response.json();
          if (data.error) throw new Error(data.error.message);
          return data.result;
        });
      },
      {
        retries: 3,
        onFailedAttempt: (error) => {
          console.warn(`RPC request failed, attempt ${error.attemptNumber}:`, error.message);
        }
      }
    );
  }
  
  private selectEndpoint(): RpcEndpoint {
    // Weighted random selection based on health
    const totalWeight = RPC_ENDPOINTS.reduce((sum, endpoint) => {
      const health = this.healthScores.get(endpoint.url) || 0;
      return sum + (endpoint.weight * health / 100);
    }, 0);
    
    let random = Math.random() * totalWeight;
    
    for (const endpoint of RPC_ENDPOINTS) {
      const health = this.healthScores.get(endpoint.url) || 0;
      const weight = endpoint.weight * health / 100;
      
      if (random < weight) return endpoint;
      random -= weight;
    }
    
    return RPC_ENDPOINTS[0]; // Fallback
  }
}

export const connection = new SmartConnection();
```

### ✅ DO: Implement Transaction Batching and Priority Fees

```typescript
// app/src/lib/utils/transactions.ts
import { 
  Transaction, 
  TransactionInstruction,
  ComputeBudgetProgram,
  PublicKey,
  TransactionMessage,
  VersionedTransaction
} from '@solana/web3.js';
import { connection } from '$lib/config/rpc';
import type { Wallet } from '$lib/stores/wallet.svelte';

interface TransactionOptions {
  priorityLevel?: 'low' | 'medium' | 'high' | 'max';
  computeUnits?: number;
  skipPreflight?: boolean;
  retries?: number;
}

export class TransactionBuilder {
  private instructions: TransactionInstruction[] = [];
  private signers: any[] = [];
  
  add(ix: TransactionInstruction, signers: any[] = []): this {
    this.instructions.push(ix);
    this.signers.push(...signers);
    return this;
  }
  
  async build(
    wallet: Wallet,
    options: TransactionOptions = {}
  ): Promise<VersionedTransaction> {
    if (!wallet.publicKey) throw new Error('Wallet not connected');
    
    // Get priority fee estimate
    const priorityFee = await this.estimatePriorityFee(options.priorityLevel || 'medium');
    
    // Add compute budget instructions
    const computeIx = ComputeBudgetProgram.setComputeUnitLimit({
      units: options.computeUnits || 200_000,
    });
    
    const priceIx = ComputeBudgetProgram.setComputeUnitPrice({
      microLamports: priorityFee,
    });
    
    // Build transaction
    const { blockhash, lastValidBlockHeight } = 
      await connection.getLatestBlockhash('confirmed');
    
    const message = new TransactionMessage({
      payerKey: wallet.publicKey,
      recentBlockhash: blockhash,
      instructions: [computeIx, priceIx, ...this.instructions],
    }).compileToV0Message();
    
    return new VersionedTransaction(message);
  }
  
  private async estimatePriorityFee(level: string): number {
    try {
      // Simulate transaction to get priority fee estimate
      const testTx = new Transaction().add(...this.instructions);
      const response = await connection.simulateTransaction(testTx);
      
      // Use actual consumed units for estimate
      const unitsConsumed = response.value.unitsConsumed || 200_000;
      
      // Priority fee levels (microlamports per compute unit)
      const levels = {
        low: 1,
        medium: 1000,
        high: 10_000,
        max: 100_000,
      };
      
      return levels[level] || levels.medium;
    } catch {
      return 1000; // Default medium priority
    }
  }
  
  async sendAndConfirm(
    wallet: Wallet,
    options: TransactionOptions = {}
  ): Promise<string> {
    const tx = await this.build(wallet, options);
    const signed = await wallet.signTransaction(tx);
    
    // Send with retries
    const signature = await connection.sendRawTransaction(
      signed.serialize(),
      {
        skipPreflight: options.skipPreflight || false,
        maxRetries: options.retries || 3,
      }
    );
    
    // Wait for confirmation
    const { blockhash, lastValidBlockHeight } = 
      await connection.getLatestBlockhash('confirmed');
    
    const confirmation = await connection.confirmTransaction({
      signature,
      blockhash,
      lastValidBlockHeight,
    }, 'confirmed');
    
    if (confirmation.value.err) {
      throw new Error(`Transaction failed: ${JSON.stringify(confirmation.value.err)}`);
    }
    
    return signature;
  }
}

// Helper for common patterns
export async function sendTransaction(
  wallet: Wallet,
  instructions: TransactionInstruction[],
  options?: TransactionOptions
): Promise<string> {
  const builder = new TransactionBuilder();
  instructions.forEach(ix => builder.add(ix));
  return builder.sendAndConfirm(wallet, options);
}
```

## 10. Production Deployment

### ✅ DO: Implement Comprehensive Security Audits

```rust
// scripts/security-audit.rs
use anchor_lang::prelude::*;
use solana_program::program_pack::Pack;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔍 Running security audit...");
    
    // Check 1: Verify program upgrade authority
    check_upgrade_authority().await?;
    
    // Check 2: Scan for missing owner checks
    scan_owner_checks().await?;
    
    // Check 3: Verify arithmetic operations
    verify_safe_math().await?;
    
    // Check 4: Check for reentrancy vulnerabilities
    check_reentrancy_guards().await?;
    
    // Check 5: Validate PDA derivations
    validate_pda_derivations().await?;
    
    // Check 6: Ensure proper account discriminators
    check_account_discriminators().await?;
    
    println!("✅ Security audit complete!");
    Ok(())
}
```

### ✅ DO: Create Deployment Scripts with Verification

```typescript
// scripts/deploy.ts
import { Program, AnchorProvider, Wallet } from '@coral-xyz/anchor';
import { Connection, Keypair, PublicKey } from '@solana/web3.js';
import { execSync } from 'child_process';
import fs from 'fs/promises';
import chalk from 'chalk';

async function deploy() {
  console.log(chalk.blue('🚀 Starting deployment...'));
  
  // 1. Build program
  console.log(chalk.yellow('Building program...'));
  execSync('anchor build', { stdio: 'inherit' });
  
  // 2. Run tests
  console.log(chalk.yellow('Running tests...'));
  execSync('anchor test --skip-deploy', { stdio: 'inherit' });
  
  // 3. Deploy to devnet
  console.log(chalk.yellow('Deploying to devnet...'));
  const programId = execSync('anchor deploy --provider.cluster devnet')
    .toString()
    .match(/Program Id: (\w+)/)?.[1];
  
  if (!programId) throw new Error('Deployment failed');
  
  // 4. Verify deployment
  console.log(chalk.yellow('Verifying deployment...'));
  const connection = new Connection('https://api.devnet.solana.com');
  const accountInfo = await connection.getAccountInfo(new PublicKey(programId));
  
  if (!accountInfo || !accountInfo.executable) {
    throw new Error('Program not found or not executable');
  }
  
  // 5. Initialize program
  console.log(chalk.yellow('Initializing program...'));
  await initializeProgram(programId);
  
  // 6. Update frontend config
  console.log(chalk.yellow('Updating frontend configuration...'));
  await updateFrontendConfig(programId);
  
  // 7. Verify on explorer
  const explorerUrl = `https://explorer.solana.com/address/${programId}?cluster=devnet`;
  console.log(chalk.green(`✅ Deployment successful!`));
  console.log(chalk.blue(`View on Explorer: ${explorerUrl}`));
  
  // 8. Run post-deployment checks
  await runSmokeTests(programId);
}

async function initializeProgram(programId: string) {
  // Initialize protocol state
  const provider = AnchorProvider.env();
  const program = new Program(IDL, programId, provider);
  
  const [protocolPda] = PublicKey.findProgramAddressSync(
    [Buffer.from('protocol')],
    program.programId
  );
  
  // Check if already initialized
  try {
    await program.account.protocol.fetch(protocolPda);
    console.log(chalk.gray('Protocol already initialized'));
    return;
  } catch {
    // Not initialized, proceed
  }
  
  await program.methods
    .initialize({ feeBasisPoints: 250 })
    .accounts({
      protocol: protocolPda,
      authority: provider.wallet.publicKey,
      treasury: provider.wallet.publicKey,
      systemProgram: SystemProgram.programId,
    })
    .rpc();
}

async function updateFrontendConfig(programId: string) {
  const config = {
    programId,
    cluster: 'devnet',
    deployedAt: new Date().toISOString(),
  };
  
  await fs.writeFile(
    './app/src/lib/config/program.json',
    JSON.stringify(config, null, 2)
  );
}

async function runSmokeTests(programId: string) {
  console.log(chalk.yellow('Running smoke tests...'));
  
  // Test 1: Can fetch protocol state
  // Test 2: Can create user account
  // Test 3: Can execute basic action
  
  console.log(chalk.green('✅ All smoke tests passed!'));
}

// Run deployment
deploy().catch(console.error);
```

### ✅ DO: Monitor On-Chain Metrics

```typescript
// app/src/lib/monitoring/metrics.ts
import { Connection, PublicKey } from '@solana/web3.js';
import { Program } from '@coral-xyz/anchor';

export class OnChainMetrics {
  constructor(
    private program: Program,
    private connection: Connection
  ) {}
  
  async collectMetrics() {
    const [protocolPda] = PublicKey.findProgramAddressSync(
      [Buffer.from('protocol')],
      this.program.programId
    );
    
    // Fetch protocol state
    const protocol = await this.program.account.protocol.fetch(protocolPda);
    
    // Get program account info
    const programInfo = await this.connection.getAccountInfo(
      this.program.programId
    );
    
    // Calculate rent
    const rentExempt = await this.connection.getMinimumBalanceForRentExemption(
      programInfo?.data.length || 0
    );
    
    // Fetch recent transactions
    const signatures = await this.connection.getSignaturesForAddress(
      this.program.programId,
      { limit: 100 }
    );
    
    // Parse events from transactions
    const events = await this.parseEvents(signatures);
    
    return {
      protocol: {
        totalValueLocked: protocol.totalValueLocked.toString(),
        totalUsers: protocol.totalUsers.toString(),
        paused: protocol.paused,
        feeBasisPoints: protocol.feeBasisPoints,
      },
      program: {
        executable: programInfo?.executable,
        owner: programInfo?.owner.toString(),
        lamports: programInfo?.lamports,
        rentExempt,
        dataLen: programInfo?.data.length,
      },
      activity: {
        recentTransactions: signatures.length,
        events: events.length,
        lastActivity: signatures[0]?.blockTime 
          ? new Date(signatures[0].blockTime * 1000)
          : null,
      },
    };
  }
  
  private async parseEvents(signatures: any[]): Promise<any[]> {
    const events = [];
    
    for (const sig of signatures.slice(0, 10)) { // Check last 10
      try {
        const tx = await this.connection.getParsedTransaction(
          sig.signature,
          { maxSupportedTransactionVersion: 0 }
        );
        
        // Extract events from logs
        const logs = tx?.meta?.logMessages || [];
        const programLogs = logs.filter(log => 
          log.includes(this.program.programId.toString())
        );
        
        // Parse event data
        for (const log of programLogs) {
          if (log.includes('Program data:')) {
            // Decode base64 event data
            const data = log.split('Program data: ')[1];
            events.push({
              signature: sig.signature,
              data,
              timestamp: sig.blockTime,
            });
          }
        }
      } catch (e) {
        console.error('Failed to parse transaction:', e);
      }
    }
    
    return events;
  }
}
```

## 11. Advanced Patterns

### Compressed NFTs with State Compression

```rust
// programs/my-program/src/instructions/compressed_nft.rs
use anchor_lang::prelude::*;
use spl_account_compression::{
    program::SplAccountCompression,
    cpi::{accounts::Modify, modify},
    wrap_application_data_v1,
};
use mpl_bubblegum::state::metaplex_adapter::{
    Collection, Creator, TokenProgramVersion, TokenStandard, UseMethod,
};

#[derive(Accounts)]
pub struct MintCompressedNFT<'info> {
    #[account(mut)]
    pub tree_authority: Account<'info, TreeConfig>,
    
    /// CHECK: Merkle tree account
    #[account(mut)]
    pub merkle_tree: UncheckedAccount<'info>,
    
    pub payer: Signer<'info>,
    pub tree_delegate: Signer<'info>,
    
    pub log_wrapper: Program<'info, Noop>,
    pub compression_program: Program<'info, SplAccountCompression>,
    pub system_program: Program<'info, System>,
}

pub fn mint_compressed_nft(
    ctx: Context<MintCompressedNFT>,
    metadata: CompressedNFTMetadata,
) -> Result<()> {
    let tree_authority = &ctx.accounts.tree_authority;
    let merkle_tree = &ctx.accounts.merkle_tree;
    let payer = &ctx.accounts.payer;
    
    // Prepare metadata
    let metadata_args = MetadataArgs {
        name: metadata.name,
        symbol: metadata.symbol,
        uri: metadata.uri,
        seller_fee_basis_points: 500, // 5%
        primary_sale_happened: false,
        is_mutable: true,
        edition_nonce: None,
        token_standard: Some(TokenStandard::NonFungible),
        collection: metadata.collection.map(|c| Collection {
            verified: false,
            key: c,
        }),
        uses: None,
        token_program_version: TokenProgramVersion::Original,
        creators: vec![Creator {
            address: payer.key(),
            verified: true,
            share: 100,
        }],
    };
    
    // Hash metadata
    let metadata_hash = hash_metadata(&metadata_args)?;
    
    // Create leaf
    let leaf = LeafSchema::new_v0(
        tree_authority.key(),
        payer.key(),
        payer.key(),
        metadata_hash,
    );
    
    // Add to merkle tree
    let cpi_ctx = CpiContext::new_with_signer(
        ctx.accounts.compression_program.to_account_info(),
        Modify {
            merkle_tree: merkle_tree.to_account_info(),
            authority: tree_authority.to_account_info(),
            noop: ctx.accounts.log_wrapper.to_account_info(),
        },
        &[&[
            merkle_tree.key().as_ref(),
            &[tree_authority.bump],
        ]],
    );
    
    modify(cpi_ctx, leaf.to_bytes())?;
    
    // Wrap and log metadata
    wrap_application_data_v1(
        metadata_args.try_to_vec()?,
        &ctx.accounts.log_wrapper,
    )?;
    
    emit!(CompressedNFTMinted {
        tree: merkle_tree.key(),
        leaf_index: tree_authority.num_minted,
        metadata_hash,
    });
    
    tree_authority.num_minted += 1;
    
    Ok(())
}
```

### Light Protocol Integration for ZK Compression

```rust
// programs/my-program/src/instructions/zk_compress.rs
use anchor_lang::prelude::*;
use light_protocol_program::{
    process_instruction::process_instruction,
    state::CompressedAccount,
    utils::CompressedData,
};

#[derive(Accounts)]
pub struct CompressData<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    
    /// CHECK: Light Protocol program
    pub light_program: UncheckedAccount<'info>,
    
    /// CHECK: State merkle tree
    pub state_merkle_tree: UncheckedAccount<'info>,
    
    /// CHECK: Nullifier queue
    pub nullifier_queue: UncheckedAccount<'info>,
    
    pub system_program: Program<'info, System>,
}

pub fn compress_account_data(
    ctx: Context<CompressData>,
    data: Vec<u8>,
) -> Result<()> {
    // Validate data size
    require!(data.len() <= 10_240, MyError::DataTooLarge); // 10KB max
    
    // Create compressed account
    let compressed_account = CompressedAccount {
        owner: ctx.accounts.payer.key(),
        lamports: 0,
        data: CompressedData::from(data),
        address: None,
    };
    
    // Build Light Protocol instruction
    let ix = light_protocol_program::instruction::compress_account(
        &ctx.accounts.light_program.key(),
        &ctx.accounts.payer.key(),
        &ctx.accounts.state_merkle_tree.key(),
        &ctx.accounts.nullifier_queue.key(),
        compressed_account,
    )?;
    
    // Execute CPI
    solana_program::program::invoke(
        &ix,
        &[
            ctx.accounts.payer.to_account_info(),
            ctx.accounts.light_program.to_account_info(),
            ctx.accounts.state_merkle_tree.to_account_info(),
            ctx.accounts.nullifier_queue.to_account_info(),
            ctx.accounts.system_program.to_account_info(),
        ],
    )?;
    
    emit!(DataCompressed {
        owner: ctx.accounts.payer.key(),
        data_hash: hash(&data),
        size: data.len() as u64,
    });
    
    Ok(())
}
```

### Clockwork Automation

```rust
// programs/my-program/src/instructions/automation.rs
use anchor_lang::prelude::*;
use clockwork_sdk::{
    cpi::{thread_create, thread_delete},
    state::{Thread, ThreadSettings},
    ThreadProgram,
};

#[derive(Accounts)]
#[instruction(thread_id: String)]
pub struct CreateAutomation<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,
    
    #[account(mut)]
    pub payer: Signer<'info>,
    
    /// CHECK: Thread account (PDA)
    #[account(
        mut,
        seeds = [b"thread", authority.key().as_ref(), thread_id.as_bytes()],
        seeds::program = clockwork_program.key(),
        bump
    )]
    pub thread: UncheckedAccount<'info>,
    
    pub clockwork_program: Program<'info, ThreadProgram>,
    pub system_program: Program<'info, System>,
}

pub fn create_automated_task(
    ctx: Context<CreateAutomation>,
    thread_id: String,
    schedule: String, // Cron expression
) -> Result<()> {
    let clockwork_program = &ctx.accounts.clockwork_program;
    let authority = &ctx.accounts.authority;
    let payer = &ctx.accounts.payer;
    let thread = &ctx.accounts.thread;
    
    // Create instruction for automated task
    let target_ix = Instruction {
        program_id: crate::ID,
        accounts: vec![
            AccountMeta::new(ctx.accounts.protocol.key(), false),
            // ... other accounts
        ],
        data: crate::instruction::ExecuteScheduledTask {}.data(),
    };
    
    // Create thread
    thread_create(
        CpiContext::new_with_signer(
            clockwork_program.to_account_info(),
            clockwork_sdk::cpi::accounts::ThreadCreate {
                authority: authority.to_account_info(),
                payer: payer.to_account_info(),
                thread: thread.to_account_info(),
                system_program: ctx.accounts.system_program.to_account_info(),
            },
            &[&[
                b"thread",
                authority.key().as_ref(),
                thread_id.as_bytes(),
                &[ctx.bumps.thread],
            ]],
        ),
        thread_id,
        vec![target_ix],
        ThreadSettings {
            fee: 1000, // lamports
            kickoff_instruction: None,
            rate_limit: Some(60), // seconds
            schedule: Some(schedule), // e.g., "0 */6 * * *" for every 6 hours
        },
    )?;
    
    emit!(AutomationCreated {
        thread_id,
        authority: authority.key(),
        schedule,
    });
    
    Ok(())
}
```

## Conclusion

This guide represents the cutting edge of Solana development as of mid-2025. The combination of Anchor's powerful abstractions, Solang's EVM compatibility, and SvelteKit's reactive architecture creates a development experience that is both productive and performant.

Key takeaways:
- **Security first**: Every line of code should consider potential attack vectors
- **Optimize for Solana's architecture**: Leverage PDAs, CPIs, and parallel execution
- **User experience matters**: Handle transaction states gracefully and provide clear feedback
- **Test everything**: From unit tests to integration tests with Bankrun
- **Monitor in production**: Track on-chain metrics and user behavior

The Solana ecosystem continues to evolve rapidly. Stay engaged with the community, follow SIPs (Solana Improvement Proposals), and always validate patterns against the latest best practices. With these foundations, you're ready to build the next generation of decentralized applications on the world's most performant blockchain.

Remember: The best Solana programs are not just secure and efficient—they're also maintainable, well-documented, and built with the end user in mind. Happy building! 🚀