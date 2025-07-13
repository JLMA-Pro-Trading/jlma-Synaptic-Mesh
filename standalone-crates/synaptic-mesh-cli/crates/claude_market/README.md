# Claude Market - Peer Compute Federation

A compliant marketplace for compute contribution where participants voluntarily share compute resources and are rewarded with tokens for successful task completions.

## Core Principles

- **No Shared API Keys**: Each contributor runs their own Claude Max account locally
- **Task Routing**: Market facilitates task distribution, not Claude access  
- **Contribution Rewards**: Tokens reward successful completions, not API access
- **Voluntary Participation**: Users maintain full control and transparency

## Features

- ⚡ **First-Accept Auctions**: Fast task assignment to qualified providers
- 💰 **Price Discovery**: Transparent market pricing with 24h averages and VWAP
- 🏆 **Reputation-Weighted Matching**: Quality providers get preference
- 📊 **SLA Enforcement**: Automatic tracking and penalty calculation
- 🔒 **Privacy Levels**: Public, Private, and Confidential task handling
- 🛡️ **Secure Escrow**: Multi-sig escrow for payment protection

## Quick Start

### Basic Usage

```rust
use claude_market::{Market, OrderType, ComputeTaskSpec, PrivacyLevel};
use libp2p::PeerId;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize market
    let market = Market::new("market.db").await?;
    market.init_schema().await?;

    let requester = PeerId::random();
    let provider = PeerId::random();

    // Create a compute task
    let task_spec = ComputeTaskSpec {
        task_type: "code_generation".to_string(),
        compute_units: 100,
        max_duration_secs: 300,
        required_capabilities: vec!["rust".to_string()],
        min_reputation: Some(50.0),
        privacy_level: PrivacyLevel::Private,
        encrypted_payload: None,
    };

    // Place compute request (starts auction)
    let request = market.place_order(
        OrderType::RequestCompute,
        requester,
        50, // 50 tokens per unit
        100, // 100 units
        task_spec.clone(),
        None,
        None,
        None,
    ).await?;

    // Provider offers compute
    let offer = market.place_order(
        OrderType::OfferCompute,
        provider,
        45, // Willing to work for 45 tokens per unit
        200, // Can handle 200 units
        task_spec,
        None,
        None,
        None,
    ).await?;

    // Check assignments
    let assignments = market.get_assignments(None, 10).await?;
    println!("Created {} assignments", assignments.len());

    Ok(())
}
```

## Examples

### 1. Market Simulation

Demonstrates real-world usage scenarios with multiple participants:

```bash
cd examples
cargo run --example market_simulation
```

This simulation shows:
- Peer compute federation in action
- Reputation-based provider selection
- SLA enforcement and quality tracking
- Price discovery for different task types

### 2. Economic Simulation

Tests market efficiency with 100+ rounds of trading:

```bash
cd examples  
cargo run --example economic_simulation
```

This simulation analyzes:
- Price convergence rates
- Market liquidity metrics
- SLA compliance effectiveness
- Reputation system impact

### Sample Output

```
🚀 Synaptic Market Simulation - Compute Contribution Trading

📋 Market Participants:
  Requesters: 2 users needing compute
  Providers: 3 users offering compute

📊 Setting up provider reputations...
  Provider 1: New contributor (50 reputation)
  Provider 2: Experienced (120 reputation)
  Provider 3: Elite contributor (175 reputation)

📝 Scenario 1: Code Generation Task
  ✅ Request placed, auction started
  ❌ Provider 1 rejected - reputation too low
  ✅ Provider 2 matched!
  💰 Effective price: 48 tokens/unit
  📦 Assigned units: 100
  💵 Total cost: 4800 tokens

📈 Price Discovery Data:
  Code Generation:
    Average price (24h): 47.50 tokens/unit
    Volume-weighted avg: 48.20 tokens/unit
    Price range: 45 - 52 tokens/unit
    Total volume: 350 units
```

## Architecture

### Core Components

1. **Market Engine** (`src/market.rs`)
   - First-accept auction implementation
   - Order matching and assignment creation
   - Price discovery calculation

2. **Reputation System** (`src/reputation.rs`)
   - Provider scoring and weighting
   - SLA violation tracking
   - Feedback and rating management

3. **Escrow Service** (`src/escrow.rs`)
   - Multi-signature transaction security
   - Dispute resolution mechanisms
   - Automatic timeout handling

4. **Wallet System** (`src/wallet.rs`)
   - Token balance management
   - Atomic transfers and locking
   - Transaction history

### Database Schema

The market uses SQLite with the following key tables:

- `orders` - Compute requests and offers
- `task_assignments` - Matched tasks with SLA tracking
- `auctions` - First-accept auction management
- `price_discovery` - Market pricing data
- `reputation_scores` - Provider reputation tracking

## Compliance

The Synaptic Market strictly follows a peer compute federation model:

### ✅ Compliant Design
- Each node uses their own Claude login locally
- Tasks are routed, not account access
- Tokens reward contribution, not access resale
- Users can opt in, audit, and cap usage

### ❌ Not Allowed
- Sharing API keys between users
- Central service using shared accounts
- Task relays through foreign keys
- Broker reselling someone's subscription

## API Reference

### Core Types

```rust
// Compute task specification
pub struct ComputeTaskSpec {
    pub task_type: String,
    pub compute_units: u64,
    pub max_duration_secs: u64,
    pub required_capabilities: Vec<String>,
    pub min_reputation: Option<f64>,
    pub privacy_level: PrivacyLevel,
    pub encrypted_payload: Option<Vec<u8>>,
}

// Task assignment with SLA tracking
pub struct TaskAssignment {
    pub id: Uuid,
    pub requester: PeerId,
    pub provider: PeerId,
    pub price_per_unit: u64,
    pub compute_units: u64,
    pub total_cost: u64,
    pub sla_metrics: SLAMetrics,
    pub status: AssignmentStatus,
}

// Price discovery data
pub struct PriceDiscovery {
    pub task_type: String,
    pub avg_price_24h: f64,
    pub vwap: f64,
    pub min_price: u64,
    pub max_price: u64,
    pub total_volume: u64,
}
```

### Key Methods

```rust
impl Market {
    // Place a compute order
    pub async fn place_order(
        order_type: OrderType,
        trader: PeerId,
        price_per_unit: u64,
        total_units: u64,
        task_spec: ComputeTaskSpec,
        sla_spec: Option<SLASpec>,
        expires_at: Option<DateTime<Utc>>,
        signing_key: Option<&SigningKey>,
    ) -> Result<Order>;

    // Start task execution
    pub async fn start_task(
        assignment_id: &Uuid,
        provider: &PeerId,
    ) -> Result<()>;

    // Complete task with quality scores
    pub async fn complete_task(
        assignment_id: &Uuid,
        provider: &PeerId,
        quality_scores: HashMap<String, f64>,
    ) -> Result<()>;

    // Get price discovery data
    pub async fn get_price_discovery(
        task_type: &str
    ) -> Result<Option<PriceDiscovery>>;
}
```

## Testing

Run the test suite:

```bash
cargo test
```

Key test coverage:
- Market order placement and matching
- Reputation-weighted provider selection
- SLA tracking and enforcement
- Price discovery calculation
- Auction lifecycle management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Ensure compliance guidelines are followed
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Documentation

- [Market Design Rationale](MARKET_DESIGN_RATIONALE.md) - Detailed design decisions and economic model
- [Issue #8 Update](ISSUE_8_UPDATE.md) - Implementation status and technical details
- [Compliance Guidelines](../../../plans/synaptic-market/compliance.md) - Legal and compliance framework