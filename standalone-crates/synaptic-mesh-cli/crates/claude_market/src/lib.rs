//! # Claude Market - Peer Compute Federation
//! 
//! A compliant marketplace for compute contribution where participants voluntarily share
//! compute resources and are rewarded with tokens for successful task completions.
//! 
//! ## Core Principles
//! 
//! - **No Shared API Keys**: Each contributor runs their own Claude Max account locally
//! - **Task Routing**: Market facilitates task distribution, not Claude access
//! - **Contribution Rewards**: Tokens reward successful completions, not API access
//! - **Voluntary Participation**: Users maintain full control and transparency
//! 
//! ## Features
//! 
//! - **First-Accept Auctions**: Fast task assignment to qualified providers
//! - **Price Discovery**: Transparent market pricing with 24h averages and VWAP
//! - **Reputation-Weighted Matching**: Quality providers get preference
//! - **SLA Enforcement**: Automatic tracking and penalty calculation
//! - **Privacy Levels**: Public, Private, and Confidential task handling
//! - **Secure Escrow**: Multi-sig escrow for payment protection
//! 
//! ## Architecture
//! 
//! The market operates as a peer compute federation similar to folding@home or BOINC,
//! where participants contribute spare compute capacity. All operations are cryptographically
//! secured and no credentials are ever shared between participants.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod error;
pub mod escrow;
pub mod ledger;
pub mod market;
pub mod reputation;
pub mod wallet;

// Re-export core types
pub use error::{MarketError, Result};
pub use escrow::{
    Escrow, EscrowState, EscrowAgreement, MultiSigType, 
    ReleaseAuth, ReleaseDecision, DisputeInfo, DisputeOutcome,
    ArbitratorDecision, AuditEntry
};
pub use ledger::{Ledger, TokenTx, TxType};
pub use market::{
    Market, Order, OrderType, OrderStatus,
    ComputeRequest, ComputeOffer, ComputeTaskSpec, PrivacyLevel,
    SLASpec, FirstAcceptAuction, AuctionStatus,
    TaskAssignment, AssignmentStatus, SLAMetrics, PriceDiscovery
};
pub use reputation::{Reputation, ReputationScore};
pub use wallet::{Wallet, TokenBalance, TokenTransfer};

use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main market instance that coordinates all components
#[derive(Clone)]
pub struct ClaudeMarket {
    /// Peer ID for this market node
    pub peer_id: PeerId,
    /// Token wallet manager (shared with escrow)
    pub wallet: Arc<Wallet>,
    /// Market order book
    pub market: Arc<RwLock<Market>>,
    /// Escrow service
    pub escrow: Arc<Escrow>,
    /// Reputation tracker
    pub reputation: Arc<RwLock<Reputation>>,
    /// Transaction ledger
    pub ledger: Arc<RwLock<Ledger>>,
}

impl ClaudeMarket {
    /// Create a new market instance
    pub async fn new(peer_id: PeerId, db_path: Option<&str>) -> Result<Self> {
        let db_path = db_path.unwrap_or(":memory:");
        
        // Create shared wallet instance
        let wallet = Arc::new(Wallet::new(db_path).await?);
        
        // Create escrow with wallet integration
        let escrow = Arc::new(Escrow::new(db_path, wallet.clone()).await?);
        
        Ok(Self {
            peer_id,
            wallet: wallet.clone(),
            market: Arc::new(RwLock::new(Market::new(db_path).await?)),
            escrow,
            reputation: Arc::new(RwLock::new(Reputation::new(db_path).await?)),
            ledger: Arc::new(RwLock::new(Ledger::new(db_path).await?)),
        })
    }

    /// Initialize database schemas
    pub async fn initialize(&self) -> Result<()> {
        self.wallet.init_schema().await?;
        self.market.write().await.init_schema().await?;
        self.escrow.init_schema().await?;
        self.reputation.write().await.init_schema().await?;
        self.ledger.write().await.init_schema().await?;
        Ok(())
    }
}

/// Market configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConfig {
    /// Minimum reputation score to trade
    pub min_reputation: f64,
    /// Escrow timeout in seconds
    pub escrow_timeout_secs: u64,
    /// Maximum trade size in tokens
    pub max_trade_size: u64,
    /// Fee percentage (0.0 - 1.0)
    pub fee_rate: f64,
}

impl Default for MarketConfig {
    fn default() -> Self {
        Self {
            min_reputation: 0.0,
            escrow_timeout_secs: 3600, // 1 hour
            max_trade_size: 1000,
            fee_rate: 0.01, // 1%
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_market_creation() {
        let peer_id = PeerId::random();
        let market = ClaudeMarket::new(peer_id, None).await.unwrap();
        market.initialize().await.unwrap();
        
        assert_eq!(market.peer_id, peer_id);
    }
}