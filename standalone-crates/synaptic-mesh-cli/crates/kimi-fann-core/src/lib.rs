//! # Kimi-FANN Core: Micro-Expert Neural Architecture
//! 
//! This crate provides the core micro-expert architecture for converting Kimi-K2's 
//! 384 experts into 50-100 micro-experts (1K-100K parameters each) using ruv-FANN 
//! neural networks compiled to WebAssembly.
//!
//! ## Architecture Overview
//!
//! The system converts Kimi-K2's massive mixture-of-experts model into a distributed
//! collection of tiny, specialized neural networks that can run efficiently in WASM
//! environments while maintaining intelligent routing and coordination capabilities.
//!
//! ## Key Features
//!
//! - **Memory Efficient**: <512MB memory usage per expert
//! - **Fast Inference**: <100ms inference time per expert  
//! - **Zero Unsafe Code**: Built on ruv-FANN's memory-safe foundation
//! - **WASM Ready**: Browser deployment with Web Workers
//! - **Synaptic Mesh Integration**: Compatible with DAA system
//!
//! ## Example Usage
//!
//! ```rust
//! use kimi_fann_core::{ExpertDomain, MicroExpert, ExpertRouter};
//! 
//! // Create a reasoning micro-expert
//! let expert = MicroExpert::new(ExpertDomain::Reasoning, 10_000)?;
//! 
//! // Initialize the expert router
//! let mut router = ExpertRouter::new();
//! router.register_expert(ExpertDomain::Reasoning, expert);
//! 
//! // Route a request to appropriate experts
//! let experts = router.route_request("Solve this logic puzzle").await?;
//! ```

use wasm_bindgen::prelude::*;

pub mod expert;
pub mod router;
pub mod memory;
pub mod domains;
pub mod runtime;
pub mod error;
pub mod enhanced_router;
pub mod async_processor;

pub use expert::{MicroExpert, ExpertConfig, ExpertParams, ExpertMetrics};
pub use router::{ExpertRouter, RoutingStrategy, RequestContext};
pub use memory::{ExpertMemoryManager, MemoryStats};
pub use domains::ExpertDomain;
pub use runtime::{KimiWasmRuntime, RuntimeConfig};
pub use error::{KimiError, Result};
pub use enhanced_router::{EnhancedExpertRouter, EnhancedRequestContext, EnhancedExpertSelection};
pub use async_processor::{AsyncProcessor, AsyncTask, TaskResult, ParallelExecutor, WorkerComm};

/// Initialize the WASM module and set up logging
#[wasm_bindgen(start)]
pub fn init() {
    console_log::init_with_level(log::Level::Info).unwrap_or(());
    log::info!("Kimi-FANN Core initialized");
}

/// Version information for the crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Maximum recommended memory usage per expert (512MB)
pub const MAX_EXPERT_MEMORY: usize = 512 * 1024 * 1024;

/// Target inference time per expert (100ms)
pub const TARGET_INFERENCE_TIME_MS: u32 = 100;

/// Default number of experts to maintain in memory
pub const DEFAULT_EXPERT_CACHE_SIZE: usize = 10;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_constants() {
        assert_eq!(MAX_EXPERT_MEMORY, 512 * 1024 * 1024);
        assert_eq!(TARGET_INFERENCE_TIME_MS, 100);
        assert_eq!(DEFAULT_EXPERT_CACHE_SIZE, 10);
    }
}