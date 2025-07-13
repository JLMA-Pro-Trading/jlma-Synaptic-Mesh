//! Error types for Kimi-FANN Core

use thiserror::Error;
use wasm_bindgen::prelude::*;

/// Result type alias for Kimi-FANN operations
pub type Result<T> = std::result::Result<T, KimiError>;

/// Main error type for Kimi-FANN operations
#[derive(Error, Debug)]
pub enum KimiError {
    #[error("Expert not found: {domain:?}")]
    ExpertNotFound { domain: String },

    #[error("Memory allocation failed: {size} bytes")]
    MemoryAllocation { size: usize },

    #[error("Neural network error: {message}")]
    NeuralNetwork { message: String },

    #[error("WASM runtime error: {message}")]
    WasmRuntime { message: String },

    #[error("Routing error: {message}")]
    Routing { message: String },

    #[error("Configuration error: {message}")]
    Configuration { message: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Unknown error: {message}")]
    Unknown { message: String },
}

impl KimiError {
    pub fn neural_network(message: impl Into<String>) -> Self {
        Self::NeuralNetwork {
            message: message.into(),
        }
    }

    pub fn wasm_runtime(message: impl Into<String>) -> Self {
        Self::WasmRuntime {
            message: message.into(),
        }
    }

    pub fn routing(message: impl Into<String>) -> Self {
        Self::Routing {
            message: message.into(),
        }
    }

    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    pub fn unknown(message: impl Into<String>) -> Self {
        Self::Unknown {
            message: message.into(),
        }
    }
}

impl From<KimiError> for JsValue {
    fn from(err: KimiError) -> Self {
        JsValue::from_str(&err.to_string())
    }
}