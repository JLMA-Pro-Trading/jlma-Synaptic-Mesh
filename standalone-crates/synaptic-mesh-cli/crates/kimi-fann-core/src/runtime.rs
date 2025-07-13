//! WASM runtime for Kimi-K2 micro-experts

use crate::{
    domains::ExpertDomain,
    expert::MicroExpert,
    memory::{ExpertMemoryManager, MemoryStats},
    router::{ExpertRouter, RequestContext},
    error::Result,
};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;
use futures::future;

/// Configuration for the WASM runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub max_memory_mb: usize,
    pub expert_cache_size: usize,
    pub enable_web_workers: bool,
    pub default_temperature: f32,
    pub max_context_length: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: 512,
            expert_cache_size: 10,
            enable_web_workers: true,
            default_temperature: 0.7,
            max_context_length: 32_000,
        }
    }
}

/// Context window for maintaining conversation state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextWindow {
    pub messages: Vec<String>,
    pub current_length: usize,
    pub max_length: usize,
}

impl ContextWindow {
    pub fn new(max_length: usize) -> Self {
        Self {
            messages: Vec::new(),
            current_length: 0,
            max_length,
        }
    }

    pub fn add_message(&mut self, message: String) {
        let message_len = message.len();
        
        // Simple length-based truncation (would be more sophisticated with tokenization)
        while self.current_length + message_len > self.max_length && !self.messages.is_empty() {
            if let Some(removed) = self.messages.remove(0) {
                self.current_length = self.current_length.saturating_sub(removed.len());
            }
        }
        
        self.messages.push(message.clone());
        self.current_length += message_len;
    }

    pub fn get_context(&self) -> String {
        self.messages.join("\n")
    }
}

/// Main WASM runtime for Kimi-K2 micro-experts
#[wasm_bindgen]
pub struct KimiWasmRuntime {
    #[wasm_bindgen(skip)]
    router: ExpertRouter,
    #[wasm_bindgen(skip)]
    memory_manager: ExpertMemoryManager,
    #[wasm_bindgen(skip)]
    context_window: ContextWindow,
    #[wasm_bindgen(skip)]
    config: RuntimeConfig,
    initialized: bool,
}

#[wasm_bindgen]
impl KimiWasmRuntime {
    /// Initialize the runtime with configuration
    #[wasm_bindgen]
    pub fn initialize(config: &JsValue) -> Result<KimiWasmRuntime> {
        let config: RuntimeConfig = if config.is_undefined() {
            RuntimeConfig::default()
        } else {
            serde_wasm_bindgen::from_value(config.clone())
                .map_err(|e| crate::error::KimiError::configuration(format!("Invalid config: {}", e)))?
        };

        let router = ExpertRouter::new();
        let memory_manager = ExpertMemoryManager::new(config.max_memory_mb);
        let context_window = ContextWindow::new(config.max_context_length);

        Ok(Self {
            router,
            memory_manager,
            context_window,
            config,
            initialized: true,
        })
    }

    /// Register a micro-expert with the runtime
    #[wasm_bindgen]
    pub fn register_expert(&mut self, domain: ExpertDomain, expert: MicroExpert) -> Result<()> {
        // Store expert in memory manager
        self.memory_manager.store_expert(expert.clone())?;
        
        // Register with router
        self.router.register_expert(domain, expert);
        
        Ok(())
    }

    /// Process a request using appropriate experts
    #[wasm_bindgen]
    pub fn process_request(&mut self, request: &str) -> js_sys::Promise {
        let request = request.to_string();
        let router = &self.router;
        let memory_manager = &mut self.memory_manager;
        let context_window = &mut self.context_window;

        // Create future for async processing
        let future = async move {
            // Add request to context
            context_window.add_message(format!("User: {}", request));
            
            // Route request to experts
            let context = RequestContext::new(&request);
            let selections = router.select_experts(&context)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            if selections.is_empty() {
                return Err(JsValue::from_str("No suitable experts found"));
            }

            // Process with selected experts
            let mut responses = Vec::new();
            for selection in selections {
                // Load expert into memory
                if !memory_manager.load_expert(selection.domain)
                    .map_err(|e| JsValue::from_str(&e.to_string()))? {
                    continue; // Skip if expert couldn't be loaded
                }

                // Get expert and process
                if let Ok(Some(expert)) = memory_manager.get_expert_mut(selection.domain) {
                    // Simple tokenization simulation
                    let input_tokens: Vec<f32> = request.chars()
                        .map(|c| c as u32 as f32 / 1000.0)
                        .take(32)
                        .collect();
                    
                    let output = expert.predict(input_tokens)
                        .map_err(|e| JsValue::from_str(&e.to_string()))?;
                    
                    // Convert output back to text (simulation)
                    let response_text = format!(
                        "{} expert response (confidence: {:.2}): Processed {} tokens",
                        selection.domain,
                        selection.confidence,
                        output.len()
                    );
                    
                    responses.push(response_text);
                }
            }

            // Combine responses
            let final_response = if responses.is_empty() {
                "No response generated".to_string()
            } else {
                responses.join("\n\n")
            };

            // Add response to context
            context_window.add_message(format!("Assistant: {}", final_response));

            Ok(JsValue::from_str(&final_response))
        };

        future_to_promise(Box::pin(future))
    }

    /// Get runtime status and statistics
    #[wasm_bindgen]
    pub fn get_status(&self) -> JsValue {
        let memory_stats = self.memory_manager.memory_stats();
        let available_domains = self.router.available_domains();
        
        let status = serde_json::json!({
            "initialized": self.initialized,
            "available_experts": available_domains.len(),
            "memory_stats": memory_stats,
            "context_length": self.context_window.current_length,
            "max_context_length": self.context_window.max_length,
            "config": self.config
        });

        serde_wasm_bindgen::to_value(&status).unwrap_or(JsValue::NULL)
    }

    /// Get memory usage statistics
    #[wasm_bindgen]
    pub fn get_memory_usage(&self) -> JsValue {
        self.memory_manager.get_memory_stats()
    }

    /// Clear context window
    #[wasm_bindgen]
    pub fn clear_context(&mut self) {
        self.context_window = ContextWindow::new(self.config.max_context_length);
    }

    /// Set routing strategy
    #[wasm_bindgen]
    pub fn set_routing_strategy(&mut self, strategy: &str) -> Result<()> {
        self.router.set_strategy(strategy)
    }

    /// Check if runtime is initialized
    #[wasm_bindgen(getter)]
    pub fn initialized(&self) -> bool {
        self.initialized
    }
}

impl KimiWasmRuntime {
    /// Native interface for processing requests
    pub async fn process_request_native(&mut self, request: &str) -> Result<String> {
        // Add request to context
        self.context_window.add_message(format!("User: {}", request));
        
        // Route request to experts
        let context = RequestContext::new(request);
        let selections = self.router.select_experts(&context)?;

        if selections.is_empty() {
            return Err(crate::error::KimiError::routing("No suitable experts found"));
        }

        // Process with selected experts
        let mut responses = Vec::new();
        for selection in selections {
            // Load expert into memory
            if !self.memory_manager.load_expert(selection.domain)? {
                continue; // Skip if expert couldn't be loaded
            }

            // Get expert and process
            if let Some(expert) = self.memory_manager.get_expert_mut(selection.domain)? {
                // Simple tokenization simulation
                let input_tokens: Vec<f32> = request.chars()
                    .map(|c| c as u32 as f32 / 1000.0)
                    .take(32)
                    .collect();
                
                let output = expert.predict(input_tokens)?;
                
                // Convert output back to text (simulation)
                let response_text = format!(
                    "{} expert response (confidence: {:.2}): Processed {} tokens",
                    selection.domain,
                    selection.confidence,
                    output.len()
                );
                
                responses.push(response_text);
            }
        }

        // Combine responses
        let final_response = if responses.is_empty() {
            "No response generated".to_string()
        } else {
            responses.join("\n\n")
        };

        // Add response to context
        self.context_window.add_message(format!("Assistant: {}", final_response));

        Ok(final_response)
    }

    /// Get context window
    pub fn context_window(&self) -> &ContextWindow {
        &self.context_window
    }

    /// Get router
    pub fn router(&self) -> &ExpertRouter {
        &self.router
    }

    /// Get memory manager
    pub fn memory_manager(&self) -> &ExpertMemoryManager {
        &self.memory_manager
    }

    /// Get configuration
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_creation() {
        let config = RuntimeConfig::default();
        let runtime = KimiWasmRuntime {
            router: ExpertRouter::new(),
            memory_manager: ExpertMemoryManager::new(config.max_memory_mb),
            context_window: ContextWindow::new(config.max_context_length),
            config,
            initialized: true,
        };
        
        assert!(runtime.initialized());
    }

    #[test]
    fn test_context_window() {
        let mut context = ContextWindow::new(100);
        
        context.add_message("Hello".to_string());
        assert_eq!(context.current_length, 5);
        
        context.add_message("World".to_string());
        assert_eq!(context.current_length, 11); // "Hello" + "\n" + "World"
    }

    #[test]
    fn test_expert_registration() {
        let config = RuntimeConfig::default();
        let mut runtime = KimiWasmRuntime {
            router: ExpertRouter::new(),
            memory_manager: ExpertMemoryManager::new(config.max_memory_mb),
            context_window: ContextWindow::new(config.max_context_length),
            config,
            initialized: true,
        };

        let expert = MicroExpert::new(ExpertDomain::Coding).unwrap();
        assert!(runtime.register_expert(ExpertDomain::Coding, expert).is_ok());
    }
}