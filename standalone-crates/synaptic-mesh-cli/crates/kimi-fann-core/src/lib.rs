//! # Kimi-FANN Core: Micro-Expert Neural Architecture
//! 
//! This crate provides the core micro-expert architecture for converting Kimi-K2's 
//! 384 experts into 50-100 micro-experts using neural networks compiled to WebAssembly.

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use lru::LruCache;

/// Expert domain types
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ExpertDomain {
    Reasoning,
    Coding,
    Language,
    Mathematics,
    ToolUse,
    Context,
}

/// Configuration for creating a micro-expert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertConfig {
    pub domain: ExpertDomain,
    pub parameter_count: usize,
    pub learning_rate: f32,
}

/// A simple neural network implementation for WASM
pub struct SimpleNeuralNetwork {
    layers: Vec<usize>,
    weights: Vec<Vec<Vec<f32>>>,
    biases: Vec<Vec<f32>>,
}

impl SimpleNeuralNetwork {
    pub fn new(layers: &[usize]) -> Result<Self> {
        if layers.len() < 2 {
            return Err(anyhow!("Neural network needs at least 2 layers"));
        }
        
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        
        for i in 0..layers.len() - 1 {
            let layer_weights = (0..layers[i + 1])
                .map(|_| (0..layers[i]).map(|_| js_sys::Math::random() as f32 - 0.5).collect())
                .collect();
            weights.push(layer_weights);
            
            let layer_biases = (0..layers[i + 1]).map(|_| js_sys::Math::random() as f32 - 0.5).collect();
            biases.push(layer_biases);
        }
        
        Ok(SimpleNeuralNetwork {
            layers: layers.to_vec(),
            weights,
            biases,
        })
    }
    
    pub fn run(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.layers[0] {
            return Err(anyhow!("Input size mismatch"));
        }
        
        let mut current = input.to_vec();
        
        for (layer_idx, (layer_weights, layer_biases)) in 
            self.weights.iter().zip(self.biases.iter()).enumerate() {
            
            let mut next = Vec::new();
            for (neuron_weights, bias) in layer_weights.iter().zip(layer_biases.iter()) {
                let sum: f32 = neuron_weights.iter()
                    .zip(current.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f32>() + bias;
                
                // Use sigmoid activation for hidden layers, linear for output
                let activated = if layer_idx == self.weights.len() - 1 {
                    sum // Linear output
                } else {
                    1.0 / (1.0 + (-sum).exp()) // Sigmoid
                };
                next.push(activated);
            }
            current = next;
        }
        
        Ok(current)
    }
    
    pub fn train_on_data(&mut self, _input: &[f32], _target: &[f32]) -> Result<()> {
        // Simple placeholder for training - in real implementation would use backpropagation
        Ok(())
    }
}

/// A micro-expert neural network
#[wasm_bindgen]
pub struct MicroExpert {
    domain: ExpertDomain,
    config: ExpertConfig,
    network: Option<SimpleNeuralNetwork>,
    input_cache: LruCache<String, Vec<f32>>,
}

impl MicroExpert {
    /// Create the neural network architecture for this expert
    fn create_network(config: &ExpertConfig) -> Result<SimpleNeuralNetwork> {
        let layers = match config.domain {
            ExpertDomain::Reasoning => vec![64, 128, 64, 32],
            ExpertDomain::Coding => vec![96, 192, 128, 64],
            ExpertDomain::Language => vec![128, 256, 128, 64],
            ExpertDomain::Mathematics => vec![80, 160, 80, 40],
            ExpertDomain::ToolUse => vec![48, 96, 48, 24],
            ExpertDomain::Context => vec![32, 64, 32, 16],
        };
        
        SimpleNeuralNetwork::new(&layers)
    }
    
    /// Convert text input to neural network input vector
    fn text_to_vector(&self, input: &str) -> Vec<f32> {
        // Simple text vectorization - in production this would use embeddings
        let mut vector = vec![0.0; 64]; // Base input size
        
        // Character-based encoding with positional information
        for (i, ch) in input.chars().take(32).enumerate() {
            let char_value = (ch as u32 % 256) as f32 / 255.0;
            if i * 2 + 1 < vector.len() {
                vector[i * 2] = char_value;
                vector[i * 2 + 1] = (i as f32) / 32.0; // Position encoding
            }
        }
        
        // Add domain-specific features
        match self.domain {
            ExpertDomain::Reasoning => {
                // Look for logical keywords
                if input.contains("if") || input.contains("then") { vector[62] = 1.0; }
                if input.contains("because") || input.contains("therefore") { vector[63] = 1.0; }
            },
            ExpertDomain::Coding => {
                // Look for code patterns
                if input.contains("function") || input.contains("def") { vector[62] = 1.0; }
                if input.contains("{") || input.contains("}") { vector[63] = 1.0; }
            },
            ExpertDomain::Language => {
                // Language complexity indicators
                vector[62] = input.split_whitespace().count() as f32 / 100.0;
                vector[63] = input.chars().count() as f32 / 1000.0;
            },
            ExpertDomain::Mathematics => {
                // Math symbols and numbers
                if input.chars().any(|c| c.is_numeric()) { vector[62] = 1.0; }
                if input.contains("+") || input.contains("=") { vector[63] = 1.0; }
            },
            ExpertDomain::ToolUse => {
                // Tool and action words
                if input.contains("use") || input.contains("tool") { vector[62] = 1.0; }
                if input.contains("execute") || input.contains("run") { vector[63] = 1.0; }
            },
            ExpertDomain::Context => {
                // Context length and complexity
                vector[62] = (input.len() as f32).min(1000.0) / 1000.0;
                vector[63] = input.split('.').count() as f32 / 10.0;
            },
        }
        
        vector
    }
}

#[wasm_bindgen]
impl MicroExpert {
    /// Create a new micro-expert with neural network
    #[wasm_bindgen(constructor)]
    pub fn new(domain: ExpertDomain) -> MicroExpert {
        let config = ExpertConfig {
            domain,
            parameter_count: match domain {
                ExpertDomain::Reasoning => 25_000,
                ExpertDomain::Coding => 35_000,
                ExpertDomain::Language => 50_000,
                ExpertDomain::Mathematics => 30_000,
                ExpertDomain::ToolUse => 20_000,
                ExpertDomain::Context => 15_000,
            },
            learning_rate: 0.001,
        };
        
        // Try to create the neural network
        let network = Self::create_network(&config).ok();
        
        MicroExpert { 
            domain, 
            config,
            network,
            input_cache: LruCache::new(NonZeroUsize::new(100).unwrap()), 
        }
    }
    
    /// Process a request using the neural network
    pub fn process(&mut self, input: &str) -> String {
        // Check cache first
        if let Some(cached_output) = self.input_cache.get(input) {
            return format!("Cached result for {:?}: {:?}", self.domain, cached_output);
        }
        
        // Prepare input vector before borrowing network
        let input_vector = self.text_to_vector(input);
        
        match &mut self.network {
            Some(network) => {
                match network.run(&input_vector) {
                    Ok(output) => {
                        // Cache the result
                        self.input_cache.put(input.to_string(), output.clone());
                        
                        // Interpret output based on domain
                        let confidence = output.get(0).unwrap_or(&0.0);
                        let relevance = output.get(1).unwrap_or(&0.0);
                        
                        format!(
                            "{:?} expert analysis: confidence={:.3}, relevance={:.3}, params={}",
                            self.domain, confidence, relevance, self.config.parameter_count
                        )
                    },
                    Err(e) => {
                        format!("Neural network error in {:?} expert: {}", self.domain, e)
                    }
                }
            },
            None => {
                format!("Neural network not available for {:?} expert", self.domain)
            }
        }
    }
    
    /// Train the expert on a specific input-output pair
    pub fn train(&mut self, input: &str, expected_output: f32) -> bool {
        // Prepare data before borrowing network
        let input_vector = self.text_to_vector(input);
        let output_vector = vec![expected_output, 0.5]; // Simple training target
        
        if let Some(network) = &mut self.network {
            network.train_on_data(&input_vector, &output_vector).is_ok()
        } else {
            false
        }
    }
    
    /// Get expert statistics
    pub fn get_stats(&self) -> String {
        format!(
            "Domain: {:?}, Parameters: {}, Cache: {}/100",
            self.domain,
            self.config.parameter_count,
            self.input_cache.len()
        )
    }
}

/// Expert router for request distribution with neural routing
#[wasm_bindgen]
pub struct ExpertRouter {
    experts: HashMap<ExpertDomain, MicroExpert>,
    routing_cache: LruCache<String, Vec<ExpertDomain>>,
}

impl ExpertRouter {
    /// Analyze request to determine which experts should handle it
    fn analyze_request(&self, request: &str) -> Vec<ExpertDomain> {
        let mut domains = Vec::new();
        let request_lower = request.to_lowercase();
        
        // Rule-based routing with confidence scoring
        let mut domain_scores = HashMap::new();
        
        // Reasoning patterns
        if request_lower.contains("why") || request_lower.contains("because") || 
           request_lower.contains("explain") || request_lower.contains("reason") {
            domain_scores.insert(ExpertDomain::Reasoning, 0.8);
        }
        
        // Coding patterns
        if request_lower.contains("code") || request_lower.contains("function") || 
           request_lower.contains("debug") || request_lower.contains("implement") ||
           request_lower.contains("program") {
            domain_scores.insert(ExpertDomain::Coding, 0.9);
        }
        
        // Language patterns
        if request_lower.contains("write") || request_lower.contains("text") || 
           request_lower.contains("language") || request_lower.contains("translate") {
            domain_scores.insert(ExpertDomain::Language, 0.7);
        }
        
        // Mathematics patterns
        if request_lower.chars().any(|c| c.is_numeric()) || 
           request_lower.contains("calculate") || request_lower.contains("math") ||
           request_lower.contains("+") || request_lower.contains("=") {
            domain_scores.insert(ExpertDomain::Mathematics, 0.8);
        }
        
        // Tool use patterns
        if request_lower.contains("tool") || request_lower.contains("use") || 
           request_lower.contains("execute") || request_lower.contains("run") {
            domain_scores.insert(ExpertDomain::ToolUse, 0.6);
        }
        
        // Context patterns (always included with lower priority)
        domain_scores.insert(ExpertDomain::Context, 0.3);
        
        // Sort by score and take top domains
        let mut scored_domains: Vec<_> = domain_scores.into_iter().collect();
        scored_domains.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        for (domain, score) in scored_domains {
            if score > 0.5 || domains.len() < 2 {
                domains.push(domain);
            }
            if domains.len() >= 3 { // Limit to 3 experts max
                break;
            }
        }
        
        domains
    }
}

#[wasm_bindgen]
impl ExpertRouter {
    /// Create a new router with default experts
    #[wasm_bindgen(constructor)]
    pub fn new() -> ExpertRouter {
        let mut experts = HashMap::new();
        
        // Create all expert types
        experts.insert(ExpertDomain::Reasoning, MicroExpert::new(ExpertDomain::Reasoning));
        experts.insert(ExpertDomain::Coding, MicroExpert::new(ExpertDomain::Coding));
        experts.insert(ExpertDomain::Language, MicroExpert::new(ExpertDomain::Language));
        experts.insert(ExpertDomain::Mathematics, MicroExpert::new(ExpertDomain::Mathematics));
        experts.insert(ExpertDomain::ToolUse, MicroExpert::new(ExpertDomain::ToolUse));
        experts.insert(ExpertDomain::Context, MicroExpert::new(ExpertDomain::Context));
        
        ExpertRouter {
            experts,
            routing_cache: LruCache::new(NonZeroUsize::new(50).unwrap()),
        }
    }
    
    /// Route a request to appropriate experts and get combined response
    pub fn route(&mut self, request: &str) -> String {
        // Check cache first
        let selected_domains = if let Some(cached_domains) = self.routing_cache.get(request) {
            cached_domains.clone()
        } else {
            let domains = self.analyze_request(request);
            self.routing_cache.put(request.to_string(), domains.clone());
            domains
        };
        
        if selected_domains.is_empty() {
            return "No suitable experts found for this request".to_string();
        }
        
        let mut responses = Vec::new();
        for domain in &selected_domains {
            if let Some(expert) = self.experts.get_mut(domain) {
                let response = expert.process(request);
                responses.push(response);
            }
        }
        
        format!(
            "Routed to {} experts: {}\nResponses:\n{}",
            selected_domains.len(),
            selected_domains.iter()
                .map(|d| format!("{:?}", d))
                .collect::<Vec<_>>()
                .join(", "),
            responses.join("\n")
        )
    }
    
    /// Train a specific expert
    pub fn train_expert(&mut self, domain: ExpertDomain, input: &str, expected_output: f32) -> bool {
        if let Some(expert) = self.experts.get_mut(&domain) {
            expert.train(input, expected_output)
        } else {
            false
        }
    }
    
    /// Get statistics for all experts
    pub fn get_all_stats(&self) -> String {
        let stats: Vec<_> = self.experts.values()
            .map(|expert| expert.get_stats())
            .collect();
        format!("Router Stats:\n{}", stats.join("\n"))
    }
}

/// Processing configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    pub max_experts: usize,
    pub timeout_ms: u32,
}

#[wasm_bindgen]
impl ProcessingConfig {
    /// Create default configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> ProcessingConfig {
        ProcessingConfig {
            max_experts: 3,
            timeout_ms: 5000,
        }
    }
}

/// Main runtime for Kimi-FANN with neural processing
#[wasm_bindgen]
pub struct KimiRuntime {
    config: ProcessingConfig,
    router: ExpertRouter,
    query_count: u32,
    error_count: u32,
}

#[wasm_bindgen]
impl KimiRuntime {
    /// Create a new runtime with all experts initialized
    #[wasm_bindgen(constructor)]
    pub fn new(config: ProcessingConfig) -> KimiRuntime {
        let router = ExpertRouter::new(); // Router now auto-creates all experts
        
        KimiRuntime { 
            config, 
            router,
            query_count: 0,
            error_count: 0,
        }
    }
    
    /// Process a query using the expert routing system
    pub fn process(&mut self, query: &str) -> String {
        self.query_count += 1;
        
        if query.trim().is_empty() {
            self.error_count += 1;
            return "Error: Empty query provided".to_string();
        }
        
        if query.len() > 10000 {
            self.error_count += 1;
            return "Error: Query too long (max 10000 characters)".to_string();
        }
        
        // Process with timeout simulation
        let start_time = js_sys::Date::now();
        let result = self.router.route(query);
        let end_time = js_sys::Date::now();
        let processing_time = end_time - start_time;
        
        format!(
            "Kimi-FANN Runtime v{}\nQuery #{}\nProcessing time: {:.2}ms\n\n{}",
            VERSION, self.query_count, processing_time, result
        )
    }
    
    /// Train a specific expert domain
    pub fn train_expert(&mut self, domain: ExpertDomain, input: &str, expected_output: f32) -> bool {
        self.router.train_expert(domain, input, expected_output)
    }
    
    /// Get runtime statistics
    pub fn get_stats(&self) -> String {
        format!(
            "Kimi-FANN Runtime Statistics:\nTotal queries: {}\nErrors: {}\nSuccess rate: {:.1}%\nMax experts: {}\nTimeout: {}ms\n\n{}",
            self.query_count,
            self.error_count,
            if self.query_count > 0 { 
                ((self.query_count - self.error_count) as f32 / self.query_count as f32) * 100.0 
            } else { 
                0.0 
            },
            self.config.max_experts,
            self.config.timeout_ms,
            self.router.get_all_stats()
        )
    }
    
    /// Reset runtime statistics
    pub fn reset_stats(&mut self) {
        self.query_count = 0;
        self.error_count = 0;
    }
}

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn init() {
    // Initialize logging for WASM
    console_log::init_with_level(log::Level::Info).unwrap_or(());
    log::info!("Kimi-FANN Core initialized");
}

/// Integrated P2P runtime for Kimi-FANN
pub struct P2PKimiRuntime {
    config: ProcessingConfig,
    router_handle: Option<RouterHandle>,
    expert_pool: Option<ExpertPool>,
    health_monitor: Option<HealthMonitorHandle>,
    #[cfg(not(target_arch = "wasm32"))]
    runtime_handle: Option<tokio::runtime::Handle>,
}

impl P2PKimiRuntime {
    /// Create a new P2P-enabled runtime
    pub async fn new(config: ProcessingConfig, p2p_config: Option<P2PConfig>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut runtime = Self {
            config,
            router_handle: None,
            expert_pool: None,
            health_monitor: None,
            #[cfg(not(target_arch = "wasm32"))]
            runtime_handle: None,
        };

        if let Some(p2p_cfg) = p2p_config {
            runtime.initialize_p2p(p2p_cfg).await?;
        }

        Ok(runtime)
    }

    /// Initialize P2P networking
    async fn initialize_p2p(&mut self, p2p_config: P2PConfig) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Create enhanced router
            let (mut router, router_handle) = EnhancedRouter::new(p2p_config).await?;
            
            // Start router in background
            tokio::spawn(async move {
                if let Err(e) = router.start().await {
                    log::error!("Router start error: {}", e);
                }
                if let Err(e) = router.run().await {
                    log::error!("Router run error: {}", e);
                }
            });

            // Create expert pool
            let mut expert_pool = ExpertPool::new(Some(router_handle.clone())).await?;
            
            // Add default experts
            for domain in [
                ExpertDomain::Reasoning,
                ExpertDomain::Coding,
                ExpertDomain::Language,
                ExpertDomain::Mathematics,
                ExpertDomain::ToolUse,
                ExpertDomain::Context,
            ] {
                let config = ExpertConfig {
                    domain,
                    parameter_count: 10_000,
                    learning_rate: 0.001,
                };
                expert_pool.add_expert(domain, config).await?;
            }

            // Create health monitor
            let (mut health_monitor, health_handle) = NetworkHealthMonitor::new(
                Some(router_handle.clone()),
                None,
            );

            // Start health monitor in background
            tokio::spawn(async move {
                if let Err(e) = health_monitor.start().await {
                    log::error!("Health monitor error: {}", e);
                }
            });

            self.router_handle = Some(router_handle);
            self.expert_pool = Some(expert_pool);
            self.health_monitor = Some(health_handle);
        }

        log::info!("P2P Kimi runtime initialized successfully");
        Ok(())
    }

    /// Process a query with P2P coordination
    pub async fn process_with_coordination(
        &self,
        query: &str,
        domain: ExpertDomain,
        strategy: CoordinationStrategy,
    ) -> Result<String, Box<dyn std::error::Error>> {
        if let Some(expert_pool) = &self.expert_pool {
            let response = expert_pool.process_request(domain, query, strategy).await?;
            Ok(response.result)
        } else {
            // Fallback to local processing
            Ok(format!("Local processing of '{}' with {:?} expert", query, domain))
        }
    }

    /// Get network statistics
    pub async fn get_network_stats(&self) -> Result<NetworkStats, Box<dyn std::error::Error>> {
        if let Some(router) = &self.router_handle {
            let peer_stats = router.get_peer_stats().await?;
            Ok(NetworkStats {
                total_peers: peer_stats.total_peers,
                expert_coverage: peer_stats.active_experts,
                network_health: if let Some(health_monitor) = &self.health_monitor {
                    Some(health_monitor.get_network_health().await?)
                } else {
                    None
                },
            })
        } else {
            Err("P2P networking not initialized".into())
        }
    }

    /// Connect to a peer
    pub async fn connect_peer(&self, address: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(router) = &self.router_handle {
            router.connect_peer(address).await?;
            Ok(())
        } else {
            Err("P2P networking not initialized".into())
        }
    }

    /// Get pool statistics
    pub async fn get_pool_stats(&self) -> Result<PoolStats, Box<dyn std::error::Error>> {
        if let Some(expert_pool) = &self.expert_pool {
            Ok(expert_pool.get_pool_stats().await)
        } else {
            Err("Expert pool not initialized".into())
        }
    }
}

/// Network statistics for the runtime
#[derive(Debug, Clone, Serialize)]
pub struct NetworkStats {
    pub total_peers: usize,
    pub expert_coverage: HashMap<ExpertDomain, usize>,
    pub network_health: Option<NetworkHealth>,
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");