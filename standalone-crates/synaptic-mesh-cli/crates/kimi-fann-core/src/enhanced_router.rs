//! Enhanced ML-based expert routing system for production use

use crate::{domains::ExpertDomain, expert::MicroExpert, error::Result};
use ruv_fann::{NeuralNetwork, TrainingData, ActivationFunction};
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::{Instant, Duration};
use std::sync::atomic::{AtomicU64, Ordering};
use wasm_bindgen::prelude::*;

/// Advanced prompt analysis features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptFeatures {
    pub length: usize,
    pub complexity_score: f32,
    pub domain_keywords: HashMap<ExpertDomain, f32>,
    pub sentiment_score: f32,
    pub urgency_level: f32,
    pub technical_depth: f32,
    pub code_presence: f32,
    pub math_symbols: f32,
    pub question_type: String,
}

/// Enhanced routing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnhancedRoutingStrategy {
    /// ML-based routing with neural networks
    MachineLearning,
    /// Hybrid ML + traditional keyword analysis
    Hybrid,
    /// Performance-weighted selection
    PerformanceWeighted,
    /// Adaptive learning from user feedback
    AdaptiveLearning,
}

/// Detailed expert selection with analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedExpertSelection {
    pub domain: ExpertDomain,
    pub confidence: f32,
    pub reasoning: String,
    pub performance_score: f32,
    pub estimated_quality: f32,
    pub load_priority: u8,
    pub expected_tokens: usize,
    pub routing_algorithm: String,
    pub feature_alignment: f32,
    pub latency_estimate_ms: u32,
}

/// Enhanced request context with ML features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedRequestContext {
    pub prompt: String,
    pub max_experts: usize,
    pub preferred_domains: Vec<ExpertDomain>,
    pub performance_threshold: f32,
    pub context_length: usize,
    pub features: Option<PromptFeatures>,
    pub previous_experts: Vec<ExpertDomain>,
    pub session_context: Vec<String>,
    pub user_feedback: Option<f32>,
    pub urgency_override: Option<f32>,
}

impl EnhancedRequestContext {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            max_experts: 3,
            preferred_domains: Vec::new(),
            performance_threshold: 0.7,
            context_length: 0,
            features: None,
            previous_experts: Vec::new(),
            session_context: Vec::new(),
            user_feedback: None,
            urgency_override: None,
        }
    }
    
    pub fn with_session_context(prompt: impl Into<String>, session_context: Vec<String>) -> Self {
        let context_length = session_context.iter().map(|s| s.len()).sum();
        Self {
            prompt: prompt.into(),
            max_experts: 3,
            preferred_domains: Vec::new(),
            performance_threshold: 0.7,
            context_length,
            features: None,
            previous_experts: Vec::new(),
            session_context,
            user_feedback: None,
            urgency_override: None,
        }
    }
}

/// Production-ready ML expert router
#[wasm_bindgen]
pub struct EnhancedExpertRouter {
    #[wasm_bindgen(skip)]
    experts: HashMap<ExpertDomain, MicroExpert>,
    #[wasm_bindgen(skip)]
    strategy: EnhancedRoutingStrategy,
    #[wasm_bindgen(skip)]
    routing_network: Option<NeuralNetwork>,
    #[wasm_bindgen(skip)]
    feature_extractor: AdvancedFeatureExtractor,
    #[wasm_bindgen(skip)]
    performance_tracker: PerformanceTracker,
    #[wasm_bindgen(skip)]
    performance_history: BTreeMap<ExpertDomain, Vec<f32>>,
    #[wasm_bindgen(skip)]
    request_count: AtomicU64,
    #[wasm_bindgen(skip)]
    successful_routes: AtomicU64,
    #[wasm_bindgen(skip)]
    last_optimization: Instant,
    #[wasm_bindgen(skip)]
    routing_cache: HashMap<String, Vec<EnhancedExpertSelection>>,
}

#[wasm_bindgen]
impl EnhancedExpertRouter {
    /// Create a new enhanced expert router
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let mut router = Self {
            experts: HashMap::new(),
            strategy: EnhancedRoutingStrategy::Hybrid,
            routing_network: None,
            feature_extractor: AdvancedFeatureExtractor::new(),
            performance_tracker: PerformanceTracker::new(),
            performance_history: BTreeMap::new(),
            request_count: AtomicU64::new(0),
            successful_routes: AtomicU64::new(0),
            last_optimization: Instant::now(),
            routing_cache: HashMap::new(),
        };
        
        // Initialize ML routing network
        if let Err(e) = router.initialize_ml_routing() {
            log::warn!("Failed to initialize ML routing: {}", e);
        }
        
        router
    }
    
    /// Initialize machine learning routing system
    fn initialize_ml_routing(&mut self) -> Result<()> {
        // Create neural network for routing decisions
        // Input: 64 features (comprehensive prompt analysis)
        // Hidden layers: 32, 16 neurons
        // Output: 6 expert domains + confidence
        let layers = vec![64, 32, 16, 7]; // 6 domains + 1 confidence
        
        let mut network = NeuralNetwork::new(&layers)
            .map_err(|e| crate::error::KimiError::neural_network(format!("ML routing network creation failed: {}", e)))?;
        
        network.set_activation_function_hidden(ActivationFunction::Sigmoid);
        network.set_activation_function_output(ActivationFunction::Sigmoid);
        network.set_learning_rate(0.001);
        network.randomize_weights(-0.1, 0.1);
        
        self.routing_network = Some(network);
        log::info!("Initialized enhanced ML routing system");
        
        Ok(())
    }
    
    /// Register an expert with enhanced analytics
    #[wasm_bindgen]
    pub fn register_expert(&mut self, domain: ExpertDomain, expert: MicroExpert) {
        self.experts.insert(domain, expert);
        self.performance_history.entry(domain).or_insert_with(Vec::new);
        log::info!("Registered {} expert with enhanced routing", domain);
    }
    
    /// Route request using enhanced ML algorithms
    #[wasm_bindgen]
    pub fn route_request_enhanced(&mut self, prompt: &str) -> Result<JsValue> {
        let start_time = Instant::now();
        
        // Check cache first
        if let Some(cached_result) = self.routing_cache.get(prompt) {
            log::debug!("Using cached routing result for prompt");
            return serde_wasm_bindgen::to_value(cached_result)
                .map_err(|e| crate::error::KimiError::routing(format!("Cache serialization failed: {}", e)));
        }
        
        let mut context = EnhancedRequestContext::new(prompt);
        
        // Extract advanced features
        context.features = Some(self.feature_extractor.extract_enhanced_features(prompt)?);
        
        // Perform ML-based routing
        let selections = self.route_with_ml(&context)?;
        
        // Cache result for repeated queries
        self.routing_cache.insert(prompt.to_string(), selections.clone());
        if self.routing_cache.len() > 1000 {
            // Clear oldest entries
            let keys_to_remove: Vec<_> = self.routing_cache.keys().take(100).cloned().collect();
            for key in keys_to_remove {
                self.routing_cache.remove(&key);
            }
        }
        
        // Update metrics
        let routing_time = start_time.elapsed();
        self.performance_tracker.record_routing_time(routing_time);
        self.request_count.fetch_add(1, Ordering::Relaxed);
        
        // Periodic optimization
        if self.should_optimize_routing() {
            if let Err(e) = self.optimize_routing_system() {
                log::warn!("Routing optimization failed: {}", e);
            }
        }
        
        log::debug!("Enhanced routing completed in {:?}: {} experts selected", routing_time, selections.len());
        
        serde_wasm_bindgen::to_value(&selections)
            .map_err(|e| crate::error::KimiError::routing(format!("Serialization failed: {}", e)))
    }
    
    /// Set routing strategy
    #[wasm_bindgen]
    pub fn set_enhanced_strategy(&mut self, strategy: &str) -> Result<()> {
        self.strategy = match strategy {
            "ml" => EnhancedRoutingStrategy::MachineLearning,
            "hybrid" => EnhancedRoutingStrategy::Hybrid,
            "performance" => EnhancedRoutingStrategy::PerformanceWeighted,
            "adaptive" => EnhancedRoutingStrategy::AdaptiveLearning,
            _ => return Err(crate::error::KimiError::configuration(format!(
                "Unknown enhanced routing strategy: {}", strategy
            ))),
        };
        
        log::info!("Set enhanced routing strategy to: {:?}", self.strategy);
        Ok(())
    }
    
    /// Update performance with user feedback
    #[wasm_bindgen]
    pub fn update_performance_with_feedback(&mut self, domain: ExpertDomain, performance_score: f32, user_rating: f32) {
        // Combine performance score with user feedback
        let combined_score = (performance_score * 0.7) + (user_rating * 0.3);
        
        let history = self.performance_history.entry(domain).or_insert_with(Vec::new);
        history.push(combined_score.max(0.0).min(1.0));
        
        // Keep rolling window of performance data
        if history.len() > 500 {
            history.remove(0);
        }
        
        // Update success counter for good performance
        if combined_score > 0.75 {
            self.successful_routes.fetch_add(1, Ordering::Relaxed);
        }
        
        // Train routing network with this feedback
        if let Err(e) = self.train_routing_network(domain, performance_score, user_rating) {
            log::warn!("Failed to train routing network: {}", e);
        }
        
        log::debug!("Updated {} expert performance: {:.3} (with user feedback: {:.3})", 
                   domain, performance_score, user_rating);
    }
    
    /// Get comprehensive routing analytics
    #[wasm_bindgen]
    pub fn get_enhanced_analytics(&self) -> JsValue {
        let total_requests = self.request_count.load(Ordering::Relaxed);
        let successful_routes = self.successful_routes.load(Ordering::Relaxed);
        let success_rate = if total_requests > 0 {
            successful_routes as f32 / total_requests as f32
        } else { 0.0 };
        
        let avg_routing_time = self.performance_tracker.get_average_routing_time();
        
        // Performance statistics per domain
        let mut domain_stats = HashMap::new();
        for (domain, history) in &self.performance_history {
            if !history.is_empty() {
                let avg_performance = history.iter().sum::<f32>() / history.len() as f32;
                let recent_performance = if history.len() >= 10 {
                    history.iter().rev().take(10).sum::<f32>() / 10.0
                } else {
                    avg_performance
                };
                
                domain_stats.insert(domain.to_string(), serde_json::json!({
                    "average_performance": avg_performance,
                    "recent_performance": recent_performance,
                    "total_requests": history.len(),
                    "trend": recent_performance - avg_performance
                }));
            }
        }
        
        let analytics = serde_json::json!({
            "total_requests": total_requests,
            "successful_routes": successful_routes,
            "success_rate": success_rate,
            "average_routing_time_ms": avg_routing_time.as_millis(),
            "routing_strategy": format!("{:?}", self.strategy),
            "ml_routing_enabled": self.routing_network.is_some(),
            "cache_size": self.routing_cache.len(),
            "available_experts": self.experts.len(),
            "domain_statistics": domain_stats,
            "system_health": self.calculate_system_health()
        });
        
        serde_wasm_bindgen::to_value(&analytics).unwrap_or(JsValue::NULL)
    }
}

impl EnhancedExpertRouter {
    /// Route using machine learning algorithms
    fn route_with_ml(&mut self, context: &EnhancedRequestContext) -> Result<Vec<EnhancedExpertSelection>> {
        let features = context.features.as_ref()
            .ok_or_else(|| crate::error::KimiError::routing("Features not extracted"))?;
        
        match &self.strategy {
            EnhancedRoutingStrategy::MachineLearning => self.route_pure_ml(context, features),
            EnhancedRoutingStrategy::Hybrid => self.route_hybrid_ml(context, features),
            EnhancedRoutingStrategy::PerformanceWeighted => self.route_performance_weighted(context, features),
            EnhancedRoutingStrategy::AdaptiveLearning => self.route_adaptive_learning(context, features),
        }
    }
    
    /// Pure ML-based routing
    fn route_pure_ml(&self, context: &EnhancedRequestContext, features: &PromptFeatures) -> Result<Vec<EnhancedExpertSelection>> {
        let mut selections = Vec::new();
        
        if let Some(ref network) = self.routing_network {
            let input = self.features_to_ml_input(features, context);
            let output = network.run(&input)
                .map_err(|e| crate::error::KimiError::routing(format!("ML routing failed: {}", e)))?;
            
            // Extract domain scores and confidence
            let domains = [ExpertDomain::Reasoning, ExpertDomain::Coding, ExpertDomain::Language,
                          ExpertDomain::Mathematics, ExpertDomain::ToolUse, ExpertDomain::Context];
            
            let overall_confidence = output.get(6).unwrap_or(&0.5);
            
            for (i, &domain) in domains.iter().enumerate() {
                if i < output.len() && self.experts.contains_key(&domain) {
                    let domain_score = output[i] as f32;
                    let confidence = domain_score * (*overall_confidence as f32);
                    
                    if confidence > context.performance_threshold {
                        selections.push(self.create_enhanced_selection(
                            domain, confidence, features, "pure-ml", context
                        )?);
                    }
                }
            }
        }
        
        selections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        selections.truncate(context.max_experts);
        
        Ok(selections)
    }
    
    /// Hybrid ML + keyword routing
    fn route_hybrid_ml(&self, context: &EnhancedRequestContext, features: &PromptFeatures) -> Result<Vec<EnhancedExpertSelection>> {
        let mut selections = Vec::new();
        
        // Get ML scores
        let ml_scores = if let Some(ref network) = self.routing_network {
            let input = self.features_to_ml_input(features, context);
            match network.run(&input) {
                Ok(output) => {
                    let domains = [ExpertDomain::Reasoning, ExpertDomain::Coding, ExpertDomain::Language,
                                  ExpertDomain::Mathematics, ExpertDomain::ToolUse, ExpertDomain::Context];
                    let mut scores = HashMap::new();
                    for (i, &domain) in domains.iter().enumerate() {
                        if i < output.len() {
                            scores.insert(domain, output[i] as f32);
                        }
                    }
                    scores
                },
                Err(_) => HashMap::new(),
            }
        } else {
            HashMap::new()
        };
        
        // Get keyword scores
        let keyword_scores = self.analyze_keywords_advanced(&context.prompt);
        
        // Combine scores (60% ML, 40% keywords)
        let mut hybrid_scores = HashMap::new();
        for domain in [ExpertDomain::Reasoning, ExpertDomain::Coding, ExpertDomain::Language,
                      ExpertDomain::Mathematics, ExpertDomain::ToolUse, ExpertDomain::Context] {
            let ml_score = ml_scores.get(&domain).unwrap_or(&0.0);
            let keyword_score = keyword_scores.get(&domain).unwrap_or(&0.0);
            let hybrid_score = (ml_score * 0.6) + (keyword_score * 0.4);
            hybrid_scores.insert(domain, hybrid_score);
        }
        
        // Apply performance adjustments
        let adjusted_scores = self.apply_performance_adjustments(hybrid_scores);
        
        // Create selections
        for (domain, score) in adjusted_scores {
            if score > context.performance_threshold && self.experts.contains_key(&domain) {
                selections.push(self.create_enhanced_selection(
                    domain, score, features, "hybrid-ml", context
                )?);
            }
        }
        
        selections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        selections.truncate(context.max_experts);
        
        Ok(selections)
    }
    
    /// Performance-weighted routing
    fn route_performance_weighted(&self, context: &EnhancedRequestContext, features: &PromptFeatures) -> Result<Vec<EnhancedExpertSelection>> {
        let keyword_scores = self.analyze_keywords_advanced(&context.prompt);
        let mut selections = Vec::new();
        
        for (domain, base_score) in keyword_scores {
            if self.experts.contains_key(&domain) {
                let performance_weight = self.get_domain_performance(domain);
                let weighted_score = base_score * (0.3 + performance_weight * 0.7);
                
                if weighted_score > context.performance_threshold {
                    selections.push(self.create_enhanced_selection(
                        domain, weighted_score, features, "performance-weighted", context
                    )?);
                }
            }
        }
        
        selections.sort_by(|a, b| b.performance_score.partial_cmp(&a.performance_score).unwrap_or(std::cmp::Ordering::Equal));
        selections.truncate(context.max_experts);
        
        Ok(selections)
    }
    
    /// Adaptive learning routing
    fn route_adaptive_learning(&self, context: &EnhancedRequestContext, features: &PromptFeatures) -> Result<Vec<EnhancedExpertSelection>> {
        // Start with hybrid approach and adapt based on recent performance
        let mut selections = self.route_hybrid_ml(context, features)?;
        
        // Adjust based on recent user feedback and performance trends
        for selection in &mut selections {
            let recent_performance = self.get_recent_performance(selection.domain);
            let feedback_adjustment = if let Some(feedback) = context.user_feedback {
                (feedback - 0.5) * 0.2 // Max ±0.2 adjustment based on feedback
            } else { 0.0 };
            
            selection.confidence = (selection.confidence + feedback_adjustment + 
                                  (recent_performance - 0.5) * 0.15).max(0.0).min(1.0);
            selection.routing_algorithm = "adaptive-learning".to_string();
        }
        
        // Re-sort after adjustments
        selections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(selections)
    }
    
    /// Create enhanced expert selection with detailed analytics
    fn create_enhanced_selection(&self, domain: ExpertDomain, confidence: f32, 
                               features: &PromptFeatures, algorithm: &str,
                               context: &EnhancedRequestContext) -> Result<EnhancedExpertSelection> {
        let performance_score = self.get_domain_performance(domain);
        let estimated_quality = self.estimate_response_quality(domain, features);
        let load_priority = self.calculate_load_priority(domain, confidence, performance_score);
        let expected_tokens = self.estimate_response_length(domain, features);
        let feature_alignment = self.calculate_feature_alignment(domain, features);
        let latency_estimate = self.estimate_latency(domain, expected_tokens);
        
        let reasoning = format!(
            "{} routing: {:.2} confidence, {:.2} performance, {:.2} quality, {:.2} alignment",
            algorithm, confidence, performance_score, estimated_quality, feature_alignment
        );
        
        Ok(EnhancedExpertSelection {
            domain,
            confidence,
            reasoning,
            performance_score,
            estimated_quality,
            load_priority,
            expected_tokens,
            routing_algorithm: algorithm.to_string(),
            feature_alignment,
            latency_estimate_ms: latency_estimate,
        })
    }
    
    /// Convert features to ML input vector
    fn features_to_ml_input(&self, features: &PromptFeatures, context: &EnhancedRequestContext) -> Vec<f64> {
        let mut input = Vec::with_capacity(64);
        
        // Basic prompt features (16 dimensions)
        input.push((features.length as f64).ln().min(10.0) / 10.0);
        input.push(features.complexity_score as f64);
        input.push(features.sentiment_score as f64);
        input.push(features.urgency_level as f64);
        input.push(features.technical_depth as f64);
        input.push(features.code_presence as f64);
        input.push(features.math_symbols as f64);
        input.push(context.context_length as f64 / 10000.0); // Normalize context length
        
        // Domain keyword scores (6 dimensions)
        for domain in [ExpertDomain::Reasoning, ExpertDomain::Coding, ExpertDomain::Language,
                      ExpertDomain::Mathematics, ExpertDomain::ToolUse, ExpertDomain::Context] {
            input.push(features.domain_keywords.get(&domain).unwrap_or(&0.0) as f64);
        }
        
        // Question type one-hot encoding (8 dimensions)
        let question_types = ["factual", "procedural", "conceptual", "analytical", "creative", "comparative", "evaluative", "synthesis"];
        for q_type in question_types {
            input.push(if features.question_type == q_type { 1.0 } else { 0.0 });
        }
        
        // Context features (10 dimensions)
        input.push(context.max_experts as f64 / 10.0);
        input.push(context.performance_threshold as f64);
        input.push(context.previous_experts.len() as f64 / 6.0);
        input.push(context.user_feedback.unwrap_or(0.5) as f64);
        input.push(context.urgency_override.unwrap_or(features.urgency_level) as f64);
        
        // Performance history features (6 dimensions)
        for domain in [ExpertDomain::Reasoning, ExpertDomain::Coding, ExpertDomain::Language,
                      ExpertDomain::Mathematics, ExpertDomain::ToolUse, ExpertDomain::Context] {
            input.push(self.get_domain_performance(domain) as f64);
        }
        
        // System state features (8 dimensions)
        input.push((self.request_count.load(Ordering::Relaxed) as f64).ln() / 10.0);
        input.push(self.get_system_load() as f64);
        input.push(self.routing_cache.len() as f64 / 1000.0);
        
        // Pad to exactly 64 dimensions
        while input.len() < 64 {
            input.push(0.0);
        }
        input.truncate(64);
        
        input
    }
    
    /// Advanced keyword analysis
    fn analyze_keywords_advanced(&self, prompt: &str) -> HashMap<ExpertDomain, f32> {
        // This would use the enhanced keyword analysis from the main router
        // For now, return basic analysis
        let mut scores = HashMap::new();
        let prompt_lower = prompt.to_lowercase();
        
        // Enhanced patterns with contextual understanding
        let patterns = [
            (ExpertDomain::Coding, vec!["code", "program", "function", "algorithm", "debug", "implement"]),
            (ExpertDomain::Mathematics, vec!["calculate", "equation", "math", "formula", "solve", "number"]),
            (ExpertDomain::Reasoning, vec!["analyze", "logic", "reason", "deduce", "conclude", "problem"]),
            (ExpertDomain::Language, vec!["write", "explain", "summarize", "translate", "grammar", "text"]),
            (ExpertDomain::ToolUse, vec!["api", "call", "execute", "tool", "service", "command"]),
            (ExpertDomain::Context, vec!["context", "reference", "previous", "relate", "history", "memory"]),
        ];
        
        for (domain, keywords) in patterns {
            let score = keywords.iter()
                .map(|&word| if prompt_lower.contains(word) { 0.2 } else { 0.0 })
                .sum::<f32>()
                .min(1.0);
            scores.insert(domain, score);
        }
        
        scores
    }
    
    /// Apply performance-based score adjustments
    fn apply_performance_adjustments(&self, scores: HashMap<ExpertDomain, f32>) -> HashMap<ExpertDomain, f32> {
        let mut adjusted = HashMap::new();
        
        for (domain, score) in scores {
            let performance_multiplier = 0.5 + (self.get_domain_performance(domain) * 0.5);
            let adjusted_score = score * performance_multiplier;
            adjusted.insert(domain, adjusted_score);
        }
        
        adjusted
    }
    
    /// Get domain performance score
    fn get_domain_performance(&self, domain: ExpertDomain) -> f32 {
        if let Some(history) = self.performance_history.get(&domain) {
            if history.is_empty() {
                0.5
            } else {
                // Exponentially weighted average favoring recent performance
                let mut weighted_sum = 0.0;
                let mut weights_sum = 0.0;
                let decay_factor = 0.9;
                
                for (i, &score) in history.iter().rev().enumerate() {
                    let weight = decay_factor.powi(i as i32);
                    weighted_sum += score * weight;
                    weights_sum += weight;
                }
                
                weighted_sum / weights_sum
            }
        } else {
            0.5
        }
    }
    
    /// Get recent performance for adaptive learning
    fn get_recent_performance(&self, domain: ExpertDomain) -> f32 {
        if let Some(history) = self.performance_history.get(&domain) {
            if history.len() >= 5 {
                history.iter().rev().take(5).sum::<f32>() / 5.0
            } else if !history.is_empty() {
                history.iter().sum::<f32>() / history.len() as f32
            } else {
                0.5
            }
        } else {
            0.5
        }
    }
    
    /// Estimate response quality
    fn estimate_response_quality(&self, domain: ExpertDomain, features: &PromptFeatures) -> f32 {
        let base_quality = self.get_domain_performance(domain);
        
        // Domain-feature alignment bonus
        let alignment = self.calculate_feature_alignment(domain, features);
        
        (base_quality * 0.7 + alignment * 0.3).min(1.0)
    }
    
    /// Calculate feature alignment with domain
    fn calculate_feature_alignment(&self, domain: ExpertDomain, features: &PromptFeatures) -> f32 {
        match domain {
            ExpertDomain::Coding => {
                (features.code_presence * 0.4 + features.technical_depth * 0.3 + 
                 features.domain_keywords.get(&domain).unwrap_or(&0.0) * 0.3)
            },
            ExpertDomain::Mathematics => {
                (features.math_symbols * 0.4 + features.technical_depth * 0.3 + 
                 features.domain_keywords.get(&domain).unwrap_or(&0.0) * 0.3)
            },
            ExpertDomain::Language => {
                (features.domain_keywords.get(&domain).unwrap_or(&0.0) * 0.5 + 
                 (1.0 - features.technical_depth) * 0.3 + 
                 (if features.question_type == "creative" { 0.2 } else { 0.0 }))
            },
            ExpertDomain::Reasoning => {
                (features.complexity_score * 0.4 + 
                 features.domain_keywords.get(&domain).unwrap_or(&0.0) * 0.3 +
                 (if features.question_type == "analytical" { 0.3 } else { 0.0 }))
            },
            _ => features.domain_keywords.get(&domain).unwrap_or(&0.0),
        }
    }
    
    /// Calculate load priority
    fn calculate_load_priority(&self, domain: ExpertDomain, confidence: f32, performance: f32) -> u8 {
        let combined_score = (confidence * 0.6 + performance * 0.4);
        (combined_score * 10.0).round().max(1.0).min(10.0) as u8
    }
    
    /// Estimate response length
    fn estimate_response_length(&self, domain: ExpertDomain, features: &PromptFeatures) -> usize {
        let base_length = match domain {
            ExpertDomain::Coding => 250,
            ExpertDomain::Mathematics => 180,
            ExpertDomain::Language => 320,
            ExpertDomain::Reasoning => 280,
            ExpertDomain::ToolUse => 120,
            ExpertDomain::Context => 220,
        };
        
        let complexity_multiplier = 1.0 + (features.complexity_score * 0.8);
        let urgency_multiplier = if features.urgency_level > 0.7 { 0.8 } else { 1.0 };
        
        ((base_length as f32) * complexity_multiplier * urgency_multiplier) as usize
    }
    
    /// Estimate latency for response generation
    fn estimate_latency(&self, domain: ExpertDomain, expected_tokens: usize) -> u32 {
        let base_latency = match domain {
            ExpertDomain::Coding => 150,
            ExpertDomain::Mathematics => 120,
            ExpertDomain::Language => 100,
            ExpertDomain::Reasoning => 180,
            ExpertDomain::ToolUse => 80,
            ExpertDomain::Context => 140,
        };
        
        let token_overhead = (expected_tokens as f32 * 0.5) as u32;
        let performance_adjustment = (self.get_domain_performance(domain) - 0.5) * -50.0; // Better performance = lower latency
        
        (base_latency + token_overhead + performance_adjustment as u32).max(50)
    }
    
    /// Train routing network with feedback
    fn train_routing_network(&mut self, domain: ExpertDomain, performance: f32, user_rating: f32) -> Result<()> {
        // This would implement online learning for the routing network
        // For now, just log the training attempt
        log::debug!("Training routing network: {} domain, performance: {:.3}, rating: {:.3}", 
                   domain, performance, user_rating);
        Ok(())
    }
    
    /// Check if routing optimization should be performed
    fn should_optimize_routing(&self) -> bool {
        let elapsed = self.last_optimization.elapsed();
        let request_count = self.request_count.load(Ordering::Relaxed);
        
        // Optimize every 5 minutes or every 1000 requests
        elapsed > Duration::from_secs(300) || request_count % 1000 == 0
    }
    
    /// Optimize routing system
    fn optimize_routing_system(&mut self) -> Result<()> {
        log::info!("Optimizing routing system...");
        
        // Clear old cache entries
        if self.routing_cache.len() > 500 {
            let keys_to_remove: Vec<_> = self.routing_cache.keys().take(200).cloned().collect();
            for key in keys_to_remove {
                self.routing_cache.remove(&key);
            }
        }
        
        // Update last optimization time
        self.last_optimization = Instant::now();
        
        // Additional optimization logic would go here
        // - Retrain routing network
        // - Adjust strategy based on performance
        // - Cleanup performance history
        
        Ok(())
    }
    
    /// Calculate overall system health
    fn calculate_system_health(&self) -> f32 {
        let total_requests = self.request_count.load(Ordering::Relaxed);
        let successful_routes = self.successful_routes.load(Ordering::Relaxed);
        
        if total_requests == 0 {
            return 1.0; // Perfect health with no data
        }
        
        let success_rate = successful_routes as f32 / total_requests as f32;
        let avg_performance: f32 = self.performance_history.values()
            .filter_map(|history| {
                if history.is_empty() { None } else {
                    Some(history.iter().sum::<f32>() / history.len() as f32)
                }
            })
            .sum::<f32>() / self.performance_history.len().max(1) as f32;
        
        let routing_efficiency = if self.performance_tracker.get_average_routing_time().as_millis() < 50 {
            1.0
        } else {
            0.8
        };
        
        (success_rate * 0.4 + avg_performance * 0.4 + routing_efficiency * 0.2).min(1.0)
    }
    
    /// Get current system load estimate
    fn get_system_load(&self) -> f32 {
        // Simple heuristic based on cache size and request frequency
        let cache_load = self.routing_cache.len() as f32 / 1000.0;
        let request_frequency = self.request_count.load(Ordering::Relaxed) as f32 / 
                               self.last_optimization.elapsed().as_secs().max(1) as f32;
        
        (cache_load * 0.3 + (request_frequency / 10.0) * 0.7).min(1.0)
    }
}

/// Advanced feature extraction system
#[derive(Debug)]
struct AdvancedFeatureExtractor {
    code_indicators: Vec<&'static str>,
    math_symbols: Vec<char>,
    technical_terms: Vec<&'static str>,
    urgency_markers: Vec<&'static str>,
    question_classifiers: HashMap<&'static str, Vec<&'static str>>,
}

impl AdvancedFeatureExtractor {
    fn new() -> Self {
        let mut question_classifiers = HashMap::new();
        question_classifiers.insert("factual", vec!["what is", "define", "explain", "who", "when", "where"]);
        question_classifiers.insert("procedural", vec!["how to", "steps", "process", "method", "procedure"]);
        question_classifiers.insert("conceptual", vec!["why", "concept", "principle", "theory", "understand"]);
        question_classifiers.insert("analytical", vec!["analyze", "compare", "evaluate", "assess", "examine"]);
        question_classifiers.insert("creative", vec!["create", "design", "brainstorm", "generate", "imagine"]);
        question_classifiers.insert("comparative", vec!["difference", "similar", "contrast", "versus", "compare"]);
        question_classifiers.insert("evaluative", vec!["judge", "critique", "rate", "assess", "opinion"]);
        question_classifiers.insert("synthesis", vec!["combine", "integrate", "merge", "synthesize", "unify"]);
        
        Self {
            code_indicators: vec!["```", "function", "class", "import", "def ", "var ", "let ", "const ", 
                                "public", "private", "static", "return", "if (", "for (", "while ("],
            math_symbols: vec!['∑', '∏', '∫', '∂', '√', '±', '≠', '≤', '≥', '∞', 'π', 'θ', 'α', 'β', 'γ'],
            technical_terms: vec!["algorithm", "implementation", "architecture", "optimization", 
                                "performance", "scalability", "framework", "library", "database",
                                "api", "interface", "protocol", "encryption", "authentication"],
            urgency_markers: vec!["urgent", "asap", "quickly", "immediately", "now", "emergency", 
                                "critical", "deadline", "rush", "priority"],
            question_classifiers,
        }
    }
    
    fn extract_enhanced_features(&self, prompt: &str) -> Result<PromptFeatures> {
        let prompt_lower = prompt.to_lowercase();
        
        Ok(PromptFeatures {
            length: prompt.len(),
            complexity_score: self.calculate_enhanced_complexity(&prompt_lower),
            domain_keywords: self.extract_domain_scores(&prompt_lower),
            sentiment_score: self.analyze_enhanced_sentiment(&prompt_lower),
            urgency_level: self.detect_enhanced_urgency(&prompt_lower),
            technical_depth: self.assess_enhanced_technical_depth(&prompt_lower),
            code_presence: self.detect_enhanced_code_presence(&prompt_lower),
            math_symbols: self.count_enhanced_math_symbols(prompt),
            question_type: self.classify_enhanced_question_type(&prompt_lower),
        })
    }
    
    fn calculate_enhanced_complexity(&self, prompt: &str) -> f32 {
        let words: Vec<&str> = prompt.split_whitespace().collect();
        let word_count = words.len() as f32;
        
        // Lexical complexity
        let avg_word_length = words.iter().map(|w| w.len()).sum::<usize>() as f32 / word_count.max(1.0);
        let unique_words = words.iter().collect::<std::collections::HashSet<_>>().len() as f32;
        let lexical_diversity = unique_words / word_count.max(1.0);
        
        // Syntactic complexity
        let sentence_count = prompt.split(|c| c == '.' || c == '!' || c == '?').count() as f32;
        let avg_sentence_length = word_count / sentence_count.max(1.0);
        
        // Technical complexity
        let technical_term_count = self.technical_terms.iter()
            .filter(|&&term| prompt.contains(term))
            .count() as f32;
        let technical_density = technical_term_count / word_count.max(1.0) * 100.0;
        
        // Combine complexity measures
        let lexical_complexity = (avg_word_length / 10.0 + lexical_diversity) / 2.0;
        let syntactic_complexity = (avg_sentence_length / 25.0).min(1.0);
        let semantic_complexity = (technical_density / 5.0).min(1.0);
        
        ((lexical_complexity + syntactic_complexity + semantic_complexity) / 3.0).min(1.0)
    }
    
    fn extract_domain_scores(&self, prompt: &str) -> HashMap<ExpertDomain, f32> {
        let mut scores = HashMap::new();
        
        // Enhanced domain detection with weighted terms
        let domain_patterns = [
            (ExpertDomain::Coding, vec![
                ("code", 0.3), ("program", 0.3), ("function", 0.25), ("algorithm", 0.3),
                ("debug", 0.25), ("implement", 0.2), ("syntax", 0.2), ("variable", 0.15),
                ("class", 0.2), ("method", 0.2), ("api", 0.15), ("framework", 0.15)
            ]),
            (ExpertDomain::Mathematics, vec![
                ("math", 0.3), ("calculate", 0.3), ("equation", 0.3), ("formula", 0.25),
                ("solve", 0.2), ("number", 0.15), ("statistics", 0.25), ("probability", 0.25),
                ("integral", 0.3), ("derivative", 0.3), ("matrix", 0.2), ("vector", 0.2)
            ]),
            (ExpertDomain::Reasoning, vec![
                ("analyze", 0.3), ("logic", 0.3), ("reason", 0.3), ("deduce", 0.25),
                ("conclude", 0.25), ("problem", 0.2), ("think", 0.2), ("inference", 0.25),
                ("argument", 0.2), ("evidence", 0.2), ("hypothesis", 0.25), ("premise", 0.25)
            ]),
            (ExpertDomain::Language, vec![
                ("write", 0.2), ("explain", 0.2), ("summarize", 0.25), ("translate", 0.3),
                ("grammar", 0.25), ("text", 0.15), ("language", 0.2), ("style", 0.2),
                ("narrative", 0.2), ("discourse", 0.25), ("rhetoric", 0.25), ("linguistics", 0.3)
            ]),
            (ExpertDomain::ToolUse, vec![
                ("api", 0.3), ("tool", 0.25), ("execute", 0.25), ("command", 0.25),
                ("service", 0.2), ("endpoint", 0.3), ("automation", 0.25), ("workflow", 0.2),
                ("integration", 0.25), ("webhook", 0.3), ("trigger", 0.2), ("invoke", 0.25)
            ]),
            (ExpertDomain::Context, vec![
                ("context", 0.3), ("reference", 0.25), ("previous", 0.2), ("relate", 0.2),
                ("history", 0.2), ("memory", 0.25), ("background", 0.15), ("connection", 0.25),
                ("relationship", 0.2), ("dependency", 0.25), ("scenario", 0.2), ("environment", 0.15)
            ]),
        ];
        
        for (domain, patterns) in domain_patterns {
            let mut score = 0.0;
            for (term, weight) in patterns {
                if prompt.contains(term) {
                    score += weight;
                }
            }
            
            // Apply pattern bonuses
            score += self.calculate_domain_pattern_bonus(prompt, domain);
            
            scores.insert(domain, score.min(1.0));
        }
        
        scores
    }
    
    fn calculate_domain_pattern_bonus(&self, prompt: &str, domain: ExpertDomain) -> f32 {
        match domain {
            ExpertDomain::Coding => {
                let mut bonus = 0.0;
                if prompt.contains("```") || prompt.contains("`") { bonus += 0.3; }
                if prompt.contains("error") || prompt.contains("exception") { bonus += 0.2; }
                if prompt.contains(".py") || prompt.contains(".js") || prompt.contains(".rs") { bonus += 0.2; }
                bonus
            },
            ExpertDomain::Mathematics => {
                let mut bonus = 0.0;
                if self.math_symbols.iter().any(|&c| prompt.contains(c)) { bonus += 0.3; }
                if prompt.matches(char::is_numeric).count() > 3 { bonus += 0.1; }
                if prompt.contains("theorem") || prompt.contains("proof") { bonus += 0.2; }
                bonus
            },
            ExpertDomain::Reasoning => {
                let mut bonus = 0.0;
                if prompt.contains("because") && prompt.contains("therefore") { bonus += 0.2; }
                if prompt.contains("if") && prompt.contains("then") { bonus += 0.2; }
                if prompt.contains("hypothesis") && prompt.contains("evidence") { bonus += 0.2; }
                bonus
            },
            _ => 0.0,
        }
    }
    
    fn analyze_enhanced_sentiment(&self, prompt: &str) -> f32 {
        let positive_indicators = ["good", "great", "excellent", "amazing", "wonderful", "fantastic",
                                  "help", "please", "thank", "appreciate", "love", "perfect"];
        let negative_indicators = ["bad", "terrible", "awful", "horrible", "broken", "error",
                                  "fail", "wrong", "hate", "frustrated", "confused", "stuck"];
        let neutral_indicators = ["explain", "describe", "show", "tell", "understand", "learn"];
        
        let positive_count = positive_indicators.iter().filter(|&&word| prompt.contains(word)).count() as f32;
        let negative_count = negative_indicators.iter().filter(|&&word| prompt.contains(word)).count() as f32;
        let neutral_count = neutral_indicators.iter().filter(|&&word| prompt.contains(word)).count() as f32;
        
        let total = positive_count + negative_count + neutral_count;
        if total == 0.0 {
            0.5 // Neutral default
        } else {
            (positive_count + neutral_count * 0.5) / total
        }
    }
    
    fn detect_enhanced_urgency(&self, prompt: &str) -> f32 {
        let mut urgency_score = 0.0;
        
        // Direct urgency markers
        let urgent_count = self.urgency_markers.iter().filter(|&&word| prompt.contains(word)).count();
        urgency_score += urgent_count as f32 * 0.3;
        
        // Punctuation indicators
        let exclamation_count = prompt.matches('!').count();
        urgency_score += exclamation_count as f32 * 0.1;
        
        // Capitalization ratio
        let caps_ratio = prompt.chars().filter(|c| c.is_uppercase()).count() as f32 / 
                        prompt.chars().filter(|c| c.is_alphabetic()).count().max(1) as f32;
        urgency_score += caps_ratio * 0.2;
        
        // Time-related terms
        let time_indicators = ["today", "now", "soon", "quick", "fast", "immediate"];
        let time_count = time_indicators.iter().filter(|&&word| prompt.contains(word)).count();
        urgency_score += time_count as f32 * 0.15;
        
        urgency_score.min(1.0)
    }
    
    fn assess_enhanced_technical_depth(&self, prompt: &str) -> f32 {
        let technical_count = self.technical_terms.iter().filter(|&&term| prompt.contains(term)).count();
        let jargon_density = technical_count as f32 / prompt.split_whitespace().count().max(1) as f32;
        
        let complexity_indicators = ["complex", "sophisticated", "advanced", "detailed", "comprehensive"];
        let complexity_count = complexity_indicators.iter().filter(|&&word| prompt.contains(word)).count();
        
        ((technical_count as f32 * 0.2) + (jargon_density * 50.0) + (complexity_count as f32 * 0.1)).min(1.0)
    }
    
    fn detect_enhanced_code_presence(&self, prompt: &str) -> f32 {
        let code_count = self.code_indicators.iter().filter(|&&indicator| prompt.contains(indicator)).count();
        let code_block_bonus = if prompt.contains("```") { 0.4 } else { 0.0 };
        let inline_code_bonus = if prompt.contains("`") && !prompt.contains("```") { 0.2 } else { 0.0 };
        
        ((code_count as f32 * 0.15) + code_block_bonus + inline_code_bonus).min(1.0)
    }
    
    fn count_enhanced_math_symbols(&self, prompt: &str) -> f32 {
        let symbol_count = self.math_symbols.iter().filter(|&&symbol| prompt.contains(symbol)).count();
        let number_density = prompt.chars().filter(|c| c.is_numeric()).count() as f32 / 
                           prompt.chars().count().max(1) as f32;
        let equation_indicators = ["=", "+", "-", "*", "/", "^"];
        let equation_count = equation_indicators.iter().filter(|&&op| prompt.contains(op)).count();
        
        ((symbol_count as f32 * 0.3) + (number_density * 2.0) + (equation_count as f32 * 0.1)).min(1.0)
    }
    
    fn classify_enhanced_question_type(&self, prompt: &str) -> String {
        let mut scores: HashMap<&str, f32> = HashMap::new();
        
        for (q_type, patterns) in &self.question_classifiers {
            let mut score = 0.0;
            for &pattern in patterns {
                if prompt.contains(pattern) {
                    score += 1.0;
                }
            }
            
            // Add contextual bonuses
            score += match *q_type {
                "factual" => if prompt.starts_with("what") || prompt.starts_with("who") { 0.5 } else { 0.0 },
                "procedural" => if prompt.contains("step") || prompt.contains("how") { 0.5 } else { 0.0 },
                "analytical" => if prompt.contains("why") || prompt.contains("reason") { 0.5 } else { 0.0 },
                "creative" => if prompt.contains("idea") || prompt.contains("invent") { 0.5 } else { 0.0 },
                _ => 0.0,
            };
            
            scores.insert(q_type, score);
        }
        
        scores.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(q_type, _)| q_type.to_string())
            .unwrap_or_else(|| "factual".to_string())
    }
}

/// Performance tracking system
#[derive(Debug)]
struct PerformanceTracker {
    routing_times: Vec<Duration>,
    total_requests: u64,
    cache_hits: u64,
    ml_predictions: u64,
    last_cleanup: Instant,
}

impl PerformanceTracker {
    fn new() -> Self {
        Self {
            routing_times: Vec::new(),
            total_requests: 0,
            cache_hits: 0,
            ml_predictions: 0,
            last_cleanup: Instant::now(),
        }
    }
    
    fn record_routing_time(&mut self, duration: Duration) {
        self.routing_times.push(duration);
        self.total_requests += 1;
        
        // Periodic cleanup
        if self.routing_times.len() > 2000 {
            self.routing_times.drain(0..1000);
        }
        
        // Performance analytics cleanup every hour
        if self.last_cleanup.elapsed() > Duration::from_secs(3600) {
            self.cleanup_old_data();
            self.last_cleanup = Instant::now();
        }
    }
    
    fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }
    
    fn record_ml_prediction(&mut self) {
        self.ml_predictions += 1;
    }
    
    fn get_average_routing_time(&self) -> Duration {
        if self.routing_times.is_empty() {
            Duration::from_millis(0)
        } else {
            let total_ms: u128 = self.routing_times.iter().map(|d| d.as_millis()).sum();
            Duration::from_millis((total_ms / self.routing_times.len() as u128) as u64)
        }
    }
    
    fn get_cache_hit_rate(&self) -> f32 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.cache_hits as f32 / self.total_requests as f32
        }
    }
    
    fn get_ml_usage_rate(&self) -> f32 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.ml_predictions as f32 / self.total_requests as f32
        }
    }
    
    fn cleanup_old_data(&mut self) {
        // Keep only recent routing times for memory efficiency
        if self.routing_times.len() > 1000 {
            self.routing_times.drain(0..500);
        }
    }
}