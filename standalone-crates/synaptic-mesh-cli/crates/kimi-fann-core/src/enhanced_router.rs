//! Enhanced routing system with market integration
//!
//! This module provides an enhanced routing system that integrates with the
//! Synaptic Market to enable dynamic expert allocation based on market conditions,
//! pricing, and reputation.

use crate::{ExpertDomain, MicroExpert, ProcessingConfig, NetworkStats};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// Enhanced router with market integration
#[wasm_bindgen]
pub struct EnhancedRouter {
    experts: HashMap<ExpertDomain, Vec<MicroExpert>>,
    market_integration: Option<MarketIntegration>,
    #[allow(dead_code)]
    config: ProcessingConfig,
    stats: NetworkStats,
}

/// Market integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketIntegration {
    /// Enable market-based routing
    pub market_routing_enabled: bool,
    /// Budget for compute purchases (in tokens)
    pub compute_budget: u64,
    /// Minimum reputation score required
    pub min_reputation: f64,
    /// Maximum price per compute unit
    pub max_price_per_unit: u64,
    /// Preferred expert domains
    pub preferred_domains: Vec<ExpertDomain>,
}

/// Expert performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertMetrics {
    /// Expert domain
    pub domain: ExpertDomain,
    /// Total queries processed
    pub queries_processed: u64,
    /// Average response time (ms)
    pub avg_response_time: f64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Quality score (0.0 to 1.0)
    pub quality_score: f64,
    /// Current load (0.0 to 1.0)
    pub current_load: f64,
}

/// Route recommendation from the enhanced router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteRecommendation {
    /// Recommended expert domain
    pub domain: ExpertDomain,
    /// Confidence in the recommendation (0.0 to 1.0)
    pub confidence: f64,
    /// Estimated cost (in tokens)
    pub estimated_cost: u64,
    /// Estimated processing time (ms)
    pub estimated_time: u32,
    /// Whether to use local or market experts
    pub use_market: bool,
    /// Reasoning for the recommendation
    pub reasoning: String,
}

/// Query classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryClassification {
    /// Primary domain for the query
    pub primary_domain: ExpertDomain,
    /// Secondary domains that might be needed
    pub secondary_domains: Vec<ExpertDomain>,
    /// Complexity score (0.0 to 1.0)
    pub complexity: f64,
    /// Urgency level (0.0 to 1.0)
    pub urgency: f64,
}

#[wasm_bindgen]
impl EnhancedRouter {
    /// Create a new enhanced router
    #[wasm_bindgen(constructor)]
    pub fn new(config: ProcessingConfig) -> EnhancedRouter {
        let mut experts = HashMap::new();
        
        // Initialize expert pools for each domain
        for domain in [
            ExpertDomain::Reasoning,
            ExpertDomain::Coding,
            ExpertDomain::Language,
            ExpertDomain::Mathematics,
            ExpertDomain::ToolUse,
            ExpertDomain::Context,
        ] {
            experts.insert(domain, vec![MicroExpert::new(domain)]);
        }

        let stats = NetworkStats {
            active_peers: 1,
            total_queries: 0,
            average_latency_ms: 0.0,
            expert_utilization: HashMap::new(),
            neural_accuracy: 0.85,
        };

        EnhancedRouter {
            experts,
            market_integration: None,
            config,
            stats,
        }
    }

    /// Enable market integration
    pub fn enable_market_integration(&mut self, integration_config: &str) -> Result<(), JsValue> {
        let config: MarketIntegration = serde_json::from_str(integration_config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
        
        self.market_integration = Some(config);
        Ok(())
    }

    /// Classify a query to determine appropriate expert domains
    pub fn classify_query(&self, query: &str) -> String {
        let classification = self.internal_classify_query(query);
        serde_json::to_string(&classification).unwrap_or_default()
    }

    /// Get routing recommendation for a query
    pub fn get_route_recommendation(&self, query: &str) -> String {
        let recommendation = self.internal_get_route_recommendation(query);
        serde_json::to_string(&recommendation).unwrap_or_default()
    }

    /// Route a query using enhanced logic
    pub fn enhanced_route(&mut self, query: &str) -> String {
        // Classify the query
        let classification = self.internal_classify_query(query);
        
        // Get route recommendation
        let recommendation = self.internal_get_route_recommendation(query);
        
        // Update statistics
        self.update_stats(&classification, &recommendation);
        
        // Process the query
        self.process_with_recommendation(query, &recommendation)
    }

    /// Get current expert metrics
    pub fn get_expert_metrics(&self) -> String {
        let metrics: Vec<ExpertMetrics> = self.experts.keys().map(|&domain| {
            let utilization = self.stats.expert_utilization.get(&domain).copied().unwrap_or(0.0);
            
            ExpertMetrics {
                domain,
                queries_processed: self.stats.total_queries / self.experts.len() as u64,
                avg_response_time: self.stats.average_latency_ms,
                success_rate: 0.95, // Would be calculated from actual metrics
                quality_score: 0.85, // Would be calculated from feedback
                current_load: utilization,
            }
        }).collect();
        
        serde_json::to_string(&metrics).unwrap_or_default()
    }

    /// Get network statistics
    pub fn get_network_stats(&self) -> String {
        serde_json::to_string(&self.stats).unwrap_or_default()
    }

    /// Update expert capacity based on market conditions
    pub fn update_capacity(&mut self, domain_str: &str, capacity: usize) -> Result<(), JsValue> {
        let domain: ExpertDomain = serde_json::from_str(&format!("\"{}\"", domain_str))
            .map_err(|e| JsValue::from_str(&format!("Invalid domain: {}", e)))?;
        
        if let Some(expert_pool) = self.experts.get_mut(&domain) {
            // Adjust the number of experts based on capacity
            if capacity > expert_pool.len() {
                // Add more experts
                for _ in expert_pool.len()..capacity {
                    expert_pool.push(MicroExpert::new(domain));
                }
            } else if capacity < expert_pool.len() {
                // Remove excess experts
                expert_pool.truncate(capacity);
            }
        }
        
        Ok(())
    }
}

impl EnhancedRouter {
    /// Internal query classification logic
    fn internal_classify_query(&self, query: &str) -> QueryClassification {
        let query_lower = query.to_lowercase();
        
        // Simple keyword-based classification (would be more sophisticated in practice)
        let (primary_domain, complexity) = if query_lower.contains("code") || query_lower.contains("function") || query_lower.contains("algorithm") {
            (ExpertDomain::Coding, 0.8)
        } else if query_lower.contains("math") || query_lower.contains("calculate") || query_lower.contains("equation") {
            (ExpertDomain::Mathematics, 0.7)
        } else if query_lower.contains("tool") || query_lower.contains("api") || query_lower.contains("function") {
            (ExpertDomain::ToolUse, 0.6)
        } else if query_lower.contains("reason") || query_lower.contains("logic") || query_lower.contains("analyze") {
            (ExpertDomain::Reasoning, 0.9)
        } else if query_lower.contains("context") || query_lower.contains("remember") || query_lower.contains("previous") {
            (ExpertDomain::Context, 0.5)
        } else {
            (ExpertDomain::Language, 0.4)
        };

        // Determine secondary domains
        let mut secondary_domains = Vec::new();
        if complexity > 0.7 {
            secondary_domains.push(ExpertDomain::Reasoning);
        }
        if query_lower.len() > 100 {
            secondary_domains.push(ExpertDomain::Context);
        }

        // Calculate urgency based on query characteristics
        let urgency = if query_lower.contains("urgent") || query_lower.contains("asap") || query_lower.contains("quickly") {
            0.9
        } else if query_lower.contains("when possible") || query_lower.contains("sometime") {
            0.2
        } else {
            0.5
        };

        QueryClassification {
            primary_domain,
            secondary_domains,
            complexity,
            urgency,
        }
    }

    /// Internal route recommendation logic
    fn internal_get_route_recommendation(&self, query: &str) -> RouteRecommendation {
        let classification = self.internal_classify_query(query);
        
        // Check local expert availability
        let local_available = self.experts.get(&classification.primary_domain)
            .map(|experts| !experts.is_empty())
            .unwrap_or(false);

        // Determine if market should be used
        let use_market = if let Some(ref market_config) = self.market_integration {
            market_config.market_routing_enabled && 
            (classification.complexity > 0.8 || classification.urgency > 0.7 || !local_available)
        } else {
            false
        };

        // Estimate cost and time
        let base_cost = (classification.complexity * 100.0) as u64;
        let estimated_cost = if use_market {
            base_cost * 2 // Market premium
        } else {
            base_cost / 10 // Local processing is cheaper
        };

        let estimated_time = if use_market {
            (classification.complexity * 5000.0) as u32 // Network latency
        } else {
            (classification.complexity * 1000.0) as u32 // Local processing
        };

        // Calculate confidence
        let confidence = if local_available && !use_market {
            0.9
        } else if use_market {
            0.7
        } else {
            0.5
        };

        // Generate reasoning
        let reasoning = if use_market {
            format!("Using market experts for {} domain due to high complexity/urgency", 
                   format!("{:?}", classification.primary_domain))
        } else {
            format!("Using local {} expert - sufficient capacity available", 
                   format!("{:?}", classification.primary_domain))
        };

        RouteRecommendation {
            domain: classification.primary_domain,
            confidence,
            estimated_cost,
            estimated_time,
            use_market,
            reasoning,
        }
    }

    /// Update internal statistics
    fn update_stats(&mut self, classification: &QueryClassification, recommendation: &RouteRecommendation) {
        self.stats.total_queries += 1;
        
        // Update expert utilization
        let current_util = self.stats.expert_utilization
            .get(&classification.primary_domain)
            .copied()
            .unwrap_or(0.0);
        
        let new_util = (current_util * 0.9) + (0.1 * if recommendation.use_market { 0.5 } else { 1.0 });
        self.stats.expert_utilization.insert(classification.primary_domain, new_util);
        
        // Update average latency (simple exponential moving average)
        let new_latency = recommendation.estimated_time as f64;
        if self.stats.average_latency_ms == 0.0 {
            self.stats.average_latency_ms = new_latency;
        } else {
            self.stats.average_latency_ms = (self.stats.average_latency_ms * 0.9) + (new_latency * 0.1);
        }
    }

    /// Process query with the given recommendation
    fn process_with_recommendation(&self, query: &str, recommendation: &RouteRecommendation) -> String {
        if recommendation.use_market {
            // In a real implementation, this would make market calls
            format!(
                "Market processing: {} (domain: {:?}, cost: {} tokens, time: {}ms)",
                query,
                recommendation.domain,
                recommendation.estimated_cost,
                recommendation.estimated_time
            )
        } else {
            // Use local expert
            if let Some(experts) = self.experts.get(&recommendation.domain) {
                if let Some(expert) = experts.first() {
                    expert.process(query)
                } else {
                    "No local expert available".to_string()
                }
            } else {
                "Domain not supported".to_string()
            }
        }
    }
}

/// Market integration utilities
pub struct MarketUtils;

impl MarketUtils {
    /// Calculate optimal market strategy based on current conditions
    pub fn calculate_market_strategy(
        local_capacity: &HashMap<ExpertDomain, usize>,
        market_prices: &HashMap<ExpertDomain, u64>,
        budget: u64,
    ) -> MarketStrategy {
        let mut recommendations = HashMap::new();
        
        for (&domain, &local_count) in local_capacity {
            let market_price = market_prices.get(&domain).copied().unwrap_or(100);
            
            let action = if local_count == 0 && market_price <= budget / 10 {
                StrategyAction::BuyFromMarket
            } else if local_count > 3 && market_price > 50 {
                StrategyAction::SellToMarket
            } else {
                StrategyAction::UseLocal
            };
            
            recommendations.insert(domain, action);
        }
        
        MarketStrategy {
            recommendations,
            total_budget: budget,
            expected_cost: market_prices.values().sum::<u64>() / market_prices.len() as u64,
        }
    }
}

/// Market strategy recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketStrategy {
    /// Action recommendations per domain
    pub recommendations: HashMap<ExpertDomain, StrategyAction>,
    /// Total available budget
    pub total_budget: u64,
    /// Expected cost of strategy
    pub expected_cost: u64,
}

/// Strategy action for a domain
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum StrategyAction {
    /// Use local experts
    UseLocal,
    /// Buy compute from market
    BuyFromMarket,
    /// Sell compute to market
    SellToMarket,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_router_creation() {
        let config = ProcessingConfig::new();
        let router = EnhancedRouter::new(config);
        
        assert_eq!(router.experts.len(), 6);
        assert!(router.market_integration.is_none());
    }

    #[test]
    fn test_query_classification() {
        let config = ProcessingConfig::new();
        let router = EnhancedRouter::new(config);
        
        let classification = router.internal_classify_query("Write a function to calculate fibonacci numbers");
        assert_eq!(classification.primary_domain, ExpertDomain::Coding);
        assert!(classification.complexity > 0.5);
    }

    #[test]
    fn test_route_recommendation() {
        let config = ProcessingConfig::new();
        let router = EnhancedRouter::new(config);
        
        let recommendation = router.internal_get_route_recommendation("Simple greeting");
        assert_eq!(recommendation.domain, ExpertDomain::Language);
        assert!(!recommendation.use_market); // Should use local for simple queries
    }

    #[test]
    fn test_market_strategy_calculation() {
        let mut local_capacity = HashMap::new();
        local_capacity.insert(ExpertDomain::Coding, 0);
        local_capacity.insert(ExpertDomain::Mathematics, 5);
        
        let mut market_prices = HashMap::new();
        market_prices.insert(ExpertDomain::Coding, 50);
        market_prices.insert(ExpertDomain::Mathematics, 100);
        
        let strategy = MarketUtils::calculate_market_strategy(&local_capacity, &market_prices, 1000);
        
        assert!(matches!(
            strategy.recommendations.get(&ExpertDomain::Coding),
            Some(StrategyAction::BuyFromMarket)
        ));
        assert!(matches!(
            strategy.recommendations.get(&ExpertDomain::Mathematics),
            Some(StrategyAction::SellToMarket)
        ));
    }
}