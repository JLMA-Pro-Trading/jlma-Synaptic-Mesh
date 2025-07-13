//! Expert routing and selection system for Kimi-K2 micro-experts

use crate::{domains::ExpertDomain, expert::MicroExpert, error::Result};
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use wasm_bindgen::prelude::*;

/// Strategy for routing requests to experts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Select the single most relevant expert
    BestMatch,
    /// Select top N experts based on relevance
    TopK(usize),
    /// Use ensemble of experts with weighted voting
    Ensemble,
    /// Adaptive selection based on context and performance
    Adaptive,
}

impl Default for RoutingStrategy {
    fn default() -> Self {
        Self::TopK(3)
    }
}

/// Context information for routing decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestContext {
    pub prompt: String,
    pub max_experts: usize,
    pub preferred_domains: Vec<ExpertDomain>,
    pub performance_threshold: f32,
    pub context_length: usize,
}

impl RequestContext {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            max_experts: 3,
            preferred_domains: Vec::new(),
            performance_threshold: 0.7,
            context_length: 0,
        }
    }
}

/// Expert selection result with confidence scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertSelection {
    pub domain: ExpertDomain,
    pub confidence: f32,
    pub reasoning: String,
}

/// Expert router that manages micro-expert selection and execution
#[wasm_bindgen]
pub struct ExpertRouter {
    #[wasm_bindgen(skip)]
    experts: HashMap<ExpertDomain, MicroExpert>,
    #[wasm_bindgen(skip)]
    strategy: RoutingStrategy,
    #[wasm_bindgen(skip)]
    performance_history: BTreeMap<ExpertDomain, Vec<f32>>,
}

#[wasm_bindgen]
impl ExpertRouter {
    /// Create a new expert router
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            experts: HashMap::new(),
            strategy: RoutingStrategy::default(),
            performance_history: BTreeMap::new(),
        }
    }

    /// Register an expert with the router
    #[wasm_bindgen]
    pub fn register_expert(&mut self, domain: ExpertDomain, expert: MicroExpert) {
        self.experts.insert(domain, expert);
        self.performance_history.entry(domain).or_insert_with(Vec::new);
    }

    /// Remove an expert from the router
    #[wasm_bindgen]
    pub fn remove_expert(&mut self, domain: ExpertDomain) -> bool {
        self.experts.remove(&domain).is_some()
    }

    /// Get list of available expert domains
    #[wasm_bindgen]
    pub fn available_domains(&self) -> Vec<ExpertDomain> {
        self.experts.keys().copied().collect()
    }

    /// Route a request to appropriate experts
    #[wasm_bindgen]
    pub fn route_request(&self, prompt: &str) -> Result<JsValue> {
        let context = RequestContext::new(prompt);
        let selections = self.select_experts(&context)?;
        serde_wasm_bindgen::to_value(&selections)
            .map_err(|e| crate::error::KimiError::routing(format!("Serialization failed: {}", e)))
    }

    /// Set the routing strategy
    #[wasm_bindgen]
    pub fn set_strategy(&mut self, strategy: &str) -> Result<()> {
        self.strategy = match strategy {
            "best_match" => RoutingStrategy::BestMatch,
            "top_k" => RoutingStrategy::TopK(3),
            "ensemble" => RoutingStrategy::Ensemble,
            "adaptive" => RoutingStrategy::Adaptive,
            _ => return Err(crate::error::KimiError::configuration(format!(
                "Unknown routing strategy: {}", strategy
            ))),
        };
        Ok(())
    }

    /// Get performance statistics for all experts
    #[wasm_bindgen]
    pub fn get_performance_stats(&self) -> JsValue {
        let stats: HashMap<String, f32> = self.performance_history
            .iter()
            .map(|(domain, scores)| {
                let avg = if scores.is_empty() {
                    0.0
                } else {
                    scores.iter().sum::<f32>() / scores.len() as f32
                };
                (domain.to_string(), avg)
            })
            .collect();
        
        serde_wasm_bindgen::to_value(&stats).unwrap_or(JsValue::NULL)
    }
}

impl ExpertRouter {
    /// Native Rust interface for expert selection
    pub fn select_experts(&self, context: &RequestContext) -> Result<Vec<ExpertSelection>> {
        let mut selections = Vec::new();
        
        // Analyze the prompt to determine relevant domains
        let domain_scores = self.analyze_prompt(&context.prompt);
        
        // Apply routing strategy
        let selected_domains = match &self.strategy {
            RoutingStrategy::BestMatch => {
                if let Some((domain, score)) = domain_scores.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
                    vec![(*domain, *score)]
                } else {
                    Vec::new()
                }
            },
            RoutingStrategy::TopK(k) => {
                let mut sorted: Vec<_> = domain_scores.iter().collect();
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                sorted.into_iter().take(*k).map(|(d, s)| (*d, *s)).collect()
            },
            RoutingStrategy::Ensemble => {
                domain_scores.iter().filter(|(_, score)| *score > 0.3).map(|(d, s)| (*d, *s)).collect()
            },
            RoutingStrategy::Adaptive => {
                self.adaptive_selection(&domain_scores, context)
            },
        };

        // Create selections with reasoning
        for (domain, confidence) in selected_domains {
            if self.experts.contains_key(&domain) {
                let reasoning = format!(
                    "Selected {} expert with {:.2} confidence based on prompt analysis",
                    domain, confidence
                );
                selections.push(ExpertSelection {
                    domain,
                    confidence,
                    reasoning,
                });
            }
        }

        // Limit to max_experts
        selections.truncate(context.max_experts);
        
        Ok(selections)
    }

    /// Analyze prompt to determine domain relevance scores
    fn analyze_prompt(&self, prompt: &str) -> HashMap<ExpertDomain, f32> {
        let mut scores = HashMap::new();
        let prompt_lower = prompt.to_lowercase();
        
        // Simple keyword-based analysis (would be replaced with actual ML model)
        let keywords = [
            (ExpertDomain::Coding, vec!["code", "program", "function", "debug", "implement", "algorithm", "bug", "syntax"]),
            (ExpertDomain::Mathematics, vec!["calculate", "equation", "math", "formula", "solve", "number", "statistics", "probability"]),
            (ExpertDomain::Reasoning, vec!["think", "analyze", "logic", "reason", "deduce", "conclude", "problem", "solve"]),
            (ExpertDomain::Language, vec!["translate", "write", "explain", "summarize", "text", "language", "grammar", "style"]),
            (ExpertDomain::ToolUse, vec!["api", "call", "execute", "tool", "function", "command", "service", "endpoint"]),
            (ExpertDomain::Context, vec!["context", "reference", "previous", "relate", "connection", "history", "memory"]),
        ];

        for (domain, words) in keywords {
            let mut score = 0.0;
            for word in words {
                if prompt_lower.contains(word) {
                    score += 0.2;
                }
            }
            scores.insert(domain, score.min(1.0));
        }

        // Boost preferred domains
        for domain in &self.experts.keys().collect::<Vec<_>>() {
            scores.entry(**domain).or_insert(0.1);
        }

        scores
    }

    /// Adaptive selection based on performance history
    fn adaptive_selection(&self, domain_scores: &HashMap<ExpertDomain, f32>, context: &RequestContext) -> Vec<(ExpertDomain, f32)> {
        let mut adjusted_scores = Vec::new();
        
        for (domain, base_score) in domain_scores {
            let performance_boost = if let Some(history) = self.performance_history.get(domain) {
                if history.is_empty() {
                    0.0
                } else {
                    let avg_performance = history.iter().sum::<f32>() / history.len() as f32;
                    (avg_performance - 0.5) * 0.3 // Boost/penalty based on historical performance
                }
            } else {
                0.0
            };
            
            let adjusted_score = (base_score + performance_boost).max(0.0).min(1.0);
            if adjusted_score > context.performance_threshold {
                adjusted_scores.push((*domain, adjusted_score));
            }
        }
        
        // Sort by adjusted score
        adjusted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        adjusted_scores
    }

    /// Update performance history for an expert
    pub fn update_performance(&mut self, domain: ExpertDomain, score: f32) {
        if let Some(history) = self.performance_history.get_mut(&domain) {
            history.push(score);
            // Keep only last 100 scores
            if history.len() > 100 {
                history.remove(0);
            }
        }
    }

    /// Get expert by domain
    pub fn get_expert(&self, domain: ExpertDomain) -> Option<&MicroExpert> {
        self.experts.get(&domain)
    }

    /// Get mutable expert by domain
    pub fn get_expert_mut(&mut self, domain: ExpertDomain) -> Option<&mut MicroExpert> {
        self.experts.get_mut(&domain)
    }

    /// Get all experts
    pub fn experts(&self) -> &HashMap<ExpertDomain, MicroExpert> {
        &self.experts
    }
}

impl Default for ExpertRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let router = ExpertRouter::new();
        assert!(router.available_domains().is_empty());
    }

    #[test]
    fn test_expert_registration() {
        let mut router = ExpertRouter::new();
        let expert = MicroExpert::new(ExpertDomain::Coding).unwrap();
        
        router.register_expert(ExpertDomain::Coding, expert);
        assert_eq!(router.available_domains().len(), 1);
        assert!(router.available_domains().contains(&ExpertDomain::Coding));
    }

    #[test]
    fn test_prompt_analysis() {
        let router = ExpertRouter::new();
        let scores = router.analyze_prompt("Write a function to calculate fibonacci numbers");
        
        // Should have high coding score
        assert!(scores.get(&ExpertDomain::Coding).unwrap_or(&0.0) > &0.0);
        assert!(scores.get(&ExpertDomain::Mathematics).unwrap_or(&0.0) > &0.0);
    }

    #[test]
    fn test_expert_selection() {
        let mut router = ExpertRouter::new();
        router.register_expert(ExpertDomain::Coding, MicroExpert::new(ExpertDomain::Coding).unwrap());
        router.register_expert(ExpertDomain::Mathematics, MicroExpert::new(ExpertDomain::Mathematics).unwrap());
        
        let context = RequestContext::new("Implement a sorting algorithm");
        let selections = router.select_experts(&context).unwrap();
        
        assert!(!selections.is_empty());
        assert!(selections.iter().any(|s| s.domain == ExpertDomain::Coding));
    }
}