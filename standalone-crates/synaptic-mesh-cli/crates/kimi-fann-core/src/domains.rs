//! Expert domain definitions for Kimi-K2 micro-experts

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Expert domains representing different AI capabilities
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpertDomain {
    /// Logical reasoning and problem-solving
    Reasoning,
    /// Code generation, debugging, and analysis
    Coding,
    /// Natural language understanding and generation
    Language,
    /// Mathematical reasoning and computation
    Mathematics,
    /// Function calling and API interaction
    ToolUse,
    /// Long-context understanding and synthesis
    Context,
}

impl ExpertDomain {
    /// Get the recommended parameter count for this domain
    pub fn default_parameter_count(&self) -> usize {
        match self {
            Self::Reasoning => 10_000,
            Self::Coding => 50_000,
            Self::Language => 25_000,
            Self::Mathematics => 20_000,
            Self::ToolUse => 15_000,
            Self::Context => 30_000,
        }
    }

    /// Get a human-readable description of this domain
    pub fn description(&self) -> &'static str {
        match self {
            Self::Reasoning => "Logical inference, problem-solving, and deductive reasoning",
            Self::Coding => "Code generation, debugging, analysis, and programming tasks",
            Self::Language => "Natural language understanding, generation, and processing",
            Self::Mathematics => "Mathematical reasoning, computation, and symbolic manipulation",
            Self::ToolUse => "Function calling, API interaction, and tool orchestration",
            Self::Context => "Long-context understanding, synthesis, and cross-reference analysis",
        }
    }

    /// Get the specialization focus areas for this domain
    pub fn specializations(&self) -> &'static [&'static str] {
        match self {
            Self::Reasoning => &["logic", "inference", "problem_solving", "deduction"],
            Self::Coding => &["generation", "debugging", "analysis", "refactoring"],
            Self::Language => &["understanding", "generation", "translation", "summarization"],
            Self::Mathematics => &["algebra", "calculus", "statistics", "geometry"],
            Self::ToolUse => &["function_calling", "api_interaction", "workflow_orchestration"],
            Self::Context => &["long_context", "synthesis", "cross_reference", "memory"],
        }
    }
}

#[wasm_bindgen]
impl ExpertDomain {
    /// Get the default parameter count for WASM
    #[wasm_bindgen(getter = default_parameter_count)]
    pub fn wasm_default_parameter_count(&self) -> usize {
        self.default_parameter_count()
    }

    /// Get the description for WASM
    #[wasm_bindgen(getter = description)]
    pub fn wasm_description(&self) -> String {
        self.description().to_string()
    }
}

impl std::fmt::Display for ExpertDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reasoning => write!(f, "reasoning"),
            Self::Coding => write!(f, "coding"),
            Self::Language => write!(f, "language"),
            Self::Mathematics => write!(f, "mathematics"),
            Self::ToolUse => write!(f, "tool-use"),
            Self::Context => write!(f, "context"),
        }
    }
}

impl std::str::FromStr for ExpertDomain {
    type Err = crate::error::KimiError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "reasoning" => Ok(Self::Reasoning),
            "coding" => Ok(Self::Coding),
            "language" => Ok(Self::Language),
            "mathematics" | "math" => Ok(Self::Mathematics),
            "tool-use" | "tool_use" | "tools" => Ok(Self::ToolUse),
            "context" => Ok(Self::Context),
            _ => Err(crate::error::KimiError::configuration(format!(
                "Unknown expert domain: {}",
                s
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_counts() {
        assert_eq!(ExpertDomain::Reasoning.default_parameter_count(), 10_000);
        assert_eq!(ExpertDomain::Coding.default_parameter_count(), 50_000);
        assert_eq!(ExpertDomain::Language.default_parameter_count(), 25_000);
        assert_eq!(ExpertDomain::Mathematics.default_parameter_count(), 20_000);
        assert_eq!(ExpertDomain::ToolUse.default_parameter_count(), 15_000);
        assert_eq!(ExpertDomain::Context.default_parameter_count(), 30_000);
    }

    #[test]
    fn test_from_str() {
        assert_eq!("reasoning".parse::<ExpertDomain>().unwrap(), ExpertDomain::Reasoning);
        assert_eq!("coding".parse::<ExpertDomain>().unwrap(), ExpertDomain::Coding);
        assert_eq!("math".parse::<ExpertDomain>().unwrap(), ExpertDomain::Mathematics);
        assert!("invalid".parse::<ExpertDomain>().is_err());
    }
}