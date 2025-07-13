//! # Micro-Expert Demo
//! 
//! This example demonstrates the micro-expert neural architecture
//! with real neural network processing using our custom implementation.

use kimi_fann_core::{
    KimiRuntime, ProcessingConfig, ExpertDomain, MicroExpert, ExpertRouter
};

fn main() {
    println!("🧠 Kimi-FANN Core Demo - Neural Network Micro-Experts");
    println!("================================================\n");

    // Create a processing configuration
    let config = ProcessingConfig::new();
    
    // Initialize the runtime with all experts
    let mut runtime = KimiRuntime::new(config);
    
    println!("📊 Initial Runtime Stats:");
    println!("{}\n", runtime.get_stats());

    // Test different types of queries
    let test_queries = vec![
        ("What is the meaning of life?", "reasoning query"),
        ("Write a function to add two numbers", "coding query"),
        ("Translate hello to Spanish", "language query"),
        ("Calculate 2 + 2", "mathematics query"),
        ("Execute this command", "tool use query"),
        ("Provide context about this conversation", "context query"),
    ];

    for (query, description) in test_queries {
        println!("🔍 Testing {}: '{}'", description, query);
        let response = runtime.process(query);
        println!("📝 Response:\n{}\n", response);
        println!("-".repeat(80));
    }

    // Test training functionality
    println!("\n🎓 Testing Expert Training:");
    let success = runtime.train_expert(
        ExpertDomain::Mathematics, 
        "Calculate 5 + 3", 
        0.9  // High confidence for math
    );
    println!("Training result: {}", if success { "Success ✅" } else { "Failed ❌" });

    // Show final stats
    println!("\n📊 Final Runtime Stats:");
    println!("{}", runtime.get_stats());

    // Test individual expert
    println!("\n🔬 Testing Individual Expert:");
    let mut coding_expert = MicroExpert::new(ExpertDomain::Coding);
    let response = coding_expert.process("def hello_world():");
    println!("Coding expert response: {}", response);
    println!("Expert stats: {}", coding_expert.get_stats());

    // Test router directly
    println!("\n🚦 Testing Expert Router:");
    let mut router = ExpertRouter::new();
    let routing_response = router.route("Why should I use Rust for web development?");
    println!("Router response: {}", routing_response);
    println!("Router stats: {}", router.get_all_stats());

    println!("\n✨ Demo completed successfully!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_creation() {
        let config = ProcessingConfig::new();
        let runtime = KimiRuntime::new(config);
        let stats = runtime.get_stats();
        assert!(stats.contains("Total queries: 0"));
    }

    #[test]
    fn test_expert_creation() {
        let expert = MicroExpert::new(ExpertDomain::Reasoning);
        let stats = expert.get_stats();
        assert!(stats.contains("Reasoning"));
    }

    #[test]
    fn test_router_creation() {
        let router = ExpertRouter::new();
        let stats = router.get_all_stats();
        assert!(stats.contains("Router Stats"));
    }
}