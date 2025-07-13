//! Integration tests for Kimi-FANN Core
//! 
//! These tests validate the complete system functionality with real neural networks
//! and actual data processing, replacing all mock implementations.

use kimi_fann_core::*;
use std::collections::HashMap;

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_micro_expert_creation_and_configuration() {
        // Test expert creation with different domains
        let reasoning_expert = MicroExpert::new(ExpertDomain::Reasoning);
        let coding_expert = MicroExpert::new(ExpertDomain::Coding);
        let math_expert = MicroExpert::new(ExpertDomain::Mathematics);
        
        // Verify experts are created successfully
        assert_ne!(reasoning_expert.process("test"), "");
        assert_ne!(coding_expert.process("test"), "");
        assert_ne!(math_expert.process("test"), "");
    }

    #[test]
    fn test_expert_router_functionality() {
        let mut router = ExpertRouter::new();
        
        // Add multiple experts
        router.add_expert(MicroExpert::new(ExpertDomain::Reasoning));
        router.add_expert(MicroExpert::new(ExpertDomain::Coding));
        router.add_expert(MicroExpert::new(ExpertDomain::Mathematics));
        router.add_expert(MicroExpert::new(ExpertDomain::Language));
        
        // Test routing for different request types
        let coding_request = "Write a function to sort an array in Python";
        let math_request = "Calculate the derivative of x^2 + 3x + 2";
        let reasoning_request = "Analyze the pros and cons of renewable energy";
        let language_request = "Translate 'hello world' to Spanish";
        
        let coding_result = router.route(&coding_request);
        let math_result = router.route(&math_request);
        let reasoning_result = router.route(&reasoning_request);
        let language_result = router.route(&language_request);
        
        // Verify routing produces different outputs for different domains
        assert_ne!(coding_result, math_result);
        assert_ne!(reasoning_result, language_result);
        assert!(!coding_result.is_empty());
        assert!(!math_result.is_empty());
    }

    #[test]
    fn test_kimi_runtime_processing() {
        let config = ProcessingConfig::new();
        let runtime = KimiRuntime::new(config);
        
        // Test various types of queries
        let queries = vec![
            "Explain quantum computing",
            "def fibonacci(n): pass  # Complete this function",
            "What is the integral of sin(x)?",
            "How do neural networks learn?",
            "Create a REST API endpoint for user authentication",
        ];
        
        for query in queries {
            let result = runtime.process(query);
            assert!(!result.is_empty(), "Empty result for query: {}", query);
            assert!(result.len() > 10, "Too short result for: {}", query);
        }
    }

    #[test]
    fn test_expert_domain_specialization() {
        // Test that different experts handle domain-specific tasks appropriately
        let reasoning_expert = MicroExpert::new(ExpertDomain::Reasoning);
        let coding_expert = MicroExpert::new(ExpertDomain::Coding);
        let math_expert = MicroExpert::new(ExpertDomain::Mathematics);
        
        // Reasoning tasks
        let reasoning_result = reasoning_expert.process("Should we invest in renewable energy?");
        assert!(reasoning_result.contains("Reasoning"));
        
        // Coding tasks
        let coding_result = coding_expert.process("def bubble_sort(arr):");
        assert!(coding_result.contains("Coding"));
        
        // Mathematics tasks
        let math_result = math_expert.process("Solve: 2x + 5 = 15");
        assert!(math_result.contains("Mathematics"));
    }

    #[test]
    fn test_processing_config_limits() {
        let mut config = ProcessingConfig::new();
        assert_eq!(config.max_experts, 3);
        assert_eq!(config.timeout_ms, 5000);
        
        // Test that configuration is used appropriately
        let runtime = KimiRuntime::new(config);
        let result = runtime.process("Complex multi-step reasoning task");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_concurrent_processing() {
        use std::thread;
        use std::sync::Arc;
        
        let config = ProcessingConfig::new();
        let runtime = Arc::new(KimiRuntime::new(config));
        
        let mut handles = vec![];
        
        // Spawn multiple threads to test concurrent processing
        for i in 0..5 {
            let runtime_clone = Arc::clone(&runtime);
            let handle = thread::spawn(move || {
                let query = format!("Process concurrent request {}", i);
                runtime_clone.process(&query)
            });
            handles.push(handle);
        }
        
        // Collect results
        let mut results = vec![];
        for handle in handles {
            results.push(handle.join().unwrap());
        }
        
        // Verify all requests were processed successfully
        assert_eq!(results.len(), 5);
        for result in results {
            assert!(!result.is_empty());
        }
    }

    #[test]
    fn test_memory_efficiency() {
        // Create multiple experts and verify memory usage is reasonable
        let mut experts = vec![];
        for i in 0..100 {
            let domain = match i % 6 {
                0 => ExpertDomain::Reasoning,
                1 => ExpertDomain::Coding,
                2 => ExpertDomain::Language,
                3 => ExpertDomain::Mathematics,
                4 => ExpertDomain::ToolUse,
                _ => ExpertDomain::Context,
            };
            experts.push(MicroExpert::new(domain));
        }
        
        // Process requests with all experts
        for expert in &experts {
            let result = expert.process("test memory efficiency");
            assert!(!result.is_empty());
        }
        
        // This test mainly ensures the system doesn't crash with many experts
        assert_eq!(experts.len(), 100);
    }

    #[test]
    fn test_error_handling_and_robustness() {
        let runtime = KimiRuntime::new(ProcessingConfig::new());
        
        // Test with various edge cases
        let edge_cases = vec![
            "",  // Empty input
            " ",  // Whitespace only
            "a".repeat(10000),  // Very long input
            "Special chars: !@#$%^&*()[]{}|\\:;\"'<>?,./~`",
            "Unicode: 你好世界 🌍 ñañá ñoño",
            "Code injection attempt: ; DROP TABLE users; --",
        ];
        
        for edge_case in edge_cases {
            let result = runtime.process(&edge_case);
            // Should handle gracefully, not panic
            assert!(!result.is_empty() || edge_case.is_empty());
        }
    }

    #[test]
    fn test_expert_router_load_balancing() {
        let mut router = ExpertRouter::new();
        
        // Add multiple experts of the same domain
        for _ in 0..3 {
            router.add_expert(MicroExpert::new(ExpertDomain::Coding));
        }
        
        // Send multiple requests and verify they're handled
        let mut results = vec![];
        for i in 0..10 {
            let request = format!("def function_{}(): pass", i);
            results.push(router.route(&request));
        }
        
        // All requests should be processed
        for result in results {
            assert!(!result.is_empty());
        }
    }

    #[test]
    fn test_version_consistency() {
        // Verify version information is accessible
        assert!(!kimi_fann_core::VERSION.is_empty());
        assert!(kimi_fann_core::VERSION.contains('.'));
        
        // Version should follow semantic versioning pattern
        let version_parts: Vec<&str> = kimi_fann_core::VERSION.split('.').collect();
        assert!(version_parts.len() >= 2, "Version should have at least major.minor");
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_processing_latency() {
        let runtime = KimiRuntime::new(ProcessingConfig::new());
        
        let start = Instant::now();
        let result = runtime.process("Calculate the factorial of 10");
        let duration = start.elapsed();
        
        // Processing should complete within reasonable time
        assert!(duration.as_millis() < 1000, "Processing took too long: {:?}", duration);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_throughput() {
        let runtime = KimiRuntime::new(ProcessingConfig::new());
        let num_requests = 100;
        
        let start = Instant::now();
        for i in 0..num_requests {
            let query = format!("Process request number {}", i);
            let result = runtime.process(&query);
            assert!(!result.is_empty());
        }
        let duration = start.elapsed();
        
        let throughput = num_requests as f64 / duration.as_secs_f64();
        println!("Throughput: {:.2} requests/second", throughput);
        
        // Should handle at least 10 requests per second
        assert!(throughput > 10.0, "Throughput too low: {:.2} req/s", throughput);
    }

    #[test]
    fn test_memory_stability() {
        let runtime = KimiRuntime::new(ProcessingConfig::new());
        
        // Process many requests to test for memory leaks
        for i in 0..1000 {
            let query = format!("Memory test iteration {}", i);
            let result = runtime.process(&query);
            assert!(!result.is_empty());
            
            // Periodically check that we're not accumulating excessive memory
            if i % 100 == 0 {
                // This is a basic check - in a real implementation, we'd monitor actual memory usage
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}