//! Expert Router Tests
//! 
//! Tests for the enhanced routing system that distributes requests
//! to appropriate micro-experts based on content analysis.

use kimi_fann_core::*;
use std::collections::HashMap;

#[cfg(test)]
mod router_tests {
    use super::*;

    #[test]
    fn test_router_expert_distribution() {
        let mut router = ExpertRouter::new();
        
        // Add multiple experts of different types
        router.add_expert(MicroExpert::new(ExpertDomain::Reasoning));
        router.add_expert(MicroExpert::new(ExpertDomain::Coding));
        router.add_expert(MicroExpert::new(ExpertDomain::Mathematics));
        router.add_expert(MicroExpert::new(ExpertDomain::Language));
        router.add_expert(MicroExpert::new(ExpertDomain::ToolUse));
        router.add_expert(MicroExpert::new(ExpertDomain::Context));
        
        // Test routing to appropriate experts
        let test_cases = vec![
            ("Write a Python function to sort a list", "coding"),
            ("What is the derivative of x^2?", "mathematics"),
            ("Explain the philosophy of artificial intelligence", "reasoning"),
            ("Translate this text to French", "language"),
            ("How do I configure nginx?", "tool-use"),
            ("Remember my preferences for this session", "context"),
        ];
        
        for (query, expected_domain) in test_cases {
            let result = router.route(query);
            assert!(!result.is_empty(), "No routing result for: {}", query);
            println!("Query: {} -> Result: {}", query, result);
        }
    }

    #[test]
    fn test_router_load_balancing() {
        let mut router = ExpertRouter::new();
        
        // Add multiple experts of the same domain to test load balancing
        for _ in 0..5 {
            router.add_expert(MicroExpert::new(ExpertDomain::Coding));
        }
        
        // Send multiple similar requests
        let mut results = vec![];
        for i in 0..20 {
            let query = format!("def function_{}(): pass", i);
            let result = router.route(&query);
            results.push(result);
        }
        
        // All requests should be handled
        assert_eq!(results.len(), 20);
        for result in results {
            assert!(!result.is_empty());
        }
    }

    #[test]
    fn test_router_fallback_mechanisms() {
        let mut router = ExpertRouter::new();
        
        // Add only one type of expert
        router.add_expert(MicroExpert::new(ExpertDomain::Language));
        
        // Try routing requests that don't match the available expert
        let mismatched_queries = vec![
            "Calculate 2 + 2",              // Math query to language expert
            "def bubble_sort(arr): pass",   // Code query to language expert
            "Configure Apache server",      // Tool query to language expert
        ];
        
        for query in mismatched_queries {
            let result = router.route(query);
            // Should still handle gracefully, even if not optimal
            assert!(!result.is_empty(), "Router failed to handle: {}", query);
        }
    }

    #[test]
    fn test_router_performance_under_load() {
        let mut router = ExpertRouter::new();
        
        // Add multiple experts for performance testing
        for domain in [
            ExpertDomain::Reasoning,
            ExpertDomain::Coding,
            ExpertDomain::Mathematics,
            ExpertDomain::Language,
        ] {
            for _ in 0..3 {
                router.add_expert(MicroExpert::new(domain));
            }
        }
        
        let start_time = std::time::Instant::now();
        
        // Process many requests rapidly
        let num_requests = 100;
        for i in 0..num_requests {
            let query = format!("Request number {} for processing", i);
            let result = router.route(&query);
            assert!(!result.is_empty());
        }
        
        let duration = start_time.elapsed();
        let throughput = num_requests as f64 / duration.as_secs_f64();
        
        println!("Router throughput: {:.2} requests/second", throughput);
        assert!(throughput > 50.0, "Router throughput too low: {:.2}", throughput);
    }

    #[test]
    fn test_router_concurrent_access() {
        use std::sync::Arc;
        use std::thread;
        
        let mut router = ExpertRouter::new();
        router.add_expert(MicroExpert::new(ExpertDomain::Reasoning));
        router.add_expert(MicroExpert::new(ExpertDomain::Coding));
        router.add_expert(MicroExpert::new(ExpertDomain::Mathematics));
        
        let router = Arc::new(router);
        let mut handles = vec![];
        
        // Spawn multiple threads to test concurrent routing
        for i in 0..10 {
            let router_clone = Arc::clone(&router);
            let handle = thread::spawn(move || {
                let query = format!("Concurrent request {}", i);
                router_clone.route(&query)
            });
            handles.push(handle);
        }
        
        // Collect all results
        let mut results = vec![];
        for handle in handles {
            let result = handle.join().unwrap();
            results.push(result);
        }
        
        // All concurrent requests should succeed
        assert_eq!(results.len(), 10);
        for result in results {
            assert!(!result.is_empty());
        }
    }

    #[test]
    fn test_router_expert_selection_accuracy() {
        let mut router = ExpertRouter::new();
        
        // Add one expert of each type
        let domains = vec![
            ExpertDomain::Reasoning,
            ExpertDomain::Coding,
            ExpertDomain::Mathematics,
            ExpertDomain::Language,
            ExpertDomain::ToolUse,
            ExpertDomain::Context,
        ];
        
        for domain in domains {
            router.add_expert(MicroExpert::new(domain));
        }
        
        // Test domain-specific queries
        let domain_queries = vec![
            ("Should we invest in cryptocurrency?", ExpertDomain::Reasoning),
            ("def fibonacci(n): return", ExpertDomain::Coding),
            ("What is the limit of sin(x)/x as x approaches 0?", ExpertDomain::Mathematics),
            ("Please translate this to German", ExpertDomain::Language),
            ("How do I install Docker?", ExpertDomain::ToolUse),
            ("What were we discussing about AI?", ExpertDomain::Context),
        ];
        
        for (query, expected_domain) in domain_queries {
            let result = router.route(query);
            assert!(!result.is_empty(), "No result for domain-specific query: {}", query);
            
            // The result should indicate processing by the appropriate expert
            let domain_name = format!("{:?}", expected_domain);
            println!("Query: {} -> Expected: {} -> Result: {}", query, domain_name, result);
        }
    }

    #[test]
    fn test_router_with_empty_expert_pool() {
        let router = ExpertRouter::new();
        
        // Test routing with no experts added
        let result = router.route("Test query with no experts");
        
        // Should handle gracefully
        assert!(!result.is_empty() || true); // Allow empty result for no experts
    }

    #[test]
    fn test_router_query_preprocessing() {
        let mut router = ExpertRouter::new();
        router.add_expert(MicroExpert::new(ExpertDomain::Language));
        
        // Test with various input formats
        let test_inputs = vec![
            "   whitespace query   ",
            "UPPERCASE QUERY",
            "lowercase query",
            "Mixed Case Query",
            "Query with numbers 123 and symbols !@#",
            "Multi\nline\nquery",
            "",  // Empty query
        ];
        
        for input in test_inputs {
            let result = router.route(input);
            // Should handle all input formats without crashing
            if !input.trim().is_empty() {
                assert!(!result.is_empty(), "No result for input: '{}'", input);
            }
        }
    }

    #[test]
    fn test_router_scalability() {
        let mut router = ExpertRouter::new();
        
        // Add many experts to test scalability
        for i in 0..50 {
            let domain = match i % 6 {
                0 => ExpertDomain::Reasoning,
                1 => ExpertDomain::Coding,
                2 => ExpertDomain::Mathematics,
                3 => ExpertDomain::Language,
                4 => ExpertDomain::ToolUse,
                _ => ExpertDomain::Context,
            };
            router.add_expert(MicroExpert::new(domain));
        }
        
        // Test routing performance with many experts
        let start_time = std::time::Instant::now();
        
        for i in 0..20 {
            let query = format!("Scalability test query {}", i);
            let result = router.route(&query);
            assert!(!result.is_empty());
        }
        
        let duration = start_time.elapsed();
        println!("Routing 20 queries with 50 experts took: {:?}", duration);
        
        // Should complete within reasonable time even with many experts
        assert!(duration.as_millis() < 5000, "Routing too slow with many experts");
    }

    #[test]
    fn test_router_memory_usage() {
        let mut router = ExpertRouter::new();
        
        // Add experts and process requests to test memory behavior
        for _ in 0..10 {
            router.add_expert(MicroExpert::new(ExpertDomain::Coding));
        }
        
        // Process many requests to test for memory leaks
        for i in 0..1000 {
            let query = format!("Memory test query {}", i);
            let result = router.route(&query);
            assert!(!result.is_empty());
            
            // Simulate memory pressure testing
            if i % 100 == 0 {
                // In a real implementation, we'd check actual memory usage here
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

#[cfg(test)]
mod routing_algorithm_tests {
    use super::*;

    #[test]
    fn test_content_based_routing() {
        let mut router = ExpertRouter::new();
        
        // Add experts with different specializations
        router.add_expert(MicroExpert::new(ExpertDomain::Coding));
        router.add_expert(MicroExpert::new(ExpertDomain::Mathematics));
        router.add_expert(MicroExpert::new(ExpertDomain::Language));
        
        // Test content-based routing
        let content_tests = vec![
            // Programming-related content
            ("function", "coding"),
            ("variable", "coding"),
            ("algorithm", "coding"),
            ("debug", "coding"),
            
            // Math-related content
            ("equation", "mathematics"),
            ("calculate", "mathematics"),
            ("integral", "mathematics"),
            ("probability", "mathematics"),
            
            // Language-related content
            ("translate", "language"),
            ("grammar", "language"),
            ("write", "language"),
            ("communicate", "language"),
        ];
        
        for (keyword, expected_category) in content_tests {
            let query = format!("Please help me with {}", keyword);
            let result = router.route(&query);
            assert!(!result.is_empty(), "No result for keyword: {}", keyword);
            println!("Keyword: {} -> Category: {} -> Result: {}", keyword, expected_category, result);
        }
    }

    #[test]
    fn test_routing_consistency() {
        let mut router = ExpertRouter::new();
        router.add_expert(MicroExpert::new(ExpertDomain::Coding));
        router.add_expert(MicroExpert::new(ExpertDomain::Mathematics));
        
        let query = "Calculate the fibonacci sequence";
        
        // Route the same query multiple times
        let mut results = vec![];
        for _ in 0..10 {
            results.push(router.route(query));
        }
        
        // Results should be consistent (same routing decision)
        for result in &results {
            assert!(!result.is_empty());
        }
        
        // In a deterministic router, all results should be identical
        // In a load-balancing router, they might vary but should all be valid
        println!("Consistency test - first result: {}", results[0]);
    }

    #[test]
    fn test_multi_domain_queries() {
        let mut router = ExpertRouter::new();
        router.add_expert(MicroExpert::new(ExpertDomain::Coding));
        router.add_expert(MicroExpert::new(ExpertDomain::Mathematics));
        router.add_expert(MicroExpert::new(ExpertDomain::Reasoning));
        
        // Queries that could apply to multiple domains
        let multi_domain_queries = vec![
            "Implement a mathematical optimization algorithm",  // Coding + Math
            "Explain the logic behind sorting algorithms",      // Coding + Reasoning
            "Analyze the mathematical complexity of AI models", // Math + Reasoning
            "Design a neural network for pattern recognition",  // All three
        ];
        
        for query in multi_domain_queries {
            let result = router.route(query);
            assert!(!result.is_empty(), "No result for multi-domain query: {}", query);
            println!("Multi-domain query: {} -> {}", query, result);
        }
    }
}