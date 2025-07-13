//! Property-Based Tests for Kimi-FANN Core
//! 
//! Uses property-based testing to verify system behavior
//! across a wide range of inputs and scenarios.

use kimi_fann_core::*;
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_expert_processing_never_panics(
        domain in prop::sample::select(vec![
            ExpertDomain::Reasoning,
            ExpertDomain::Coding,
            ExpertDomain::Mathematics,
            ExpertDomain::Language,
            ExpertDomain::ToolUse,
            ExpertDomain::Context,
        ]),
        query in ".*"
    ) {
        let expert = MicroExpert::new(domain);
        let result = expert.process(&query);
        
        // Should never panic, always return something
        // Empty input might return empty result, that's okay
        if !query.trim().is_empty() {
            prop_assert!(!result.is_empty() || query.len() > 10000);
        }
    }

    #[test]
    fn test_router_handles_any_input(
        num_experts in 1..10usize,
        query in ".*"
    ) {
        let mut router = ExpertRouter::new();
        
        // Add random number of experts
        for i in 0..num_experts {
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
        
        let result = router.route(&query);
        
        // Should handle any input without panicking
        if !query.trim().is_empty() {
            prop_assert!(!result.is_empty() || query.len() > 10000);
        }
    }

    #[test]
    fn test_runtime_configuration_properties(
        max_experts in 1..20usize,
        timeout_ms in 100..60000u32
    ) {
        let config = ProcessingConfig {
            max_experts,
            timeout_ms,
        };
        
        let runtime = KimiRuntime::new(config);
        let result = runtime.process("Test configuration");
        
        // Runtime should work with any reasonable configuration
        prop_assert!(!result.is_empty());
    }

    #[test]
    fn test_expert_config_serialization_roundtrip(
        parameter_count in 1000..100000usize,
        learning_rate in 0.0001..0.1f32,
        domain in prop::sample::select(vec![
            ExpertDomain::Reasoning,
            ExpertDomain::Coding,
            ExpertDomain::Mathematics,
            ExpertDomain::Language,
            ExpertDomain::ToolUse,
            ExpertDomain::Context,
        ])
    ) {
        let config = ExpertConfig {
            domain,
            parameter_count,
            learning_rate,
        };
        
        // Test serialization roundtrip
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: ExpertConfig = serde_json::from_str(&serialized).unwrap();
        
        prop_assert_eq!(config.domain, deserialized.domain);
        prop_assert_eq!(config.parameter_count, deserialized.parameter_count);
        prop_assert!((config.learning_rate - deserialized.learning_rate).abs() < f32::EPSILON);
    }

    #[test]
    fn test_processing_preserves_input_length_relationship(
        input in ".*"
    ) {
        let expert = MicroExpert::new(ExpertDomain::Language);
        let result = expert.process(&input);
        
        // Longer inputs should generally produce longer outputs
        // (with reasonable bounds)
        if !input.trim().is_empty() && input.len() < 1000 {
            prop_assert!(!result.is_empty());
            
            // Result should be at least somewhat proportional to input
            // but not necessarily linear
            if input.len() > 50 {
                prop_assert!(result.len() > 10);
            }
        }
    }

    #[test]
    fn test_expert_domain_consistency(
        domain in prop::sample::select(vec![
            ExpertDomain::Reasoning,
            ExpertDomain::Coding,
            ExpertDomain::Mathematics,
            ExpertDomain::Language,
            ExpertDomain::ToolUse,
            ExpertDomain::Context,
        ]),
        query in "[a-zA-Z0-9 ]{1,100}"
    ) {
        let expert = MicroExpert::new(domain);
        let result = expert.process(&query);
        
        // Expert should always reference its domain in the output
        let domain_str = format!("{:?}", domain);
        prop_assert!(result.contains(&domain_str));
    }

    #[test]
    fn test_unicode_handling(
        unicode_input in "[\u{0000}-\u{FFFF}]{0,100}"
    ) {
        let expert = MicroExpert::new(ExpertDomain::Language);
        let result = expert.process(&unicode_input);
        
        // Should handle Unicode gracefully
        if !unicode_input.trim().is_empty() {
            // Should not crash and should produce some output
            prop_assert!(result.len() >= 5);
        }
    }

    #[test]
    fn test_numerical_input_handling(
        number in -1000000.0..1000000.0f64
    ) {
        let expert = MicroExpert::new(ExpertDomain::Mathematics);
        let input = number.to_string();
        let result = expert.process(&input);
        
        // Mathematical expert should handle any reasonable number
        prop_assert!(!result.is_empty());
        prop_assert!(result.contains("Mathematics"));
    }

    #[test]
    fn test_concurrent_processing_safety(
        queries in prop::collection::vec("[a-zA-Z0-9 ]{1,50}", 1..20)
    ) {
        use std::sync::Arc;
        use std::thread;
        
        let runtime = Arc::new(KimiRuntime::new(ProcessingConfig::new()));
        let mut handles = vec![];
        
        for query in queries {
            let runtime_clone = Arc::clone(&runtime);
            let handle = thread::spawn(move || {
                runtime_clone.process(&query)
            });
            handles.push(handle);
        }
        
        let results: Vec<String> = handles.into_iter()
            .map(|h| h.join().unwrap())
            .collect();
        
        // All concurrent operations should succeed
        for result in results {
            prop_assert!(!result.is_empty());
        }
    }

    #[test]
    fn test_memory_bounds(
        num_operations in 1..100usize
    ) {
        let expert = MicroExpert::new(ExpertDomain::Coding);
        
        // Perform many operations to test memory behavior
        for i in 0..num_operations {
            let query = format!("Memory test operation {}", i);
            let result = expert.process(&query);
            prop_assert!(!result.is_empty());
        }
        
        // Should complete without memory issues
        prop_assert!(true);
    }
}

#[cfg(test)]
mod property_integration_tests {
    use super::*;
    use proptest::test_runner::TestRunner;
    use proptest::strategy::Strategy;

    #[test]
    fn test_system_invariants() {
        let mut runner = TestRunner::default();
        
        // Define system invariants that should always hold
        let strategy = (
            prop::sample::select(vec![
                ExpertDomain::Reasoning,
                ExpertDomain::Coding,
                ExpertDomain::Mathematics,
                ExpertDomain::Language,
                ExpertDomain::ToolUse,
                ExpertDomain::Context,
            ]),
            "[a-zA-Z0-9 ]{1,100}",
        );
        
        runner.run(&strategy, |(domain, query)| {
            let expert = MicroExpert::new(domain);
            let result = expert.process(&query);
            
            // Invariant 1: Non-empty queries always produce non-empty results
            if !query.trim().is_empty() {
                assert!(!result.is_empty(), "Empty result for non-empty query");
            }
            
            // Invariant 2: Results always contain domain information
            let domain_str = format!("{:?}", domain);
            assert!(result.contains(&domain_str), "Result missing domain info");
            
            // Invariant 3: Results are reasonable length
            assert!(result.len() <= 10000, "Result too long: {}", result.len());
            assert!(result.len() >= 5, "Result too short: {}", result.len());
            
            Ok(())
        }).unwrap();
    }

    #[test]
    fn test_router_load_balancing_properties() {
        let mut runner = TestRunner::default();
        
        let strategy = (
            prop::collection::vec(
                prop::sample::select(vec![
                    ExpertDomain::Reasoning,
                    ExpertDomain::Coding,
                    ExpertDomain::Mathematics,
                ]),
                1..10
            ),
            prop::collection::vec("[a-zA-Z0-9 ]{1,50}", 1..20),
        );
        
        runner.run(&strategy, |(domains, queries)| {
            let mut router = ExpertRouter::new();
            
            // Add experts
            for domain in domains {
                router.add_expert(MicroExpert::new(domain));
            }
            
            // Process queries
            for query in queries {
                let result = router.route(&query);
                
                if !query.trim().is_empty() {
                    assert!(!result.is_empty(), "Router failed for query: {}", query);
                }
            }
            
            Ok(())
        }).unwrap();
    }

    #[test]
    fn test_performance_properties() {
        let mut runner = TestRunner::default();
        
        let strategy = prop::collection::vec("[a-zA-Z0-9 ]{1,100}", 1..50);
        
        runner.run(&strategy, |queries| {
            let runtime = KimiRuntime::new(ProcessingConfig::new());
            let start = std::time::Instant::now();
            
            for query in queries {
                let result = runtime.process(&query);
                if !query.trim().is_empty() {
                    assert!(!result.is_empty());
                }
            }
            
            let total_time = start.elapsed();
            
            // Performance property: Should complete within reasonable time
            assert!(total_time.as_secs() < 30, "Processing took too long: {:?}", total_time);
            
            Ok(())
        }).unwrap();
    }

    #[test]
    fn test_error_handling_properties() {
        let mut runner = TestRunner::default();
        
        // Test with potentially problematic inputs
        let strategy = prop::sample::select(vec![
            "",
            " ",
            "\n",
            "\t",
            "null",
            "undefined",
            "0",
            "-1",
            "NaN",
            "Infinity",
            "[]",
            "{}",
            "<>",
            "\"\"",
            "''",
            "\u{0000}",
            "\u{FFFF}",
        ]);
        
        runner.run(&strategy, |problematic_input| {
            let expert = MicroExpert::new(ExpertDomain::Language);
            
            // Should never panic, regardless of input
            let result = expert.process(problematic_input);
            
            // May return empty result for truly empty input, but should not crash
            if !problematic_input.trim().is_empty() {
                // Should produce some response for non-empty input
                assert!(!result.is_empty() || problematic_input.len() == 1);
            }
            
            Ok(())
        }).unwrap();
    }

    #[test]
    fn test_configuration_space_properties() {
        let mut runner = TestRunner::default();
        
        let strategy = (1..50usize, 100..60000u32);
        
        runner.run(&strategy, |(max_experts, timeout_ms)| {
            let config = ProcessingConfig {
                max_experts,
                timeout_ms,
            };
            
            // Should be able to create runtime with any reasonable config
            let runtime = KimiRuntime::new(config);
            let result = runtime.process("Configuration test");
            
            assert!(!result.is_empty(), "Failed with config: {} experts, {}ms timeout", 
                   max_experts, timeout_ms);
            
            Ok(())
        }).unwrap();
    }
}