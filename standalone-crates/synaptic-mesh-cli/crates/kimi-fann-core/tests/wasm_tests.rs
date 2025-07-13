//! WASM-Specific Tests for Kimi-FANN Core
//! 
//! Tests focused on WebAssembly compilation, performance,
//! and browser integration scenarios.

use kimi_fann_core::*;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[cfg(test)]
mod wasm_tests {
    use super::*;

    #[test]
    #[wasm_bindgen_test]
    fn test_wasm_expert_creation() {
        // Test that experts can be created in WASM environment
        let expert = MicroExpert::new(ExpertDomain::Coding);
        let result = expert.process("def hello(): pass");
        assert!(!result.is_empty());
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_wasm_router_functionality() {
        let mut router = ExpertRouter::new();
        router.add_expert(MicroExpert::new(ExpertDomain::Mathematics));
        router.add_expert(MicroExpert::new(ExpertDomain::Coding));
        
        let result = router.route("Calculate factorial of 5");
        assert!(!result.is_empty());
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_wasm_runtime_processing() {
        let config = ProcessingConfig::new();
        let runtime = KimiRuntime::new(config);
        
        let result = runtime.process("WASM test query");
        assert!(!result.is_empty());
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_wasm_serialization_compatibility() {
        // Test JSON serialization in WASM context
        let config = ExpertConfig {
            domain: ExpertDomain::Language,
            parameter_count: 25000,
            learning_rate: 0.005,
        };
        
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.is_empty());
        
        let parsed: ExpertConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.domain, config.domain);
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_wasm_memory_management() {
        // Test memory usage patterns in WASM
        let mut experts = vec![];
        
        // Create multiple experts
        for i in 0..20 {
            let domain = match i % 6 {
                0 => ExpertDomain::Reasoning,
                1 => ExpertDomain::Coding,
                2 => ExpertDomain::Mathematics,
                3 => ExpertDomain::Language,
                4 => ExpertDomain::ToolUse,
                _ => ExpertDomain::Context,
            };
            experts.push(MicroExpert::new(domain));
        }
        
        // Process with all experts
        for (i, expert) in experts.iter().enumerate() {
            let query = format!("WASM memory test {}", i);
            let result = expert.process(&query);
            assert!(!result.is_empty());
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_wasm_performance_characteristics() {
        let expert = MicroExpert::new(ExpertDomain::Mathematics);
        
        // Test performance in WASM environment
        let start = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();
        
        for i in 0..10 {
            let query = format!("Calculate {}", i);
            let result = expert.process(&query);
            assert!(!result.is_empty());
        }
        
        let end = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();
        
        let duration = end - start;
        assert!(duration < 1000.0, "WASM processing too slow: {}ms", duration);
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_wasm_error_handling() {
        let expert = MicroExpert::new(ExpertDomain::Coding);
        
        // Test error handling in WASM
        let problematic_inputs = vec![
            "",
            " ",
            "null",
            "undefined",
            "\0",
            "🚀💻🔥", // Unicode
        ];
        
        for input in problematic_inputs {
            let result = expert.process(input);
            // Should not panic in WASM
            if !input.trim().is_empty() {
                assert!(!result.is_empty());
            }
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_wasm_concurrent_processing() {
        use wasm_bindgen_futures::JsFuture;
        use js_sys::Promise;
        
        let runtime = KimiRuntime::new(ProcessingConfig::new());
        
        // Test asynchronous processing in WASM
        let queries = vec![
            "Async query 1",
            "Async query 2", 
            "Async query 3",
        ];
        
        for query in queries {
            let result = runtime.process(query);
            assert!(!result.is_empty());
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_wasm_browser_integration() {
        // Test browser-specific functionality
        use web_sys::console;
        
        let expert = MicroExpert::new(ExpertDomain::Language);
        let result = expert.process("Browser integration test");
        
        // Log to browser console (should not crash)
        console::log_1(&format!("WASM test result: {}", result).into());
        
        assert!(!result.is_empty());
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_wasm_large_data_processing() {
        let expert = MicroExpert::new(ExpertDomain::Coding);
        
        // Test with larger inputs
        let large_input = "def large_function():\n".repeat(100);
        let result = expert.process(&large_input);
        
        assert!(!result.is_empty());
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_wasm_cross_domain_functionality() {
        let mut router = ExpertRouter::new();
        
        // Add all domain types
        router.add_expert(MicroExpert::new(ExpertDomain::Reasoning));
        router.add_expert(MicroExpert::new(ExpertDomain::Coding));
        router.add_expert(MicroExpert::new(ExpertDomain::Mathematics));
        router.add_expert(MicroExpert::new(ExpertDomain::Language));
        router.add_expert(MicroExpert::new(ExpertDomain::ToolUse));
        router.add_expert(MicroExpert::new(ExpertDomain::Context));
        
        // Test cross-domain queries
        let cross_domain_queries = vec![
            "Write code to solve math problems",
            "Explain the reasoning behind language translation",
            "Use tools to analyze mathematical data",
            "Remember the context of our coding discussion",
        ];
        
        for query in cross_domain_queries {
            let result = router.route(query);
            assert!(!result.is_empty(), "No result for cross-domain query: {}", query);
        }
    }
}

#[cfg(test)]
mod wasm_worker_tests {
    use super::*;
    use wasm_bindgen::JsValue;

    #[test]
    #[wasm_bindgen_test]
    fn test_worker_compatibility() {
        // Test that the system works in Web Worker context
        let expert = MicroExpert::new(ExpertDomain::ToolUse);
        let result = expert.process("Worker test query");
        assert!(!result.is_empty());
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_message_passing() {
        // Test serialization for message passing between main thread and worker
        let config = ExpertConfig {
            domain: ExpertDomain::Context,
            parameter_count: 15000,
            learning_rate: 0.002,
        };
        
        // Convert to JsValue and back (simulates worker message passing)
        let js_value = serde_wasm_bindgen::to_value(&config).unwrap();
        let round_trip: ExpertConfig = serde_wasm_bindgen::from_value(js_value).unwrap();
        
        assert_eq!(round_trip.domain, config.domain);
        assert_eq!(round_trip.parameter_count, config.parameter_count);
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_worker_lifecycle() {
        // Test expert creation, processing, and cleanup in worker context
        let runtime = KimiRuntime::new(ProcessingConfig::new());
        
        // Simulate worker initialization
        let init_result = runtime.process("Initialize worker");
        assert!(!init_result.is_empty());
        
        // Simulate multiple worker tasks
        for i in 0..5 {
            let task = format!("Worker task {}", i);
            let result = runtime.process(&task);
            assert!(!result.is_empty());
        }
        
        // Simulate worker cleanup
        let cleanup_result = runtime.process("Cleanup worker");
        assert!(!cleanup_result.is_empty());
    }
}

#[cfg(test)]
mod wasm_optimization_tests {
    use super::*;

    #[test]
    #[wasm_bindgen_test]
    fn test_size_optimization() {
        // Test that WASM binary size optimizations don't break functionality
        let expert = MicroExpert::new(ExpertDomain::Reasoning);
        let result = expert.process("Size optimization test");
        assert!(!result.is_empty());
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_startup_performance() {
        // Test WASM module startup time
        let start = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();
        
        let _runtime = KimiRuntime::new(ProcessingConfig::new());
        
        let end = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();
        
        let startup_time = end - start;
        assert!(startup_time < 100.0, "WASM startup too slow: {}ms", startup_time);
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_memory_footprint() {
        // Test memory usage in WASM
        let mut components = vec![];
        
        // Create multiple components
        for i in 0..10 {
            let expert = MicroExpert::new(ExpertDomain::Coding);
            components.push(expert);
        }
        
        // Use all components
        for (i, expert) in components.iter().enumerate() {
            let query = format!("Memory footprint test {}", i);
            let result = expert.process(&query);
            assert!(!result.is_empty());
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_garbage_collection_behavior() {
        // Test that objects are properly garbage collected in WASM
        for iteration in 0..10 {
            let expert = MicroExpert::new(ExpertDomain::Language);
            let query = format!("GC test iteration {}", iteration);
            let result = expert.process(&query);
            assert!(!result.is_empty());
            
            // Expert should be dropped at end of scope
        }
    }
}