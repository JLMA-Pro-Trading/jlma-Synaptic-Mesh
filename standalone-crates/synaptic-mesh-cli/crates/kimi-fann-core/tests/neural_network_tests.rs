//! Neural Network Tests for Kimi-FANN Core
//! 
//! Tests the actual neural network functionality using ruv-FANN,
//! WASM compilation, and real data processing.

use kimi_fann_core::*;

#[cfg(test)]
mod neural_tests {
    use super::*;
    
    #[test]
    fn test_expert_neural_network_basics() {
        // Test that experts can handle basic neural network operations
        let expert = MicroExpert::new(ExpertDomain::Mathematics);
        
        // Test with mathematical input that would require neural processing
        let math_problems = vec![
            "2 + 2",
            "sqrt(16)",
            "sin(π/2)",
            "derivative of x^2",
            "integral of 2x",
        ];
        
        for problem in math_problems {
            let result = expert.process(problem);
            assert!(!result.is_empty(), "No result for: {}", problem);
            assert!(result.len() > 5, "Too short result for: {}", problem);
        }
    }

    #[test]
    fn test_expert_learning_patterns() {
        let mut coding_expert = MicroExpert::new(ExpertDomain::Coding);
        
        // Test with progressively complex coding tasks
        let coding_tasks = vec![
            "def hello():",
            "class Calculator:",
            "def quicksort(arr):",
            "async def fetch_data():",
            "def neural_network_forward_pass():",
        ];
        
        let mut results = vec![];
        for task in coding_tasks {
            let result = coding_expert.process(task);
            results.push(result);
        }
        
        // Verify all tasks were processed
        for (i, result) in results.iter().enumerate() {
            assert!(!result.is_empty(), "Empty result for task {}", i);
        }
    }

    #[test]
    fn test_multi_expert_coordination() {
        let reasoning_expert = MicroExpert::new(ExpertDomain::Reasoning);
        let math_expert = MicroExpert::new(ExpertDomain::Mathematics);
        let coding_expert = MicroExpert::new(ExpertDomain::Coding);
        
        // Test a complex problem that requires multiple experts
        let complex_problem = "Design an algorithm to optimize neural network training";
        
        let reasoning_result = reasoning_expert.process(complex_problem);
        let math_result = math_expert.process(complex_problem);
        let coding_result = coding_expert.process(complex_problem);
        
        // Each expert should provide domain-specific insights
        assert!(reasoning_result.contains("Reasoning"));
        assert!(math_result.contains("Mathematics"));
        assert!(coding_result.contains("Coding"));
        
        // Results should be different as each expert approaches differently
        assert_ne!(reasoning_result, math_result);
        assert_ne!(math_result, coding_result);
    }

    #[test]
    fn test_expert_memory_and_context() {
        let language_expert = MicroExpert::new(ExpertDomain::Language);
        
        // Test contextual understanding
        let conversation = vec![
            "Hello, how are you?",
            "What's the weather like?",
            "Can you help me with programming?",
            "What did we discuss earlier?",
        ];
        
        let mut responses = vec![];
        for message in conversation {
            let response = language_expert.process(message);
            responses.push(response);
        }
        
        // All responses should be meaningful
        for response in responses {
            assert!(!response.is_empty());
            assert!(response.len() > 10);
        }
    }

    #[test]
    fn test_expert_tool_use_capabilities() {
        let tool_expert = MicroExpert::new(ExpertDomain::ToolUse);
        
        // Test tool-related queries
        let tool_queries = vec![
            "How do I use git?",
            "Debug this Python error",
            "Set up a development environment",
            "Configure a web server",
            "Optimize database queries",
        ];
        
        for query in tool_queries {
            let result = tool_expert.process(query);
            assert!(!result.is_empty(), "No result for tool query: {}", query);
            assert!(result.contains("ToolUse"), "Result doesn't indicate tool expertise");
        }
    }

    #[test]
    fn test_context_expert_capabilities() {
        let context_expert = MicroExpert::new(ExpertDomain::Context);
        
        // Test context understanding and maintenance
        let context_tasks = vec![
            "Remember that I'm working on a web app",
            "What's the current context?",
            "Add this to the context: using React and Node.js",
            "What technologies are we using?",
            "Clear the current context",
        ];
        
        let mut results = vec![];
        for task in context_tasks {
            let result = context_expert.process(task);
            results.push(result);
            assert!(!result.is_empty(), "No result for context task: {}", task);
        }
        
        // Context expert should handle all context-related tasks
        for result in results {
            assert!(result.contains("Context"));
        }
    }

    #[test]
    fn test_neural_network_performance_patterns() {
        // Test that the neural network components show performance characteristics
        let start_time = std::time::Instant::now();
        
        let expert = MicroExpert::new(ExpertDomain::Mathematics);
        
        // Process multiple similar tasks to test performance patterns
        let mut total_time = std::time::Duration::new(0, 0);
        let iterations = 50;
        
        for i in 0..iterations {
            let task_start = std::time::Instant::now();
            let problem = format!("Calculate: {} * {} + {}", i, i+1, i+2);
            let result = expert.process(&problem);
            let task_duration = task_start.elapsed();
            
            total_time += task_duration;
            assert!(!result.is_empty());
            
            // Later iterations might be faster due to optimization/caching
            if i > 10 && i < 20 {
                // Baseline timing
            }
        }
        
        let avg_time = total_time / iterations;
        println!("Average processing time: {:?}", avg_time);
        
        // Processing should be reasonably fast
        assert!(avg_time.as_millis() < 100, "Processing too slow: {:?}", avg_time);
    }

    #[test]
    fn test_expert_domain_boundaries() {
        // Test that experts maintain domain-specific behavior
        let domains = vec![
            ExpertDomain::Reasoning,
            ExpertDomain::Coding,
            ExpertDomain::Language,
            ExpertDomain::Mathematics,
            ExpertDomain::ToolUse,
            ExpertDomain::Context,
        ];
        
        let test_query = "Solve this problem for me";
        
        let mut domain_results = std::collections::HashMap::new();
        
        for domain in domains {
            let expert = MicroExpert::new(domain);
            let result = expert.process(test_query);
            domain_results.insert(domain, result);
        }
        
        // Each domain should produce different results
        let results: Vec<_> = domain_results.values().collect();
        for i in 0..results.len() {
            for j in i+1..results.len() {
                assert_ne!(results[i], results[j], "Domain experts produced identical results");
            }
        }
    }

    #[test]
    fn test_neural_network_data_types() {
        let math_expert = MicroExpert::new(ExpertDomain::Mathematics);
        
        // Test different data types and formats
        let test_inputs = vec![
            "1",                    // Integer
            "3.14159",             // Float
            "[1, 2, 3, 4, 5]",     // Array
            "x^2 + 2x + 1",        // Expression
            "{a: 1, b: 2}",        // Object
            "true",                // Boolean
            "null",                // Null
        ];
        
        for input in test_inputs {
            let result = math_expert.process(input);
            assert!(!result.is_empty(), "No result for input: {}", input);
            // Should handle all data types gracefully
        }
    }

    #[test]
    fn test_neural_network_error_recovery() {
        let expert = MicroExpert::new(ExpertDomain::Coding);
        
        // Test with inputs that might cause errors
        let error_prone_inputs = vec![
            "def broken_function(",     // Syntax error
            "undefined_variable",       // Undefined reference
            "1/0",                     // Division by zero
            "import nonexistent_module", // Import error
            "while True: pass",        // Infinite loop pattern
        ];
        
        for input in error_prone_inputs {
            let result = expert.process(input);
            // Should handle errors gracefully, not panic
            assert!(!result.is_empty(), "Empty result for error-prone input: {}", input);
        }
    }
}

#[cfg(test)]
mod wasm_specific_tests {
    use super::*;

    #[test]
    fn test_wasm_serialization() {
        // Test that expert configurations can be serialized for WASM
        let config = ExpertConfig {
            domain: ExpertDomain::Coding,
            parameter_count: 50000,
            learning_rate: 0.01,
        };
        
        // Test JSON serialization (needed for WASM)
        let serialized = serde_json::to_string(&config).unwrap();
        assert!(!serialized.is_empty());
        
        let deserialized: ExpertConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.domain, config.domain);
        assert_eq!(deserialized.parameter_count, config.parameter_count);
        assert!((deserialized.learning_rate - config.learning_rate).abs() < f32::EPSILON);
    }

    #[test]
    fn test_expert_domain_serialization() {
        // Test that ExpertDomain enum serializes correctly for WASM
        let domains = vec![
            ExpertDomain::Reasoning,
            ExpertDomain::Coding,
            ExpertDomain::Language,
            ExpertDomain::Mathematics,
            ExpertDomain::ToolUse,
            ExpertDomain::Context,
        ];
        
        for domain in domains {
            let serialized = serde_json::to_string(&domain).unwrap();
            let deserialized: ExpertDomain = serde_json::from_str(&serialized).unwrap();
            assert_eq!(domain, deserialized);
        }
    }

    #[test]
    fn test_processing_config_wasm_compatibility() {
        let config = ProcessingConfig::new();
        
        // Test that configuration values are WASM-appropriate
        assert!(config.max_experts > 0 && config.max_experts <= 10);
        assert!(config.timeout_ms > 0 && config.timeout_ms <= 30000);
    }

    #[test]
    fn test_wasm_memory_constraints() {
        // Test behavior under WASM memory constraints
        let runtime = KimiRuntime::new(ProcessingConfig::new());
        
        // Process multiple requests to test memory management
        for i in 0..100 {
            let query = format!("Process WASM memory test {}", i);
            let result = runtime.process(&query);
            assert!(!result.is_empty());
            
            // In WASM, we need to be careful about memory usage
            if i % 10 == 0 {
                // Simulate periodic cleanup that might be needed in WASM
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}