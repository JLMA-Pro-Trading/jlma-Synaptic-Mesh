//! Micro-expert implementation for Kimi-K2 WASM conversion

use crate::{domains::ExpertDomain, error::Result};
use ruv_fann::{NeuralNetwork, TrainingData, ActivationFunction, TrainAlgorithm};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use std::time::Instant;
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Configuration for creating a micro-expert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertConfig {
    pub domain: ExpertDomain,
    pub parameter_count: usize,
    pub learning_rate: f32,
    pub activation_function: String,
    pub layers: Vec<usize>,
}

impl Default for ExpertConfig {
    fn default() -> Self {
        Self {
            domain: ExpertDomain::Reasoning,
            parameter_count: 10_000,
            learning_rate: 0.001,
            activation_function: "relu".to_string(),
            layers: vec![128, 64, 32],
        }
    }
}

/// Runtime parameters for expert execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertParams {
    pub temperature: f32,
    pub max_tokens: usize,
    pub top_p: f32,
    pub frequency_penalty: f32,
}

impl Default for ExpertParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            max_tokens: 1024,
            top_p: 0.9,
            frequency_penalty: 0.0,
        }
    }
}

/// Performance metrics for expert execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertMetrics {
    pub inference_time_ms: u32,
    pub memory_usage_bytes: usize,
    pub confidence_score: f32,
    pub accuracy: f32,
    pub total_invocations: u64,
}

impl Default for ExpertMetrics {
    fn default() -> Self {
        Self {
            inference_time_ms: 0,
            memory_usage_bytes: 0,
            confidence_score: 0.0,
            accuracy: 0.0,
            total_invocations: 0,
        }
    }
}

/// A micro-expert neural network specialized for a specific domain
#[wasm_bindgen]
pub struct MicroExpert {
    domain: ExpertDomain,
    config: ExpertConfig,
    params: ExpertParams,
    metrics: ExpertMetrics,
    #[wasm_bindgen(skip)]
    pub network: Option<NeuralNetwork>,
    #[wasm_bindgen(skip)]
    pub training_data: Option<TrainingData>,
    #[wasm_bindgen(skip)]
    performance_history: Vec<f64>,
    #[wasm_bindgen(skip)]
    last_prediction_time: Option<Instant>,
}

#[wasm_bindgen]
impl MicroExpert {
    /// Create a new micro-expert with default configuration
    #[wasm_bindgen(constructor)]
    pub fn new(domain: ExpertDomain) -> Result<MicroExpert> {
        let config = ExpertConfig {
            domain,
            parameter_count: domain.default_parameter_count(),
            ..Default::default()
        };
        
        let mut expert = Self {
            domain,
            config: config.clone(),
            params: ExpertParams::default(),
            metrics: ExpertMetrics::default(),
            network: None,
            training_data: None,
            performance_history: Vec::new(),
            last_prediction_time: None,
        };
        
        // Initialize the neural network
        expert.initialize_network()?;
        Ok(expert)
    }

    /// Create a micro-expert with custom configuration
    #[wasm_bindgen]
    pub fn with_config(config: &JsValue) -> Result<MicroExpert> {
        let config: ExpertConfig = serde_wasm_bindgen::from_value(config.clone())
            .map_err(|e| crate::error::KimiError::configuration(format!("Invalid config: {}", e)))?;
        
        let mut expert = Self {
            domain: config.domain,
            config: config.clone(),
            params: ExpertParams::default(),
            metrics: ExpertMetrics::default(),
            network: None,
            training_data: None,
            performance_history: Vec::new(),
            last_prediction_time: None,
        };
        
        // Initialize the neural network
        expert.initialize_network()?;
        Ok(expert)
    }

    /// Get the expert's domain
    #[wasm_bindgen(getter)]
    pub fn domain(&self) -> ExpertDomain {
        self.domain
    }

    /// Get the parameter count
    #[wasm_bindgen(getter = parameter_count)]
    pub fn parameter_count(&self) -> usize {
        self.config.parameter_count
    }

    /// Predict/inference with the micro-expert
    #[wasm_bindgen]
    pub fn predict(&mut self, input: Vec<f32>) -> Result<Vec<f32>> {
        let start_time = Instant::now();
        self.last_prediction_time = Some(start_time);
        
        let network = self.network.as_ref()
            .ok_or_else(|| crate::error::KimiError::neural_network("Neural network not initialized"))?;
        
        // Validate input size
        let input_size = network.get_num_input();
        if input.len() != input_size {
            return Err(crate::error::KimiError::neural_network(
                format!("Input size mismatch: expected {}, got {}", input_size, input.len())
            ));
        }
        
        // Convert to f64 for ruv-FANN
        let input_f64: Vec<f64> = input.iter().map(|&x| x as f64).collect();
        
        // Run neural network inference
        let output_f64 = network.run(&input_f64)
            .map_err(|e| crate::error::KimiError::neural_network(format!("Prediction failed: {}", e)))?;
        
        // Convert back to f32
        let output: Vec<f32> = output_f64.iter().map(|&x| x as f32).collect();
        
        // Calculate confidence based on output distribution
        let confidence = self.calculate_confidence(&output);
        
        // Update metrics
        let inference_time = start_time.elapsed();
        self.metrics.inference_time_ms = inference_time.as_millis() as u32;
        self.metrics.total_invocations += 1;
        self.metrics.confidence_score = confidence;
        self.metrics.memory_usage_bytes = self.estimate_memory_usage();
        
        // Store performance history
        self.performance_history.push(confidence as f64);
        if self.performance_history.len() > 1000 {
            self.performance_history.remove(0);
        }
        
        // Apply temperature scaling if needed
        let scaled_output = if self.params.temperature != 1.0 {
            self.apply_temperature_scaling(output, self.params.temperature)
        } else {
            output
        };
        
        Ok(scaled_output)
    }

    /// Get confidence score for the last prediction
    #[wasm_bindgen(getter = confidence)]
    pub fn get_confidence(&self) -> f32 {
        self.metrics.confidence_score
    }

    /// Get performance metrics
    #[wasm_bindgen]
    pub fn get_metrics(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.metrics).unwrap_or(JsValue::NULL)
    }

    /// Update runtime parameters
    #[wasm_bindgen]
    pub fn set_params(&mut self, params: &JsValue) -> Result<()> {
        self.params = serde_wasm_bindgen::from_value(params.clone())
            .map_err(|e| crate::error::KimiError::configuration(format!("Invalid params: {}", e)))?;
        Ok(())
    }
}

impl MicroExpert {
    /// Create expert with native Rust interface
    pub fn new_with_config(config: ExpertConfig) -> Result<Self> {
        let mut expert = Self {
            domain: config.domain,
            config: config.clone(),
            params: ExpertParams::default(),
            metrics: ExpertMetrics::default(),
            network: None,
            training_data: None,
            performance_history: Vec::new(),
            last_prediction_time: None,
        };
        
        expert.initialize_network()?;
        Ok(expert)
    }
    
    /// Initialize the neural network based on configuration
    fn initialize_network(&mut self) -> Result<()> {
        let layers = &self.config.layers;
        if layers.len() < 2 {
            return Err(crate::error::KimiError::neural_network(
                "At least 2 layers (input and output) are required"
            ));
        }
        
        // Create network topology
        let mut network = NeuralNetwork::new(&layers)
            .map_err(|e| crate::error::KimiError::neural_network(format!("Failed to create network: {}", e)))?;
        
        // Set activation function
        let activation = match self.config.activation_function.as_str() {
            "relu" => ActivationFunction::Linear, // ruv-FANN doesn't have ReLU, use linear approximation
            "sigmoid" => ActivationFunction::Sigmoid,
            "tanh" => ActivationFunction::Tanh,
            "linear" => ActivationFunction::Linear,
            _ => ActivationFunction::Sigmoid, // Default fallback
        };
        
        network.set_activation_function_hidden(activation);
        network.set_activation_function_output(activation);
        
        // Set learning parameters
        network.set_learning_rate(self.config.learning_rate as f64);
        network.set_train_algorithm(TrainAlgorithm::BackProp);
        
        // Randomize weights
        network.randomize_weights(-0.1, 0.1);
        
        self.network = Some(network);
        
        log::info!("Initialized {} expert with {} parameters", 
                  self.domain, self.config.parameter_count);
        
        Ok(())
    }
    
    /// Calculate confidence score based on output distribution
    fn calculate_confidence(&self, output: &[f32]) -> f32 {
        if output.is_empty() {
            return 0.0;
        }
        
        // For single output, use absolute value
        if output.len() == 1 {
            return output[0].abs().min(1.0);
        }
        
        // For multiple outputs, calculate entropy-based confidence
        let sum: f32 = output.iter().map(|&x| x.abs()).sum();
        if sum == 0.0 {
            return 0.0;
        }
        
        let probs: Vec<f32> = output.iter().map(|&x| x.abs() / sum).collect();
        let entropy: f32 = -probs.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum();
        
        let max_entropy = (output.len() as f32).ln();
        if max_entropy == 0.0 {
            return 1.0;
        }
        
        1.0 - (entropy / max_entropy)
    }
    
    /// Apply temperature scaling to outputs
    fn apply_temperature_scaling(&self, output: Vec<f32>, temperature: f32) -> Vec<f32> {
        if temperature <= 0.0 {
            return output;
        }
        
        output.iter().map(|&x| x / temperature).collect()
    }
    
    /// Estimate current memory usage
    fn estimate_memory_usage(&self) -> usize {
        let base_size = std::mem::size_of::<Self>();
        let network_size = if let Some(ref network) = self.network {
            // Estimate network memory usage
            let weights = network.get_total_connections() * std::mem::size_of::<f64>();
            let neurons = network.get_total_neurons() * std::mem::size_of::<f64>();
            weights + neurons
        } else {
            0
        };
        let history_size = self.performance_history.len() * std::mem::size_of::<f64>();
        
        base_size + network_size + history_size
    }
    
    /// Train the expert with provided data
    pub fn train(&mut self, training_inputs: &[Vec<f32>], training_outputs: &[Vec<f32>], epochs: u32) -> Result<f32> {
        let network = self.network.as_mut()
            .ok_or_else(|| crate::error::KimiError::neural_network("Neural network not initialized"))?;
        
        if training_inputs.len() != training_outputs.len() {
            return Err(crate::error::KimiError::neural_network(
                "Training inputs and outputs must have the same length"
            ));
        }
        
        // Convert training data to f64
        let mut train_data = Vec::new();
        for (input, output) in training_inputs.iter().zip(training_outputs.iter()) {
            let input_f64: Vec<f64> = input.iter().map(|&x| x as f64).collect();
            let output_f64: Vec<f64> = output.iter().map(|&x| x as f64).collect();
            train_data.push((input_f64, output_f64));
        }
        
        // Create training data object
        let mut fann_data = TrainingData::new();
        for (input, output) in &train_data {
            fann_data.add_train_data(input, output)
                .map_err(|e| crate::error::KimiError::neural_network(format!("Failed to add training data: {}", e)))?;
        }
        
        // Train the network
        let mse = network.train_on_data(&fann_data, epochs, 0, 0.001)
            .map_err(|e| crate::error::KimiError::neural_network(format!("Training failed: {}", e)))?;
        
        self.training_data = Some(fann_data);
        
        log::info!("Training completed for {} expert: MSE = {:.6}", self.domain, mse);
        
        Ok(mse as f32)
    }
    
    /// Get performance statistics
    pub fn get_performance_stats(&self) -> (f32, f32, u64) {
        let avg_confidence = if self.performance_history.is_empty() {
            0.0
        } else {
            self.performance_history.iter().sum::<f64>() / self.performance_history.len() as f64
        };
        
        let accuracy = self.metrics.accuracy;
        let invocations = self.metrics.total_invocations;
        
        (avg_confidence as f32, accuracy, invocations)
    }
    
    /// Save network to binary data
    pub fn save_network(&self) -> Result<Vec<u8>> {
        let network = self.network.as_ref()
            .ok_or_else(|| crate::error::KimiError::neural_network("Neural network not initialized"))?;
        
        // For now, serialize to JSON (would use binary format in production)
        let serialized = serde_json::to_vec(&self.config)
            .map_err(|e| crate::error::KimiError::neural_network(format!("Serialization failed: {}", e)))?;
        
        Ok(serialized)
    }
    
    /// Load network from binary data
    pub fn load_network(&mut self, data: &[u8]) -> Result<()> {
        let config: ExpertConfig = serde_json::from_slice(data)
            .map_err(|e| crate::error::KimiError::neural_network(format!("Deserialization failed: {}", e)))?;
        
        self.config = config;
        self.initialize_network()?;
        
        Ok(())
    }

    /// Get the current configuration
    pub fn config(&self) -> &ExpertConfig {
        &self.config
    }

    /// Get the current parameters
    pub fn params(&self) -> &ExpertParams {
        &self.params
    }

    /// Get the current metrics
    pub fn metrics(&self) -> &ExpertMetrics {
        &self.metrics
    }

    /// Update the configuration
    pub fn update_config(&mut self, config: ExpertConfig) -> Result<()> {
        self.config = config;
        self.initialize_network()?;
        Ok(())
    }

    /// Update the parameters
    pub fn update_params(&mut self, params: ExpertParams) {
        self.params = params;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_creation() {
        let expert = MicroExpert::new(ExpertDomain::Reasoning).unwrap();
        assert_eq!(expert.domain(), ExpertDomain::Reasoning);
        assert_eq!(expert.parameter_count(), 10_000);
    }

    #[test]
    fn test_expert_prediction() {
        let mut expert = MicroExpert::new(ExpertDomain::Coding).unwrap();
        let input = vec![1.0, 2.0, 3.0];
        let output = expert.predict(input.clone()).unwrap();
        
        assert_eq!(output.len(), input.len());
        assert!(expert.get_confidence() > 0.0);
    }

    #[test]
    fn test_custom_config() {
        let config = ExpertConfig {
            domain: ExpertDomain::Mathematics,
            parameter_count: 15_000,
            learning_rate: 0.01,
            activation_function: "tanh".to_string(),
            layers: vec![64, 32, 16],
        };
        
        let expert = MicroExpert::new_with_config(config.clone()).unwrap();
        assert_eq!(expert.config().parameter_count, 15_000);
        assert_eq!(expert.config().learning_rate, 0.01);
    }
}