//! Async processing and Web Workers support for browser-compatible execution

use crate::{
    domains::ExpertDomain, 
    expert::MicroExpert, 
    enhanced_router::{EnhancedExpertRouter, EnhancedRequestContext, EnhancedExpertSelection},
    memory::ExpertMemoryManager,
    error::Result
};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{future_to_promise, JsFuture};
use js_sys::{Promise, Array, Object, Reflect};
use web_sys::{Worker, MessageEvent, DedicatedWorkerGlobalScope, console};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use futures::future::{self, BoxFuture};
use std::rc::Rc;
use std::cell::RefCell;

/// Task types for async processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AsyncTaskType {
    ExpertInference,
    ModelTraining,
    MemoryCompression,
    PerformanceAnalysis,
    DataPreprocessing,
}

/// Async task configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncTask {
    pub id: String,
    pub task_type: AsyncTaskType,
    pub priority: u8,
    pub timeout_ms: u32,
    pub retry_count: u8,
    pub payload: serde_json::Value,
}

/// Task execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: String,
    pub success: bool,
    pub result: serde_json::Value,
    pub execution_time_ms: u32,
    pub memory_used: usize,
    pub error_message: Option<String>,
}

/// Worker pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerPoolConfig {
    pub max_workers: usize,
    pub task_queue_size: usize,
    pub worker_timeout_ms: u32,
    pub enable_shared_array_buffer: bool,
    pub worker_script_url: String,
}

impl Default for WorkerPoolConfig {
    fn default() -> Self {
        Self {
            max_workers: 4,
            task_queue_size: 100,
            worker_timeout_ms: 30000,
            enable_shared_array_buffer: false,
            worker_script_url: "kimi-worker.js".to_string(),
        }
    }
}

/// Async processing engine with Web Workers support
#[wasm_bindgen]
pub struct AsyncProcessor {
    #[wasm_bindgen(skip)]
    config: WorkerPoolConfig,
    #[wasm_bindgen(skip)]
    workers: Vec<Rc<RefCell<WorkerInstance>>>,
    #[wasm_bindgen(skip)]
    task_queue: Vec<AsyncTask>,
    #[wasm_bindgen(skip)]
    active_tasks: HashMap<String, AsyncTask>,
    #[wasm_bindgen(skip)]
    completed_tasks: HashMap<String, TaskResult>,
    #[wasm_bindgen(skip)]
    router: Rc<RefCell<EnhancedExpertRouter>>,
    #[wasm_bindgen(skip)]
    memory_manager: Rc<RefCell<ExpertMemoryManager>>,
    task_counter: u32,
}

/// Individual worker instance
#[derive(Debug)]
struct WorkerInstance {
    worker: Worker,
    is_busy: bool,
    current_task: Option<String>,
    created_at: js_sys::Date,
}

#[wasm_bindgen]
impl AsyncProcessor {
    /// Create new async processor
    #[wasm_bindgen(constructor)]
    pub fn new(router: EnhancedExpertRouter, memory_manager: ExpertMemoryManager) -> Result<AsyncProcessor> {
        let config = WorkerPoolConfig::default();
        let mut processor = Self {
            config: config.clone(),
            workers: Vec::new(),
            task_queue: Vec::new(),
            active_tasks: HashMap::new(),
            completed_tasks: HashMap::new(),
            router: Rc::new(RefCell::new(router)),
            memory_manager: Rc::new(RefCell::new(memory_manager)),
            task_counter: 0,
        };
        
        // Initialize worker pool
        processor.initialize_workers()?;
        
        Ok(processor)
    }
    
    /// Initialize Web Workers pool
    fn initialize_workers(&mut self) -> Result<()> {
        for i in 0..self.config.max_workers {
            match self.create_worker(i) {
                Ok(worker_instance) => {
                    self.workers.push(Rc::new(RefCell::new(worker_instance)));
                },
                Err(e) => {
                    log::warn!("Failed to create worker {}: {}", i, e);
                    // Continue with fewer workers rather than failing completely
                }
            }
        }
        
        if self.workers.is_empty() {
            log::warn!("No Web Workers available, falling back to main thread execution");
        } else {
            log::info!("Initialized {} Web Workers for async processing", self.workers.len());
        }
        
        Ok(())
    }
    
    /// Create individual worker
    fn create_worker(&self, worker_id: usize) -> Result<WorkerInstance> {
        let worker = Worker::new(&self.config.worker_script_url)
            .map_err(|e| crate::error::KimiError::wasm_runtime(format!("Worker creation failed: {:?}", e)))?;
        
        // Set up worker message handling
        let worker_clone = worker.clone();
        let closure = Closure::wrap(Box::new(move |event: MessageEvent| {
            Self::handle_worker_message(event);
        }) as Box<dyn FnMut(_)>);
        
        worker.set_onmessage(Some(closure.as_ref().unchecked_ref()));
        closure.forget(); // Keep closure alive
        
        let worker_instance = WorkerInstance {
            worker,
            is_busy: false,
            current_task: None,
            created_at: js_sys::Date::new_0(),
        };
        
        log::debug!("Created worker {} successfully", worker_id);
        
        Ok(worker_instance)
    }
    
    /// Handle worker messages
    fn handle_worker_message(event: MessageEvent) {
        if let Ok(data) = event.data().dyn_into::<js_sys::Object>() {
            // Process worker response
            if let Ok(task_id) = Reflect::get(&data, &"taskId".into()) {
                if let Some(task_id_str) = task_id.as_string() {
                    log::debug!("Received worker response for task: {}", task_id_str);
                    // Additional processing logic would go here
                }
            }
        }
    }
    
    /// Submit async task for processing
    #[wasm_bindgen]
    pub fn submit_task(&mut self, task_type: &str, payload: &JsValue, priority: u8) -> Result<String> {
        let task_id = format!("task_{}", self.task_counter);
        self.task_counter += 1;
        
        let task_type_enum = match task_type {
            "inference" => AsyncTaskType::ExpertInference,
            "training" => AsyncTaskType::ModelTraining,
            "compression" => AsyncTaskType::MemoryCompression,
            "analysis" => AsyncTaskType::PerformanceAnalysis,
            "preprocessing" => AsyncTaskType::DataPreprocessing,
            _ => return Err(crate::error::KimiError::configuration(format!("Unknown task type: {}", task_type))),
        };
        
        let payload_json = serde_wasm_bindgen::from_value(payload.clone())
            .map_err(|e| crate::error::KimiError::configuration(format!("Invalid payload: {}", e)))?;
        
        let task = AsyncTask {
            id: task_id.clone(),
            task_type: task_type_enum,
            priority,
            timeout_ms: self.config.worker_timeout_ms,
            retry_count: 0,
            payload: payload_json,
        };
        
        // Add to queue
        self.task_queue.push(task);
        self.task_queue.sort_by(|a, b| b.priority.cmp(&a.priority)); // Higher priority first
        
        // Try to execute immediately if workers available
        if let Err(e) = self.try_execute_queued_tasks() {
            log::warn!("Failed to execute queued tasks: {}", e);
        }
        
        log::debug!("Submitted task {} with priority {}", task_id, priority);
        
        Ok(task_id)
    }
    
    /// Try to execute queued tasks
    fn try_execute_queued_tasks(&mut self) -> Result<()> {
        while !self.task_queue.is_empty() {
            // Find available worker
            let available_worker = self.workers.iter()
                .position(|worker| !worker.borrow().is_busy);
            
            if let Some(worker_idx) = available_worker {
                let task = self.task_queue.remove(0);
                let task_id = task.id.clone();
                
                // Execute task
                match self.execute_task_on_worker(worker_idx, task) {
                    Ok(_) => {
                        log::debug!("Started execution of task {} on worker {}", task_id, worker_idx);
                    },
                    Err(e) => {
                        log::error!("Failed to execute task {} on worker {}: {}", task_id, worker_idx, e);
                        // Re-queue task for retry
                        self.requeue_task_for_retry(task_id)?;
                    }
                }
            } else {
                // No available workers
                break;
            }
        }
        
        Ok(())
    }
    
    /// Execute task on specific worker
    fn execute_task_on_worker(&mut self, worker_idx: usize, task: AsyncTask) -> Result<()> {
        if worker_idx >= self.workers.len() {
            return Err(crate::error::KimiError::configuration("Invalid worker index"));
        }
        
        let worker_rc = self.workers[worker_idx].clone();
        let mut worker = worker_rc.borrow_mut();
        
        if worker.is_busy {
            return Err(crate::error::KimiError::configuration("Worker is busy"));
        }
        
        // Check if we should use Web Worker or main thread
        if self.should_use_main_thread(&task) {
            // Execute on main thread
            drop(worker); // Release borrow
            self.execute_task_main_thread(task)
        } else {
            // Execute on Web Worker
            let message = self.create_worker_message(&task)?;
            
            worker.worker.post_message(&message)
                .map_err(|e| crate::error::KimiError::wasm_runtime(format!("Worker message failed: {:?}", e)))?;
            
            worker.is_busy = true;
            worker.current_task = Some(task.id.clone());
            
            self.active_tasks.insert(task.id.clone(), task);
            
            Ok(())
        }
    }
    
    /// Determine if task should run on main thread
    fn should_use_main_thread(&self, task: &AsyncTask) -> bool {
        // Use main thread for quick tasks or when no workers available
        match task.task_type {
            AsyncTaskType::ExpertInference => false, // Can benefit from worker
            AsyncTaskType::ModelTraining => false,   // Definitely use worker
            AsyncTaskType::MemoryCompression => false, // Use worker
            AsyncTaskType::PerformanceAnalysis => true, // Quick, use main thread
            AsyncTaskType::DataPreprocessing => false, // Use worker
        }
    }
    
    /// Execute task on main thread
    fn execute_task_main_thread(&mut self, task: AsyncTask) -> Result<()> {
        let start_time = js_sys::Date::now();
        
        let result = match task.task_type {
            AsyncTaskType::ExpertInference => self.execute_inference_task(&task),
            AsyncTaskType::ModelTraining => self.execute_training_task(&task),
            AsyncTaskType::MemoryCompression => self.execute_compression_task(&task),
            AsyncTaskType::PerformanceAnalysis => self.execute_analysis_task(&task),
            AsyncTaskType::DataPreprocessing => self.execute_preprocessing_task(&task),
        };
        
        let execution_time = (js_sys::Date::now() - start_time) as u32;
        
        let task_result = match result {
            Ok(result_data) => TaskResult {
                task_id: task.id.clone(),
                success: true,
                result: result_data,
                execution_time_ms: execution_time,
                memory_used: 0, // Would calculate actual memory usage
                error_message: None,
            },
            Err(e) => TaskResult {
                task_id: task.id.clone(),
                success: false,
                result: serde_json::Value::Null,
                execution_time_ms: execution_time,
                memory_used: 0,
                error_message: Some(e.to_string()),
            },
        };
        
        self.completed_tasks.insert(task.id.clone(), task_result);
        
        log::debug!("Completed main thread task {} in {}ms", task.id, execution_time);
        
        Ok(())
    }
    
    /// Execute expert inference task
    fn execute_inference_task(&mut self, task: &AsyncTask) -> Result<serde_json::Value> {
        // Extract input data from payload
        let prompt = task.payload.get("prompt")
            .and_then(|p| p.as_str())
            .ok_or_else(|| crate::error::KimiError::configuration("Missing prompt in inference task"))?;
        
        let max_experts = task.payload.get("max_experts")
            .and_then(|e| e.as_u64())
            .unwrap_or(3) as usize;
        
        // Create request context
        let mut context = EnhancedRequestContext::new(prompt);
        context.max_experts = max_experts;
        
        // Perform routing
        let mut router = self.router.borrow_mut();
        let selections = router.route_with_ml(&context)
            .map_err(|e| crate::error::KimiError::routing(format!("Inference routing failed: {}", e)))?;
        
        // Process with selected experts
        let mut results = Vec::new();
        for selection in selections {
            // Load expert
            let mut memory_manager = self.memory_manager.borrow_mut();
            if memory_manager.load_expert(selection.domain)? {
                if let Some(expert) = memory_manager.get_expert_mut(selection.domain)? {
                    // Simulate expert inference
                    let input_tokens: Vec<f32> = prompt.chars()
                        .map(|c| c as u32 as f32 / 1000.0)
                        .take(32)
                        .collect();
                    
                    let output = expert.predict(input_tokens)?;
                    
                    results.push(serde_json::json!({
                        "domain": selection.domain.to_string(),
                        "confidence": selection.confidence,
                        "output_size": output.len(),
                        "processing_time_ms": selection.latency_estimate_ms
                    }));
                }
            }
        }
        
        Ok(serde_json::json!({
            "inference_results": results,
            "total_experts": results.len()
        }))
    }
    
    /// Execute model training task
    fn execute_training_task(&mut self, task: &AsyncTask) -> Result<serde_json::Value> {
        // Extract training parameters
        let domain_str = task.payload.get("domain")
            .and_then(|d| d.as_str())
            .ok_or_else(|| crate::error::KimiError::configuration("Missing domain in training task"))?;
        
        let domain: ExpertDomain = domain_str.parse()
            .map_err(|e| crate::error::KimiError::configuration(format!("Invalid domain: {}", e)))?;
        
        let epochs = task.payload.get("epochs")
            .and_then(|e| e.as_u64())
            .unwrap_or(10) as u32;
        
        // Simulate training data
        let training_inputs = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];
        let training_outputs = vec![
            vec![0.9],
            vec![0.8],
            vec![0.7],
        ];
        
        // Load and train expert
        let mut memory_manager = self.memory_manager.borrow_mut();
        if memory_manager.load_expert(domain)? {
            if let Some(expert) = memory_manager.get_expert_mut(domain)? {
                let mse = expert.train(&training_inputs, &training_outputs, epochs)?;
                
                return Ok(serde_json::json!({
                    "domain": domain.to_string(),
                    "epochs": epochs,
                    "final_mse": mse,
                    "training_samples": training_inputs.len()
                }));
            }
        }
        
        Err(crate::error::KimiError::configuration("Expert not found for training"))
    }
    
    /// Execute memory compression task
    fn execute_compression_task(&mut self, task: &AsyncTask) -> Result<serde_json::Value> {
        let algorithm = task.payload.get("algorithm")
            .and_then(|a| a.as_str())
            .unwrap_or("lz4");
        
        let mut memory_manager = self.memory_manager.borrow_mut();
        
        // Set compression algorithm
        memory_manager.set_compression_algorithm(algorithm)?;
        
        // Trigger memory optimization
        memory_manager.optimize_cache()?;
        memory_manager.defragment_memory()?;
        
        let stats = memory_manager.get_compression_stats();
        
        Ok(serde_json::json!({
            "compression_algorithm": algorithm,
            "optimization_completed": true,
            "compression_stats": stats
        }))
    }
    
    /// Execute performance analysis task
    fn execute_analysis_task(&mut self, task: &AsyncTask) -> Result<serde_json::Value> {
        let analysis_type = task.payload.get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("routing");
        
        match analysis_type {
            "routing" => {
                let router = self.router.borrow();
                let analytics = router.get_enhanced_analytics();
                Ok(serde_wasm_bindgen::from_value(analytics)?)
            },
            "memory" => {
                let memory_manager = self.memory_manager.borrow();
                let stats = memory_manager.get_memory_stats();
                let performance = memory_manager.get_performance_metrics();
                Ok(serde_json::json!({
                    "memory_stats": stats,
                    "performance_metrics": performance
                }))
            },
            _ => Err(crate::error::KimiError::configuration(format!("Unknown analysis type: {}", analysis_type)))
        }
    }
    
    /// Execute data preprocessing task
    fn execute_preprocessing_task(&mut self, task: &AsyncTask) -> Result<serde_json::Value> {
        let data_type = task.payload.get("data_type")
            .and_then(|t| t.as_str())
            .unwrap_or("text");
        
        let input_data = task.payload.get("data")
            .ok_or_else(|| crate::error::KimiError::configuration("Missing data in preprocessing task"))?;
        
        match data_type {
            "text" => {
                if let Some(text) = input_data.as_str() {
                    // Simulate text preprocessing
                    let tokens = text.split_whitespace().collect::<Vec<_>>();
                    let features = self.extract_text_features(text);
                    
                    Ok(serde_json::json!({
                        "token_count": tokens.len(),
                        "features": features,
                        "preprocessed": true
                    }))
                } else {
                    Err(crate::error::KimiError::configuration("Invalid text data"))
                }
            },
            _ => Err(crate::error::KimiError::configuration(format!("Unsupported data type: {}", data_type)))
        }
    }
    
    /// Extract text features for preprocessing
    fn extract_text_features(&self, text: &str) -> serde_json::Value {
        let word_count = text.split_whitespace().count();
        let char_count = text.chars().count();
        let sentence_count = text.split(|c| c == '.' || c == '!' || c == '?').count();
        
        serde_json::json!({
            "word_count": word_count,
            "character_count": char_count,
            "sentence_count": sentence_count,
            "average_word_length": if word_count > 0 { char_count as f32 / word_count as f32 } else { 0.0 }
        })
    }
    
    /// Create worker message
    fn create_worker_message(&self, task: &AsyncTask) -> Result<JsValue> {
        let message = serde_json::json!({
            "taskId": task.id,
            "taskType": format!("{:?}", task.task_type),
            "priority": task.priority,
            "timeout": task.timeout_ms,
            "payload": task.payload
        });
        
        serde_wasm_bindgen::to_value(&message)
            .map_err(|e| crate::error::KimiError::configuration(format!("Message serialization failed: {}", e)))
    }
    
    /// Requeue task for retry
    fn requeue_task_for_retry(&mut self, task_id: String) -> Result<()> {
        if let Some(mut task) = self.active_tasks.remove(&task_id) {
            task.retry_count += 1;
            
            if task.retry_count < 3 {
                // Reduce priority for retry
                task.priority = task.priority.saturating_sub(1);
                self.task_queue.push(task);
                self.task_queue.sort_by(|a, b| b.priority.cmp(&a.priority));
                log::debug!("Requeued task {} for retry (attempt {})", task_id, task.retry_count);
            } else {
                // Mark as failed
                let failed_result = TaskResult {
                    task_id: task_id.clone(),
                    success: false,
                    result: serde_json::Value::Null,
                    execution_time_ms: 0,
                    memory_used: 0,
                    error_message: Some("Max retries exceeded".to_string()),
                };
                self.completed_tasks.insert(task_id, failed_result);
                log::warn!("Task {} failed after {} retries", task_id, task.retry_count);
            }
        }
        
        Ok(())
    }
    
    /// Get task result
    #[wasm_bindgen]
    pub fn get_task_result(&self, task_id: &str) -> JsValue {
        if let Some(result) = self.completed_tasks.get(task_id) {
            serde_wasm_bindgen::to_value(result).unwrap_or(JsValue::NULL)
        } else if self.active_tasks.contains_key(task_id) {
            serde_wasm_bindgen::to_value(&serde_json::json!({
                "status": "running",
                "task_id": task_id
            })).unwrap_or(JsValue::NULL)
        } else if self.task_queue.iter().any(|t| t.id == task_id) {
            serde_wasm_bindgen::to_value(&serde_json::json!({
                "status": "queued",
                "task_id": task_id
            })).unwrap_or(JsValue::NULL)
        } else {
            serde_wasm_bindgen::to_value(&serde_json::json!({
                "status": "not_found",
                "task_id": task_id
            })).unwrap_or(JsValue::NULL)
        }
    }
    
    /// Get processing statistics
    #[wasm_bindgen]
    pub fn get_processing_stats(&self) -> JsValue {
        let stats = serde_json::json!({
            "active_workers": self.workers.len(),
            "busy_workers": self.workers.iter().filter(|w| w.borrow().is_busy).count(),
            "queued_tasks": self.task_queue.len(),
            "active_tasks": self.active_tasks.len(),
            "completed_tasks": self.completed_tasks.len(),
            "queue_capacity": self.config.task_queue_size,
            "worker_timeout_ms": self.config.worker_timeout_ms
        });
        
        serde_wasm_bindgen::to_value(&stats).unwrap_or(JsValue::NULL)
    }
    
    /// Configure worker pool
    #[wasm_bindgen]
    pub fn configure_workers(&mut self, config: &JsValue) -> Result<()> {
        let new_config: WorkerPoolConfig = serde_wasm_bindgen::from_value(config.clone())
            .map_err(|e| crate::error::KimiError::configuration(format!("Invalid config: {}", e)))?;
        
        // Update configuration
        self.config = new_config;
        
        // Reinitialize workers if needed
        if self.workers.len() != self.config.max_workers {
            log::info!("Reinitializing worker pool with {} workers", self.config.max_workers);
            self.workers.clear();
            self.initialize_workers()?;
        }
        
        Ok(())
    }
    
    /// Shutdown async processor
    #[wasm_bindgen]
    pub fn shutdown(&mut self) {
        // Terminate all workers
        for worker_rc in &self.workers {
            let worker = worker_rc.borrow();
            worker.worker.terminate();
        }
        
        self.workers.clear();
        self.task_queue.clear();
        self.active_tasks.clear();
        
        log::info!("Async processor shutdown completed");
    }
}

/// Parallel task execution utilities
#[wasm_bindgen]
pub struct ParallelExecutor;

#[wasm_bindgen]
impl ParallelExecutor {
    /// Execute multiple tasks in parallel
    #[wasm_bindgen]
    pub fn execute_parallel(tasks: &Array) -> Promise {
        let task_promises: Vec<Promise> = (0..tasks.length())
            .filter_map(|i| tasks.get(i).dyn_into::<Promise>().ok())
            .collect();
        
        if task_promises.is_empty() {
            return Promise::resolve(&JsValue::from(Array::new()));
        }
        
        Promise::all(&Array::from_iter(task_promises.iter()))
    }
    
    /// Execute tasks with timeout
    #[wasm_bindgen]
    pub fn execute_with_timeout(task: Promise, timeout_ms: u32) -> Promise {
        let timeout_promise = Self::create_timeout_promise(timeout_ms);
        Promise::race(&Array::of2(&task, &timeout_promise))
    }
    
    /// Create timeout promise
    fn create_timeout_promise(timeout_ms: u32) -> Promise {
        let timeout_ms_val = JsValue::from(timeout_ms);
        js_sys::Promise::new(&mut |resolve, _reject| {
            let timeout_id = web_sys::window()
                .unwrap()
                .set_timeout_with_callback_and_timeout_and_arguments_0(
                    &resolve,
                    timeout_ms as i32
                )
                .unwrap();
            
            // Store timeout ID for potential cleanup
            let _ = timeout_id;
        })
    }
    
    /// Batch process multiple items
    #[wasm_bindgen]
    pub fn batch_process(items: &Array, batch_size: usize, processor_fn: &js_sys::Function) -> Promise {
        future_to_promise(async move {
            let mut results = Array::new();
            let total_items = items.length() as usize;
            
            for batch_start in (0..total_items).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(total_items);
                let batch = Array::new();
                
                for i in batch_start..batch_end {
                    if let Some(item) = items.get(i as u32).as_ref() {
                        batch.push(item);
                    }
                }
                
                // Process batch
                let batch_result = processor_fn.call1(&JsValue::NULL, &batch)
                    .map_err(|e| JsValue::from_str(&format!("Batch processing failed: {:?}", e)))?;
                
                if let Ok(promise) = batch_result.dyn_into::<Promise>() {
                    let awaited_result = JsFuture::from(promise).await?;
                    results.push(&awaited_result);
                } else {
                    results.push(&batch_result);
                }
            }
            
            Ok(results.into())
        })
    }
}

/// Worker communication utilities
#[wasm_bindgen]
pub struct WorkerComm;

#[wasm_bindgen]
impl WorkerComm {
    /// Send message to worker with response handling
    #[wasm_bindgen]
    pub fn send_message_with_response(worker: &Worker, message: &JsValue, timeout_ms: u32) -> Promise {
        let worker_clone = worker.clone();
        let message_clone = message.clone();
        
        future_to_promise(async move {
            let (sender, receiver) = futures::channel::oneshot::channel();
            let sender = Rc::new(RefCell::new(Some(sender)));
            
            // Set up message handler
            let closure = {
                let sender = sender.clone();
                Closure::wrap(Box::new(move |event: MessageEvent| {
                    if let Some(sender) = sender.borrow_mut().take() {
                        let _ = sender.send(event.data());
                    }
                }) as Box<dyn FnMut(_)>)
            };
            
            worker_clone.set_onmessage(Some(closure.as_ref().unchecked_ref()));
            
            // Send message
            worker_clone.post_message(&message_clone)
                .map_err(|e| JsValue::from_str(&format!("Failed to send message: {:?}", e)))?;
            
            // Wait for response with timeout
            let timeout = gloo_timers::future::TimeoutFuture::new(timeout_ms);
            
            match futures::future::select(receiver, timeout).await {
                futures::future::Either::Left((response, _)) => {
                    closure.forget();
                    response.map_err(|e| JsValue::from_str(&format!("Response error: {:?}", e)))
                },
                futures::future::Either::Right((_, _)) => {
                    closure.forget();
                    Err(JsValue::from_str("Worker response timeout"))
                }
            }
        })
    }
    
    /// Broadcast message to multiple workers
    #[wasm_bindgen]
    pub fn broadcast_message(workers: &Array, message: &JsValue) -> Promise {
        let worker_promises: Vec<Promise> = (0..workers.length())
            .filter_map(|i| {
                workers.get(i)
                    .dyn_into::<Worker>()
                    .ok()
                    .map(|worker| {
                        let message_clone = message.clone();
                        Promise::resolve(&message_clone).then(&mut |_| {
                            worker.post_message(&message_clone).unwrap_or(());
                            JsValue::from(true)
                        })
                    })
            })
            .collect();
        
        Promise::all(&Array::from_iter(worker_promises.iter()))
    }
}

// Required for gloo_timers feature
use gloo_timers;