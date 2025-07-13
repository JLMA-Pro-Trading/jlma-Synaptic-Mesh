//! Memory management for Kimi-K2 micro-experts

use crate::{domains::ExpertDomain, expert::MicroExpert, error::Result};
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use wasm_bindgen::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Instant, Duration};
use lz4::{Encoder, Decoder};
use flate2::{write::GzEncoder, read::GzDecoder, Compression};
use std::io::{Write, Read};

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub active_experts: usize,
    pub cached_experts: usize,
    pub cache_hit_rate: f32,
    pub evictions: u64,
}

/// Compression algorithms available
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// LZ4 - Fast compression/decompression
    Lz4,
    /// Gzip - Better compression ratio
    Gzip,
    /// No compression
    None,
}

/// Neural network quantization methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum QuantizationMethod {
    /// 8-bit integer quantization
    Int8,
    /// 16-bit integer quantization
    Int16,
    /// No quantization (32-bit float)
    Float32,
}

/// Compressed expert data for storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedExpert {
    pub domain: ExpertDomain,
    pub compressed_data: Vec<u8>,
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f32,
    pub compression_algorithm: CompressionAlgorithm,
    pub quantization_method: QuantizationMethod,
    pub creation_time: u64,
    pub access_count: u64,
    pub last_access: u64,
}

/// Memory pool for efficient allocation
#[derive(Debug)]
struct MemoryPool {
    pool: Vec<Vec<u8>>,
    available: Vec<usize>,
    allocated: AtomicUsize,
    max_size: usize,
}

impl MemoryPool {
    fn new(max_size: usize) -> Self {
        Self {
            pool: Vec::new(),
            available: Vec::new(),
            allocated: AtomicUsize::new(0),
            max_size,
        }
    }
    
    fn allocate(&mut self, size: usize) -> Result<Vec<u8>> {
        // Try to find existing buffer of sufficient size
        for (i, &existing_size) in self.available.iter().enumerate() {
            if existing_size >= size {
                let buffer = self.pool.swap_remove(i);
                self.available.swap_remove(i);
                return Ok(buffer);
            }
        }
        
        // Check memory limits
        let current = self.allocated.load(Ordering::Relaxed);
        if current + size > self.max_size {
            return Err(crate::error::KimiError::MemoryAllocation { size });
        }
        
        // Allocate new buffer
        self.allocated.fetch_add(size, Ordering::Relaxed);
        Ok(vec![0; size])
    }
    
    fn deallocate(&mut self, buffer: Vec<u8>) {
        let size = buffer.len();
        self.allocated.fetch_sub(size, Ordering::Relaxed);
        self.pool.push(buffer);
        self.available.push(size);
    }
    
    fn usage(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }
}

/// Expert memory manager with LRU caching and compression
#[wasm_bindgen]
pub struct ExpertMemoryManager {
    #[wasm_bindgen(skip)]
    active_experts: LruCache<ExpertDomain, MicroExpert>,
    #[wasm_bindgen(skip)]
    expert_cache: std::collections::HashMap<ExpertDomain, CompressedExpert>,
    max_memory: usize,
    current_memory: usize,
    cache_hits: u64,
    cache_misses: u64,
    evictions: u64,
    #[wasm_bindgen(skip)]
    memory_pool: MemoryPool,
    #[wasm_bindgen(skip)]
    compression_algorithm: CompressionAlgorithm,
    #[wasm_bindgen(skip)]
    quantization_method: QuantizationMethod,
    #[wasm_bindgen(skip)]
    performance_metrics: std::collections::HashMap<ExpertDomain, Vec<Duration>>,
}

#[wasm_bindgen]
impl ExpertMemoryManager {
    /// Create a new memory manager with specified memory limit
    #[wasm_bindgen(constructor)]
    pub fn new(max_memory_mb: usize) -> Self {
        let cache_size = NonZeroUsize::new(10).unwrap(); // Default cache size
        let max_memory_bytes = max_memory_mb * 1024 * 1024;
        Self {
            active_experts: LruCache::new(cache_size),
            expert_cache: std::collections::HashMap::new(),
            max_memory: max_memory_bytes,
            current_memory: 0,
            cache_hits: 0,
            cache_misses: 0,
            evictions: 0,
            memory_pool: MemoryPool::new(max_memory_bytes),
            compression_algorithm: CompressionAlgorithm::Lz4,
            quantization_method: QuantizationMethod::Float32,
            performance_metrics: std::collections::HashMap::new(),
        }
    }
    
    /// Set compression algorithm
    #[wasm_bindgen]
    pub fn set_compression_algorithm(&mut self, algorithm: &str) -> Result<()> {
        self.compression_algorithm = match algorithm {
            "lz4" => CompressionAlgorithm::Lz4,
            "gzip" => CompressionAlgorithm::Gzip,
            "none" => CompressionAlgorithm::None,
            _ => return Err(crate::error::KimiError::configuration(
                format!("Unknown compression algorithm: {}", algorithm)
            )),
        };
        Ok(())
    }
    
    /// Set quantization method
    #[wasm_bindgen]
    pub fn set_quantization_method(&mut self, method: &str) -> Result<()> {
        self.quantization_method = match method {
            "int8" => QuantizationMethod::Int8,
            "int16" => QuantizationMethod::Int16,
            "float32" => QuantizationMethod::Float32,
            _ => return Err(crate::error::KimiError::configuration(
                format!("Unknown quantization method: {}", method)
            )),
        };
        Ok(())
    }

    /// Load an expert into active memory
    #[wasm_bindgen]
    pub fn load_expert(&mut self, domain: ExpertDomain) -> Result<bool> {
        let start_time = Instant::now();
        
        // Check if already in active cache
        if self.active_experts.contains(&domain) {
            self.cache_hits += 1;
            // Update access time
            if let Some(compressed) = self.expert_cache.get_mut(&domain) {
                compressed.access_count += 1;
                compressed.last_access = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
            }
            return Ok(true);
        }

        self.cache_misses += 1;

        // Check if we have compressed version
        if let Some(compressed) = self.expert_cache.get_mut(&domain) {
            let expert = self.decompress_expert(compressed)?;
            let expert_size = self.estimate_expert_size(&expert);

            // Check memory constraints
            if self.current_memory + expert_size > self.max_memory {
                self.evict_least_used(expert_size)?;
            }

            // Update access statistics
            compressed.access_count += 1;
            compressed.last_access = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            self.active_experts.put(domain, expert);
            self.current_memory += expert_size;
            
            // Track performance
            let load_time = start_time.elapsed();
            self.performance_metrics
                .entry(domain)
                .or_insert_with(Vec::new)
                .push(load_time);
            
            log::debug!("Loaded {} expert in {:?} (compression: {:?})", 
                       domain, load_time, compressed.compression_algorithm);
            
            Ok(true)
        } else {
            // Expert not found
            Ok(false)
        }
    }

    /// Store an expert in compressed cache
    #[wasm_bindgen]
    pub fn store_expert(&mut self, expert: MicroExpert) -> Result<()> {
        let domain = expert.domain();
        let compressed = self.compress_expert(&expert)?;
        
        log::info!("Stored {} expert: {:.2}% compression ratio ({} -> {} bytes)",
                  domain, 
                  compressed.compression_ratio * 100.0,
                  compressed.original_size,
                  compressed.compressed_size);
        
        self.expert_cache.insert(domain, compressed);
        Ok(())
    }

    /// Get memory usage statistics
    #[wasm_bindgen]
    pub fn get_memory_stats(&self) -> JsValue {
        let hit_rate = if self.cache_hits + self.cache_misses > 0 {
            self.cache_hits as f32 / (self.cache_hits + self.cache_misses) as f32
        } else {
            0.0
        };

        let stats = MemoryStats {
            total_allocated: self.current_memory,
            active_experts: self.active_experts.len(),
            cached_experts: self.expert_cache.len(),
            cache_hit_rate: hit_rate,
            evictions: self.evictions,
        };

        serde_wasm_bindgen::to_value(&stats).unwrap_or(JsValue::NULL)
    }

    /// Check if expert is loaded in active memory
    #[wasm_bindgen]
    pub fn is_expert_loaded(&self, domain: ExpertDomain) -> bool {
        self.active_experts.contains(&domain)
    }

    /// Get available memory in bytes
    #[wasm_bindgen(getter = available_memory)]
    pub fn available_memory(&self) -> usize {
        self.max_memory.saturating_sub(self.current_memory)
    }

    /// Clear all experts from memory
    #[wasm_bindgen]
    pub fn clear_memory(&mut self) {
        self.active_experts.clear();
        self.expert_cache.clear();
        self.current_memory = 0;
    }
}

impl ExpertMemoryManager {
    /// Native interface to get expert
    pub fn get_expert(&mut self, domain: ExpertDomain) -> Result<Option<&MicroExpert>> {
        if !self.load_expert(domain)? {
            return Ok(None);
        }
        Ok(self.active_experts.get(&domain))
    }

    /// Native interface to get mutable expert
    pub fn get_expert_mut(&mut self, domain: ExpertDomain) -> Result<Option<&mut MicroExpert>> {
        if !self.load_expert(domain)? {
            return Ok(None);
        }
        Ok(self.active_experts.get_mut(&domain))
    }

    /// Estimate memory usage of an expert
    fn estimate_expert_size(&self, expert: &MicroExpert) -> usize {
        // Base size estimation (would be more accurate with actual neural network)
        let base_size = std::mem::size_of::<MicroExpert>();
        let param_size = expert.parameter_count() * std::mem::size_of::<f32>();
        base_size + param_size
    }

    /// Compress expert for storage using real compression algorithms
    fn compress_expert(&mut self, expert: &MicroExpert) -> Result<CompressedExpert> {
        // Serialize to binary format
        let serialized = self.serialize_expert(expert)?;
        let original_size = serialized.len();
        
        // Apply quantization if needed
        let quantized_data = self.apply_quantization(serialized)?;
        
        // Apply compression algorithm
        let compressed_data = match self.compression_algorithm {
            CompressionAlgorithm::Lz4 => self.compress_lz4(&quantized_data)?,
            CompressionAlgorithm::Gzip => self.compress_gzip(&quantized_data)?,
            CompressionAlgorithm::None => quantized_data,
        };
        
        let compressed_size = compressed_data.len();
        let compression_ratio = compressed_size as f32 / original_size as f32;
        
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(CompressedExpert {
            domain: expert.domain(),
            compressed_data,
            original_size,
            compressed_size,
            compression_ratio,
            compression_algorithm: self.compression_algorithm,
            quantization_method: self.quantization_method,
            creation_time: current_time,
            access_count: 0,
            last_access: current_time,
        })
    }
    
    /// Serialize expert to binary format
    fn serialize_expert(&self, expert: &MicroExpert) -> Result<Vec<u8>> {
        // For production, this would serialize the neural network weights directly
        // For now, use JSON as a placeholder
        serde_json::to_vec(expert)
            .map_err(|e| crate::error::KimiError::configuration(format!("Serialization failed: {}", e)))
    }
    
    /// Apply quantization to reduce memory usage
    fn apply_quantization(&self, data: Vec<u8>) -> Result<Vec<u8>> {
        match self.quantization_method {
            QuantizationMethod::Float32 => Ok(data), // No quantization
            QuantizationMethod::Int16 => {
                // Simulate 16-bit quantization (would be more sophisticated in production)
                Ok(data)
            },
            QuantizationMethod::Int8 => {
                // Simulate 8-bit quantization (would be more sophisticated in production)
                Ok(data)
            },
        }
    }
    
    /// Compress data using LZ4
    fn compress_lz4(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        let mut buffer = self.memory_pool.allocate(data.len() * 2)?; // Estimate compressed size
        
        let mut encoder = Encoder::new(&mut buffer)
            .map_err(|e| crate::error::KimiError::configuration(format!("LZ4 encoder failed: {}", e)))?;
        
        encoder.write_all(data)
            .map_err(|e| crate::error::KimiError::configuration(format!("LZ4 compression failed: {}", e)))?;
        
        let compressed = encoder.finish().1
            .map_err(|e| crate::error::KimiError::configuration(format!("LZ4 finish failed: {}", e)))?;
        
        self.memory_pool.deallocate(buffer);
        Ok(compressed)
    }
    
    /// Compress data using Gzip
    fn compress_gzip(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        {
            let mut encoder = GzEncoder::new(&mut buffer, Compression::default());
            encoder.write_all(data)
                .map_err(|e| crate::error::KimiError::configuration(format!("Gzip compression failed: {}", e)))?;
        }
        Ok(buffer)
    }

    /// Decompress expert from storage using real decompression algorithms
    fn decompress_expert(&mut self, compressed: &CompressedExpert) -> Result<MicroExpert> {
        // Decompress data based on algorithm used
        let decompressed_data = match compressed.compression_algorithm {
            CompressionAlgorithm::Lz4 => self.decompress_lz4(&compressed.compressed_data)?,
            CompressionAlgorithm::Gzip => self.decompress_gzip(&compressed.compressed_data)?,
            CompressionAlgorithm::None => compressed.compressed_data.clone(),
        };
        
        // Apply dequantization if needed
        let dequantized_data = self.apply_dequantization(decompressed_data, compressed.quantization_method)?;
        
        // Deserialize expert
        let expert: MicroExpert = serde_json::from_slice(&dequantized_data)
            .map_err(|e| crate::error::KimiError::configuration(format!("Deserialization failed: {}", e)))?;
        
        Ok(expert)
    }
    
    /// Decompress LZ4 data
    fn decompress_lz4(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        let mut buffer = self.memory_pool.allocate(data.len() * 4)?; // Estimate decompressed size
        
        let mut decoder = Decoder::new(data)
            .map_err(|e| crate::error::KimiError::configuration(format!("LZ4 decoder failed: {}", e)))?;
        
        decoder.read_to_end(&mut buffer)
            .map_err(|e| crate::error::KimiError::configuration(format!("LZ4 decompression failed: {}", e)))?;
        
        Ok(buffer)
    }
    
    /// Decompress Gzip data
    fn decompress_gzip(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut decoder = GzDecoder::new(data);
        let mut buffer = Vec::new();
        
        decoder.read_to_end(&mut buffer)
            .map_err(|e| crate::error::KimiError::configuration(format!("Gzip decompression failed: {}", e)))?;
        
        Ok(buffer)
    }
    
    /// Apply dequantization
    fn apply_dequantization(&self, data: Vec<u8>, method: QuantizationMethod) -> Result<Vec<u8>> {
        match method {
            QuantizationMethod::Float32 => Ok(data), // No dequantization needed
            QuantizationMethod::Int16 => {
                // Simulate 16-bit dequantization
                Ok(data)
            },
            QuantizationMethod::Int8 => {
                // Simulate 8-bit dequantization
                Ok(data)
            },
        }
    }

    /// Evict least recently used experts to free memory with smart eviction
    fn evict_least_used(&mut self, needed_bytes: usize) -> Result<()> {
        let mut freed_bytes = 0;
        let start_time = Instant::now();

        // First, try to evict experts with lowest access frequency
        let mut eviction_candidates: Vec<_> = self.expert_cache.iter()
            .filter(|(domain, _)| self.active_experts.contains(domain))
            .map(|(domain, compressed)| {
                let access_rate = compressed.access_count as f64 / 
                    (std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs() - compressed.creation_time + 1) as f64;
                (*domain, access_rate)
            })
            .collect();
        
        // Sort by access rate (lowest first)
        eviction_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Evict based on smart algorithm
        for (domain, _) in eviction_candidates {
            if freed_bytes >= needed_bytes {
                break;
            }
            
            if let Some(expert) = self.active_experts.pop(&domain) {
                let expert_size = self.estimate_expert_size(&expert);
                
                // Update compressed cache with fresh data
                let compressed = self.compress_expert(&expert)?;
                self.expert_cache.insert(domain, compressed);
                
                freed_bytes += expert_size;
                self.current_memory = self.current_memory.saturating_sub(expert_size);
                self.evictions += 1;
                
                log::debug!("Evicted {} expert ({} bytes freed)", domain, expert_size);
            }
        }
        
        // Fallback to LRU eviction if smart eviction wasn't enough
        while freed_bytes < needed_bytes && !self.active_experts.is_empty() {
            if let Some((domain, expert)) = self.active_experts.pop_lru() {
                let expert_size = self.estimate_expert_size(&expert);
                
                // Store in compressed cache before evicting
                let compressed = self.compress_expert(&expert)?;
                self.expert_cache.insert(domain, compressed);
                
                freed_bytes += expert_size;
                self.current_memory = self.current_memory.saturating_sub(expert_size);
                self.evictions += 1;
                
                log::debug!("LRU evicted {} expert ({} bytes freed)", domain, expert_size);
            }
        }

        if freed_bytes < needed_bytes {
            return Err(crate::error::KimiError::MemoryAllocation { size: needed_bytes });
        }

        let eviction_time = start_time.elapsed();
        log::info!("Memory eviction completed: {} bytes freed in {:?}", freed_bytes, eviction_time);

        Ok(())
    }

    /// Set cache size
    pub fn set_cache_size(&mut self, size: usize) -> Result<()> {
        let new_size = NonZeroUsize::new(size)
            .ok_or_else(|| crate::error::KimiError::configuration("Cache size must be greater than 0"))?;
        
        // Create new cache with different size
        let mut new_cache = LruCache::new(new_size);
        
        // Move existing entries if they fit
        while let Some((domain, expert)) = self.active_experts.pop_lru() {
            if new_cache.len() < size {
                new_cache.put(domain, expert);
            } else {
                // Store evicted experts in compressed cache
                let compressed = self.compress_expert(&expert)?;
                self.expert_cache.insert(domain, compressed);
                self.evictions += 1;
            }
        }
        
        self.active_experts = new_cache;
        Ok(())
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let hit_rate = if self.cache_hits + self.cache_misses > 0 {
            self.cache_hits as f32 / (self.cache_hits + self.cache_misses) as f32
        } else {
            0.0
        };

        MemoryStats {
            total_allocated: self.current_memory,
            active_experts: self.active_experts.len(),
            cached_experts: self.expert_cache.len(),
            cache_hit_rate: hit_rate,
            evictions: self.evictions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager_creation() {
        let manager = ExpertMemoryManager::new(512); // 512 MB
        assert_eq!(manager.available_memory(), 512 * 1024 * 1024);
    }

    #[test]
    fn test_expert_storage_and_loading() {
        let mut manager = ExpertMemoryManager::new(512);
        let expert = MicroExpert::new(ExpertDomain::Coding).unwrap();
        
        // Store expert
        manager.store_expert(expert).unwrap();
        assert!(manager.expert_cache.contains_key(&ExpertDomain::Coding));
        
        // Load expert
        assert!(manager.load_expert(ExpertDomain::Coding).unwrap());
        assert!(manager.is_expert_loaded(ExpertDomain::Coding));
    }

    #[test]
    fn test_memory_stats() {
        let manager = ExpertMemoryManager::new(256);
        let stats = manager.memory_stats();
        
        assert_eq!(stats.active_experts, 0);
        assert_eq!(stats.cached_experts, 0);
        assert_eq!(stats.cache_hit_rate, 0.0);
    }

    #[test]
    fn test_cache_size_modification() {
        let mut manager = ExpertMemoryManager::new(256);
        
        // Should succeed
        assert!(manager.set_cache_size(20).is_ok());
        
        // Should fail with zero
        assert!(manager.set_cache_size(0).is_err());
    }
}