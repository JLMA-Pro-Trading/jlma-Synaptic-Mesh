//! # Persistence Demo
//! 
//! Demonstrates the RocksDB persistence layer for Kimi-FANN Core

use kimi_fann_core::{
    PersistentKimiRuntime, ProcessingConfig, PersistenceConfig, P2PSyncConfig,
    ExpertDomain, StorageManager, storage::*,
};
use anyhow::Result;
use std::time::{SystemTime, UNIX_EPOCH};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    println!("🧠 Kimi-FANN Core Persistence Demo");
    println!("===================================");
    
    // Create temporary directory for demo
    let temp_dir = tempfile::tempdir()?;
    let storage_path = temp_dir.path().to_str().unwrap();
    
    println!("📁 Storage path: {}", storage_path);
    
    // Demo 1: Basic Storage Operations
    demo_basic_storage(storage_path).await?;
    
    // Demo 2: Neural Network Persistence
    demo_neural_persistence(storage_path).await?;
    
    // Demo 3: Persistent Runtime
    demo_persistent_runtime(storage_path).await?;
    
    // Demo 4: P2P Sync Configuration
    demo_p2p_sync_config().await?;
    
    // Demo 5: Storage Statistics and Maintenance
    demo_storage_maintenance(storage_path).await?;
    
    println!("\n✅ All demos completed successfully!");
    
    Ok(())
}

async fn demo_basic_storage(storage_path: &str) -> Result<()> {
    println!("\n🔧 Demo 1: Basic Storage Operations");
    println!("-----------------------------------");
    
    // Create storage manager
    let storage_manager = StorageManager::new(storage_path, false)?;
    
    // Store some neural weights
    let expert_id = "reasoning_expert_001";
    let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let biases = vec![0.01, 0.02];
    
    storage_manager.store_expert_weights(expert_id, weights.clone(), biases.clone(), 0)?;
    println!("✅ Stored weights for expert: {}", expert_id);
    
    // Retrieve weights
    if let Some((retrieved_weights, retrieved_biases)) = storage_manager.load_expert_weights(expert_id, 0)? {
        println!("✅ Retrieved weights: {:?}", retrieved_weights);
        println!("✅ Retrieved biases: {:?}", retrieved_biases);
        
        assert_eq!(weights, retrieved_weights);
        assert_eq!(biases, retrieved_biases);
        println!("✅ Weight verification passed!");
    } else {
        println!("❌ Failed to retrieve weights");
    }
    
    // Store expert configuration
    let config_data = ExpertConfigData {
        expert_id: expert_id.to_string(),
        domain: "Reasoning".to_string(),
        parameter_count: 25000,
        learning_rate: 0.001,
        architecture: "micro_expert_v1".to_string(),
        created_at: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        updated_at: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
    };
    
    storage_manager.backend().store_expert_config(&config_data)?;
    println!("✅ Stored expert configuration");
    
    // Create a system checkpoint
    let checkpoint = StateCheckpoint {
        checkpoint_id: String::new(),
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        system_version: "1.0.0".to_string(),
        expert_states: vec![ExpertState {
            expert_id: expert_id.to_string(),
            activation_count: 100,
            success_rate: 0.95,
            last_updated: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            memory_usage: 1024 * 1024,
        }],
        global_metrics: GlobalMetrics {
            total_requests: 500,
            average_response_time: 0.25,
            memory_usage: 128 * 1024 * 1024,
            disk_usage: 64 * 1024 * 1024,
            uptime: 3600,
        },
    };
    
    let checkpoint_id = storage_manager.backend().create_checkpoint(&checkpoint)?;
    println!("✅ Created checkpoint: {}", checkpoint_id);
    
    Ok(())
}

async fn demo_neural_persistence(storage_path: &str) -> Result<()> {
    println!("\n🧠 Demo 2: Neural Network Persistence");
    println!("-------------------------------------");
    
    let persistence_config = PersistenceConfig {
        auto_save_interval: 60, // 1 minute for demo
        max_checkpoints: 5,
        compression_enabled: true,
        backup_enabled: true,
        sync_with_peers: false,
    };
    
    let mut neural_persistence = kimi_fann_core::persistence::NeuralPersistence::new(
        storage_path, 
        persistence_config
    )?;
    
    println!("✅ Created neural persistence manager");
    
    // Check if auto-save should be triggered
    if neural_persistence.should_auto_save() {
        println!("✅ Auto-save is ready (first time)");
    }
    
    // Create a system checkpoint
    let expert_ids = vec![
        "reasoning_expert".to_string(),
        "coding_expert".to_string(),
        "language_expert".to_string(),
    ];
    
    let checkpoint_id = neural_persistence.create_checkpoint("1.0.1", &expert_ids)?;
    println!("✅ Created neural checkpoint: {}", checkpoint_id);
    
    // Restore from checkpoint
    let restored_checkpoint = neural_persistence.restore_checkpoint(&checkpoint_id)?;
    println!("✅ Restored checkpoint with {} expert states", 
             restored_checkpoint.expert_states.len());
    
    // Get storage statistics
    let stats = neural_persistence.get_storage_stats()?;
    println!("✅ Storage stats: {} total, {} checkpoints", 
             stats.human_readable_size(), stats.checkpoint_count);
    
    // Verify storage integrity
    let is_valid = neural_persistence.verify_storage_integrity()?;
    println!("✅ Storage integrity: {}", if is_valid { "VALID" } else { "INVALID" });
    
    Ok(())
}

async fn demo_persistent_runtime(storage_path: &str) -> Result<()> {
    println!("\n🚀 Demo 3: Persistent Runtime");
    println!("-----------------------------");
    
    let config = ProcessingConfig::new();
    let persistence_config = PersistenceConfig::default();
    
    let mut runtime = PersistentKimiRuntime::new_with_storage(
        config,
        storage_path,
        persistence_config,
    )?;
    
    println!("✅ Created persistent runtime");
    
    // Process some queries with persistence
    let queries = vec![
        "How does neural network training work?",
        "Write a function to calculate fibonacci numbers",
        "Translate 'hello world' to Spanish",
        "What is 2 + 2 * 3?",
        "Use a calculator tool to compute 15 * 7",
        "Summarize the context of this conversation",
    ];
    
    println!("\n📝 Processing queries with persistence:");
    for (i, query) in queries.iter().enumerate() {
        let result = runtime.process_with_persistence(query).await?;
        println!("\nQuery {}: {}", i + 1, query);
        println!("Response preview: {}...", 
                 result.lines().next().unwrap_or("").chars().take(80).collect::<String>());
    }
    
    // Save current state
    let checkpoint_id = runtime.save_state()?;
    println!("\n✅ Saved runtime state to checkpoint: {}", checkpoint_id);
    
    // Create a backup
    let backup_dir = format!("{}/backup", storage_path);
    let backup_path = runtime.create_backup(&backup_dir)?;
    println!("✅ Created backup at: {}", backup_path);
    
    // Get comprehensive statistics
    let stats = runtime.get_comprehensive_stats();
    println!("\n📊 Comprehensive Statistics:");
    println!("{}", stats);
    
    // Verify integrity
    let is_valid = runtime.verify_integrity()?;
    println!("\n✅ Runtime integrity check: {}", if is_valid { "PASSED" } else { "FAILED" });
    
    // Compact storage
    runtime.compact_storage()?;
    println!("✅ Storage compaction completed");
    
    Ok(())
}

async fn demo_p2p_sync_config() -> Result<()> {
    println!("\n🌐 Demo 4: P2P Sync Configuration");
    println!("----------------------------------");
    
    let sync_config = P2PSyncConfig {
        node_id: "demo_node_001".to_string(),
        sync_interval_seconds: 30,
        max_peers: 5,
        conflict_resolution: kimi_fann_core::p2p_sync::ConflictResolution::LatestTimestamp,
        enable_weight_sync: true,
        enable_config_sync: true,
        enable_checkpoint_sync: true,
        heartbeat_interval_seconds: 10,
        peer_timeout_seconds: 60,
    };
    
    println!("✅ P2P sync configuration:");
    println!("   Node ID: {}", sync_config.node_id);
    println!("   Sync interval: {}s", sync_config.sync_interval_seconds);
    println!("   Max peers: {}", sync_config.max_peers);
    println!("   Weight sync: {}", sync_config.enable_weight_sync);
    println!("   Config sync: {}", sync_config.enable_config_sync);
    println!("   Checkpoint sync: {}", sync_config.enable_checkpoint_sync);
    
    let sync_manager = kimi_fann_core::P2PSyncManager::new(sync_config);
    println!("✅ Created P2P sync manager");
    
    // Demonstrate sync statistics
    let sync_stats = sync_manager.get_sync_stats();
    println!("✅ Initial sync stats: {} weights sent, {} received", 
             sync_stats.weights_sent, sync_stats.weights_received);
    
    Ok(())
}

async fn demo_storage_maintenance(storage_path: &str) -> Result<()> {
    println!("\n🔧 Demo 5: Storage Maintenance");
    println!("------------------------------");
    
    let storage_manager = StorageManager::new(storage_path, false)?;
    
    // Get database statistics
    let stats_str = storage_manager.backend().get_stats()?;
    println!("📊 RocksDB Statistics (first 200 chars):");
    println!("{}", stats_str.chars().take(200).collect::<String>());
    
    // Get size information
    let (total_size, file_count) = storage_manager.backend().get_size_info()?;
    println!("\n📁 Storage Size Info:");
    println!("   Total size: {} bytes", total_size);
    println!("   File count: {}", file_count);
    
    // Verify integrity
    let is_valid = storage_manager.backend().verify_integrity()?;
    println!("✅ Integrity check: {}", if is_valid { "PASSED" } else { "FAILED" });
    
    // Compact database
    storage_manager.backend().compact_database()?;
    println!("✅ Database compaction completed");
    
    // List all checkpoints
    let checkpoints = storage_manager.backend().list_checkpoints()?;
    println!("📋 Available checkpoints: {}", checkpoints.len());
    for (i, checkpoint_id) in checkpoints.iter().take(3).enumerate() {
        println!("   {}: {}", i + 1, checkpoint_id);
    }
    
    Ok(())
}