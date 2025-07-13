//! P2P networking layer for QuDAG
//! 
//! Implements peer-to-peer networking using libp2p with support for
//! peer discovery, message routing, and DAG synchronization.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use libp2p::{
    gossipsub::{self, IdentTopic, MessageAuthenticity, ValidationMode},
    identify,
    kad::{self, record::Key, Kademlia, KademliaEvent},
    mdns,
    multiaddr::Multiaddr,
    noise,
    ping,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp,
    yamux,
    PeerId, Swarm, Transport,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;

use crate::{DAGNode, QuDAGError, Result};

/// QuDAG network manager
#[derive(Debug)]
pub struct QuDAGNetwork {
    swarm: Swarm<NetworkBehavior>,
    event_sender: mpsc::UnboundedSender<NetworkEvent>,
    event_receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<NetworkEvent>>>>,
    peer_manager: Arc<RwLock<PeerManager>>,
    message_cache: Arc<RwLock<HashMap<String, CachedMessage>>>,
}

impl QuDAGNetwork {
    /// Create a new QuDAG network
    pub async fn new(
        listen_addr: Multiaddr,
        keypair: libp2p::identity::Keypair,
    ) -> Result<Self> {
        // Build transport
        let transport = tcp::tokio::Transport::new(tcp::Config::default().nodelay(true))
            .upgrade(libp2p::core::upgrade::Version::V1)
            .authenticate(noise::NoiseAuthenticated::xx(&keypair)?)
            .multiplex(yamux::YamuxConfig::default())
            .boxed();

        // Create network behavior
        let behavior = NetworkBehavior::new(&keypair).await?;

        // Create swarm
        let mut swarm = Swarm::with_tokio_executor(transport, behavior, keypair.public().to_peer_id());
        swarm.listen_on(listen_addr)?;

        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        let peer_manager = Arc::new(RwLock::new(PeerManager::new()));
        let message_cache = Arc::new(RwLock::new(HashMap::new()));

        Ok(Self {
            swarm,
            event_sender,
            event_receiver: Arc::new(RwLock::new(Some(event_receiver))),
            peer_manager,
            message_cache,
        })
    }

    /// Start the network
    pub async fn start(&mut self) -> Result<()> {
        let event_receiver = self.event_receiver.write().await.take()
            .ok_or(QuDAGError::NetworkError("Network already started".to_string()))?;

        let peer_manager = Arc::clone(&self.peer_manager);
        let message_cache = Arc::clone(&self.message_cache);

        // Spawn network event processing task
        tokio::spawn(async move {
            Self::process_network_events(event_receiver, peer_manager, message_cache).await;
        });

        // Start the main network loop
        tokio::spawn(async move {
            // This would be the main swarm polling loop
            // For brevity, this is simplified
        });

        tracing::info!("QuDAG network started");
        Ok(())
    }

    /// Stop the network
    pub async fn stop(&mut self) -> Result<()> {
        // Implementation would gracefully shut down network connections
        tracing::info!("QuDAG network stopped");
        Ok(())
    }

    /// Broadcast a DAG node to all peers
    pub async fn broadcast_dag_node(&mut self, node: DAGNode) -> Result<()> {
        let message = NetworkMessage::DAGNode(node);
        let serialized = bincode::serialize(&message)
            .map_err(|e| QuDAGError::NetworkError(format!("Serialization error: {}", e)))?;

        // Publish to gossipsub topic
        self.swarm
            .behaviour_mut()
            .gossipsub
            .publish(IdentTopic::new("qudag-dag-nodes"), serialized)
            .map_err(|e| QuDAGError::NetworkError(format!("Broadcast error: {}", e)))?;

        Ok(())
    }

    /// Request a DAG node from peers
    pub async fn request_dag_node(&mut self, node_id: Uuid) -> Result<()> {
        let message = NetworkMessage::DAGNodeRequest(node_id);
        let serialized = bincode::serialize(&message)
            .map_err(|e| QuDAGError::NetworkError(format!("Serialization error: {}", e)))?;

        self.swarm
            .behaviour_mut()
            .gossipsub
            .publish(IdentTopic::new("qudag-requests"), serialized)
            .map_err(|e| QuDAGError::NetworkError(format!("Request error: {}", e)))?;

        Ok(())
    }

    /// Connect to a peer
    pub async fn connect_peer(&mut self, peer_addr: Multiaddr) -> Result<()> {
        self.swarm.dial(peer_addr)
            .map_err(|e| QuDAGError::NetworkError(format!("Connection error: {}", e)))?;
        Ok(())
    }

    /// Get connected peers
    pub async fn connected_peers(&self) -> Vec<PeerId> {
        self.peer_manager.read().await.connected_peers()
    }

    /// Get network statistics
    pub async fn get_stats(&self) -> NetworkStats {
        let peer_manager = self.peer_manager.read().await;
        NetworkStats {
            connected_peers: peer_manager.connected_peers().len(),
            total_messages: peer_manager.total_messages(),
            uptime: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Process incoming network events
    async fn process_network_events(
        mut event_receiver: mpsc::UnboundedReceiver<NetworkEvent>,
        peer_manager: Arc<RwLock<PeerManager>>,
        message_cache: Arc<RwLock<HashMap<String, CachedMessage>>>,
    ) {
        while let Some(event) = event_receiver.recv().await {
            match event {
                NetworkEvent::PeerConnected(peer_id) => {
                    peer_manager.write().await.add_peer(peer_id);
                    tracing::info!("Peer connected: {}", peer_id);
                }
                NetworkEvent::PeerDisconnected(peer_id) => {
                    peer_manager.write().await.remove_peer(&peer_id);
                    tracing::info!("Peer disconnected: {}", peer_id);
                }
                NetworkEvent::MessageReceived { from, message } => {
                    Self::handle_message(from, message, &message_cache).await;
                }
                NetworkEvent::DAGNodeReceived(node) => {
                    tracing::debug!("Received DAG node: {}", node.id);
                    // This would trigger DAG validation and storage
                }
            }
        }
    }

    /// Handle incoming network messages
    async fn handle_message(
        from: PeerId,
        message: NetworkMessage,
        message_cache: &Arc<RwLock<HashMap<String, CachedMessage>>>,
    ) {
        match message {
            NetworkMessage::DAGNode(node) => {
                tracing::debug!("Received DAG node {} from {}", node.id, from);
                // Validate and process DAG node
            }
            NetworkMessage::DAGNodeRequest(node_id) => {
                tracing::debug!("Received DAG node request for {} from {}", node_id, from);
                // Look up and respond with DAG node if available
            }
            NetworkMessage::ConsensusVote { node_id, vote } => {
                tracing::debug!("Received consensus vote for {} from {}", node_id, from);
                // Process consensus vote
            }
            NetworkMessage::PeerDiscovery(peer_info) => {
                tracing::debug!("Received peer discovery from {}", from);
                // Update peer information
            }
        }
    }
}

/// LibP2P network behavior combining multiple protocols
#[derive(NetworkBehaviour)]
#[behaviour(to_swarm = "NetworkBehaviorEvent")]
pub struct NetworkBehavior {
    pub gossipsub: gossipsub::Behaviour,
    pub kad: Kademlia<kad::store::MemoryStore>,
    pub identify: identify::Behaviour,
    pub ping: ping::Behaviour,
    pub mdns: mdns::tokio::Behaviour,
}

impl NetworkBehavior {
    pub async fn new(keypair: &libp2p::identity::Keypair) -> Result<Self> {
        // Configure Gossipsub
        let gossipsub_config = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(Duration::from_secs(10))
            .validation_mode(ValidationMode::Strict)
            .message_authenticity(MessageAuthenticity::Signed(keypair.clone()))
            .build()
            .map_err(|e| QuDAGError::NetworkError(format!("Gossipsub config error: {}", e)))?;

        let mut gossipsub = gossipsub::Behaviour::new(
            MessageAuthenticity::Signed(keypair.clone()),
            gossipsub_config,
        )
        .map_err(|e| QuDAGError::NetworkError(format!("Gossipsub creation error: {}", e)))?;

        // Subscribe to topics
        gossipsub.subscribe(&IdentTopic::new("qudag-dag-nodes"))
            .map_err(|e| QuDAGError::NetworkError(format!("Topic subscription error: {}", e)))?;
        gossipsub.subscribe(&IdentTopic::new("qudag-requests"))
            .map_err(|e| QuDAGError::NetworkError(format!("Topic subscription error: {}", e)))?;
        gossipsub.subscribe(&IdentTopic::new("qudag-consensus"))
            .map_err(|e| QuDAGError::NetworkError(format!("Topic subscription error: {}", e)))?;

        // Configure Kademlia DHT
        let kad_store = kad::store::MemoryStore::new(keypair.public().to_peer_id());
        let kad = Kademlia::new(keypair.public().to_peer_id(), kad_store);

        // Configure Identify
        let identify = identify::Behaviour::new(identify::Config::new(
            "/qudag/1.0.0".to_string(),
            keypair.public(),
        ));

        // Configure Ping
        let ping = ping::Behaviour::new(ping::Config::new());

        // Configure mDNS for local discovery
        let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), keypair.public().to_peer_id())
            .map_err(|e| QuDAGError::NetworkError(format!("mDNS creation error: {}", e)))?;

        Ok(Self {
            gossipsub,
            kad,
            identify,
            ping,
            mdns,
        })
    }
}

/// Network behavior events
#[derive(Debug)]
pub enum NetworkBehaviorEvent {
    Gossipsub(gossipsub::Event),
    Kad(KademliaEvent),
    Identify(identify::Event),
    Ping(ping::Event),
    Mdns(mdns::Event),
}

impl From<gossipsub::Event> for NetworkBehaviorEvent {
    fn from(event: gossipsub::Event) -> Self {
        NetworkBehaviorEvent::Gossipsub(event)
    }
}

impl From<KademliaEvent> for NetworkBehaviorEvent {
    fn from(event: KademliaEvent) -> Self {
        NetworkBehaviorEvent::Kad(event)
    }
}

impl From<identify::Event> for NetworkBehaviorEvent {
    fn from(event: identify::Event) -> Self {
        NetworkBehaviorEvent::Identify(event)
    }
}

impl From<ping::Event> for NetworkBehaviorEvent {
    fn from(event: ping::Event) -> Self {
        NetworkBehaviorEvent::Ping(event)
    }
}

impl From<mdns::Event> for NetworkBehaviorEvent {
    fn from(event: mdns::Event) -> Self {
        NetworkBehaviorEvent::Mdns(event)
    }
}

/// Network events
#[derive(Debug, Clone)]
pub enum NetworkEvent {
    PeerConnected(PeerId),
    PeerDisconnected(PeerId),
    MessageReceived {
        from: PeerId,
        message: NetworkMessage,
    },
    DAGNodeReceived(DAGNode),
}

/// Network messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkMessage {
    DAGNode(DAGNode),
    DAGNodeRequest(Uuid),
    ConsensusVote {
        node_id: Uuid,
        vote: crate::consensus::Vote,
    },
    PeerDiscovery(PeerInfo),
}

/// Peer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub peer_id: PeerId,
    pub addresses: Vec<Multiaddr>,
    pub protocols: Vec<String>,
    pub agent_version: String,
}

/// Peer manager for tracking connections
#[derive(Debug)]
pub struct PeerManager {
    connected_peers: HashSet<PeerId>,
    peer_info: HashMap<PeerId, PeerInfo>,
    message_count: u64,
}

impl PeerManager {
    pub fn new() -> Self {
        Self {
            connected_peers: HashSet::new(),
            peer_info: HashMap::new(),
            message_count: 0,
        }
    }

    pub fn add_peer(&mut self, peer_id: PeerId) {
        self.connected_peers.insert(peer_id);
    }

    pub fn remove_peer(&mut self, peer_id: &PeerId) {
        self.connected_peers.remove(peer_id);
        self.peer_info.remove(peer_id);
    }

    pub fn connected_peers(&self) -> Vec<PeerId> {
        self.connected_peers.iter().cloned().collect()
    }

    pub fn total_messages(&self) -> u64 {
        self.message_count
    }

    pub fn increment_messages(&mut self) {
        self.message_count += 1;
    }
}

/// Cached network message
#[derive(Debug, Clone)]
struct CachedMessage {
    message: NetworkMessage,
    timestamp: u64,
    from: PeerId,
}

/// Network statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    pub connected_peers: usize,
    pub total_messages: u64,
    pub uptime: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_network_creation() {
        let keypair = libp2p::identity::Keypair::generate_ed25519();
        let listen_addr: Multiaddr = "/ip4/127.0.0.1/tcp/0".parse().unwrap();
        
        let network = QuDAGNetwork::new(listen_addr, keypair).await;
        assert!(network.is_ok());
    }

    #[tokio::test]
    async fn test_peer_manager() {
        let mut manager = PeerManager::new();
        let peer_id = PeerId::random();
        
        assert_eq!(manager.connected_peers().len(), 0);
        
        manager.add_peer(peer_id);
        assert_eq!(manager.connected_peers().len(), 1);
        
        manager.remove_peer(&peer_id);
        assert_eq!(manager.connected_peers().len(), 0);
    }

    #[test]
    fn test_network_message_serialization() {
        let node = DAGNode::genesis(b"test".to_vec());
        let message = NetworkMessage::DAGNode(node);
        
        let serialized = bincode::serialize(&message).unwrap();
        let deserialized: NetworkMessage = bincode::deserialize(&serialized).unwrap();
        
        match deserialized {
            NetworkMessage::DAGNode(n) => assert_eq!(n.data, b"test"),
            _ => panic!("Wrong message type"),
        }
    }
}