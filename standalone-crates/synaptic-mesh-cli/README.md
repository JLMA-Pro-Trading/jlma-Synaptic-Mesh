# Synaptic Mesh CLI

Complete CLI library for the Synaptic Neural Mesh project, integrating all components.

## Features

- **Complete Integration**: All Synaptic Neural Mesh components in one CLI
- **P2P Operations**: Launch and manage mesh nodes
- **Neural Networks**: Train and run WASM neural networks
- **Swarm Management**: Control distributed agent swarms
- **QuDAG Networking**: Quantum-resistant DAG operations

## Installation

```bash
cargo install synaptic-mesh-cli
```

## Usage

```bash
# Start a mesh node
synaptic-mesh node start

# Create a swarm
synaptic-mesh swarm create --agents 100

# Train a neural network
synaptic-mesh neural train --model mymodel.json

# Query mesh status
synaptic-mesh status
```

## Library Usage

```rust
use synaptic_mesh_cli::{MeshCommand, execute_command};

let cmd = MeshCommand::NodeStart { port: 8080 };
execute_command(cmd).await?;
```

## License

MIT OR Apache-2.0