# Synaptic Neural Mesh - MCP Integration

A comprehensive Model Context Protocol (MCP) implementation for the Synaptic Neural Mesh distributed neural fabric. This integration enables AI assistants to interact with the mesh through standardized JSON-RPC 2.0 protocols.

## 🌟 Features

- **27+ Neural Mesh Tools** - Complete suite of MCP tools for distributed neural operations
- **Multi-Transport Support** - stdio, HTTP, and WebSocket communication layers
- **Real-time Event Streaming** - Live updates and notifications for mesh activities
- **WASM Bridge Integration** - Direct connection to Rust WASM modules (QuDAG, ruv-swarm, DAA)
- **Authentication & Security** - API key management, rate limiting, and encryption
- **High Performance** - Connection pooling, caching, and optimized processing
- **Comprehensive Testing** - Full test suite with integration and performance tests

## 🚀 Quick Start

### Installation

```bash
cd src/mcp
npm install
```

### Basic Usage

```bash
# Start MCP server with stdio transport
npm start

# Start with HTTP transport
node cli.js start --transport http --port 3000

# Start with authentication enabled
node cli.js start --auth --config ./config.json

# Show available tools
node cli.js tools

# Run tests
npm test
```

### Configuration

Create a configuration file:

```bash
node cli.js config init
```

Example configuration:

```json
{
  "transport": "stdio",
  "port": 3000,
  "enableAuth": false,
  "enableEvents": true,
  "wasmEnabled": true,
  "logLevel": "info",
  "rateLimits": {
    "requests": 100,
    "window": 60000
  },
  "apiKeys": [
    {
      "key": "your-api-key-here",
      "name": "main-key", 
      "permissions": ["neural_*", "mesh_*"]
    }
  ]
}
```

## 🛠️ Neural Mesh Tools

### Core Tools

#### `neural_mesh_init`
Initialize a neural mesh topology with specified configuration.

```json
{
  "topology": "mesh|hierarchical|ring|star|hybrid",
  "maxAgents": 10,
  "strategy": "parallel|sequential|adaptive|balanced",
  "enableConsensus": true,
  "cryptoLevel": "basic|quantum|post-quantum"
}
```

#### `neural_agent_spawn`
Spawn specialized neural agents in the mesh.

```json
{
  "meshId": "mesh-id",
  "type": "coordinator|researcher|coder|analyst|architect|tester|reviewer|optimizer|documenter|monitor|specialist",
  "name": "agent-name",
  "capabilities": ["capability1", "capability2"],
  "neuralModel": "model-type",
  "resources": {"cpu": 1, "memory": 512}
}
```

#### `neural_consensus`
Coordinate neural decisions through DAG consensus.

```json
{
  "meshId": "mesh-id",
  "proposal": {"action": "update", "value": 123},
  "agents": ["agent1", "agent2", "agent3"],
  "consensusType": "majority|supermajority|unanimous|weighted"
}
```

### Memory Operations

#### `mesh_memory_store`
Store data in distributed mesh memory.

```json
{
  "key": "storage-key",
  "value": {"any": "data"},
  "namespace": "default",
  "ttl": 3600,
  "replicas": 3
}
```

#### `mesh_memory_retrieve`
Retrieve data from distributed mesh memory.

```json
{
  "key": "storage-key",
  "namespace": "default"
}
```

### Training & Performance

#### `neural_train`
Coordinate distributed neural training across mesh.

```json
{
  "meshId": "mesh-id",
  "modelType": "feedforward|cnn|rnn|transformer",
  "trainingData": {"inputs": [], "outputs": []},
  "epochs": 100,
  "distributionStrategy": "data_parallel|model_parallel|pipeline"
}
```

#### `mesh_performance`
Get real-time performance metrics.

```json
{
  "meshId": "mesh-id",
  "metrics": ["cpu", "memory", "throughput"],
  "timeframe": "1h|24h|7d"
}
```

### Advanced Tools

- `neural_pattern_recognize` - Pattern recognition in neural mesh data
- `mesh_topology_optimize` - Dynamic topology optimization
- `neural_ensemble_create` - Multi-model coordination
- `mesh_fault_tolerance` - Fault tolerance configuration
- `load_balance` - Computational load distribution
- `security_encrypt/decrypt` - Mesh security operations
- `mesh_autoscale` - Automatic scaling configuration
- `mesh_analytics` - Advanced analytics and insights

[See full tool documentation](./docs/tools-reference.md)

## 🔄 Event Streaming

Real-time event streaming for mesh activities:

```javascript
import SynapticMeshMCP from './index.js';

const mcp = new SynapticMeshMCP();
await mcp.initialize();

// Create event stream
const streamId = mcp.events.streamNeuralMeshEvents({
  'data.meshId': 'specific-mesh'
});

// Subscribe to events
mcp.events.subscribe(streamId, (event) => {
  console.log('Event:', event.type, event.data);
});
```

## 🔐 Authentication

### API Key Management

```bash
# Create API key
node cli.js auth create-key --name "my-key" --permissions "neural_*"

# List API keys
node cli.js auth list-keys

# Revoke API key
node cli.js auth revoke-key --key "api-key-id"
```

### Usage with Authentication

```bash
# Set authorization header
export MCP_API_KEY="your-api-key"
node your-mcp-client.js
```

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
npm test

# Run specific test file
node --test tests/mcp-integration.test.js

# Run with coverage
npm run test:coverage
```

### Performance Testing

```bash
# Run benchmark
node cli.js perf benchmark --iterations 1000 --concurrent 50

# Monitor performance
node cli.js perf monitor --interval 1000 --duration 60
```

## 🌐 Transport Layers

### stdio (Default)
Standard input/output communication, ideal for command-line tools and MCP clients.

### HTTP
RESTful HTTP API with JSON-RPC 2.0 over HTTP POST.

```bash
curl -X POST http://localhost:3000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "neural_mesh_init",
      "arguments": {
        "topology": "mesh",
        "maxAgents": 5,
        "strategy": "parallel"
      }
    },
    "id": 1
  }'
```

### WebSocket
Real-time bidirectional communication with event streaming support.

```javascript
const ws = new WebSocket('ws://localhost:3000/ws');
ws.send(JSON.stringify({
  jsonrpc: "2.0",
  method: "tools/call",
  params: { name: "mesh_performance" },
  id: 1
}));
```

## 🦀 WASM Integration

The MCP integration connects directly to Rust WASM modules:

- **QuDAG** - DAG consensus and cryptographic operations
- **ruv-swarm** - Neural agent coordination and SIMD optimization
- **DAA** - Distributed algorithm execution
- **CUDA-WASM** - GPU-accelerated computation (optional)

WASM modules are automatically detected and loaded:

```javascript
// Check WASM capabilities
const metrics = mcp.wasmBridge.getPerformanceMetrics();
console.log('SIMD Support:', metrics.capabilities.simd);
console.log('Threads Support:', metrics.capabilities.threads);
```

## 📊 Monitoring & Analytics

### Real-time Monitoring

```bash
# Monitor server status
node cli.js status --json

# Monitor performance metrics
node cli.js perf monitor

# View active connections
node cli.js connections list
```

### Analytics Dashboard

Generate analytics reports:

```bash
# Performance analysis
node cli.js analytics performance --timeframe 24h

# Usage statistics
node cli.js analytics usage --export csv

# Health check
node cli.js health --detailed
```

## 🔧 Development

### Building from Source

```bash
# Install dependencies
npm install

# Run linting
npm run lint

# Format code
npm run format

# Build documentation
npm run docs
```

### Creating Custom Tools

```javascript
// Add custom tool to neural-mesh-tools.js
customTool: {
  name: 'custom_tool',
  description: 'Custom tool description',
  inputSchema: z.object({
    param1: z.string().describe('Parameter description'),
    param2: z.number().optional()
  }),
  handler: this.customToolHandler.bind(this)
}

async customToolHandler({ param1, param2 }) {
  // Implementation
  return { success: true, result: 'Custom result' };
}
```

## 🐛 Troubleshooting

### Common Issues

1. **WASM Module Loading Fails**
   ```bash
   # Check WASM module paths
   node cli.js dev validate
   
   # Disable WASM for testing
   export DISABLE_WASM=true
   ```

2. **Authentication Errors**
   ```bash
   # Verify API key format
   node cli.js auth validate-key --key "your-key"
   
   # Check permissions
   node cli.js auth check-permissions --key "your-key" --tool "neural_mesh_init"
   ```

3. **Transport Issues**
   ```bash
   # Test connectivity
   node cli.js test --transport http --port 3000
   
   # Check port availability
   netstat -an | grep 3000
   ```

### Debug Mode

```bash
# Enable debug logging
DEBUG=synaptic-mesh:* node cli.js start

# Verbose output
node cli.js start --log-level debug
```

## 📚 Documentation

- [API Reference](./docs/api-reference.md)
- [Tool Reference](./docs/tools-reference.md)
- [Transport Guide](./docs/transport-guide.md)
- [Authentication Guide](./docs/auth-guide.md)
- [WASM Integration](./docs/wasm-integration.md)
- [Performance Tuning](./docs/performance-tuning.md)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`npm test`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io/) specification
- [Anthropic](https://anthropic.com/) for MCP development
- Rust WASM community for optimization insights
- Neural mesh research community

---

🧠 **Synaptic Neural Mesh** - Where distributed intelligence meets seamless integration.