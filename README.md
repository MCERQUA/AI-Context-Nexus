# AI Context Nexus 🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/mikecerqua/ai-context-nexus)

## 🚀 Overview

AI Context Nexus is a revolutionary distributed memory and context management system that enables multiple AI agents to share knowledge, maintain persistent context, and collaborate effectively. Built on a novel "Git-as-Memory" architecture, it transforms version control into a powerful semantic knowledge graph.

## ✨ Core Innovations

### 1. **Git-as-Memory (GaM)** 
Transform git commits into a persistent, versioned memory system where each commit represents a context chunk with full semantic metadata embedded in commit messages.

### 2. **Three-Tier Memory Hierarchy**
- **L1 Cache**: In-memory hot data (< 1ms access) with LRU eviction
- **L2 Cache**: LZ4-compressed JSON files (< 10ms access)
- **L3 Storage**: Full git history with semantic indexing (< 100ms access)

### 3. **Semantic Knowledge Graph**
NetworkX-based graph connecting contexts by semantic similarity, enabling graph-based queries, clustering, and relationship discovery.

### 4. **Universal Agent Protocol**
Standardized interface allowing any LLM (Claude, GPT, Llama, etc.) to participate in the shared memory system through the AgentOrchestrator.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Context Nexus                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Agent 1    │  │   Agent 2    │  │   Agent N    │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                  │                  │            │
│  ┌──────▼──────────────────▼──────────────────▼──────┐    │
│  │            Context Synchronization Layer           │    │
│  └────────────────────────┬───────────────────────────┘    │
│                           │                                │
│  ┌────────────────────────▼───────────────────────────┐    │
│  │              Memory Hierarchy Manager              │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │    │
│  │  │   L1    │  │   L2    │  │       L3        │   │    │
│  │  │ Hot RAM │  │  JSON   │  │  Git History    │   │    │
│  │  └─────────┘  └─────────┘  └─────────────────┘   │    │
│  └────────────────────────┬───────────────────────────┘    │
│                           │                                │
│  ┌────────────────────────▼───────────────────────────┐    │
│  │           Version Control Layer (Git + JJ)         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Context Manager (core/context_manager.py)
- Handles context serialization/deserialization
- Manages context chunks in git commits
- Implements the novel "Semantic Commit Graph" algorithm

### 2. Memory Hierarchy (memory/)
- **L1 Cache**: In-memory hot data (< 1ms access)
- **L2 Cache**: JSON-indexed recent contexts (< 10ms access)
- **L3 Storage**: Full git history (< 100ms access)

### 3. Agent Protocol (agents/)
- Standardized interface for any LLM
- Built-in adapters for Claude, GPT, Llama, etc.
- Real-time synchronization via shared memory

### 4. Tmux Integration (scripts/tmux_orchestrator.sh)
- Automatic session management
- Process isolation and monitoring
- Failure recovery and restart

## 🛠️ Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/mikecerqua/ai-context-nexus
cd ai-context-nexus

# Run the quickstart script
./quickstart.sh
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize the system
python scripts/nexus_cli.py system init
```

### Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

## Quick Start

```python
from ai_context_nexus import ContextNexus, Agent

# Initialize the nexus
nexus = ContextNexus()

# Create agents
agent1 = Agent("claude", api_key="...")
agent2 = Agent("gpt4", api_key="...")

# Register agents with nexus
nexus.register(agent1)
nexus.register(agent2)

# Share context
context = agent1.generate_context("Analyze this codebase")
nexus.commit_context(context, agent1.id)

# Another agent retrieves context
shared_context = nexus.get_context(query="codebase analysis")
response = agent2.process_with_context(shared_context, "Continue the analysis")
```

## Advanced Features

### Semantic Commit Graph
Our novel algorithm creates a knowledge graph from git commits, enabling:
- Semantic search across all historical contexts
- Context relevance scoring
- Automatic context pruning and compression

### Distributed Consensus
Multiple agents can work on the same problem with:
- Byzantine fault tolerance
- Eventual consistency guarantees
- Conflict resolution via CRDT principles

### Performance Optimization
- Memory-mapped files for large contexts
- Bloom filters for quick existence checks
- Parallel git operations via libgit2

## System Requirements

- Linux VPS with 4GB+ RAM
- Git 2.30+
- Jujutsu (JJ) 0.10+
- Python 3.9+
- Tmux 3.0+
- Redis (optional, for distributed setup)

## Documentation

- [Architecture Deep Dive](docs/architecture.md)
- [Agent Protocol Specification](docs/agent_protocol.md)
- [Memory Hierarchy Design](docs/memory_hierarchy.md)
- [Failure Recovery](docs/failure_recovery.md)
- [Performance Benchmarks](docs/benchmarks.md)

## Research & Expansion

See the `research/` directory for:
- Theoretical proofs of consistency
- Scalability analysis
- Future enhancement proposals
- Integration possibilities with other systems

## 🚀 Use Cases

- **Multi-Agent Collaboration**: Enable multiple AI agents to work on complex problems with shared context
- **Persistent Memory**: Maintain context across sessions and agent restarts
- **Knowledge Graph Building**: Automatically construct semantic knowledge graphs from agent interactions
- **Distributed AI Systems**: Scale across multiple nodes with Redis-backed synchronization
- **Research & Development**: Track and version AI reasoning processes for analysis

## 📊 Performance Benchmarks

| Operation | L1 Cache | L2 Cache | L3 Storage |
|-----------|----------|----------|------------|
| Read | < 1ms | < 10ms | < 100ms |
| Write | < 5ms | < 20ms | < 200ms |
| Search | < 10ms | < 50ms | < 500ms |
| Graph Query | - | < 100ms | < 1000ms |

## 🔒 Security Features

- **AES-256-GCM Encryption**: Optional encryption for sensitive contexts
- **JWT Authentication**: Secure API access control
- **TLS Support**: Encrypted network communication
- **Audit Logging**: Complete activity tracking
- **Rate Limiting**: Built-in protection against abuse

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 core/ agents/ scripts/

# Run type checking
mypy core/ agents/ scripts/
```

## 📖 Documentation

- [📚 Full Documentation](https://ai-context-nexus.readthedocs.io)
- [🏗️ Architecture Guide](docs/architecture.md)
- [🤖 Agent Protocol](docs/agent_protocol.md)
- [💾 Memory Hierarchy](docs/memory_hierarchy.md)
- [🔧 API Reference](docs/api_reference.md)
- [📊 Benchmarks](docs/benchmarks.md)

## 🗺️ Roadmap

- [ ] Production-ready FAISS integration
- [ ] Real embedding models (BERT, GPT)
- [ ] Web UI for system monitoring
- [ ] Kubernetes Helm charts
- [ ] Multi-cloud deployment support
- [ ] Advanced agent orchestration strategies
- [ ] Semantic search optimizations
- [ ] Graph pruning algorithms

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built with inspiration from distributed systems research
- Leveraging the power of git's content-addressable storage
- Standing on the shoulders of the open-source community

## 📬 Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/mikecerqua/ai-context-nexus/issues)
- **Discussions**: [Join the conversation](https://github.com/mikecerqua/ai-context-nexus/discussions)
- **Email**: mikecerqua@example.com

---

<p align="center">
  <i>Transforming version control into intelligent memory systems</i>
</p>
