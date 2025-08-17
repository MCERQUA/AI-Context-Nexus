# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Context Nexus is a distributed memory and context management system that enables multiple AI agents to share knowledge and collaborate effectively. It uses Git as a persistent versioned storage backend with a novel "Git-as-Memory" architecture, combined with a three-tier memory hierarchy for optimized performance.

## Core Architecture

The system implements several innovative concepts:

1. **Git-as-Memory (GaM)**: Every context is stored as a git commit with extensive metadata embedded in commit messages, enabling versioned context retrieval and semantic search across history.

2. **Three-Tier Memory Hierarchy**:
   - L1 Cache: In-memory hot data (< 1ms access) - OrderedDict with LRU eviction
   - L2 Cache: JSON files with LZ4 compression (< 10ms access) 
   - L3 Storage: Full git history (< 100ms access)

3. **Semantic Graph**: NetworkX-based graph connecting contexts by semantic similarity, enabling graph-based queries and clustering.

4. **Agent Protocol**: Standardized interface allowing any LLM to participate in the system through the AgentOrchestrator.

## Key Components

### Core Context Manager (`core/context_manager.py`)
- `ContextManager`: Main orchestrator handling context storage, retrieval, and semantic indexing
- `Context`: Data class representing shareable context chunks with embeddings and metadata
- `GitRepository`: Wrapper for git operations optimized for context storage
- `HierarchicalCache`: Implements the three-tier memory system with automatic promotion/eviction
- `SemanticIndex`: Manages embeddings and similarity search (placeholder for FAISS in production)

### Agent Protocol (`agents/agent_protocol.py`)
- `AgentOrchestrator`: Manages multiple agents and broadcasts contexts
- `LocalAgent`: Default local processing agent
- `AgentConfig`: Configuration for agent capabilities and rate limits

### CLI Tool (`scripts/nexus_cli.py`)
- Rich-based interactive CLI for system management
- Commands for context creation, search, agent management, and system control
- Interactive mode for exploratory usage

## Development Commands

```bash
# Quick setup and initialization
./quickstart.sh

# Install dependencies (after creating virtual environment)
python3 -m venv venv
source venv/bin/activate
pip install aiohttp pyyaml click rich gitpython redis numpy lz4 networkx tabulate

# Run the CLI
python scripts/nexus_cli.py --help

# Interactive mode
python scripts/nexus_cli.py interactive

# Context operations
python scripts/nexus_cli.py context create "Your context content" -t analysis
python scripts/nexus_cli.py context search "search query" -l 10
python scripts/nexus_cli.py context get ctx_id
python scripts/nexus_cli.py context chain ctx_id
python scripts/nexus_cli.py context list

# Agent operations  
python scripts/nexus_cli.py agent list
python scripts/nexus_cli.py agent process ctx_id --agent-id local_processor

# System management (requires tmux_orchestrator.sh)
python scripts/nexus_cli.py system start
python scripts/nexus_cli.py system status
python scripts/nexus_cli.py system stop

# Memory statistics
python scripts/nexus_cli.py memory stats
```

## Configuration

The system uses a comprehensive JSON configuration (`config/config.json`) with sections for:
- Repository settings (git/jujutsu integration)
- Memory hierarchy parameters (cache sizes, eviction policies)
- Agent rate limits and load balancing
- Semantic indexing (dimension, metric type)
- Networking (distributed mode, Redis)
- Security (encryption, authentication)
- Monitoring (metrics, health checks)
- Recovery (auto-recovery, backups)

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose build
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

## Testing

```bash
# Run the context manager directly to test basic functionality
python core/context_manager.py

# This will create a test repository at ./test_context_repo and demonstrate:
# - Context creation and storage
# - Semantic search
# - Context chains
# - Graph insights
```

## Important Implementation Notes

1. **Embeddings**: Currently uses random embeddings for demonstration. In production, integrate a real embedding model (BERT, GPT, etc.) in `_generate_embeddings()`.

2. **Semantic Index**: Uses in-memory similarity search. For production, integrate FAISS or similar vector database in the `SemanticIndex` class.

3. **Distributed Mode**: Redis integration is optional but recommended for multi-node deployments. Enable with `enable_redis=True`.

4. **Jujutsu Integration**: The `JujutsuWrapper` class provides advanced branching for parallel agent workflows but requires jj to be installed separately.

5. **Commit Message Format**: The system embeds full context in git commit messages using a structured format with YAML metadata and content sections. This enables reconstruction of contexts from git history alone.

## Performance Considerations

- The L1 cache uses OrderedDict for O(1) access with LRU eviction
- L2 cache uses LZ4 compression for fast decompression  
- Git packfiles and delta compression optimize L3 storage
- The semantic graph can grow large; consider pruning strategies for production
- Batch operations are supported through the ThreadPoolExecutor

## Error Handling

- All git operations are wrapped with error handling in `GitRepository`
- The cache hierarchy gracefully handles missing files and corrupted data
- Agent failures trigger circuit breakers (when configured)
- The system supports automatic recovery through checkpointing

## Security Notes

- The configuration supports AES-256-GCM encryption for sensitive contexts
- JWT authentication can be enabled for API access
- TLS support is available for network communication
- Audit logging tracks all context operations when enabled