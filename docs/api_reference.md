# AI Context Nexus API Reference

## Table of Contents

- [Core Classes](#core-classes)
  - [ContextNexus](#contextnexus)
  - [Context](#context)
  - [Agent](#agent)
- [Memory Management](#memory-management)
  - [HierarchicalCache](#hierarchicalcache)
  - [SemanticIndex](#semanticindex)
- [Agent Protocol](#agent-protocol)
  - [AgentOrchestrator](#agentorchestrator)
  - [AgentConfig](#agentconfig)
- [Utilities](#utilities)
  - [GitRepository](#gitrepository)
  - [ContextChain](#contextchain)

---

## Core Classes

### ContextNexus

The main orchestrator for the AI Context Nexus system.

```python
class ContextNexus:
    def __init__(self, config_path: str = "config/config.json")
```

#### Methods

##### `create_context(content: str, **kwargs) -> str`

Create and store a new context in the system.

**Parameters:**
- `content` (str): The context content to store
- `context_type` (str, optional): Type of context (e.g., "analysis", "conversation")
- `tags` (List[str], optional): Tags for categorization
- `agent_id` (str, optional): ID of the agent creating the context
- `metadata` (dict, optional): Additional metadata

**Returns:**
- `str`: Unique context ID (format: `ctx_[hash]`)

**Example:**
```python
context_id = nexus.create_context(
    content="Analysis of system architecture",
    context_type="technical",
    tags=["architecture", "analysis"],
    agent_id="agent_001"
)
```

##### `get_context(context_id: str) -> Context`

Retrieve a specific context by ID.

**Parameters:**
- `context_id` (str): The unique context identifier

**Returns:**
- `Context`: Context object containing content and metadata

**Raises:**
- `ContextNotFoundError`: If context doesn't exist

##### `search(query: str, limit: int = 10, threshold: float = 0.7) -> List[Context]`

Perform semantic search across all contexts.

**Parameters:**
- `query` (str): Natural language search query
- `limit` (int): Maximum number of results (default: 10)
- `threshold` (float): Similarity threshold 0-1 (default: 0.7)

**Returns:**
- `List[Context]`: Sorted list of matching contexts

**Example:**
```python
results = nexus.search("database optimization", limit=5)
for context in results:
    print(f"{context.id}: {context.similarity_score}")
```

##### `get_context_chain(context_id: str, depth: int = 5) -> ContextChain`

Retrieve a chain of related contexts.

**Parameters:**
- `context_id` (str): Starting context ID
- `depth` (int): Maximum chain depth (default: 5)

**Returns:**
- `ContextChain`: Linked list of related contexts

##### `get_graph_insights() -> dict`

Get insights from the semantic knowledge graph.

**Returns:**
- `dict`: Graph statistics and clustering information

**Example:**
```python
insights = nexus.get_graph_insights()
print(f"Total nodes: {insights['node_count']}")
print(f"Clusters: {insights['clusters']}")
```

---

### Context

Represents a single context unit in the system.

```python
@dataclass
class Context:
    id: str
    content: str
    timestamp: datetime
    context_type: str
    agent_id: str
    tags: List[str]
    embedding: np.ndarray
    metadata: dict
```

#### Properties

- `id` (str): Unique identifier
- `content` (str): The actual context content
- `timestamp` (datetime): Creation timestamp
- `context_type` (str): Context classification
- `agent_id` (str): Creating agent's ID
- `tags` (List[str]): Associated tags
- `embedding` (np.ndarray): Vector embedding (768 dimensions)
- `metadata` (dict): Additional metadata

#### Methods

##### `to_dict() -> dict`

Convert context to dictionary representation.

##### `from_dict(data: dict) -> Context`

Create context from dictionary (class method).

---

### Agent

Represents an AI agent that can interact with the nexus.

```python
class Agent:
    def __init__(self, agent_id: str, capabilities: List[str] = None)
```

#### Methods

##### `submit_context(content: str, **kwargs) -> str`

Submit a new context to the nexus.

**Parameters:**
- `content` (str): Context content
- `**kwargs`: Additional context parameters

**Returns:**
- `str`: Context ID

##### `retrieve_contexts(query: str, limit: int = 5) -> List[Context]`

Retrieve relevant contexts for the agent.

**Parameters:**
- `query` (str): Search query
- `limit` (int): Maximum results

**Returns:**
- `List[Context]`: Relevant contexts

##### `subscribe_to_updates(filter_func: Callable) -> AsyncIterator[Context]`

Subscribe to real-time context updates.

**Parameters:**
- `filter_func` (Callable): Function to filter contexts

**Returns:**
- `AsyncIterator[Context]`: Stream of matching contexts

---

## Memory Management

### HierarchicalCache

Manages the three-tier memory hierarchy.

```python
class HierarchicalCache:
    def __init__(self, config: dict)
```

#### Methods

##### `get(key: str) -> Optional[Any]`

Retrieve item from cache (checks L1 → L2 → L3).

**Parameters:**
- `key` (str): Cache key

**Returns:**
- `Any`: Cached value or None

##### `put(key: str, value: Any, tier: int = 1)`

Store item in specified cache tier.

**Parameters:**
- `key` (str): Cache key
- `value` (Any): Value to cache
- `tier` (int): Target tier (1, 2, or 3)

##### `promote(key: str)`

Promote item to higher cache tier based on access patterns.

##### `evict(tier: int = 1)`

Trigger eviction in specified tier.

##### `get_stats() -> dict`

Get cache performance statistics.

**Returns:**
```python
{
    "l1": {"size": 1024, "hits": 5000, "misses": 100},
    "l2": {"size": 10240, "hits": 2000, "misses": 500},
    "l3": {"size": 102400, "hits": 1000, "misses": 50}
}
```

---

### SemanticIndex

Manages semantic embeddings and similarity search.

```python
class SemanticIndex:
    def __init__(self, dimension: int = 768)
```

#### Methods

##### `add_embedding(context_id: str, embedding: np.ndarray)`

Add context embedding to index.

**Parameters:**
- `context_id` (str): Context identifier
- `embedding` (np.ndarray): Vector embedding

##### `search(query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]`

Find k-nearest neighbors by embedding similarity.

**Parameters:**
- `query_embedding` (np.ndarray): Query vector
- `k` (int): Number of neighbors

**Returns:**
- `List[Tuple[str, float]]`: List of (context_id, similarity_score)

##### `build_graph(threshold: float = 0.7) -> nx.Graph`

Build semantic similarity graph.

**Parameters:**
- `threshold` (float): Minimum similarity for edge creation

**Returns:**
- `nx.Graph`: NetworkX graph object

---

## Agent Protocol

### AgentOrchestrator

Manages multiple agents and coordinates their interactions.

```python
class AgentOrchestrator:
    def __init__(self, nexus: ContextNexus)
```

#### Methods

##### `register_agent(agent: Agent, config: AgentConfig)`

Register a new agent with the orchestrator.

**Parameters:**
- `agent` (Agent): Agent instance
- `config` (AgentConfig): Agent configuration

##### `broadcast_context(context: Context, exclude: List[str] = None)`

Broadcast context to all registered agents.

**Parameters:**
- `context` (Context): Context to broadcast
- `exclude` (List[str]): Agent IDs to exclude

##### `route_request(request: dict) -> Agent`

Route request to appropriate agent based on capabilities.

**Parameters:**
- `request` (dict): Request specification

**Returns:**
- `Agent`: Selected agent for handling request

##### `get_agent_stats() -> dict`

Get statistics for all registered agents.

---

### AgentConfig

Configuration for agent behavior and limits.

```python
@dataclass
class AgentConfig:
    rate_limit: int = 100  # Requests per minute
    max_context_size: int = 10000  # Characters
    capabilities: List[str] = None
    priority: int = 1
    timeout: int = 30  # Seconds
```

---

## Utilities

### GitRepository

Wrapper for git operations optimized for context storage.

```python
class GitRepository:
    def __init__(self, path: str)
```

#### Methods

##### `commit_context(context: Context) -> str`

Commit context as git commit.

**Parameters:**
- `context` (Context): Context to commit

**Returns:**
- `str`: Git commit SHA

##### `get_context_by_sha(sha: str) -> Context`

Retrieve context from git commit.

**Parameters:**
- `sha` (str): Git commit SHA

**Returns:**
- `Context`: Reconstructed context

##### `search_history(pattern: str) -> List[str]`

Search git history for pattern.

**Parameters:**
- `pattern` (str): Search pattern

**Returns:**
- `List[str]`: Matching commit SHAs

---

### ContextChain

Represents a chain of related contexts.

```python
class ContextChain:
    def __init__(self, root: Context)
```

#### Methods

##### `add_child(context: Context)`

Add child context to chain.

##### `get_ancestors(depth: int = None) -> List[Context]`

Get ancestor contexts up to specified depth.

##### `get_descendants(depth: int = None) -> List[Context]`

Get descendant contexts up to specified depth.

##### `to_graph() -> nx.DiGraph`

Convert chain to directed graph.

---

## Error Handling

### Custom Exceptions

```python
class ContextNexusError(Exception):
    """Base exception for Context Nexus"""

class ContextNotFoundError(ContextNexusError):
    """Raised when context doesn't exist"""

class CacheError(ContextNexusError):
    """Raised on cache operation failure"""

class AgentError(ContextNexusError):
    """Raised on agent operation failure"""

class IndexError(ContextNexusError):
    """Raised on semantic index failure"""
```

---

## Configuration

### Configuration File Structure

```json
{
  "repository": {
    "path": "./context_repo",
    "auto_commit": true,
    "branch_strategy": "single"
  },
  "cache": {
    "l1_size_mb": 100,
    "l2_size_mb": 1000,
    "l3_size_mb": 10000,
    "eviction_policy": "lru",
    "compression": "lz4"
  },
  "semantic": {
    "embedding_dim": 768,
    "metric": "cosine",
    "index_type": "hnsw",
    "similarity_threshold": 0.7
  },
  "agents": {
    "max_agents": 100,
    "default_timeout": 30,
    "rate_limit": 1000
  },
  "security": {
    "encryption": false,
    "auth_required": false,
    "audit_logging": true
  }
}
```

---

## Usage Examples

### Complete Workflow Example

```python
from ai_context_nexus import ContextNexus, Agent, AgentConfig

# Initialize system
nexus = ContextNexus(config_path="config/config.json")

# Create agents
analyst = Agent("analyst_01", capabilities=["analysis", "research"])
writer = Agent("writer_01", capabilities=["documentation", "summary"])

# Register agents
orchestrator = nexus.get_orchestrator()
orchestrator.register_agent(analyst, AgentConfig(rate_limit=50))
orchestrator.register_agent(writer, AgentConfig(rate_limit=30))

# Analyst creates context
context_id = analyst.submit_context(
    "System shows 30% performance degradation under load",
    context_type="analysis",
    tags=["performance", "issue"]
)

# Writer retrieves and builds upon it
related = writer.retrieve_contexts("performance analysis", limit=5)
summary_id = writer.submit_context(
    f"Based on analysis {context_id}: Recommend immediate optimization",
    context_type="summary",
    parent_id=context_id
)

# Get the context chain
chain = nexus.get_context_chain(summary_id, depth=10)
print(f"Chain length: {len(chain)}")

# Analyze the knowledge graph
insights = nexus.get_graph_insights()
print(f"Knowledge clusters: {insights['clusters']}")
```

### Async Operations Example

```python
import asyncio
from ai_context_nexus import ContextNexus

async def monitor_contexts(nexus):
    """Monitor real-time context updates"""
    async for context in nexus.subscribe_to_updates():
        if "urgent" in context.tags:
            print(f"URGENT: {context.id} - {context.content[:50]}")
        
async def main():
    nexus = ContextNexus()
    await monitor_contexts(nexus)

asyncio.run(main())
```

---

## Performance Tips

1. **Batch Operations**: Use batch methods when processing multiple contexts
2. **Embedding Cache**: Pre-compute and cache embeddings for frequent queries
3. **Graph Pruning**: Periodically prune the semantic graph to maintain performance
4. **Compression**: Enable LZ4 compression for L2 cache to reduce memory usage
5. **Index Optimization**: Use HNSW index for production with > 10k contexts

---

## Rate Limits

Default rate limits per tier:

| Operation | Rate Limit | Burst |
|-----------|------------|-------|
| Context Creation | 100/min | 200 |
| Context Retrieval | 1000/min | 2000 |
| Semantic Search | 50/min | 100 |
| Graph Query | 20/min | 40 |
| Bulk Operations | 10/min | 20 |

---

## Versioning

API follows Semantic Versioning (SemVer):
- **Major**: Breaking changes
- **Minor**: New features, backwards compatible
- **Patch**: Bug fixes

Current version: `1.0.0`