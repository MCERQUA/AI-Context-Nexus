# AI Context Nexus - Complete Usage Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Agent Integration](#agent-integration)
4. [Context Management](#context-management)
5. [Memory Hierarchy Usage](#memory-hierarchy-usage)
6. [Git Integration](#git-integration)
7. [API Reference](#api-reference)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ai-context-nexus
cd ai-context-nexus

# Run the installation script
chmod +x scripts/install.sh
./scripts/install.sh

# Configure your API keys
cp config/agents.example.json config/agents.json
# Edit config/agents.json with your actual API keys

# Start the system
./nexus start

# Check status
./nexus status
```

### Basic Usage Example
```python
from ai_context_nexus import ContextNexus, Agent, Context, ContextType

# Initialize the nexus
nexus = ContextNexus()

# Create and register agents
claude = Agent("claude", api_key="your_key")
gpt = Agent("gpt", api_key="your_key")

nexus.register(claude)
nexus.register(gpt)

# Create a context
context = Context(
    type=ContextType.ANALYSIS,
    content="Analyze this Python codebase for security vulnerabilities",
    metadata={"project": "my_app", "priority": "high"}
)

# Process with multiple agents
responses = await nexus.broadcast_context(context)

# Get consensus response
consensus = await nexus.get_consensus(context, min_agents=2)
```

## Core Concepts

### 1. Contexts
Contexts are the fundamental unit of information in the system:

```python
from datetime import datetime, timezone
from ai_context_nexus import Context, ContextType

# Create different types of contexts
code_context = Context(
    id="ctx_001",
    type=ContextType.CODE,
    content="""
    def vulnerable_function(user_input):
        # SQL injection vulnerability
        query = f"SELECT * FROM users WHERE id = {user_input}"
        return execute_query(query)
    """,
    metadata={
        "language": "python",
        "vulnerability": "sql_injection",
        "severity": "high"
    },
    agent_id="security_analyzer",
    timestamp=datetime.now(timezone.utc)
)

# Chain contexts (parent-child relationship)
fix_context = Context(
    id="ctx_002",
    type=ContextType.DECISION,
    content="Use parameterized queries to fix SQL injection",
    parent_id="ctx_001",  # Links to the code context
    agent_id="security_fixer",
    timestamp=datetime.now(timezone.utc)
)
```

### 2. Agents
Agents are AI entities that process contexts:

```python
from ai_context_nexus import AgentConfig, ClaudeAgent, GPTAgent, LocalAgent

# Configure different agent types
claude_config = AgentConfig(
    name="claude_analyzer",
    type="claude",
    api_key="sk-ant-...",
    model_name="claude-3-opus-20240229",
    max_tokens=4096,
    temperature=0.7,
    capabilities=[
        AgentCapability.ANALYSIS,
        AgentCapability.CODE_GENERATION
    ]
)

gpt_config = AgentConfig(
    name="gpt_coder",
    type="gpt",
    api_key="sk-...",
    model_name="gpt-4-turbo-preview",
    capabilities=[AgentCapability.CODE_GENERATION]
)

# Create agents
claude = ClaudeAgent(claude_config, context_manager)
gpt = GPTAgent(gpt_config, context_manager)

# Custom local agent with custom processing
def custom_processor(context):
    # Your custom logic here
    return f"Processed: {context.type.value}"

local = LocalAgent(
    config=AgentConfig(name="custom", type="local"),
    context_manager=context_manager,
    process_function=custom_processor
)
```

### 3. Memory Hierarchy
Three-tier memory system for optimal performance:

```python
# Memory tiers are managed automatically, but you can interact directly:

# Store in specific tier
await memory_manager.put_l1("hot_key", hot_data)  # RAM
await memory_manager.put_l2("warm_key", warm_data)  # SSD/JSON
await memory_manager.put_l3("cold_key", cold_data)  # Git

# Retrieve with automatic tier promotion
data = await memory_manager.get("any_key")
# If found in L3, automatically promoted to L2 and L1
```

## Agent Integration

### Registering Multiple LLM Providers

```python
# config/agents.json
{
  "agents": [
    {
      "name": "claude_primary",
      "type": "claude",
      "api_key": "${CLAUDE_API_KEY}",
      "model_name": "claude-3-opus-20240229",
      "rate_limit": 60,
      "capabilities": ["analysis", "code_generation"]
    },
    {
      "name": "gpt4_secondary",
      "type": "gpt",
      "api_key": "${OPENAI_API_KEY}",
      "model_name": "gpt-4-turbo-preview",
      "rate_limit": 60,
      "capabilities": ["text_generation", "summarization"]
    },
    {
      "name": "llama_local",
      "type": "local_llm",
      "model_path": "/models/llama-2-70b",
      "device": "cuda",
      "capabilities": ["text_generation"]
    },
    {
      "name": "cohere_embed",
      "type": "cohere",
      "api_key": "${COHERE_API_KEY}",
      "model_name": "embed-english-v3.0",
      "capabilities": ["embedding_generation"]
    }
  ]
}
```

### Agent Collaboration Patterns

```python
# Sequential Processing (Pipeline)
async def pipeline_example():
    # Stage 1: Analysis
    analysis = await analyzer_agent.process_context(raw_context)
    
    # Stage 2: Enhancement
    enhanced = await enhancer_agent.process_context(
        Context(content=analysis.content, parent_id=raw_context.id)
    )
    
    # Stage 3: Validation
    validated = await validator_agent.process_context(
        Context(content=enhanced.content, parent_id=enhanced.context.id)
    )
    
    return validated

# Parallel Processing (Map-Reduce)
async def parallel_example():
    # Map: Process with multiple agents in parallel
    tasks = [
        agent.process_context(context)
        for agent in agent_pool
    ]
    responses = await asyncio.gather(*tasks)
    
    # Reduce: Combine responses
    combined = combine_responses(responses)
    return combined

# Consensus Building
async def consensus_example():
    responses = await orchestrator.broadcast_context(context)
    
    # Vote on best response
    votes = {}
    for response in responses:
        score = evaluate_response(response)
        votes[response.agent_id] = score
    
    # Select winner
    winner = max(votes, key=votes.get)
    return responses[winner]
```

## Context Management

### Creating Rich Contexts

```python
# Multi-modal context with code, documentation, and analysis
rich_context = Context(
    id=generate_uuid(),
    type=ContextType.ANALYSIS,
    content={
        "code": """
        class DataProcessor:
            def process(self, data):
                # Complex processing logic
                return transformed_data
        """,
        "documentation": """
        The DataProcessor class handles transformation of raw data
        into the format required by the ML pipeline.
        """,
        "analysis": """
        Performance metrics:
        - Average processing time: 230ms
        - Memory usage: 156MB
        - Throughput: 4,300 records/sec
        """,
        "recommendations": [
            "Consider using batch processing for better throughput",
            "Implement caching for frequently accessed data",
            "Add error handling for edge cases"
        ]
    },
    metadata={
        "project": "data_pipeline",
        "version": "2.1.0",
        "author": "alice@example.com",
        "tags": ["performance", "optimization", "ml"],
        "priority": "high",
        "deadline": "2024-02-01"
    },
    agent_id="analyzer_001",
    timestamp=datetime.now(timezone.utc),
    embeddings=generate_embeddings(content)  # For semantic search
)

# Commit to the system
commit_hash = context_manager.add_context(rich_context)
```

### Context Chains and Graphs

```python
# Building a decision tree of contexts
root_context = Context(
    type=ContextType.DECISION,
    content="Should we migrate to microservices?"
)

# Branch 1: Yes path
yes_context = Context(
    type=ContextType.ANALYSIS,
    content="Benefits of microservices: scalability, independence...",
    parent_id=root_context.id
)

yes_implementation = Context(
    type=ContextType.CODE,
    content="Implementation plan for microservices migration",
    parent_id=yes_context.id
)

# Branch 2: No path
no_context = Context(
    type=ContextType.ANALYSIS,
    content="Reasons to stay monolithic: simplicity, cost...",
    parent_id=root_context.id
)

no_optimization = Context(
    type=ContextType.CODE,
    content="Monolith optimization strategies",
    parent_id=no_context.id
)

# Traverse the decision tree
def traverse_decision_tree(context_id):
    context = context_manager.get_context(context_id)
    children = context_manager.get_children(context_id)
    
    print(f"Decision: {context.content}")
    for child in children:
        traverse_decision_tree(child.id)
```

### Semantic Search

```python
# Search for similar contexts
query = "How to optimize database queries for better performance"

# Semantic search returns related contexts even without exact matches
similar_contexts = context_manager.search_contexts(
    query=query,
    max_results=10,
    similarity_threshold=0.7
)

for ctx in similar_contexts:
    print(f"Score: {ctx.similarity_score:.2f} - {ctx.content[:100]}...")

# Search with filters
filtered_results = context_manager.search_contexts(
    query=query,
    filters={
        "type": ContextType.CODE,
        "metadata.language": "sql",
        "timestamp": {"$gte": datetime(2024, 1, 1)}
    },
    max_results=5
)
```

## Memory Hierarchy Usage

### Direct Memory Management

```python
from ai_context_nexus.memory import MemoryManager, MemoryTier

memory = MemoryManager()

# Store with automatic tiering
await memory.put("key1", large_data)
# System automatically decides tier based on size and access patterns

# Force specific tier
await memory.put_tier("key2", data, MemoryTier.L1_HOT)  # Keep in RAM
await memory.put_tier("key3", data, MemoryTier.L2_WARM)  # Store on SSD
await memory.put_tier("key4", data, MemoryTier.L3_COLD)  # Archive to git

# Bulk operations
batch_data = {f"key_{i}": data[i] for i in range(1000)}
await memory.put_batch(batch_data)

# Get with statistics
data, stats = await memory.get_with_stats("key1")
print(f"Retrieved from: {stats['tier']}, Time: {stats['access_time_ms']}ms")

# Prefetch for better performance
await memory.prefetch(["key1", "key2", "key3"])  # Load into L1 cache

# Clear specific tier
await memory.clear_tier(MemoryTier.L1_HOT)
```

### Cache Warming Strategies

```python
# Preload frequently accessed contexts
async def warm_cache_daily():
    # Get most accessed contexts from last 24 hours
    popular_contexts = await context_manager.get_popular_contexts(
        time_range="24h",
        limit=100
    )
    
    # Preload into L1 cache
    for ctx in popular_contexts:
        await memory.prefetch(ctx.id)
    
    print(f"Cache warmed with {len(popular_contexts)} contexts")

# Schedule cache warming
scheduler.add_job(warm_cache_daily, trigger="cron", hour=6)
```

## Git Integration

### Working with Git History

```python
# Access git history directly
from ai_context_nexus.git import GitManager

git = GitManager(repo_path="./context_repo")

# Get commit history
commits = git.get_commits(limit=100)
for commit in commits:
    print(f"{commit.hash[:8]} - {commit.message}")
    context = git.extract_context(commit)
    print(f"  Type: {context.type}, Agent: {context.agent_id}")

# Time travel - get context at specific point
historical_context = git.get_context_at_time(
    context_id="ctx_001",
    timestamp=datetime(2024, 1, 15)
)

# Diff between versions
diff = git.diff_contexts(
    context_id="ctx_001",
    from_commit="abc123",
    to_commit="def456"
)
print(f"Changes: {diff}")

# Branch management with JJ
jj = JujutsuManager(repo_path="./context_repo")

# Create feature branch
jj.create_branch("experiment/new_algorithm")

# Work on branch
experimental_context = Context(...)
context_manager.add_context(experimental_context)

# Merge back
jj.merge_branch("experiment/new_algorithm", "main")
```

### Git Hooks for Validation

```bash
# .git/hooks/pre-commit
#!/bin/bash
# Validate contexts before commit

python -c "
from ai_context_nexus import validate_context
import sys
import json

commit_msg = sys.stdin.read()
if 'CONTEXT' in commit_msg:
    # Extract and validate context
    context_data = extract_context_from_commit(commit_msg)
    if not validate_context(context_data):
        print('Invalid context format')
        sys.exit(1)
"
```

## API Reference

### REST API Endpoints

```python
# Start the API server
from ai_context_nexus.api import create_app

app = create_app()
app.run(host="0.0.0.0", port=8080)
```

#### Context Endpoints
```bash
# Create context
POST /api/v1/contexts
{
  "type": "analysis",
  "content": "...",
  "metadata": {...}
}

# Get context
GET /api/v1/contexts/{context_id}

# Search contexts
POST /api/v1/contexts/search
{
  "query": "optimization techniques",
  "limit": 10,
  "filters": {...}
}

# Get context chain
GET /api/v1/contexts/{context_id}/chain

# Update context metadata
PATCH /api/v1/contexts/{context_id}
{
  "metadata": {...}
}
```

#### Agent Endpoints
```bash
# List agents
GET /api/v1/agents

# Get agent details
GET /api/v1/agents/{agent_id}

# Process context with specific agent
POST /api/v1/agents/{agent_id}/process
{
  "context_id": "ctx_001"
}

# Get agent metrics
GET /api/v1/agents/{agent_id}/metrics
```

#### Memory Endpoints
```bash
# Get memory statistics
GET /api/v1/memory/stats

# Clear cache
POST /api/v1/memory/clear
{
  "tier": "L1"  # or "L2", "L3", "all"
}

# Prefetch contexts
POST /api/v1/memory/prefetch
{
  "context_ids": ["ctx_001", "ctx_002"]
}
```

### WebSocket API

```javascript
// Real-time context updates
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = () => {
  // Subscribe to context updates
  ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['context_updates', 'agent_status']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case 'context_created':
      console.log('New context:', data.context);
      break;
    case 'agent_response':
      console.log('Agent response:', data.response);
      break;
    case 'consensus_reached':
      console.log('Consensus:', data.consensus);
      break;
  }
};

// Send context for processing
ws.send(JSON.stringify({
  type: 'process_context',
  context: {
    type: 'query',
    content: 'What is the system status?'
  }
}));
```

### gRPC API

```python
# Using gRPC for high-performance communication
import grpc
from ai_context_nexus.grpc import context_pb2, context_pb2_grpc

# Create channel and stub
channel = grpc.insecure_channel('localhost:50051')
stub = context_pb2_grpc.ContextServiceStub(channel)

# Create context via gRPC
request = context_pb2.CreateContextRequest(
    type=context_pb2.ANALYSIS,
    content="Analyze system performance",
    metadata={"priority": "high"}
)

response = stub.CreateContext(request)
print(f"Created context: {response.context_id}")

# Stream contexts
for context in stub.StreamContexts(context_pb2.Empty()):
    print(f"Received: {context.id} - {context.content[:50]}...")
```

## Advanced Usage

### Distributed Deployment

```yaml
# docker-compose.yml for distributed setup
version: '3.8'

services:
  nexus-master:
    image: ai-context-nexus:latest
    environment:
      - NEXUS_ROLE=master
      - REDIS_HOST=redis
      - DB_HOST=postgres
    ports:
      - "8080:8080"
    depends_on:
      - redis
      - postgres
  
  nexus-worker-1:
    image: ai-context-nexus:latest
    environment:
      - NEXUS_ROLE=worker
      - MASTER_HOST=nexus-master
    depends_on:
      - nexus-master
  
  nexus-worker-2:
    image: ai-context-nexus:latest
    environment:
      - NEXUS_ROLE=worker
      - MASTER_HOST=nexus-master
    depends_on:
      - nexus-master
  
  redis:
    image: redis:alpine
    volumes:
      - redis-data:/data
  
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=nexus
      - POSTGRES_USER=nexus
      - POSTGRES_PASSWORD=nexus_pass
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  redis-data:
  postgres-data:
```

### Custom Agent Implementation

```python
from ai_context_nexus import AgentProtocol, Context, AgentResponse

class CustomAnalyzerAgent(AgentProtocol):
    """Custom agent for specialized analysis."""
    
    def __init__(self, config, context_manager):
        super().__init__(config, context_manager)
        self.analyzer = self.load_custom_model()
    
    def load_custom_model(self):
        # Load your custom ML model
        return YourCustomModel()
    
    async def process_context(self, context: Context) -> AgentResponse:
        # Custom processing logic
        start_time = time.time()
        
        try:
            # Preprocess
            processed_input = self.preprocess(context)
            
            # Run analysis
            analysis_result = self.analyzer.analyze(processed_input)
            
            # Post-process
            formatted_result = self.format_result(analysis_result)
            
            # Create response
            return AgentResponse(
                agent_id=self.agent_id,
                content=formatted_result,
                context_used=context,
                processing_time=time.time() - start_time,
                success=True
            )
        except Exception as e:
            return AgentResponse(
                agent_id=self.agent_id,
                content="",
                success=False,
                error_message=str(e)
            )
    
    def preprocess(self, context):
        # Your preprocessing logic
        return context.content
    
    def format_result(self, result):
        # Format the analysis result
        return json.dumps(result, indent=2)
    
    async def generate_context(self, prompt: str) -> Context:
        # Generate new context based on analysis
        analysis = self.analyzer.analyze(prompt)
        
        return Context(
            id=generate_uuid(),
            type=ContextType.ANALYSIS,
            content=str(analysis),
            metadata={"generated_by": "custom_analyzer"},
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc)
        )
    
    def serialize_state(self) -> bytes:
        # Serialize model state
        return pickle.dumps({
            'model_state': self.analyzer.get_state(),
            'config': self.config,
            'metrics': self.get_metrics()
        })
    
    def deserialize_state(self, data: bytes):
        # Restore model state
        state = pickle.loads(data)
        self.analyzer.load_state(state['model_state'])
```

### Performance Optimization

```python
# Batch processing for efficiency
async def batch_process_contexts(contexts: List[Context], batch_size=10):
    results = []
    
    for i in range(0, len(contexts), batch_size):
        batch = contexts[i:i + batch_size]
        
        # Process batch in parallel
        tasks = [
            agent.process_context(ctx)
            for ctx in batch
            for agent in agent_pool
        ]
        
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    
    return results

# Caching with TTL
from functools import lru_cache
from datetime import timedelta

@lru_cache(maxsize=1000)
def get_cached_embedding(text: str, ttl: timedelta = timedelta(hours=1)):
    # Cache embeddings to avoid recomputation
    return generate_embedding(text)

# Connection pooling
class ConnectionPool:
    def __init__(self, size=10):
        self.pool = Queue(maxsize=size)
        for _ in range(size):
            self.pool.put(self.create_connection())
    
    def get_connection(self):
        return self.pool.get()
    
    def return_connection(self, conn):
        self.pool.put(conn)
    
    @contextmanager
    def connection(self):
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.return_connection(conn)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. High Memory Usage
```python
# Monitor memory usage
import psutil

def check_memory():
    memory = psutil.virtual_memory()
    if memory.percent > 80:
        # Trigger cache eviction
        memory_manager.reduce_cache_sizes()
        
        # Force garbage collection
        import gc
        gc.collect()

# Set up monitoring
scheduler.add_job(check_memory, 'interval', seconds=30)
```

#### 2. Slow Context Retrieval
```python
# Diagnose slow queries
async def diagnose_performance():
    # Check cache hit rates
    stats = memory_manager.get_all_stats()
    
    for tier, tier_stats in stats['tiers'].items():
        if tier_stats['hit_rate'] < 0.5:
            print(f"Low hit rate in {tier}: {tier_stats['hit_rate']}")
    
    # Check index fragmentation
    if stats['l2_cache']['fragmentation'] > 100:
        print("High L2 cache fragmentation, consider defragmentation")
    
    # Rebuild semantic index if needed
    if context_manager.index.search_time_ms > 100:
        print("Rebuilding semantic index...")
        await context_manager.rebuild_index()
```

#### 3. Agent Failures
```python
# Implement circuit breaker pattern
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'half-open'
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
            
            raise e

# Use circuit breaker with agents
breaker = CircuitBreaker()
response = await breaker.call(agent.process_context, context)
```

## Best Practices

### 1. Context Design
- Keep contexts focused and single-purpose
- Include rich metadata for better searchability
- Use appropriate context types for clarity
- Link related contexts with parent_id

### 2. Agent Management
- Implement health checks for all agents
- Use circuit breakers for fault tolerance
- Monitor rate limits and quotas
- Distribute load across multiple agents

### 3. Memory Optimization
- Regularly monitor cache hit rates
- Adjust tier sizes based on usage patterns
- Implement cache warming for predictable workloads
- Use compression for large contexts

### 4. Git Integration
- Commit contexts atomically
- Use meaningful commit messages
- Tag important context milestones
- Regular backup of git repository

### 5. Security
- Encrypt sensitive contexts at rest
- Use API keys securely (environment variables)
- Implement access control for multi-user setups
- Audit log all context access

### 6. Performance
- Batch operations when possible
- Use async/await for I/O operations
- Implement connection pooling
- Profile and optimize hot paths

### 7. Monitoring
- Set up comprehensive logging
- Track key metrics (latency, throughput, errors)
- Implement alerting for critical issues
- Regular performance reviews

## Conclusion

The AI Context Nexus provides a powerful framework for multi-agent AI collaboration with persistent memory. By following this guide and best practices, you can build sophisticated AI systems that maintain context across sessions and leverage multiple AI models effectively.

For more information:
- GitHub: https://github.com/yourusername/ai-context-nexus
- Documentation: https://docs.ai-context-nexus.io
- Support: support@ai-context-nexus.io

Happy context sharing! 🚀
