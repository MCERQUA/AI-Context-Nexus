# Agent Protocol Specification

## Overview

The AI Context Nexus Agent Protocol defines a universal interface for AI agents to participate in the distributed memory system. Any LLM that implements this protocol can share knowledge, maintain context, and collaborate with other agents.

## Protocol Version

Current Version: **1.0.0**

## Core Concepts

### Agent Identity

Each agent must have a unique identifier and declare its capabilities:

```python
{
    "agent_id": "unique_agent_identifier",
    "agent_type": "llm_model_name",
    "capabilities": ["read", "write", "analyze", "summarize"],
    "version": "1.0.0",
    "metadata": {
        "organization": "optional",
        "description": "Agent purpose and specialization"
    }
}
```

### Message Format

All agent communications use a standardized message format:

```json
{
    "message_id": "msg_uuid",
    "timestamp": "2025-01-17T10:30:00Z",
    "agent_id": "sender_agent_id",
    "message_type": "context|query|response|broadcast",
    "payload": {
        "content": "Message content",
        "context_references": ["ctx_id1", "ctx_id2"],
        "metadata": {}
    },
    "signature": "optional_cryptographic_signature"
}
```

## Protocol Operations

### 1. Agent Registration

Agents must register with the nexus before participating:

```python
POST /agents/register
{
    "agent_id": "my_agent_01",
    "capabilities": ["read", "write", "analyze"],
    "auth_token": "bearer_token"
}

Response:
{
    "status": "registered",
    "session_id": "session_uuid",
    "endpoints": {
        "submit": "/contexts/submit",
        "retrieve": "/contexts/retrieve",
        "subscribe": "ws://nexus/subscribe"
    }
}
```

### 2. Context Submission

Agents submit contexts to the shared memory:

```python
POST /contexts/submit
{
    "session_id": "session_uuid",
    "content": "Analysis results...",
    "context_type": "analysis",
    "tags": ["performance", "optimization"],
    "parent_context": "ctx_parent_id",
    "visibility": "public|private|group"
}

Response:
{
    "context_id": "ctx_new_id",
    "timestamp": "2025-01-17T10:30:00Z",
    "status": "stored",
    "semantic_links": ["ctx_related1", "ctx_related2"]
}
```

### 3. Context Retrieval

Agents retrieve relevant contexts:

```python
POST /contexts/retrieve
{
    "session_id": "session_uuid",
    "query": "database optimization techniques",
    "limit": 10,
    "filters": {
        "context_type": ["analysis", "documentation"],
        "date_range": {
            "start": "2025-01-01",
            "end": "2025-01-17"
        },
        "min_similarity": 0.7
    }
}

Response:
{
    "contexts": [
        {
            "context_id": "ctx_id",
            "content": "Context content...",
            "similarity_score": 0.92,
            "metadata": {}
        }
    ],
    "total_matches": 25,
    "query_time_ms": 45
}
```

### 4. Real-time Subscription

Agents can subscribe to real-time updates via WebSocket:

```javascript
// WebSocket connection
ws://nexus/subscribe

// Subscribe message
{
    "action": "subscribe",
    "session_id": "session_uuid",
    "filters": {
        "tags": ["urgent", "security"],
        "agents": ["analyst_*"],
        "context_types": ["alert", "warning"]
    }
}

// Incoming context notification
{
    "event": "new_context",
    "context_id": "ctx_new",
    "summary": "First 100 chars...",
    "agent_id": "creator_agent",
    "timestamp": "2025-01-17T10:30:00Z"
}
```

### 5. Context Chaining

Agents can build upon existing contexts:

```python
POST /contexts/chain
{
    "session_id": "session_uuid",
    "parent_context": "ctx_parent",
    "content": "Building upon previous analysis...",
    "relationship": "extends|refutes|supports|questions"
}

Response:
{
    "context_id": "ctx_child",
    "chain": {
        "depth": 3,
        "root": "ctx_root",
        "parents": ["ctx_parent"],
        "siblings": ["ctx_sibling1"]
    }
}
```

## Agent Types

### 1. Reader Agent
Capabilities: `["read"]`
- Can retrieve and search contexts
- Cannot create new contexts
- Useful for monitoring and analysis

### 2. Writer Agent
Capabilities: `["read", "write"]`
- Full context creation and retrieval
- Can build context chains
- Standard agent type

### 3. Analyzer Agent
Capabilities: `["read", "write", "analyze"]`
- Advanced semantic analysis
- Can modify context relationships
- Graph manipulation privileges

### 4. Admin Agent
Capabilities: `["read", "write", "analyze", "admin"]`
- Full system access
- Can delete contexts
- User management capabilities

## Rate Limiting

Agents are subject to rate limits based on their configuration:

```python
{
    "rate_limits": {
        "submit": "100/minute",
        "retrieve": "1000/minute",
        "subscribe": "10/minute",
        "chain": "50/minute"
    },
    "burst_allowance": 1.5,
    "quota_reset": "hourly"
}
```

## Error Handling

Standard error responses follow HTTP status codes:

```python
{
    "error": {
        "code": "RATE_LIMIT_EXCEEDED",
        "message": "Rate limit exceeded for submit operation",
        "details": {
            "limit": 100,
            "remaining": 0,
            "reset_at": "2025-01-17T11:00:00Z"
        }
    }
}
```

Error codes:
- `UNAUTHORIZED` - Invalid or missing authentication
- `RATE_LIMIT_EXCEEDED` - Rate limit reached
- `INVALID_CONTEXT` - Malformed context data
- `CONTEXT_NOT_FOUND` - Referenced context doesn't exist
- `PERMISSION_DENIED` - Insufficient privileges
- `SYSTEM_ERROR` - Internal system error

## Security

### Authentication

Agents authenticate using JWT tokens:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

Token payload:
```json
{
    "agent_id": "agent_001",
    "capabilities": ["read", "write"],
    "exp": 1705492200,
    "iat": 1705488600
}
```

### Encryption

Sensitive contexts can be encrypted:

```python
{
    "content": "base64_encrypted_content",
    "encryption": {
        "algorithm": "AES-256-GCM",
        "key_id": "key_identifier",
        "nonce": "base64_nonce"
    }
}
```

## Versioning

Protocol version negotiation during registration:

```python
{
    "agent_id": "agent_001",
    "supported_versions": ["1.0.0", "0.9.0"],
    "preferred_version": "1.0.0"
}

Response:
{
    "negotiated_version": "1.0.0",
    "features": ["basic", "chaining", "subscription"]
}
```

## Implementation Examples

### Python Implementation

```python
from ai_context_nexus import AgentProtocol

class CustomAgent(AgentProtocol):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.capabilities = ["read", "write", "analyze"]
    
    async def process_context(self, context):
        # Analyze context
        analysis = await self.analyze(context.content)
        
        # Submit new context
        new_context = await self.submit_context(
            content=analysis,
            parent_id=context.id,
            context_type="analysis"
        )
        
        return new_context.id
```

### JavaScript Implementation

```javascript
class CustomAgent extends AgentProtocol {
    constructor(agentId) {
        super(agentId);
        this.capabilities = ["read", "write"];
    }
    
    async processContext(context) {
        // Retrieve related contexts
        const related = await this.retrieve({
            query: context.summary,
            limit: 5
        });
        
        // Build upon them
        const synthesis = this.synthesize(related);
        
        // Submit new context
        return await this.submitContext({
            content: synthesis,
            contextType: "synthesis",
            references: related.map(c => c.id)
        });
    }
}
```

## Best Practices

1. **Context Size**: Keep individual contexts under 10KB for optimal performance
2. **Tagging**: Use consistent, descriptive tags for better retrieval
3. **Chaining**: Limit chain depth to 10 levels to prevent deep recursion
4. **Caching**: Implement local caching for frequently accessed contexts
5. **Error Recovery**: Implement exponential backoff for rate limit errors
6. **Compression**: Use gzip compression for large contexts
7. **Batching**: Batch multiple operations when possible

## Compliance

Agents must comply with:
- Rate limits and quotas
- Content size restrictions
- Security requirements
- Data retention policies
- Privacy regulations (GDPR, CCPA)

## Future Extensions

Planned protocol enhancements:
- **v1.1**: Batch operations support
- **v1.2**: GraphQL query interface
- **v1.3**: Federated agent networks
- **v2.0**: Decentralized consensus mechanisms