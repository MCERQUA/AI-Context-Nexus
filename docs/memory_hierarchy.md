# Memory Hierarchy Design

## Overview

The AI Context Nexus implements a sophisticated three-tier memory hierarchy inspired by CPU cache architectures, optimized for AI context management. This design ensures sub-second access times while maintaining unlimited storage capacity.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Memory Hierarchy                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │              L1 Cache (RAM)                      │   │
│  │  • Size: 100 MB                                  │   │
│  │  • Access: < 1ms                                 │   │
│  │  • Storage: OrderedDict (LRU)                    │   │
│  │  • Hit Rate: ~85%                                │   │
│  └────────────────┬─────────────────────────────────┘   │
│                   ↕ Promotion/Eviction                   │
│  ┌──────────────────────────────────────────────────┐   │
│  │           L2 Cache (Compressed JSON)             │   │
│  │  • Size: 1 GB                                    │   │
│  │  • Access: < 10ms                                │   │
│  │  • Storage: LZ4 Compressed Files                 │   │
│  │  • Hit Rate: ~12%                                │   │
│  └────────────────┬─────────────────────────────────┘   │
│                   ↕ Promotion/Eviction                   │
│  ┌──────────────────────────────────────────────────┐   │
│  │           L3 Storage (Git Repository)            │   │
│  │  • Size: Unlimited                               │   │
│  │  • Access: < 100ms                               │   │
│  │  • Storage: Git Commits                          │   │
│  │  • Hit Rate: ~3%                                 │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Tier Specifications

### L1 Cache - Hot Memory

**Purpose**: Store frequently accessed contexts for instant retrieval.

**Implementation**:
```python
class L1Cache:
    def __init__(self, max_size_mb: int = 100):
        self.cache = OrderedDict()
        self.max_size = max_size_mb * 1024 * 1024
        self.current_size = 0
        self.hits = 0
        self.misses = 0
```

**Characteristics**:
- **Data Structure**: OrderedDict for O(1) access with LRU ordering
- **Eviction Policy**: Least Recently Used (LRU)
- **Capacity**: 100 MB default (configurable)
- **Typical Contents**: ~1,000 contexts
- **Use Cases**: Active conversation contexts, recent queries

**Performance Metrics**:
| Operation | Time | Complexity |
|-----------|------|------------|
| Read | < 1ms | O(1) |
| Write | < 1ms | O(1) |
| Evict | < 1ms | O(1) |
| Search | < 5ms | O(n) |

### L2 Cache - Warm Storage

**Purpose**: Store moderately accessed contexts with compression.

**Implementation**:
```python
class L2Cache:
    def __init__(self, cache_dir: str, max_size_mb: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size_mb * 1024 * 1024
        self.compression = lz4.frame
        self.index = {}  # In-memory index
```

**Characteristics**:
- **Storage Format**: JSON files with LZ4 compression
- **Compression Ratio**: ~4:1 average
- **Eviction Policy**: Least Frequently Used (LFU)
- **Capacity**: 1 GB default (configurable)
- **Typical Contents**: ~10,000 contexts
- **File Organization**: Sharded by context ID hash

**Directory Structure**:
```
l2_cache/
├── 00/
│   ├── ctx_00abc123.json.lz4
│   └── ctx_00def456.json.lz4
├── 01/
│   └── ctx_01abc789.json.lz4
├── ...
└── ff/
    └── ctx_ffxyz123.json.lz4
```

**Performance Metrics**:
| Operation | Time | Notes |
|-----------|------|-------|
| Read | < 10ms | Includes decompression |
| Write | < 20ms | Includes compression |
| Evict | < 5ms | File deletion |
| Search | < 50ms | Index-based |

### L3 Storage - Cold Archive

**Purpose**: Permanent storage with unlimited capacity.

**Implementation**:
```python
class L3Storage:
    def __init__(self, repo_path: str):
        self.repo = git.Repo(repo_path)
        self.commit_index = {}  # SHA to context mapping
```

**Characteristics**:
- **Storage Format**: Git commits with structured messages
- **Compression**: Git delta compression
- **Eviction Policy**: Never (permanent storage)
- **Capacity**: Unlimited
- **Typical Contents**: All historical contexts
- **Organization**: Chronological with semantic branching

**Commit Structure**:
```yaml
# Commit message format
---
id: ctx_abc123def456
timestamp: 2025-01-17T10:30:00Z
agent: agent_001
type: analysis
tags: [performance, optimization]
embedding_dim: 768
checksum: sha256:abc123...
---
[Context content here]
```

**Performance Metrics**:
| Operation | Time | Notes |
|-----------|------|-------|
| Read | < 100ms | Git object retrieval |
| Write | < 200ms | Commit creation |
| Search | < 500ms | Git log search |
| History | < 1000ms | Full chain retrieval |

## Cache Management Algorithms

### Promotion Algorithm

Contexts are promoted from lower to higher tiers based on access patterns:

```python
def promote_context(context_id: str, access_count: int):
    if access_count > L1_THRESHOLD and context not in L1:
        if L1.is_full():
            L1.evict_lru()
        L1.add(context)
        
    elif access_count > L2_THRESHOLD and context not in L2:
        if L2.is_full():
            L2.evict_lfu()
        L2.add(context)
```

**Promotion Thresholds**:
- L3 → L2: 3 accesses within 1 hour
- L2 → L1: 5 accesses within 10 minutes
- Direct to L1: Real-time subscription contexts

### Eviction Algorithm

Each tier uses different eviction strategies:

**L1 - LRU Eviction**:
```python
def evict_lru(self):
    if self.cache:
        oldest_key = next(iter(self.cache))
        evicted = self.cache.pop(oldest_key)
        self.current_size -= len(evicted)
        # Demote to L2
        self.l2_cache.add(oldest_key, evicted)
```

**L2 - LFU Eviction**:
```python
def evict_lfu(self):
    min_freq = min(self.frequency_map.values())
    candidates = [k for k, v in self.frequency_map.items() if v == min_freq]
    # Evict oldest among least frequent
    victim = min(candidates, key=lambda x: self.access_time[x])
    self.remove(victim)
```

### Access Pattern Optimization

The system adapts to different access patterns:

1. **Sequential Access**: Pre-fetch next contexts in chain
2. **Random Access**: Increase L1 cache size temporarily
3. **Burst Access**: Batch promotions to reduce overhead
4. **Semantic Clustering**: Keep related contexts in same tier

## Memory Efficiency

### Compression Strategies

**L2 Compression (LZ4)**:
- **Speed**: 750 MB/s compression, 3500 MB/s decompression
- **Ratio**: 4:1 average for text contexts
- **CPU Usage**: < 5% overhead

**L3 Compression (Git)**:
- **Delta Compression**: Store only differences
- **Pack Files**: Efficient storage of similar objects
- **Deduplication**: Automatic via content addressing

### Memory Layout

```python
# L1 Memory Layout (RAM)
{
    "ctx_id": {
        "content": "...",
        "embedding": numpy.array([...]),
        "metadata": {...},
        "access_count": 10,
        "last_access": timestamp
    }
}

# L2 File Format (Compressed JSON)
{
    "version": "1.0",
    "context": {
        "id": "ctx_id",
        "content": "...",
        "embedding": [...],  # List for JSON serialization
        "metadata": {...}
    },
    "cache_metadata": {
        "compressed_size": 1024,
        "original_size": 4096,
        "compression_ratio": 4.0
    }
}
```

## Performance Tuning

### Configuration Parameters

```json
{
    "cache": {
        "l1": {
            "size_mb": 100,
            "eviction_policy": "lru",
            "promotion_threshold": 5,
            "ttl_seconds": 3600
        },
        "l2": {
            "size_mb": 1000,
            "eviction_policy": "lfu",
            "promotion_threshold": 3,
            "compression": "lz4",
            "shard_count": 256
        },
        "l3": {
            "auto_gc": true,
            "pack_size_mb": 100,
            "prune_days": 0
        }
    }
}
```

### Optimization Tips

1. **L1 Size**: Set to 10% of available RAM
2. **L2 Size**: Set to available SSD space minus OS needs
3. **Sharding**: Use 256 shards for L2 to reduce file contention
4. **Batch Operations**: Group L3 commits for better performance
5. **Index Caching**: Keep L2/L3 indices in RAM

## Monitoring

### Metrics Collection

```python
class CacheMetrics:
    def __init__(self):
        self.metrics = {
            "l1_hits": 0,
            "l1_misses": 0,
            "l2_hits": 0,
            "l2_misses": 0,
            "l3_hits": 0,
            "promotions": 0,
            "evictions": 0,
            "avg_access_time": 0
        }
    
    def get_hit_rate(self, tier: str) -> float:
        hits = self.metrics[f"{tier}_hits"]
        misses = self.metrics[f"{tier}_misses"]
        return hits / (hits + misses) if (hits + misses) > 0 else 0
```

### Performance Dashboard

```
┌─────────────────────────────────────────┐
│          Cache Performance              │
├─────────────────────────────────────────┤
│ L1 Cache:                               │
│   Hit Rate: 85.3%                       │
│   Size: 87.2 MB / 100 MB                │
│   Contexts: 891                         │
│   Avg Access: 0.8ms                     │
│                                          │
│ L2 Cache:                               │
│   Hit Rate: 12.1%                       │
│   Size: 734 MB / 1000 MB                │
│   Contexts: 8,421                       │
│   Avg Access: 8.3ms                     │
│                                          │
│ L3 Storage:                             │
│   Total Contexts: 52,381                │
│   Repository Size: 2.3 GB               │
│   Avg Access: 67ms                      │
└─────────────────────────────────────────┘
```

## Failure Recovery

### L1 Recovery
- Volatile memory - rebuilt from L2/L3 on restart
- Periodic snapshots to disk (optional)

### L2 Recovery
- Checksums for corruption detection
- Rebuild from L3 if corrupted
- Redundant index files

### L3 Recovery
- Git's built-in integrity checking
- Multiple remotes for backup
- Incremental backups via git bundle

## Future Enhancements

1. **L0 Cache**: CPU cache optimization for embeddings
2. **Distributed L2**: Redis/Memcached for multi-node setups
3. **L3 Sharding**: Multiple git repositories for scale
4. **Predictive Prefetching**: ML-based access prediction
5. **Compression Optimization**: Context-aware compression
6. **NUMA Awareness**: Optimize for multi-socket systems