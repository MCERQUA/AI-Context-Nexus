#!/usr/bin/env python3
"""
AI Context Nexus - Memory Manager Server

This module implements the memory management server that handles
the three-tier memory hierarchy and provides optimized access to contexts.
"""

import os
import sys
import json
import asyncio
import logging
import time
import mmap
import struct
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import OrderedDict, defaultdict
import numpy as np
import lz4.frame
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from aiohttp import web
import psutil
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryTier(Enum):
    """Memory tier levels."""
    L1_HOT = "l1_hot"       # In-memory, frequently accessed
    L2_WARM = "l2_warm"      # SSD/JSON, recently accessed
    L3_COLD = "l3_cold"      # Git history, rarely accessed


@dataclass
class MemoryStats:
    """Statistics for memory usage."""
    tier: MemoryTier
    total_capacity: int
    used_capacity: int
    hit_count: int
    miss_count: int
    eviction_count: int
    avg_access_time_ms: float
    
    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    @property
    def usage_percent(self) -> float:
        return (self.used_capacity / self.total_capacity * 100) if self.total_capacity > 0 else 0.0


class MemoryMappedCache:
    """
    Memory-mapped file cache for zero-copy access.
    """
    
    def __init__(self, cache_dir: str, max_size: int = 1024 * 1024 * 1024):  # 1GB default
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.mmap_files = {}
        self.index = {}
        self.lock = threading.RLock()
        
        # Create main cache file
        self.cache_file = self.cache_dir / "cache.mmap"
        self.init_cache_file()
    
    def init_cache_file(self):
        """Initialize memory-mapped cache file."""
        if not self.cache_file.exists():
            # Create file with initial size
            with open(self.cache_file, 'wb') as f:
                f.write(b'\x00' * self.max_size)
        
        # Open memory-mapped file
        self.file_handle = open(self.cache_file, 'r+b')
        self.mmap = mmap.mmap(
            self.file_handle.fileno(), 
            0,  # Map entire file
            access=mmap.ACCESS_WRITE
        )
        
        # Initialize free space tracking
        self.free_blocks = [(0, self.max_size)]  # (offset, size) tuples
        self.used_blocks = {}  # key -> (offset, size)
    
    def put(self, key: str, data: bytes) -> bool:
        """Store data in memory-mapped cache."""
        with self.lock:
            # Check if key already exists
            if key in self.used_blocks:
                self.delete(key)
            
            # Find suitable free block
            data_size = len(data)
            block = self.find_free_block(data_size + 8)  # +8 for size header
            
            if not block:
                return False  # No space available
            
            offset, block_size = block
            
            # Write size header and data
            self.mmap[offset:offset+8] = struct.pack('<Q', data_size)
            self.mmap[offset+8:offset+8+data_size] = data
            
            # Update tracking
            self.used_blocks[key] = (offset, data_size + 8)
            
            # Update free blocks
            self.update_free_blocks(offset, data_size + 8, block_size)
            
            return True
    
    def get(self, key: str) -> Optional[bytes]:
        """Retrieve data from memory-mapped cache."""
        with self.lock:
            if key not in self.used_blocks:
                return None
            
            offset, size = self.used_blocks[key]
            
            # Read size header
            size_bytes = self.mmap[offset:offset+8]
            data_size = struct.unpack('<Q', size_bytes)[0]
            
            # Read data
            data = bytes(self.mmap[offset+8:offset+8+data_size])
            
            return data
    
    def delete(self, key: str):
        """Delete data from cache."""
        with self.lock:
            if key not in self.used_blocks:
                return
            
            offset, size = self.used_blocks[key]
            
            # Mark block as free
            self.free_blocks.append((offset, size))
            self.free_blocks.sort()  # Keep sorted by offset
            
            # Coalesce adjacent free blocks
            self.coalesce_free_blocks()
            
            del self.used_blocks[key]
    
    def find_free_block(self, size: int) -> Optional[Tuple[int, int]]:
        """Find a free block of sufficient size."""
        for i, (offset, block_size) in enumerate(self.free_blocks):
            if block_size >= size:
                # Remove from free list
                self.free_blocks.pop(i)
                return (offset, block_size)
        return None
    
    def update_free_blocks(self, offset: int, used_size: int, block_size: int):
        """Update free blocks after allocation."""
        if used_size < block_size:
            # Add remaining space back to free blocks
            self.free_blocks.append((offset + used_size, block_size - used_size))
            self.free_blocks.sort()
    
    def coalesce_free_blocks(self):
        """Merge adjacent free blocks."""
        if len(self.free_blocks) <= 1:
            return
        
        merged = []
        current_offset, current_size = self.free_blocks[0]
        
        for offset, size in self.free_blocks[1:]:
            if offset == current_offset + current_size:
                # Adjacent blocks, merge them
                current_size += size
            else:
                # Non-adjacent, save current and start new
                merged.append((current_offset, current_size))
                current_offset, current_size = offset, size
        
        merged.append((current_offset, current_size))
        self.free_blocks = merged
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_used = sum(size for _, size in self.used_blocks.values())
            total_free = sum(size for _, size in self.free_blocks)
            
            return {
                'total_capacity': self.max_size,
                'used_capacity': total_used,
                'free_capacity': total_free,
                'num_items': len(self.used_blocks),
                'fragmentation': len(self.free_blocks),
                'largest_free_block': max((s for _, s in self.free_blocks), default=0)
            }
    
    def cleanup(self):
        """Clean up memory-mapped files."""
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'file_handle'):
            self.file_handle.close()


class BloomFilter:
    """
    Bloom filter for quick negative cache checks.
    """
    
    def __init__(self, capacity: int = 1000000, error_rate: float = 0.01):
        # Calculate optimal parameters
        self.capacity = capacity
        self.error_rate = error_rate
        
        # Optimal bit array size
        self.size = int(-capacity * np.log(error_rate) / (np.log(2) ** 2))
        
        # Optimal number of hash functions
        self.num_hashes = int(self.size * np.log(2) / capacity)
        
        # Bit array
        self.bits = np.zeros(self.size, dtype=bool)
        
        logger.info(f"Bloom filter initialized: size={self.size}, hashes={self.num_hashes}")
    
    def _hash(self, key: str, seed: int) -> int:
        """Generate hash with seed."""
        h = hashlib.sha256(f"{key}{seed}".encode()).digest()
        return int.from_bytes(h[:8], 'big') % self.size
    
    def add(self, key: str):
        """Add key to bloom filter."""
        for i in range(self.num_hashes):
            self.bits[self._hash(key, i)] = True
    
    def contains(self, key: str) -> bool:
        """Check if key might be in the set."""
        return all(self.bits[self._hash(key, i)] for i in range(self.num_hashes))
    
    def clear(self):
        """Clear the bloom filter."""
        self.bits.fill(False)
    
    def load_factor(self) -> float:
        """Get the load factor (percentage of bits set)."""
        return np.mean(self.bits)


class AdaptiveCachePolicy:
    """
    Adaptive caching policy that learns access patterns.
    """
    
    def __init__(self):
        self.access_history = defaultdict(list)
        self.frequency_scores = defaultdict(float)
        self.recency_scores = defaultdict(float)
        self.prediction_model = None
        self.time_window = 3600  # 1 hour window
    
    def record_access(self, key: str, timestamp: float):
        """Record an access event."""
        self.access_history[key].append(timestamp)
        
        # Update frequency score
        self.frequency_scores[key] = len(self.access_history[key])
        
        # Update recency score (exponential decay)
        current_time = time.time()
        recency = 1.0 / (1.0 + current_time - timestamp)
        self.recency_scores[key] = recency
        
        # Trim old history
        cutoff = current_time - self.time_window
        self.access_history[key] = [
            t for t in self.access_history[key] if t > cutoff
        ]
    
    def get_eviction_score(self, key: str) -> float:
        """
        Calculate eviction score (lower = more likely to evict).
        
        Combines frequency, recency, and predicted future access.
        """
        freq_score = self.frequency_scores.get(key, 0)
        rec_score = self.recency_scores.get(key, 0)
        
        # Weighted combination
        score = 0.4 * freq_score + 0.6 * rec_score
        
        # Apply prediction if model is trained
        if self.prediction_model:
            predicted_access = self.predict_future_access(key)
            score *= (1.0 + predicted_access)
        
        return score
    
    def predict_future_access(self, key: str) -> float:
        """Predict probability of future access."""
        if not self.prediction_model:
            return 0.5  # Default probability
        
        # Extract features
        features = self.extract_features(key)
        
        # Predict
        probability = self.prediction_model.predict_proba([features])[0, 1]
        
        return probability
    
    def extract_features(self, key: str) -> np.ndarray:
        """Extract features for prediction."""
        history = self.access_history.get(key, [])
        
        if not history:
            return np.zeros(10)  # Default features
        
        current_time = time.time()
        
        features = [
            len(history),  # Total accesses
            self.frequency_scores.get(key, 0),  # Frequency score
            self.recency_scores.get(key, 0),  # Recency score
            current_time - min(history) if history else 0,  # Age
            current_time - max(history) if history else 0,  # Time since last access
            np.std(np.diff(history)) if len(history) > 1 else 0,  # Access interval variance
            len([h for h in history if current_time - h < 60]),  # Accesses in last minute
            len([h for h in history if current_time - h < 300]),  # Accesses in last 5 minutes
            len([h for h in history if current_time - h < 900]),  # Accesses in last 15 minutes
            len([h for h in history if current_time - h < 3600]),  # Accesses in last hour
        ]
        
        return np.array(features)
    
    def train_prediction_model(self):
        """Train the access prediction model."""
        # This would be implemented with actual ML training
        # For now, we'll use a simple heuristic
        pass
    
    def select_eviction_candidates(self, items: List[str], count: int) -> List[str]:
        """Select items for eviction."""
        # Calculate scores for all items
        scores = [(item, self.get_eviction_score(item)) for item in items]
        
        # Sort by score (ascending - lower scores evicted first)
        scores.sort(key=lambda x: x[1])
        
        # Return top candidates for eviction
        return [item for item, _ in scores[:count]]


class MemoryManager:
    """
    Main memory manager that coordinates all memory tiers.
    """
    
    def __init__(self, config_path: str = "./config/config.json"):
        self.config = self.load_config(config_path)
        self.stats = {
            MemoryTier.L1_HOT: MemoryStats(
                tier=MemoryTier.L1_HOT,
                total_capacity=self.config['memory_hierarchy']['l1_cache']['size_mb'] * 1024 * 1024,
                used_capacity=0,
                hit_count=0,
                miss_count=0,
                eviction_count=0,
                avg_access_time_ms=0.0
            ),
            MemoryTier.L2_WARM: MemoryStats(
                tier=MemoryTier.L2_WARM,
                total_capacity=self.config['memory_hierarchy']['l2_cache']['size_mb'] * 1024 * 1024,
                used_capacity=0,
                hit_count=0,
                miss_count=0,
                eviction_count=0,
                avg_access_time_ms=0.0
            ),
            MemoryTier.L3_COLD: MemoryStats(
                tier=MemoryTier.L3_COLD,
                total_capacity=float('inf'),  # Unlimited
                used_capacity=0,
                hit_count=0,
                miss_count=0,
                eviction_count=0,
                avg_access_time_ms=0.0
            )
        }
        
        # Initialize components
        self.l1_cache = OrderedDict()
        self.l2_cache = MemoryMappedCache(
            cache_dir=self.config['memory_hierarchy']['l2_cache']['path'],
            max_size=self.config['memory_hierarchy']['l2_cache']['size_mb'] * 1024 * 1024
        )
        self.bloom_filter = BloomFilter()
        self.cache_policy = AdaptiveCachePolicy()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=self.config['system']['max_workers'])
        
        # Start background tasks
        self.running = True
        self.start_background_tasks()
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    async def get(self, key: str) -> Optional[bytes]:
        """Get value from memory hierarchy."""
        start_time = time.time()
        
        # Quick negative check with bloom filter
        if not self.bloom_filter.contains(key):
            return None
        
        # Check L1 cache
        if key in self.l1_cache:
            self.l1_cache.move_to_end(key)  # LRU update
            self.stats[MemoryTier.L1_HOT].hit_count += 1
            self.cache_policy.record_access(key, time.time())
            
            access_time = (time.time() - start_time) * 1000
            self.update_avg_access_time(MemoryTier.L1_HOT, access_time)
            
            return self.l1_cache[key]
        
        self.stats[MemoryTier.L1_HOT].miss_count += 1
        
        # Check L2 cache
        data = await asyncio.get_event_loop().run_in_executor(
            self.executor, self.l2_cache.get, key
        )
        
        if data:
            self.stats[MemoryTier.L2_WARM].hit_count += 1
            self.cache_policy.record_access(key, time.time())
            
            # Promote to L1
            await self.promote_to_l1(key, data)
            
            access_time = (time.time() - start_time) * 1000
            self.update_avg_access_time(MemoryTier.L2_WARM, access_time)
            
            return data
        
        self.stats[MemoryTier.L2_WARM].miss_count += 1
        
        # Check L3 (git) - would be implemented with actual git integration
        # For now, return None
        self.stats[MemoryTier.L3_COLD].miss_count += 1
        
        return None
    
    async def put(self, key: str, data: bytes):
        """Store value in memory hierarchy."""
        # Add to bloom filter
        self.bloom_filter.add(key)
        
        # Add to L1 cache
        await self.add_to_l1(key, data)
        
        # Async write to L2
        asyncio.create_task(self.write_to_l2(key, data))
        
        # Record access
        self.cache_policy.record_access(key, time.time())
    
    async def add_to_l1(self, key: str, data: bytes):
        """Add item to L1 cache with eviction if needed."""
        data_size = len(data)
        
        # Check if eviction needed
        while self.stats[MemoryTier.L1_HOT].used_capacity + data_size > \
              self.stats[MemoryTier.L1_HOT].total_capacity:
            
            if not self.l1_cache:
                break
            
            # Evict based on policy
            evict_key = self.select_eviction_candidate_l1()
            evicted_data = self.l1_cache.pop(evict_key)
            
            self.stats[MemoryTier.L1_HOT].used_capacity -= len(evicted_data)
            self.stats[MemoryTier.L1_HOT].eviction_count += 1
            
            # Demote to L2
            await self.write_to_l2(evict_key, evicted_data)
        
        # Add to cache
        self.l1_cache[key] = data
        self.stats[MemoryTier.L1_HOT].used_capacity += data_size
    
    def select_eviction_candidate_l1(self) -> str:
        """Select item to evict from L1."""
        if self.config['memory_hierarchy']['l1_cache']['eviction_policy'] == 'lru':
            # LRU: first item in OrderedDict
            return next(iter(self.l1_cache))
        else:
            # Use adaptive policy
            candidates = list(self.l1_cache.keys())
            eviction_list = self.cache_policy.select_eviction_candidates(candidates, 1)
            return eviction_list[0] if eviction_list else next(iter(self.l1_cache))
    
    async def promote_to_l1(self, key: str, data: bytes):
        """Promote item from L2 to L1."""
        await self.add_to_l1(key, data)
    
    async def write_to_l2(self, key: str, data: bytes):
        """Write item to L2 cache."""
        # Compress if configured
        if self.config['memory_hierarchy']['l2_cache']['compression'] == 'lz4':
            data = lz4.frame.compress(data)
        
        success = await asyncio.get_event_loop().run_in_executor(
            self.executor, self.l2_cache.put, key, data
        )
        
        if success:
            self.stats[MemoryTier.L2_WARM].used_capacity += len(data)
        else:
            logger.warning(f"Failed to write {key} to L2 cache")
    
    def update_avg_access_time(self, tier: MemoryTier, access_time_ms: float):
        """Update average access time with exponential moving average."""
        alpha = 0.1  # Smoothing factor
        stats = self.stats[tier]
        
        if stats.avg_access_time_ms == 0:
            stats.avg_access_time_ms = access_time_ms
        else:
            stats.avg_access_time_ms = (
                alpha * access_time_ms + 
                (1 - alpha) * stats.avg_access_time_ms
            )
    
    def start_background_tasks(self):
        """Start background maintenance tasks."""
        asyncio.create_task(self.monitor_memory_pressure())
        asyncio.create_task(self.train_cache_policy())
        asyncio.create_task(self.report_stats())
    
    async def monitor_memory_pressure(self):
        """Monitor system memory pressure and adjust caches."""
        while self.running:
            try:
                memory = psutil.virtual_memory()
                
                if memory.percent > 90:
                    # High memory pressure, reduce cache sizes
                    logger.warning(f"High memory pressure: {memory.percent}%")
                    await self.reduce_cache_sizes()
                elif memory.percent < 50:
                    # Low pressure, can increase caches
                    await self.increase_cache_sizes()
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error monitoring memory: {e}")
                await asyncio.sleep(30)
    
    async def reduce_cache_sizes(self):
        """Reduce cache sizes under memory pressure."""
        # Evict 20% of L1 cache
        evict_count = len(self.l1_cache) // 5
        
        for _ in range(evict_count):
            if not self.l1_cache:
                break
            
            key = self.select_eviction_candidate_l1()
            data = self.l1_cache.pop(key)
            
            self.stats[MemoryTier.L1_HOT].used_capacity -= len(data)
            self.stats[MemoryTier.L1_HOT].eviction_count += 1
            
            await self.write_to_l2(key, data)
        
        logger.info(f"Evicted {evict_count} items from L1 cache")
    
    async def increase_cache_sizes(self):
        """Increase cache sizes when memory available."""
        # Could implement prefetching here
        pass
    
    async def train_cache_policy(self):
        """Periodically train the adaptive cache policy."""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Train every hour
                
                # Train prediction model with recent access patterns
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.cache_policy.train_prediction_model
                )
                
                logger.info("Cache policy model updated")
                
            except Exception as e:
                logger.error(f"Error training cache policy: {e}")
    
    async def report_stats(self):
        """Periodically report statistics."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                for tier, stats in self.stats.items():
                    logger.info(
                        f"{tier.value}: "
                        f"usage={stats.usage_percent:.1f}%, "
                        f"hit_rate={stats.hit_rate:.3f}, "
                        f"avg_time={stats.avg_access_time_ms:.2f}ms, "
                        f"evictions={stats.eviction_count}"
                    )
                
                # L2 cache stats
                l2_stats = self.l2_cache.get_stats()
                logger.info(
                    f"L2 mmap: "
                    f"items={l2_stats['num_items']}, "
                    f"fragmentation={l2_stats['fragmentation']}, "
                    f"largest_free={l2_stats['largest_free_block'] / 1024 / 1024:.1f}MB"
                )
                
                # Bloom filter stats
                logger.info(f"Bloom filter load: {self.bloom_filter.load_factor():.3f}")
                
            except Exception as e:
                logger.error(f"Error reporting stats: {e}")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'tiers': {
                tier.value: {
                    'usage_percent': stats.usage_percent,
                    'hit_rate': stats.hit_rate,
                    'avg_access_time_ms': stats.avg_access_time_ms,
                    'hit_count': stats.hit_count,
                    'miss_count': stats.miss_count,
                    'eviction_count': stats.eviction_count
                }
                for tier, stats in self.stats.items()
            },
            'l2_cache': self.l2_cache.get_stats(),
            'bloom_filter': {
                'load_factor': self.bloom_filter.load_factor(),
                'size': self.bloom_filter.size,
                'num_hashes': self.bloom_filter.num_hashes
            },
            'system': {
                'memory_percent': psutil.virtual_memory().percent,
                'cpu_percent': psutil.cpu_percent(),
                'disk_usage': psutil.disk_usage('/').percent
            }
        }
    
    async def cleanup(self):
        """Clean up resources."""
        self.running = False
        self.l2_cache.cleanup()
        self.executor.shutdown(wait=True)
        logger.info("Memory manager cleaned up")


# Web API
class MemoryManagerAPI:
    """Web API for memory manager."""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.app = web.Application()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes."""
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/stats', self.get_stats)
        self.app.router.add_get('/get/{key}', self.get_value)
        self.app.router.add_post('/put/{key}', self.put_value)
        self.app.router.add_delete('/delete/{key}', self.delete_value)
        self.app.router.add_post('/clear', self.clear_cache)
    
    async def health_check(self, request):
        """Health check endpoint."""
        return web.json_response({
            'status': 'healthy',
            'timestamp': time.time()
        })
    
    async def get_stats(self, request):
        """Get memory statistics."""
        stats = self.memory_manager.get_all_stats()
        return web.json_response(stats)
    
    async def get_value(self, request):
        """Get value from memory."""
        key = request.match_info['key']
        
        data = await self.memory_manager.get(key)
        
        if data:
            return web.Response(body=data)
        else:
            return web.Response(status=404, text="Key not found")
    
    async def put_value(self, request):
        """Store value in memory."""
        key = request.match_info['key']
        data = await request.read()
        
        await self.memory_manager.put(key, data)
        
        return web.json_response({'status': 'stored', 'key': key})
    
    async def delete_value(self, request):
        """Delete value from memory."""
        key = request.match_info['key']
        
        # Implementation would remove from all tiers
        # For now, just remove from L1
        if key in self.memory_manager.l1_cache:
            del self.memory_manager.l1_cache[key]
            return web.json_response({'status': 'deleted', 'key': key})
        
        return web.Response(status=404, text="Key not found")
    
    async def clear_cache(self, request):
        """Clear all caches."""
        tier = request.query.get('tier', 'all')
        
        if tier in ['l1', 'all']:
            self.memory_manager.l1_cache.clear()
            self.memory_manager.stats[MemoryTier.L1_HOT].used_capacity = 0
        
        if tier in ['l2', 'all']:
            # Would clear L2 cache
            pass
        
        if tier == 'all':
            self.memory_manager.bloom_filter.clear()
        
        return web.json_response({'status': 'cleared', 'tier': tier})


async def main():
    """Main entry point."""
    # Initialize memory manager
    memory_manager = MemoryManager()
    
    # Initialize API
    api = MemoryManagerAPI(memory_manager)
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Shutting down memory manager...")
        asyncio.create_task(memory_manager.cleanup())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start web server
    runner = web.AppRunner(api.app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', 8081)
    await site.start()
    
    logger.info("Memory manager server started on http://0.0.0.0:8081")
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        await memory_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
