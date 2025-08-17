#!/usr/bin/env python3
"""
AI Context Nexus - Core Context Manager Implementation

This module implements the novel context management system that uses git
as a persistent, versioned storage backend for AI agent contexts.
"""

import os
import json
import hashlib
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import yaml
import numpy as np
from pathlib import Path
import mmap
import pickle
import lz4.frame
import threading
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from collections import OrderedDict, deque
import redis
import msgpack

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of contexts that can be stored."""
    CONVERSATION = "conversation"
    CODE = "code"
    DOCUMENT = "document"
    ANALYSIS = "analysis"
    DECISION = "decision"
    MEMORY = "memory"
    INSTRUCTION = "instruction"


@dataclass
class Context:
    """
    Represents a context chunk that can be shared between agents.
    """
    id: str
    type: ContextType
    content: str
    metadata: Dict[str, Any]
    agent_id: str
    timestamp: datetime
    parent_id: Optional[str] = None
    embeddings: Optional[np.ndarray] = None
    semantic_hash: Optional[str] = None
    compression_ratio: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert context to dictionary for serialization."""
        data = asdict(self)
        data['type'] = self.type.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.embeddings is not None:
            data['embeddings'] = self.embeddings.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Context':
        """Create context from dictionary."""
        data['type'] = ContextType(data['type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('embeddings'):
            data['embeddings'] = np.array(data['embeddings'])
        return cls(**data)


class SemanticIndex:
    """
    Implements semantic indexing for fast context retrieval.
    """
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = {}  # In production, use FAISS or similar
        self.embeddings = []
        self.context_ids = []
        
    def add(self, context_id: str, embedding: np.ndarray):
        """Add a context to the semantic index."""
        self.context_ids.append(context_id)
        self.embeddings.append(embedding)
        
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for semantically similar contexts.
        Returns list of (context_id, similarity_score) tuples.
        """
        if not self.embeddings:
            return []
        
        # Calculate cosine similarities
        embeddings_matrix = np.array(self.embeddings)
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        
        similarities = np.dot(embeddings_norm, query_norm)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append((self.context_ids[idx], float(similarities[idx])))
        
        return results


class HierarchicalCache:
    """
    Implements the three-tier memory hierarchy.
    """
    
    def __init__(self, l1_size_mb: int = 256, l2_size_mb: int = 10240):
        self.l1_cache = OrderedDict()  # Hot RAM cache
        self.l1_max_size = l1_size_mb * 1024 * 1024
        self.l1_current_size = 0
        
        self.l2_path = Path("./memory/l2_cache")
        self.l2_path.mkdir(parents=True, exist_ok=True)
        self.l2_index = {}
        self.l2_max_size = l2_size_mb * 1024 * 1024
        self.l2_current_size = 0
        
        self.lock = threading.RLock()
        
    def get(self, context_id: str) -> Optional[Context]:
        """Retrieve context from cache hierarchy."""
        with self.lock:
            # Check L1
            if context_id in self.l1_cache:
                # Move to end (LRU)
                self.l1_cache.move_to_end(context_id)
                return self.l1_cache[context_id]
            
            # Check L2
            if context_id in self.l2_index:
                context = self._load_from_l2(context_id)
                if context:
                    self._promote_to_l1(context_id, context)
                return context
            
            return None
    
    def put(self, context: Context):
        """Store context in cache hierarchy."""
        with self.lock:
            context_size = len(json.dumps(context.to_dict()).encode())
            
            # Add to L1
            if self.l1_current_size + context_size > self.l1_max_size:
                self._evict_from_l1()
            
            self.l1_cache[context.id] = context
            self.l1_current_size += context_size
            
            # Also persist to L2
            self._save_to_l2(context)
    
    def _evict_from_l1(self):
        """Evict least recently used items from L1 cache."""
        while self.l1_current_size > self.l1_max_size * 0.8:  # Keep 20% free
            if not self.l1_cache:
                break
            
            context_id, context = self.l1_cache.popitem(last=False)
            context_size = len(json.dumps(context.to_dict()).encode())
            self.l1_current_size -= context_size
    
    def _save_to_l2(self, context: Context):
        """Save context to L2 cache (JSON files)."""
        file_path = self.l2_path / f"{context.id}.json.lz4"
        
        # Compress and save
        data = json.dumps(context.to_dict()).encode()
        compressed = lz4.frame.compress(data)
        
        with open(file_path, 'wb') as f:
            f.write(compressed)
        
        self.l2_index[context.id] = {
            'path': str(file_path),
            'size': len(compressed),
            'timestamp': context.timestamp.isoformat()
        }
        
        self.l2_current_size += len(compressed)
        
        # Evict if necessary
        if self.l2_current_size > self.l2_max_size:
            self._evict_from_l2()
    
    def _load_from_l2(self, context_id: str) -> Optional[Context]:
        """Load context from L2 cache."""
        if context_id not in self.l2_index:
            return None
        
        file_path = Path(self.l2_index[context_id]['path'])
        if not file_path.exists():
            del self.l2_index[context_id]
            return None
        
        with open(file_path, 'rb') as f:
            compressed = f.read()
        
        data = lz4.frame.decompress(compressed)
        context_dict = json.loads(data.decode())
        
        return Context.from_dict(context_dict)
    
    def _evict_from_l2(self):
        """Evict oldest items from L2 cache."""
        # Sort by timestamp and remove oldest
        sorted_items = sorted(
            self.l2_index.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        while self.l2_current_size > self.l2_max_size * 0.8:
            if not sorted_items:
                break
            
            context_id, info = sorted_items.pop(0)
            file_path = Path(info['path'])
            
            if file_path.exists():
                file_path.unlink()
                self.l2_current_size -= info['size']
            
            del self.l2_index[context_id]
    
    def _promote_to_l1(self, context_id: str, context: Context):
        """Promote a context from L2 to L1."""
        context_size = len(json.dumps(context.to_dict()).encode())
        
        if self.l1_current_size + context_size > self.l1_max_size:
            self._evict_from_l1()
        
        self.l1_cache[context_id] = context
        self.l1_current_size += context_size


class GitRepository:
    """
    Wrapper for git operations optimized for context storage.
    """
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.repo_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize git repo if not exists
        if not (self.repo_path / ".git").exists():
            self._run_git(["init"])
            self._run_git(["config", "user.name", "AI Context Nexus"])
            self._run_git(["config", "user.email", "nexus@ai.local"])
    
    def _run_git(self, args: List[str]) -> str:
        """Run a git command and return output."""
        cmd = ["git"] + args
        result = subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    
    def commit_context(self, context: Context, message: str) -> str:
        """
        Commit a context to git with extensive metadata.
        
        Returns the commit hash.
        """
        # Create a unique file for this context
        context_file = self.repo_path / f"contexts/{context.agent_id}/{context.id}.yaml"
        context_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save context to file
        with open(context_file, 'w') as f:
            yaml.dump(context.to_dict(), f, default_flow_style=False)
        
        # Stage the file
        self._run_git(["add", str(context_file.relative_to(self.repo_path))])
        
        # Create detailed commit message
        commit_message = self._create_commit_message(context, message)
        
        # Commit with the detailed message
        self._run_git(["commit", "-m", commit_message])
        
        # Get the commit hash
        commit_hash = self._run_git(["rev-parse", "HEAD"]).strip()
        
        return commit_hash
    
    def _create_commit_message(self, context: Context, summary: str) -> str:
        """
        Create a detailed commit message with context embedded.
        
        Format:
        [CONTEXT] Summary (50 chars max)
        
        Type: {context_type}
        Agent: {agent_id}
        Timestamp: {iso_timestamp}
        Parent: {parent_id}
        Semantic-Hash: {hash}
        
        ---METADATA---
        {yaml_metadata}
        
        ---CONTENT---
        {full_content}
        """
        # Truncate summary to 50 chars
        summary = summary[:50]
        
        lines = [
            f"[CONTEXT] {summary}",
            "",
            f"Type: {context.type.value}",
            f"Agent: {context.agent_id}",
            f"Timestamp: {context.timestamp.isoformat()}",
            f"Parent: {context.parent_id or 'none'}",
            f"Semantic-Hash: {context.semantic_hash or 'pending'}",
            "",
            "---METADATA---",
            yaml.dump(context.metadata, default_flow_style=False),
            "",
            "---CONTENT---",
            context.content
        ]
        
        return "\n".join(lines)
    
    def get_context_from_commit(self, commit_hash: str) -> Optional[Context]:
        """Extract context from a git commit."""
        try:
            # Get commit message
            message = self._run_git(["show", "-s", "--format=%B", commit_hash])
            
            # Parse the commit message to extract context
            context = self._parse_commit_message(message)
            
            return context
        except subprocess.CalledProcessError:
            logger.error(f"Failed to get commit {commit_hash}")
            return None
    
    def _parse_commit_message(self, message: str) -> Optional[Context]:
        """Parse a commit message to extract context."""
        lines = message.split('\n')
        
        # Find the different sections
        metadata_start = None
        content_start = None
        
        for i, line in enumerate(lines):
            if line == "---METADATA---":
                metadata_start = i + 1
            elif line == "---CONTENT---":
                content_start = i + 1
        
        if content_start is None:
            return None
        
        # Extract metadata
        metadata_lines = lines[metadata_start:content_start-2] if metadata_start else []
        metadata = yaml.safe_load('\n'.join(metadata_lines)) if metadata_lines else {}
        
        # Extract content
        content = '\n'.join(lines[content_start:])
        
        # Parse header fields
        type_line = next((l for l in lines if l.startswith("Type: ")), None)
        agent_line = next((l for l in lines if l.startswith("Agent: ")), None)
        timestamp_line = next((l for l in lines if l.startswith("Timestamp: ")), None)
        parent_line = next((l for l in lines if l.startswith("Parent: ")), None)
        hash_line = next((l for l in lines if l.startswith("Semantic-Hash: ")), None)
        
        if not all([type_line, agent_line, timestamp_line]):
            return None
        
        # Create context object
        context = Context(
            id=hashlib.sha256(content.encode()).hexdigest()[:16],
            type=ContextType(type_line.split(": ")[1]),
            content=content,
            metadata=metadata,
            agent_id=agent_line.split(": ")[1],
            timestamp=datetime.fromisoformat(timestamp_line.split(": ")[1]),
            parent_id=parent_line.split(": ")[1] if parent_line and parent_line.split(": ")[1] != "none" else None,
            semantic_hash=hash_line.split(": ")[1] if hash_line else None
        )
        
        return context
    
    def search_commits(self, query: str, max_results: int = 100) -> List[str]:
        """Search git history for contexts matching query."""
        try:
            # Use git log with grep
            result = self._run_git([
                "log",
                "--grep", query,
                "--format=%H",
                f"-{max_results}"
            ])
            
            commit_hashes = result.strip().split('\n') if result.strip() else []
            return commit_hashes
        except subprocess.CalledProcessError:
            return []


class JujutsuWrapper:
    """
    Wrapper for Jujutsu (jj) operations for advanced branching.
    """
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        
        # Initialize jj if not already initialized
        if not (self.repo_path / ".jj").exists():
            self._run_jj(["init", "--git-repo", "."])
    
    def _run_jj(self, args: List[str]) -> str:
        """Run a jj command and return output."""
        cmd = ["jj"] + args
        result = subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    
    def create_parallel_branch(self, branch_name: str) -> str:
        """Create a new parallel branch for agent work."""
        self._run_jj(["branch", "create", branch_name])
        return branch_name
    
    def merge_branches(self, branches: List[str]) -> str:
        """Merge multiple branches with automatic conflict resolution."""
        # Create a new merge commit
        merge_args = ["new"] + [f"-r {b}" for b in branches]
        result = self._run_jj(merge_args)
        
        # Get the new commit ID
        commit_id = self._run_jj(["log", "-r", "@", "--no-graph", "--format", "commit_id"]).strip()
        
        return commit_id


class ContextManager:
    """
    Main context manager that orchestrates all components.
    """
    
    def __init__(self, repo_path: str = "./context_repo", 
                 use_jj: bool = True,
                 enable_redis: bool = False):
        self.repo_path = Path(repo_path)
        self.repo = GitRepository(str(self.repo_path))
        self.jj = JujutsuWrapper(str(self.repo_path)) if use_jj else None
        self.index = SemanticIndex()
        self.cache = HierarchicalCache()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Optional Redis for distributed mode
        self.redis = redis.Redis() if enable_redis else None
        
        # Semantic graph for advanced queries
        self.semantic_graph = nx.DiGraph()
        
        # Load existing contexts into index
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild the semantic index from git history."""
        logger.info("Rebuilding semantic index from git history...")
        
        # Get all commits
        try:
            commits = self.repo._run_git(["log", "--format=%H"]).strip().split('\n')
            
            for commit_hash in commits:
                if not commit_hash:
                    continue
                    
                context = self.repo.get_context_from_commit(commit_hash)
                if context and context.embeddings is not None:
                    self.index.add(context.id, context.embeddings)
                    
                    # Add to semantic graph
                    self.semantic_graph.add_node(
                        context.id,
                        context=context,
                        commit_hash=commit_hash
                    )
        except subprocess.CalledProcessError:
            logger.info("No existing commits found, starting fresh")
    
    def add_context(self, context: Context) -> str:
        """
        Add a new context to the system.
        
        Returns the commit hash.
        """
        # Generate semantic hash if not present
        if not context.semantic_hash:
            context.semantic_hash = self._generate_semantic_hash(context)
        
        # Generate embeddings if not present (in real implementation, use actual embeddings)
        if context.embeddings is None:
            context.embeddings = self._generate_embeddings(context.content)
        
        # Add to cache
        self.cache.put(context)
        
        # Add to semantic index
        self.index.add(context.id, context.embeddings)
        
        # Commit to git
        summary = f"{context.type.value}: {context.content[:30]}..."
        commit_hash = self.repo.commit_context(context, summary)
        
        # Add to semantic graph
        self.semantic_graph.add_node(
            context.id,
            context=context,
            commit_hash=commit_hash
        )
        
        # Update edges in semantic graph
        self._update_semantic_edges(context)
        
        # Notify other agents if in distributed mode
        if self.redis:
            self._notify_agents(context)
        
        logger.info(f"Added context {context.id} with commit {commit_hash}")
        
        return commit_hash
    
    def get_context(self, context_id: str) -> Optional[Context]:
        """Retrieve a context by ID."""
        # Check cache first
        context = self.cache.get(context_id)
        if context:
            return context
        
        # Search in git history
        commits = self.repo.search_commits(context_id, max_results=10)
        for commit_hash in commits:
            context = self.repo.get_context_from_commit(commit_hash)
            if context and context.id == context_id:
                # Add to cache for future access
                self.cache.put(context)
                return context
        
        return None
    
    def search_contexts(self, query: str, max_results: int = 10) -> List[Context]:
        """
        Search for contexts using semantic similarity.
        """
        # Generate query embedding
        query_embedding = self._generate_embeddings(query)
        
        # Search in semantic index
        results = self.index.search(query_embedding, k=max_results)
        
        contexts = []
        for context_id, score in results:
            context = self.get_context(context_id)
            if context:
                contexts.append(context)
        
        return contexts
    
    def get_context_chain(self, context_id: str) -> List[Context]:
        """
        Get the full chain of contexts (following parent links).
        """
        chain = []
        current_id = context_id
        
        while current_id:
            context = self.get_context(current_id)
            if not context:
                break
            
            chain.append(context)
            current_id = context.parent_id
        
        return list(reversed(chain))  # Return in chronological order
    
    def _generate_semantic_hash(self, context: Context) -> str:
        """Generate a semantic hash for deduplication."""
        # Combine content with metadata for hashing
        hash_input = f"{context.type.value}:{context.content}:{json.dumps(context.metadata, sort_keys=True)}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def _generate_embeddings(self, text: str) -> np.ndarray:
        """
        Generate embeddings for text.
        
        In production, this would use a real embedding model like BERT or GPT.
        For now, we'll create random embeddings for demonstration.
        """
        # Simulate embedding generation
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(768)
    
    def _update_semantic_edges(self, new_context: Context):
        """Update edges in the semantic graph based on similarity."""
        for node_id in self.semantic_graph.nodes():
            if node_id == new_context.id:
                continue
            
            other_context = self.semantic_graph.nodes[node_id]['context']
            
            # Calculate similarity
            if other_context.embeddings is not None and new_context.embeddings is not None:
                similarity = self._calculate_similarity(
                    new_context.embeddings,
                    other_context.embeddings
                )
                
                # Add edge if similarity is high enough
                if similarity > 0.7:  # Threshold
                    self.semantic_graph.add_edge(
                        new_context.id,
                        node_id,
                        weight=similarity
                    )
    
    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _notify_agents(self, context: Context):
        """Notify other agents about new context (distributed mode)."""
        if self.redis:
            notification = {
                'type': 'new_context',
                'context_id': context.id,
                'agent_id': context.agent_id,
                'timestamp': context.timestamp.isoformat()
            }
            
            self.redis.publish('context_updates', json.dumps(notification))
    
    def get_graph_insights(self) -> Dict[str, Any]:
        """Get insights from the semantic graph."""
        return {
            'total_contexts': self.semantic_graph.number_of_nodes(),
            'total_connections': self.semantic_graph.number_of_edges(),
            'avg_connections': self.semantic_graph.number_of_edges() / max(self.semantic_graph.number_of_nodes(), 1),
            'most_connected': self._get_most_connected_contexts(),
            'clusters': self._identify_clusters()
        }
    
    def _get_most_connected_contexts(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """Get the most connected contexts in the graph."""
        degree_centrality = nx.degree_centrality(self.semantic_graph)
        sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_n]
    
    def _identify_clusters(self) -> List[List[str]]:
        """Identify clusters of related contexts."""
        if self.semantic_graph.number_of_nodes() == 0:
            return []
        
        # Convert to undirected for clustering
        undirected = self.semantic_graph.to_undirected()
        
        # Find connected components
        clusters = list(nx.connected_components(undirected))
        
        return [list(cluster) for cluster in clusters]
    
    def cleanup(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
        if self.redis:
            self.redis.close()


# Example usage and testing
if __name__ == "__main__":
    # Initialize the context manager
    manager = ContextManager(repo_path="./test_context_repo")
    
    # Create a sample context
    context1 = Context(
        id="ctx_001",
        type=ContextType.ANALYSIS,
        content="This is a detailed analysis of the codebase architecture...",
        metadata={
            "project": "ai-context-nexus",
            "version": "1.0.0",
            "tags": ["architecture", "analysis"]
        },
        agent_id="agent_claude",
        timestamp=datetime.now(timezone.utc)
    )
    
    # Add the context
    commit_hash = manager.add_context(context1)
    print(f"Added context with commit: {commit_hash}")
    
    # Create a follow-up context
    context2 = Context(
        id="ctx_002",
        type=ContextType.DECISION,
        content="Based on the analysis, we should refactor the memory module...",
        metadata={
            "project": "ai-context-nexus",
            "decision": "refactor",
            "priority": "high"
        },
        agent_id="agent_gpt",
        timestamp=datetime.now(timezone.utc),
        parent_id=context1.id
    )
    
    # Add the follow-up context
    commit_hash2 = manager.add_context(context2)
    print(f"Added follow-up context with commit: {commit_hash2}")
    
    # Search for contexts
    results = manager.search_contexts("architecture analysis", max_results=5)
    print(f"Found {len(results)} matching contexts")
    
    # Get context chain
    chain = manager.get_context_chain(context2.id)
    print(f"Context chain has {len(chain)} contexts")
    
    # Get graph insights
    insights = manager.get_graph_insights()
    print(f"Graph insights: {insights}")
    
    # Cleanup
    manager.cleanup()
