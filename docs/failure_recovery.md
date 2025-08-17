# Failure Recovery Mechanisms

## Overview

The AI Context Nexus implements a comprehensive failure recovery system designed to handle various failure modes without data loss or service interruption. This document details the failure modes, detection mechanisms, and recovery strategies.

## Failure Taxonomy

### 1. Component-Level Failures

#### 1.1 Agent Failures
**Failure Modes:**
- API timeout/unavailability
- Rate limiting
- Authentication failures
- Model errors
- Memory exhaustion

**Detection:**
```python
class AgentHealthMonitor:
    def __init__(self, agent: AgentProtocol):
        self.agent = agent
        self.heartbeat_interval = 30  # seconds
        self.failure_threshold = 3
        self.consecutive_failures = 0
    
    async def monitor(self):
        while True:
            try:
                # Send heartbeat
                response = await self.agent.process_context(
                    Context(
                        id="heartbeat",
                        type=ContextType.INSTRUCTION,
                        content="ping",
                        metadata={"heartbeat": True},
                        agent_id="monitor",
                        timestamp=datetime.now(timezone.utc)
                    )
                )
                
                if response.success:
                    self.consecutive_failures = 0
                else:
                    self.consecutive_failures += 1
                    
            except Exception as e:
                self.consecutive_failures += 1
                logger.error(f"Agent {self.agent.agent_id} heartbeat failed: {e}")
            
            if self.consecutive_failures >= self.failure_threshold:
                await self.trigger_recovery()
            
            await asyncio.sleep(self.heartbeat_interval)
```

**Recovery Strategy:**
1. **Immediate Retry**: Retry failed operations with exponential backoff
2. **State Restoration**: Restore agent from last known good state
3. **Failover**: Route requests to backup agents
4. **Circuit Breaker**: Temporarily disable failing agents

#### 1.2 Context Manager Failures
**Failure Modes:**
- Git repository corruption
- Memory cache corruption
- Index inconsistency
- Disk full

**Detection:**
- Periodic integrity checks using checksums
- Git fsck on startup
- Cache validation on read

**Recovery Strategy:**
```python
class ContextManagerRecovery:
    def recover_git_repository(self):
        """Recover from git corruption."""
        try:
            # Try git fsck first
            subprocess.run(["git", "fsck", "--full"], check=True)
        except:
            # Clone from backup
            self.restore_from_backup()
    
    def rebuild_index(self):
        """Rebuild semantic index from git history."""
        commits = self.get_all_commits()
        new_index = SemanticIndex()
        
        for commit in commits:
            context = self.extract_context(commit)
            if context:
                new_index.add(context.id, context.embeddings)
        
        self.index = new_index
    
    def validate_cache(self):
        """Validate and repair cache."""
        for context_id in list(self.cache.l1_cache.keys()):
            context = self.cache.l1_cache[context_id]
            
            # Verify checksum
            if not self.verify_checksum(context):
                # Remove corrupted entry
                del self.cache.l1_cache[context_id]
                
                # Try to reload from git
                fresh_context = self.load_from_git(context_id)
                if fresh_context:
                    self.cache.l1_cache[context_id] = fresh_context
```

### 2. System-Level Failures

#### 2.1 Process Crashes
**Detection:**
- Process monitoring via systemd/supervisor
- PID file checking
- Tmux pane status monitoring

**Recovery:**
```bash
# Automatic process restart using systemd
[Unit]
Description=AI Context Nexus Context Manager
After=network.target

[Service]
Type=simple
User=nexus
WorkingDirectory=/opt/ai-context-nexus
Environment="PATH=/opt/ai-context-nexus/venv/bin"
ExecStart=/opt/ai-context-nexus/venv/bin/python core/context_manager_server.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/nexus/context_manager.log
StandardError=append:/var/log/nexus/context_manager.error.log

[Install]
WantedBy=multi-user.target
```

#### 2.2 Memory Exhaustion
**Detection:**
- Resource monitoring
- OOM killer detection
- Performance degradation metrics

**Recovery:**
```python
class MemoryManager:
    def __init__(self, threshold_mb: int = 3072):
        self.threshold = threshold_mb * 1024 * 1024
        
    def check_memory(self):
        """Check available memory and trigger cleanup if needed."""
        import psutil
        
        available = psutil.virtual_memory().available
        
        if available < self.threshold:
            self.emergency_cleanup()
    
    def emergency_cleanup(self):
        """Emergency memory cleanup."""
        # 1. Clear L1 cache
        self.cache.l1_cache.clear()
        self.cache.l1_current_size = 0
        
        # 2. Force garbage collection
        import gc
        gc.collect()
        
        # 3. Reduce cache sizes
        self.cache.l1_max_size //= 2
        
        # 4. Notify agents to reduce memory usage
        for agent in self.agents.values():
            agent.reduce_memory_usage()
```

### 3. Network Failures

#### 3.1 Network Partition (Distributed Mode)
**Detection:**
- Heartbeat timeout between nodes
- Split-brain detection via quorum

**Recovery:**
```python
class PartitionRecovery:
    def __init__(self, node_id: str, cluster_nodes: List[str]):
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.quorum_size = len(cluster_nodes) // 2 + 1
    
    async def detect_partition(self) -> bool:
        """Detect if we're in a network partition."""
        reachable = 0
        
        for node in self.cluster_nodes:
            if await self.ping_node(node):
                reachable += 1
        
        # We're in minority partition
        if reachable < self.quorum_size:
            return True
        
        return False
    
    async def handle_partition(self):
        """Handle network partition."""
        if await self.detect_partition():
            # We're in minority - become read-only
            self.set_read_only_mode(True)
            
            # Keep trying to rejoin majority
            while await self.detect_partition():
                await asyncio.sleep(5)
            
            # Rejoin and sync
            await self.sync_with_majority()
            self.set_read_only_mode(False)
```

### 4. Data Corruption

#### 4.1 Context Corruption
**Detection:**
- Semantic hash verification
- Checksum validation
- Structural validation

**Recovery:**
```python
class DataIntegrityManager:
    def verify_context(self, context: Context) -> bool:
        """Verify context integrity."""
        # 1. Check semantic hash
        expected_hash = self.calculate_semantic_hash(context)
        if context.semantic_hash != expected_hash:
            return False
        
        # 2. Validate structure
        if not self.validate_structure(context):
            return False
        
        # 3. Check parent chain
        if context.parent_id:
            parent = self.get_context(context.parent_id)
            if not parent:
                return False
        
        return True
    
    def repair_context_chain(self, context_id: str):
        """Repair broken context chain."""
        chain = []
        current_id = context_id
        
        while current_id:
            context = self.get_context(current_id)
            
            if not context:
                # Try to recover from git
                context = self.recover_from_git(current_id)
                
                if not context:
                    # Chain is broken, mark as orphaned
                    if chain:
                        chain[-1].parent_id = None
                    break
            
            chain.append(context)
            current_id = context.parent_id
        
        # Rebuild chain with correct parent links
        for i, context in enumerate(chain):
            if i > 0:
                context.parent_id = chain[i-1].id
            
            self.update_context(context)
```

## Recovery Workflows

### 1. Cascading Failure Recovery

When multiple components fail simultaneously:

```python
class CascadingFailureRecovery:
    def __init__(self):
        self.recovery_order = [
            "git_repository",
            "context_manager",
            "memory_cache",
            "semantic_index",
            "agents"
        ]
    
    async def recover_system(self):
        """Recover from cascading failure."""
        recovery_status = {}
        
        for component in self.recovery_order:
            try:
                if component == "git_repository":
                    self.recover_git()
                elif component == "context_manager":
                    self.recover_context_manager()
                elif component == "memory_cache":
                    self.rebuild_cache()
                elif component == "semantic_index":
                    self.rebuild_index()
                elif component == "agents":
                    await self.restart_agents()
                
                recovery_status[component] = "recovered"
                
            except Exception as e:
                recovery_status[component] = f"failed: {e}"
                
                # Some components are critical
                if component in ["git_repository", "context_manager"]:
                    raise SystemRecoveryError(f"Critical component {component} recovery failed")
        
        return recovery_status
```

### 2. Point-in-Time Recovery

Restore system to a specific point in time:

```python
class PointInTimeRecovery:
    def restore_to_timestamp(self, timestamp: datetime):
        """Restore system to specific timestamp."""
        # 1. Find git commit at timestamp
        commit = self.find_commit_at_timestamp(timestamp)
        
        # 2. Create recovery branch
        subprocess.run([
            "git", "checkout", "-b", 
            f"recovery_{timestamp.isoformat()}", 
            commit
        ])
        
        # 3. Rebuild all derived data
        self.rebuild_index_from_commit(commit)
        self.rebuild_cache_from_commit(commit)
        
        # 4. Notify agents of recovery
        for agent in self.agents.values():
            agent.notify_recovery(timestamp)
```

### 3. Distributed Recovery with Consensus

For distributed deployments:

```python
class DistributedRecovery:
    async def coordinate_recovery(self):
        """Coordinate recovery across multiple nodes."""
        # 1. Elect recovery coordinator
        coordinator = await self.elect_coordinator()
        
        if self.node_id == coordinator:
            # We're the coordinator
            recovery_plan = self.create_recovery_plan()
            
            # 2. Distribute recovery plan
            for node in self.cluster_nodes:
                await self.send_recovery_plan(node, recovery_plan)
            
            # 3. Execute recovery in phases
            for phase in recovery_plan.phases:
                # All nodes execute phase
                results = await self.execute_phase_distributed(phase)
                
                # Verify phase completion
                if not self.verify_phase_completion(results):
                    await self.rollback_phase(phase)
                    raise RecoveryError(f"Phase {phase.name} failed")
            
            # 4. Verify system consistency
            await self.verify_distributed_consistency()
```

## Monitoring and Alerting

### Health Check Endpoints

```python
@app.route('/health')
def health_check():
    """Comprehensive health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {}
    }
    
    # Check each component
    checks = [
        ("context_manager", check_context_manager),
        ("git_repository", check_git_repo),
        ("memory_cache", check_cache),
        ("agents", check_agents),
        ("semantic_index", check_index)
    ]
    
    for component_name, check_func in checks:
        try:
            result = check_func()
            health_status["components"][component_name] = {
                "status": "healthy",
                "details": result
            }
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["components"][component_name] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    return jsonify(health_status)
```

### Metrics Collection

```python
class MetricsCollector:
    def __init__(self):
        self.metrics = {
            "recovery_attempts": 0,
            "recovery_successes": 0,
            "recovery_failures": 0,
            "mean_recovery_time": 0,
            "component_failures": defaultdict(int),
            "last_failure": None,
            "uptime": 0
        }
    
    def record_recovery(self, component: str, success: bool, duration: float):
        """Record recovery attempt."""
        self.metrics["recovery_attempts"] += 1
        
        if success:
            self.metrics["recovery_successes"] += 1
        else:
            self.metrics["recovery_failures"] += 1
            
        self.metrics["component_failures"][component] += 1
        
        # Update mean recovery time
        n = self.metrics["recovery_successes"]
        if n > 0:
            old_mean = self.metrics["mean_recovery_time"]
            self.metrics["mean_recovery_time"] = (old_mean * (n-1) + duration) / n
```

## Testing Recovery Mechanisms

### Chaos Engineering

```python
class ChaosMonkey:
    """Inject failures to test recovery mechanisms."""
    
    def __init__(self, system):
        self.system = system
        self.failure_scenarios = [
            self.kill_random_agent,
            self.corrupt_random_context,
            self.fill_disk_space,
            self.simulate_network_latency,
            self.trigger_memory_pressure
        ]
    
    async def run_chaos_test(self, duration_minutes: int = 60):
        """Run chaos test for specified duration."""
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time:
            # Pick random failure scenario
            scenario = random.choice(self.failure_scenarios)
            
            logger.info(f"Chaos Monkey: Executing {scenario.__name__}")
            
            try:
                await scenario()
            except Exception as e:
                logger.error(f"Chaos scenario failed: {e}")
            
            # Wait before next failure
            await asyncio.sleep(random.randint(30, 300))
    
    async def kill_random_agent(self):
        """Kill a random agent process."""
        agents = list(self.system.agents.values())
        if agents:
            victim = random.choice(agents)
            logger.info(f"Killing agent {victim.agent_id}")
            victim.status = AgentStatus.ERROR
            await victim.cleanup()
```

## Recovery Best Practices

### 1. Defense in Depth
- Multiple layers of protection
- Redundant backup systems
- Graceful degradation

### 2. Fast Detection
- Sub-second heartbeats for critical components
- Continuous integrity checking
- Predictive failure detection

### 3. Automated Recovery
- Self-healing systems
- Minimal manual intervention
- Clear escalation paths

### 4. Testing and Validation
- Regular disaster recovery drills
- Chaos engineering in production
- Recovery time objectives (RTO) validation

### 5. Documentation and Runbooks
- Clear recovery procedures
- Automated runbook execution
- Post-mortem analysis

## Conclusion

The AI Context Nexus implements comprehensive failure recovery mechanisms that ensure system resilience and data integrity. Through multiple layers of detection, automated recovery procedures, and continuous monitoring, the system can handle various failure scenarios while maintaining service availability and data consistency.

The recovery system is designed to be:
- **Automatic**: Minimal manual intervention required
- **Fast**: Sub-second detection and recovery initiation
- **Comprehensive**: Covers all identified failure modes
- **Tested**: Regular chaos engineering validates recovery paths
- **Documented**: Clear procedures and metrics for all scenarios
