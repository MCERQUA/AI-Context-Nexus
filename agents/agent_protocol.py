#!/usr/bin/env python3
"""
AI Context Nexus - Agent Protocol Implementation

This module defines the protocol for AI agents to participate in the
context sharing ecosystem. It includes base classes and implementations
for various LLM providers.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
import hashlib
import pickle
import numpy as np
from pathlib import Path
import aiohttp
import backoff
import threading
from queue import Queue, Empty
import traceback

# Import context manager
from context_manager import Context, ContextType, ContextManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Status of an AI agent."""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"


class AgentCapability(Enum):
    """Capabilities that an agent can have."""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    DECISION_MAKING = "decision_making"
    MEMORY_MANAGEMENT = "memory_management"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    EMBEDDING_GENERATION = "embedding_generation"


@dataclass
class AgentConfig:
    """Configuration for an AI agent."""
    name: str
    type: str  # claude, gpt4, llama, etc.
    api_key: Optional[str] = None
    api_endpoint: Optional[str] = None
    model_name: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    capabilities: List[AgentCapability] = field(default_factory=list)
    retry_attempts: int = 3
    timeout: int = 30
    rate_limit: int = 60  # requests per minute
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Response from an AI agent."""
    agent_id: str
    content: str
    context_used: Optional[Context] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    tokens_used: int = 0
    success: bool = True
    error_message: Optional[str] = None


class AgentProtocol(ABC):
    """
    Abstract base class defining the protocol for AI agents.
    """
    
    def __init__(self, config: AgentConfig, context_manager: ContextManager):
        self.config = config
        self.context_manager = context_manager
        self.agent_id = f"{config.type}_{config.name}_{uuid.uuid4().hex[:8]}"
        self.status = AgentStatus.IDLE
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        self.total_processing_time = 0.0
        
        # Rate limiting
        self.request_times: List[float] = []
        self.rate_limit_lock = threading.Lock()
        
        # State management
        self._state = {}
        self._state_lock = threading.Lock()
        
        # Message queue for async processing
        self.message_queue = Queue()
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize the agent."""
        logger.info(f"Initializing agent {self.agent_id}")
    
    @abstractmethod
    async def process_context(self, context: Context) -> AgentResponse:
        """
        Process a context and generate a response.
        
        This is the main method that each agent must implement.
        """
        pass
    
    @abstractmethod
    async def generate_context(self, prompt: str, parent_context: Optional[Context] = None) -> Context:
        """
        Generate a new context from a prompt.
        """
        pass
    
    @abstractmethod
    def serialize_state(self) -> bytes:
        """
        Serialize the agent's internal state for persistence.
        """
        pass
    
    @abstractmethod
    def deserialize_state(self, data: bytes) -> None:
        """
        Restore the agent's internal state from serialized data.
        """
        pass
    
    async def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        with self.rate_limit_lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.request_times = [t for t in self.request_times if now - t < 60]
            
            if len(self.request_times) >= self.config.rate_limit:
                return False
            
            self.request_times.append(now)
            return True
    
    async def _make_api_request(self, endpoint: str, payload: Dict) -> Dict:
        """Make an API request with retry logic."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        @backoff.on_exception(
            backoff.expo,
            (aiohttp.ClientError, asyncio.TimeoutError),
            max_tries=self.config.retry_attempts
        )
        async def _request():
            headers = self._get_headers()
            
            async with self.session.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                response.raise_for_status()
                return await response.json()
        
        return await _request()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {"Content-Type": "application/json"}
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        return headers
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        return {
            "agent_id": self.agent_id,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(self.total_requests, 1),
            "total_tokens": self.total_tokens,
            "avg_processing_time": self.total_processing_time / max(self.total_requests, 1),
            "status": self.status.value
        }
    
    def update_state(self, key: str, value: Any):
        """Update agent state."""
        with self._state_lock:
            self._state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get agent state."""
        with self._state_lock:
            return self._state.get(key, default)
    
    async def collaborate_with(self, other_agent: 'AgentProtocol', context: Context) -> AgentResponse:
        """
        Collaborate with another agent on a context.
        
        This enables multi-agent workflows.
        """
        # Process context with this agent first
        my_response = await self.process_context(context)
        
        # Create a new context from our response
        collaboration_context = Context(
            id=f"collab_{uuid.uuid4().hex[:8]}",
            type=ContextType.CONVERSATION,
            content=f"Agent {self.agent_id} says: {my_response.content}",
            metadata={
                "collaboration": True,
                "agents": [self.agent_id, other_agent.agent_id]
            },
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            parent_id=context.id
        )
        
        # Have the other agent process our response
        their_response = await other_agent.process_context(collaboration_context)
        
        # Combine responses
        combined_response = AgentResponse(
            agent_id=f"{self.agent_id}+{other_agent.agent_id}",
            content=f"Collaborative response:\n\n{self.agent_id}:\n{my_response.content}\n\n{other_agent.agent_id}:\n{their_response.content}",
            context_used=collaboration_context,
            metadata={
                "collaboration": True,
                "agent_responses": {
                    self.agent_id: my_response.content,
                    other_agent.agent_id: their_response.content
                }
            },
            processing_time=my_response.processing_time + their_response.processing_time,
            tokens_used=my_response.tokens_used + their_response.tokens_used,
            success=my_response.success and their_response.success
        )
        
        return combined_response
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()


class ClaudeAgent(AgentProtocol):
    """
    Implementation for Claude AI agents.
    """
    
    def __init__(self, config: AgentConfig, context_manager: ContextManager):
        # Set Claude-specific defaults
        config.api_endpoint = config.api_endpoint or "https://api.anthropic.com/v1/messages"
        config.model_name = config.model_name or "claude-3-opus-20240229"
        super().__init__(config, context_manager)
    
    async def process_context(self, context: Context) -> AgentResponse:
        """Process context using Claude API."""
        start_time = time.time()
        
        try:
            # Check rate limit
            if not await self._check_rate_limit():
                return AgentResponse(
                    agent_id=self.agent_id,
                    content="Rate limit exceeded",
                    success=False,
                    error_message="Rate limit exceeded"
                )
            
            # Update status
            self.status = AgentStatus.PROCESSING
            
            # Prepare the prompt with context
            prompt = self._prepare_prompt(context)
            
            # Make API request
            payload = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }
            
            response = await self._make_api_request(self.config.api_endpoint, payload)
            
            # Extract response
            content = response.get("content", [{}])[0].get("text", "")
            tokens_used = response.get("usage", {}).get("total_tokens", 0)
            
            # Update metrics
            self.total_requests += 1
            self.successful_requests += 1
            self.total_tokens += tokens_used
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            # Update status
            self.status = AgentStatus.IDLE
            
            return AgentResponse(
                agent_id=self.agent_id,
                content=content,
                context_used=context,
                metadata={"model": self.config.model_name},
                processing_time=processing_time,
                tokens_used=tokens_used,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error processing context with Claude: {e}")
            self.failed_requests += 1
            self.status = AgentStatus.ERROR
            
            return AgentResponse(
                agent_id=self.agent_id,
                content="",
                context_used=context,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def generate_context(self, prompt: str, parent_context: Optional[Context] = None) -> Context:
        """Generate a new context using Claude."""
        # Create a temporary context for the prompt
        temp_context = Context(
            id=f"temp_{uuid.uuid4().hex[:8]}",
            type=ContextType.INSTRUCTION,
            content=prompt,
            metadata={"generated": True},
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            parent_id=parent_context.id if parent_context else None
        )
        
        # Process the context
        response = await self.process_context(temp_context)
        
        # Create a new context from the response
        new_context = Context(
            id=f"gen_{uuid.uuid4().hex[:8]}",
            type=ContextType.CONVERSATION,
            content=response.content,
            metadata={
                "generated_by": self.agent_id,
                "prompt": prompt,
                "model": self.config.model_name
            },
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            parent_id=parent_context.id if parent_context else None
        )
        
        # Store in context manager
        self.context_manager.add_context(new_context)
        
        return new_context
    
    def _prepare_prompt(self, context: Context) -> str:
        """Prepare a prompt with context for Claude."""
        # Build context chain
        chain = self.context_manager.get_context_chain(context.id)
        
        prompt_parts = []
        
        # Add context chain
        if len(chain) > 1:
            prompt_parts.append("Previous context:")
            for ctx in chain[:-1]:  # Exclude current context
                prompt_parts.append(f"[{ctx.type.value}] {ctx.content[:500]}...")
            prompt_parts.append("")
        
        # Add current context
        prompt_parts.append("Current context:")
        prompt_parts.append(f"[{context.type.value}] {context.content}")
        prompt_parts.append("")
        
        # Add instructions based on context type
        if context.type == ContextType.CODE:
            prompt_parts.append("Please analyze this code and provide insights.")
        elif context.type == ContextType.ANALYSIS:
            prompt_parts.append("Please continue or enhance this analysis.")
        elif context.type == ContextType.DECISION:
            prompt_parts.append("Please evaluate this decision and provide recommendations.")
        else:
            prompt_parts.append("Please process this context and provide a relevant response.")
        
        return "\n".join(prompt_parts)
    
    def serialize_state(self) -> bytes:
        """Serialize agent state."""
        state_data = {
            "agent_id": self.agent_id,
            "config": {
                "name": self.config.name,
                "type": self.config.type,
                "model_name": self.config.model_name
            },
            "metrics": self.get_metrics(),
            "internal_state": self._state
        }
        return pickle.dumps(state_data)
    
    def deserialize_state(self, data: bytes) -> None:
        """Deserialize agent state."""
        state_data = pickle.loads(data)
        
        # Restore metrics
        metrics = state_data.get("metrics", {})
        self.total_requests = metrics.get("total_requests", 0)
        self.successful_requests = metrics.get("successful_requests", 0)
        self.failed_requests = metrics.get("failed_requests", 0)
        self.total_tokens = metrics.get("total_tokens", 0)
        
        # Restore internal state
        self._state = state_data.get("internal_state", {})


class GPTAgent(AgentProtocol):
    """
    Implementation for GPT agents.
    """
    
    def __init__(self, config: AgentConfig, context_manager: ContextManager):
        # Set GPT-specific defaults
        config.api_endpoint = config.api_endpoint or "https://api.openai.com/v1/chat/completions"
        config.model_name = config.model_name or "gpt-4-turbo-preview"
        super().__init__(config, context_manager)
    
    async def process_context(self, context: Context) -> AgentResponse:
        """Process context using OpenAI API."""
        start_time = time.time()
        
        try:
            # Check rate limit
            if not await self._check_rate_limit():
                return AgentResponse(
                    agent_id=self.agent_id,
                    content="Rate limit exceeded",
                    success=False,
                    error_message="Rate limit exceeded"
                )
            
            # Update status
            self.status = AgentStatus.PROCESSING
            
            # Prepare messages
            messages = self._prepare_messages(context)
            
            # Make API request
            payload = {
                "model": self.config.model_name,
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }
            
            response = await self._make_api_request(self.config.api_endpoint, payload)
            
            # Extract response
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            tokens_used = response.get("usage", {}).get("total_tokens", 0)
            
            # Update metrics
            self.total_requests += 1
            self.successful_requests += 1
            self.total_tokens += tokens_used
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            # Update status
            self.status = AgentStatus.IDLE
            
            return AgentResponse(
                agent_id=self.agent_id,
                content=content,
                context_used=context,
                metadata={"model": self.config.model_name},
                processing_time=processing_time,
                tokens_used=tokens_used,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error processing context with GPT: {e}")
            self.failed_requests += 1
            self.status = AgentStatus.ERROR
            
            return AgentResponse(
                agent_id=self.agent_id,
                content="",
                context_used=context,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def generate_context(self, prompt: str, parent_context: Optional[Context] = None) -> Context:
        """Generate a new context using GPT."""
        # Similar implementation to ClaudeAgent
        temp_context = Context(
            id=f"temp_{uuid.uuid4().hex[:8]}",
            type=ContextType.INSTRUCTION,
            content=prompt,
            metadata={"generated": True},
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            parent_id=parent_context.id if parent_context else None
        )
        
        response = await self.process_context(temp_context)
        
        new_context = Context(
            id=f"gen_{uuid.uuid4().hex[:8]}",
            type=ContextType.CONVERSATION,
            content=response.content,
            metadata={
                "generated_by": self.agent_id,
                "prompt": prompt,
                "model": self.config.model_name
            },
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            parent_id=parent_context.id if parent_context else None
        )
        
        self.context_manager.add_context(new_context)
        
        return new_context
    
    def _prepare_messages(self, context: Context) -> List[Dict[str, str]]:
        """Prepare messages for GPT API."""
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": "You are a helpful AI assistant participating in a multi-agent context sharing system."
        })
        
        # Add context chain as conversation history
        chain = self.context_manager.get_context_chain(context.id)
        
        for ctx in chain:
            role = "assistant" if ctx.agent_id == self.agent_id else "user"
            messages.append({
                "role": role,
                "content": f"[{ctx.type.value}] {ctx.content}"
            })
        
        return messages
    
    def serialize_state(self) -> bytes:
        """Serialize agent state."""
        state_data = {
            "agent_id": self.agent_id,
            "config": {
                "name": self.config.name,
                "type": self.config.type,
                "model_name": self.config.model_name
            },
            "metrics": self.get_metrics(),
            "internal_state": self._state
        }
        return pickle.dumps(state_data)
    
    def deserialize_state(self, data: bytes) -> None:
        """Deserialize agent state."""
        state_data = pickle.loads(data)
        
        metrics = state_data.get("metrics", {})
        self.total_requests = metrics.get("total_requests", 0)
        self.successful_requests = metrics.get("successful_requests", 0)
        self.failed_requests = metrics.get("failed_requests", 0)
        self.total_tokens = metrics.get("total_tokens", 0)
        
        self._state = state_data.get("internal_state", {})


class LocalAgent(AgentProtocol):
    """
    Implementation for local/custom agents that don't use external APIs.
    
    This can be used for local LLMs, rule-based systems, or mock agents for testing.
    """
    
    def __init__(self, config: AgentConfig, context_manager: ContextManager, 
                 process_function: Optional[Callable] = None):
        super().__init__(config, context_manager)
        self.process_function = process_function or self._default_process
    
    async def process_context(self, context: Context) -> AgentResponse:
        """Process context using local function."""
        start_time = time.time()
        
        try:
            self.status = AgentStatus.PROCESSING
            
            # Call the processing function
            if asyncio.iscoroutinefunction(self.process_function):
                content = await self.process_function(context)
            else:
                content = self.process_function(context)
            
            processing_time = time.time() - start_time
            
            self.total_requests += 1
            self.successful_requests += 1
            
            self.status = AgentStatus.IDLE
            
            return AgentResponse(
                agent_id=self.agent_id,
                content=content,
                context_used=context,
                metadata={"local": True},
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in local agent: {e}")
            self.failed_requests += 1
            self.status = AgentStatus.ERROR
            
            return AgentResponse(
                agent_id=self.agent_id,
                content="",
                context_used=context,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def generate_context(self, prompt: str, parent_context: Optional[Context] = None) -> Context:
        """Generate context locally."""
        content = f"Local agent response to: {prompt}"
        
        new_context = Context(
            id=f"local_{uuid.uuid4().hex[:8]}",
            type=ContextType.CONVERSATION,
            content=content,
            metadata={"local": True},
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            parent_id=parent_context.id if parent_context else None
        )
        
        self.context_manager.add_context(new_context)
        
        return new_context
    
    def _default_process(self, context: Context) -> str:
        """Default processing function."""
        return f"Processed context of type {context.type.value} with {len(context.content)} characters"
    
    def serialize_state(self) -> bytes:
        """Serialize agent state."""
        state_data = {
            "agent_id": self.agent_id,
            "metrics": self.get_metrics(),
            "internal_state": self._state
        }
        return pickle.dumps(state_data)
    
    def deserialize_state(self, data: bytes) -> None:
        """Deserialize agent state."""
        state_data = pickle.loads(data)
        
        metrics = state_data.get("metrics", {})
        self.total_requests = metrics.get("total_requests", 0)
        self.successful_requests = metrics.get("successful_requests", 0)
        self.failed_requests = metrics.get("failed_requests", 0)
        
        self._state = state_data.get("internal_state", {})


class AgentOrchestrator:
    """
    Orchestrates multiple agents and manages their interactions.
    """
    
    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager
        self.agents: Dict[str, AgentProtocol] = {}
        self.active_workflows: Dict[str, Any] = {}
    
    def register_agent(self, agent: AgentProtocol):
        """Register an agent with the orchestrator."""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent {agent.agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent {agent_id}")
    
    async def broadcast_context(self, context: Context) -> List[AgentResponse]:
        """
        Broadcast a context to all agents and collect responses.
        """
        tasks = []
        for agent in self.agents.values():
            if AgentCapability.TEXT_GENERATION in agent.config.capabilities:
                tasks.append(agent.process_context(context))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_responses = []
        for response in responses:
            if isinstance(response, AgentResponse):
                valid_responses.append(response)
            else:
                logger.error(f"Agent failed with error: {response}")
        
        return valid_responses
    
    async def consensus_response(self, context: Context, min_agents: int = 3) -> AgentResponse:
        """
        Get a consensus response from multiple agents.
        
        This implements a simple majority voting mechanism.
        """
        responses = await self.broadcast_context(context)
        
        if len(responses) < min_agents:
            return AgentResponse(
                agent_id="orchestrator",
                content="Not enough agents for consensus",
                success=False,
                error_message=f"Only {len(responses)} agents responded, need {min_agents}"
            )
        
        # For simplicity, we'll combine all responses
        # In production, implement proper consensus algorithms
        combined_content = "\n\n".join([
            f"{r.agent_id}: {r.content}" for r in responses
        ])
        
        return AgentResponse(
            agent_id="consensus",
            content=combined_content,
            context_used=context,
            metadata={"agents": [r.agent_id for r in responses]},
            success=True
        )
    
    async def pipeline_workflow(self, initial_context: Context, 
                               agent_sequence: List[str]) -> List[AgentResponse]:
        """
        Execute a pipeline workflow where agents process in sequence.
        """
        responses = []
        current_context = initial_context
        
        for agent_id in agent_sequence:
            if agent_id not in self.agents:
                logger.error(f"Agent {agent_id} not found")
                continue
            
            agent = self.agents[agent_id]
            response = await agent.process_context(current_context)
            responses.append(response)
            
            # Create new context from response for next agent
            current_context = Context(
                id=f"pipeline_{uuid.uuid4().hex[:8]}",
                type=ContextType.CONVERSATION,
                content=response.content,
                metadata={"pipeline": True, "step": len(responses)},
                agent_id=agent_id,
                timestamp=datetime.now(timezone.utc),
                parent_id=current_context.id
            )
            
            # Add to context manager
            self.context_manager.add_context(current_context)
        
        return responses
    
    def get_agent_metrics(self) -> Dict[str, Dict]:
        """Get metrics for all agents."""
        return {
            agent_id: agent.get_metrics()
            for agent_id, agent in self.agents.items()
        }
    
    async def cleanup(self):
        """Clean up all agents."""
        tasks = [agent.cleanup() for agent in self.agents.values()]
        await asyncio.gather(*tasks)


# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize context manager
        context_manager = ContextManager(repo_path="./test_agent_repo")
        
        # Create orchestrator
        orchestrator = AgentOrchestrator(context_manager)
        
        # Create and register agents
        claude_config = AgentConfig(
            name="claude_analyzer",
            type="claude",
            api_key="your_api_key_here",  # Replace with actual key
            capabilities=[AgentCapability.ANALYSIS, AgentCapability.TEXT_GENERATION]
        )
        claude_agent = ClaudeAgent(claude_config, context_manager)
        orchestrator.register_agent(claude_agent)
        
        # Create a local agent for testing
        local_config = AgentConfig(
            name="local_processor",
            type="local",
            capabilities=[AgentCapability.TEXT_GENERATION]
        )
        local_agent = LocalAgent(local_config, context_manager)
        orchestrator.register_agent(local_agent)
        
        # Create a test context
        test_context = Context(
            id="test_001",
            type=ContextType.ANALYSIS,
            content="Analyze the benefits of multi-agent AI systems",
            metadata={"test": True},
            agent_id="user",
            timestamp=datetime.now(timezone.utc)
        )
        
        # Add to context manager
        context_manager.add_context(test_context)
        
        # Test broadcast
        print("Broadcasting context to all agents...")
        responses = await orchestrator.broadcast_context(test_context)
        for response in responses:
            print(f"{response.agent_id}: {response.content[:100]}...")
        
        # Test pipeline
        print("\nRunning pipeline workflow...")
        pipeline_responses = await orchestrator.pipeline_workflow(
            test_context,
            [local_agent.agent_id, local_agent.agent_id]  # Use local agent twice
        )
        for i, response in enumerate(pipeline_responses):
            print(f"Step {i+1} ({response.agent_id}): {response.content[:100]}...")
        
        # Get metrics
        print("\nAgent metrics:")
        metrics = orchestrator.get_agent_metrics()
        for agent_id, agent_metrics in metrics.items():
            print(f"{agent_id}: {agent_metrics}")
        
        # Cleanup
        await orchestrator.cleanup()
        context_manager.cleanup()
    
    # Run the example
    asyncio.run(main())
