#!/usr/bin/env python3
"""
AI Context Nexus - Interactive CLI Tool

A comprehensive command-line interface for interacting with the AI Context Nexus system.
"""

import os
import sys
import json
import asyncio
import click
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone, timedelta
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.panel import Panel
from rich.tree import Tree
from rich import print as rprint
import aiohttp
import subprocess
from tabulate import tabulate

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.context_manager import ContextManager, Context, ContextType
from agents.agent_protocol import AgentOrchestrator, AgentConfig, LocalAgent

# Initialize Rich console
console = Console()

# Configuration
DEFAULT_CONFIG = Path.home() / ".nexus" / "config.yaml"
DEFAULT_API_URL = "http://localhost:8080"


class NexusCLI:
    """Main CLI class for AI Context Nexus."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_CONFIG
        self.config = self.load_config()
        self.api_url = self.config.get('api_url', DEFAULT_API_URL)
        self.context_manager = None
        self.orchestrator = None
    
    def load_config(self) -> Dict:
        """Load configuration from file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def save_config(self):
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
    
    async def init_local(self):
        """Initialize local context manager and orchestrator."""
        if not self.context_manager:
            repo_path = self.config.get('repo_path', './context_repo')
            self.context_manager = ContextManager(repo_path=repo_path)
        
        if not self.orchestrator:
            self.orchestrator = AgentOrchestrator(self.context_manager)
    
    async def api_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make API request to the nexus server."""
        async with aiohttp.ClientSession() as session:
            url = f"{self.api_url}{endpoint}"
            async with session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"API request failed: {response.status}")


# Create CLI instance
cli = NexusCLI()


@click.group()
@click.option('--config', '-c', type=click.Path(), help='Configuration file path')
@click.option('--api-url', '-u', help='API server URL')
@click.pass_context
def nexus(ctx, config, api_url):
    """AI Context Nexus CLI - Manage contexts, agents, and memory."""
    if config:
        cli.config_path = Path(config)
        cli.config = cli.load_config()
    
    if api_url:
        cli.api_url = api_url
        cli.config['api_url'] = api_url
        cli.save_config()
    
    ctx.obj = cli


@nexus.group()
def context():
    """Manage contexts."""
    pass


@context.command()
@click.argument('content')
@click.option('--type', '-t', type=click.Choice(['code', 'analysis', 'decision', 'conversation', 'document', 'memory']), default='conversation')
@click.option('--metadata', '-m', multiple=True, help='Metadata in key=value format')
@click.option('--parent', '-p', help='Parent context ID')
@click.option('--agent', '-a', default='cli', help='Agent ID')
@click.pass_obj
async def create(cli_obj, content, type, metadata, parent, agent):
    """Create a new context."""
    await cli_obj.init_local()
    
    # Parse metadata
    meta = {}
    for m in metadata:
        key, value = m.split('=', 1)
        meta[key] = value
    
    # Create context
    context = Context(
        id=f"ctx_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        type=ContextType[type.upper()],
        content=content,
        metadata=meta,
        agent_id=agent,
        timestamp=datetime.now(timezone.utc),
        parent_id=parent
    )
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Creating context...", total=None)
        
        commit_hash = cli_obj.context_manager.add_context(context)
        
        progress.update(task, completed=True)
    
    console.print(f"[green]✓[/green] Context created: {context.id}")
    console.print(f"  Commit: {commit_hash[:8]}")
    console.print(f"  Type: {type}")
    console.print(f"  Agent: {agent}")


@context.command()
@click.argument('context_id')
@click.pass_obj
async def get(cli_obj, context_id):
    """Get a context by ID."""
    await cli_obj.init_local()
    
    context = cli_obj.context_manager.get_context(context_id)
    
    if context:
        # Create panel with context details
        panel_content = f"""[bold]ID:[/bold] {context.id}
[bold]Type:[/bold] {context.type.value}
[bold]Agent:[/bold] {context.agent_id}
[bold]Timestamp:[/bold] {context.timestamp.isoformat()}
[bold]Parent:[/bold] {context.parent_id or 'None'}
[bold]Semantic Hash:[/bold] {context.semantic_hash[:16] if context.semantic_hash else 'None'}

[bold]Metadata:[/bold]
{yaml.dump(context.metadata, default_flow_style=False)}

[bold]Content:[/bold]
{context.content}"""
        
        panel = Panel(panel_content, title=f"Context: {context_id}", border_style="blue")
        console.print(panel)
    else:
        console.print(f"[red]Context not found: {context_id}[/red]")


@context.command()
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Maximum results')
@click.option('--type', '-t', help='Filter by context type')
@click.pass_obj
async def search(cli_obj, query, limit, type):
    """Search for contexts."""
    await cli_obj.init_local()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Searching...", total=None)
        
        results = cli_obj.context_manager.search_contexts(query, max_results=limit)
        
        progress.update(task, completed=True)
    
    if results:
        table = Table(title=f"Search Results for: {query}")
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Agent", style="green")
        table.add_column("Content Preview", style="white")
        table.add_column("Timestamp", style="yellow")
        
        for ctx in results:
            if type and ctx.type.value != type:
                continue
            
            content_preview = ctx.content[:50] + "..." if len(ctx.content) > 50 else ctx.content
            table.add_row(
                ctx.id,
                ctx.type.value,
                ctx.agent_id,
                content_preview,
                ctx.timestamp.strftime("%Y-%m-%d %H:%M")
            )
        
        console.print(table)
    else:
        console.print(f"[yellow]No results found for: {query}[/yellow]")


@context.command()
@click.argument('context_id')
@click.pass_obj
async def chain(cli_obj, context_id):
    """Show the context chain (parent-child relationships)."""
    await cli_obj.init_local()
    
    chain = cli_obj.context_manager.get_context_chain(context_id)
    
    if chain:
        tree = Tree(f"[bold]Context Chain for {context_id}[/bold]")
        
        for i, ctx in enumerate(chain):
            node_text = f"[cyan]{ctx.id}[/cyan] ({ctx.type.value}) - {ctx.content[:30]}..."
            if i == 0:
                current_node = tree.add(node_text)
            else:
                current_node = current_node.add(node_text)
        
        console.print(tree)
    else:
        console.print(f"[red]Context not found: {context_id}[/red]")


@context.command()
@click.option('--limit', '-l', default=10, help='Number of recent contexts')
@click.pass_obj
async def list(cli_obj, limit):
    """List recent contexts."""
    await cli_obj.init_local()
    
    # Get recent commits from git
    try:
        result = subprocess.run(
            ['git', 'log', '--oneline', f'-{limit}'],
            cwd=cli_obj.context_manager.repo.repo_path,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            table = Table(title="Recent Contexts")
            table.add_column("Commit", style="cyan")
            table.add_column("Context Info", style="white")
            
            for line in result.stdout.strip().split('\n'):
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    commit_hash, message = parts
                    table.add_row(commit_hash, message)
            
            console.print(table)
    except Exception as e:
        console.print(f"[red]Error listing contexts: {e}[/red]")


@nexus.group()
def agent():
    """Manage agents."""
    pass


@agent.command()
@click.pass_obj
async def list(cli_obj):
    """List all registered agents."""
    try:
        response = await cli_obj.api_request('GET', '/api/v1/agents')
        agents = response.get('agents', [])
        
        table = Table(title="Registered Agents")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Type", style="magenta")
        table.add_column("Status", style="yellow")
        table.add_column("Capabilities", style="white")
        
        for agent in agents:
            capabilities = ', '.join(agent.get('capabilities', []))
            table.add_row(
                agent['id'],
                agent['name'],
                agent['type'],
                agent['status'],
                capabilities
            )
        
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error listing agents: {e}[/red]")


@agent.command()
@click.argument('agent_id')
@click.pass_obj
async def status(cli_obj, agent_id):
    """Get agent status and metrics."""
    try:
        response = await cli_obj.api_request('GET', f'/api/v1/agents/{agent_id}/metrics')
        metrics = response.get('metrics', {})
        
        panel_content = f"""[bold]Agent ID:[/bold] {agent_id}
[bold]Status:[/bold] {metrics.get('status', 'unknown')}

[bold]Performance Metrics:[/bold]
  Total Requests: {metrics.get('total_requests', 0)}
  Successful: {metrics.get('successful_requests', 0)}
  Failed: {metrics.get('failed_requests', 0)}
  Success Rate: {metrics.get('success_rate', 0):.2%}
  Total Tokens: {metrics.get('total_tokens', 0):,}
  Avg Processing Time: {metrics.get('avg_processing_time', 0):.3f}s"""
        
        panel = Panel(panel_content, title=f"Agent Status: {agent_id}", border_style="green")
        console.print(panel)
    except Exception as e:
        console.print(f"[red]Error getting agent status: {e}[/red]")


@agent.command()
@click.argument('context_id')
@click.option('--agent-id', '-a', help='Specific agent to use')
@click.option('--all', 'use_all', is_flag=True, help='Process with all agents')
@click.pass_obj
async def process(cli_obj, context_id, agent_id, use_all):
    """Process a context with agent(s)."""
    await cli_obj.init_local()
    
    context = cli_obj.context_manager.get_context(context_id)
    if not context:
        console.print(f"[red]Context not found: {context_id}[/red]")
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing context...", total=None)
        
        if use_all:
            # Broadcast to all agents
            responses = await cli_obj.orchestrator.broadcast_context(context)
            
            progress.update(task, completed=True)
            
            console.print(f"[green]Received {len(responses)} responses[/green]")
            
            for response in responses:
                panel = Panel(
                    response.content,
                    title=f"Response from {response.agent_id}",
                    border_style="blue"
                )
                console.print(panel)
        else:
            # Process with specific agent
            if not agent_id:
                agent_id = "local_processor"
            
            # Create local agent if needed
            if agent_id not in cli_obj.orchestrator.agents:
                local_agent = LocalAgent(
                    AgentConfig(name=agent_id, type="local"),
                    cli_obj.context_manager
                )
                cli_obj.orchestrator.register_agent(local_agent)
            
            agent = cli_obj.orchestrator.agents[agent_id]
            response = await agent.process_context(context)
            
            progress.update(task, completed=True)
            
            panel = Panel(
                response.content,
                title=f"Response from {agent_id}",
                border_style="green"
            )
            console.print(panel)


@nexus.group()
def memory():
    """Manage memory hierarchy."""
    pass


@memory.command()
@click.pass_obj
async def stats(cli_obj):
    """Show memory statistics."""
    try:
        response = await cli_obj.api_request('GET', '/api/v1/memory/stats')
        stats = response
        
        # Tier statistics
        console.print("[bold]Memory Tier Statistics:[/bold]")
        
        tier_data = []
        for tier_name, tier_stats in stats.get('tiers', {}).items():
            tier_data.append([
                tier_name,
                f"{tier_stats['usage_percent']:.1f}%",
                f"{tier_stats['hit_rate']:.3f}",
                f"{tier_stats['avg_access_time_ms']:.2f}ms",
                tier_stats['hit_count'],
                tier_stats['miss_count'],
                tier_stats['eviction_count']
            ])
        
        headers = ["Tier", "Usage", "Hit Rate", "Avg Time", "Hits", "Misses", "Evictions"]
        console.print(tabulate(tier_data, headers=headers, tablefmt="grid"))
        
        # System statistics
        system_stats = stats.get('system', {})
        console.print("\n[bold]System Resources:[/bold]")
        console.print(f"  Memory: {system_stats.get('memory_percent', 0):.1f}%")
        console.print(f"  CPU: {system_stats.get('cpu_percent', 0):.1f}%")
        console.print(f"  Disk: {system_stats.get('disk_usage', 0):.1f}%")
        
        # Bloom filter stats
        bloom_stats = stats.get('bloom_filter', {})
        console.print("\n[bold]Bloom Filter:[/bold]")
        console.print(f"  Load Factor: {bloom_stats.get('load_factor', 0):.3f}")
        console.print(f"  Size: {bloom_stats.get('size', 0):,} bits")
        console.print(f"  Hash Functions: {bloom_stats.get('num_hashes', 0)}")
        
    except Exception as e:
        console.print(f"[red]Error getting memory stats: {e}[/red]")


@memory.command()
@click.option('--tier', '-t', type=click.Choice(['L1', 'L2', 'L3', 'all']), default='all')
@click.pass_obj
async def clear(cli_obj, tier):
    """Clear memory cache."""
    if click.confirm(f"Are you sure you want to clear {tier} cache?"):
        try:
            response = await cli_obj.api_request(
                'POST', 
                '/api/v1/memory/clear',
                json={'tier': tier}
            )
            
            console.print(f"[green]✓[/green] Cache cleared: {tier}")
        except Exception as e:
            console.print(f"[red]Error clearing cache: {e}[/red]")


@memory.command()
@click.argument('context_ids', nargs=-1, required=True)
@click.pass_obj
async def prefetch(cli_obj, context_ids):
    """Prefetch contexts into cache."""
    try:
        response = await cli_obj.api_request(
            'POST',
            '/api/v1/memory/prefetch',
            json={'context_ids': list(context_ids)}
        )
        
        console.print(f"[green]✓[/green] Prefetched {len(context_ids)} contexts")
    except Exception as e:
        console.print(f"[red]Error prefetching: {e}[/red]")


@nexus.group()
def system():
    """System management commands."""
    pass


@system.command()
def start():
    """Start the AI Context Nexus system."""
    console.print("[bold]Starting AI Context Nexus...[/bold]")
    
    try:
        result = subprocess.run(
            ['./scripts/tmux_orchestrator.sh', 'start'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            console.print("[green]✓[/green] System started successfully")
            console.print("  Run 'nexus system attach' to view tmux session")
        else:
            console.print(f"[red]Failed to start system:[/red]\n{result.stderr}")
    except Exception as e:
        console.print(f"[red]Error starting system: {e}[/red]")


@system.command()
def stop():
    """Stop the AI Context Nexus system."""
    if click.confirm("Are you sure you want to stop the system?"):
        console.print("[bold]Stopping AI Context Nexus...[/bold]")
        
        try:
            result = subprocess.run(
                ['./scripts/tmux_orchestrator.sh', 'stop'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                console.print("[green]✓[/green] System stopped")
            else:
                console.print(f"[red]Failed to stop system:[/red]\n{result.stderr}")
        except Exception as e:
            console.print(f"[red]Error stopping system: {e}[/red]")


@system.command()
def restart():
    """Restart the AI Context Nexus system."""
    console.print("[bold]Restarting AI Context Nexus...[/bold]")
    
    try:
        result = subprocess.run(
            ['./scripts/tmux_orchestrator.sh', 'restart'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            console.print("[green]✓[/green] System restarted")
        else:
            console.print(f"[red]Failed to restart system:[/red]\n{result.stderr}")
    except Exception as e:
        console.print(f"[red]Error restarting system: {e}[/red]")


@system.command()
def status():
    """Show system status."""
    try:
        result = subprocess.run(
            ['./scripts/tmux_orchestrator.sh', 'status'],
            capture_output=True,
            text=True
        )
        
        console.print(result.stdout)
    except Exception as e:
        console.print(f"[red]Error getting status: {e}[/red]")


@system.command()
def attach():
    """Attach to tmux session."""
    try:
        subprocess.run(['tmux', 'attach-session', '-t', 'ai-context-nexus'])
    except Exception as e:
        console.print(f"[red]Error attaching to session: {e}[/red]")


@system.command()
def health():
    """Check system health."""
    try:
        result = subprocess.run(
            ['./scripts/tmux_orchestrator.sh', 'health'],
            capture_output=True,
            text=True
        )
        
        if "healthy" in result.stdout.lower():
            console.print("[green]✓[/green] System is healthy")
        else:
            console.print("[yellow]⚠[/yellow] System health check failed")
        
        console.print(result.stdout)
    except Exception as e:
        console.print(f"[red]Error checking health: {e}[/red]")


@system.command()
def backup():
    """Create system backup."""
    console.print("[bold]Creating backup...[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Backing up...", total=None)
        
        try:
            result = subprocess.run(
                ['./scripts/tmux_orchestrator.sh', 'backup'],
                capture_output=True,
                text=True
            )
            
            progress.update(task, completed=True)
            
            if result.returncode == 0:
                console.print("[green]✓[/green] Backup created successfully")
                console.print(result.stdout)
            else:
                console.print(f"[red]Backup failed:[/red]\n{result.stderr}")
        except Exception as e:
            console.print(f"[red]Error creating backup: {e}[/red]")


@nexus.command()
def interactive():
    """Start interactive mode."""
    console.print("[bold]AI Context Nexus Interactive Mode[/bold]")
    console.print("Type 'help' for commands, 'exit' to quit\n")
    
    while True:
        try:
            command = console.input("[bold cyan]nexus>[/bold cyan] ")
            
            if command.lower() in ['exit', 'quit', 'q']:
                break
            elif command.lower() in ['help', 'h', '?']:
                console.print("""
Available commands:
  create <content>    - Create a new context
  get <id>           - Get context by ID
  search <query>     - Search contexts
  list               - List recent contexts
  agents             - List agents
  stats              - Show memory stats
  status             - System status
  help               - Show this help
  exit               - Exit interactive mode
                """)
            elif command.startswith('create '):
                content = command[7:]
                # Create context with default settings
                asyncio.run(create_context_interactive(content))
            elif command.startswith('get '):
                context_id = command[4:]
                asyncio.run(get_context_interactive(context_id))
            elif command.startswith('search '):
                query = command[7:]
                asyncio.run(search_contexts_interactive(query))
            elif command == 'list':
                asyncio.run(list_contexts_interactive())
            elif command == 'agents':
                asyncio.run(list_agents_interactive())
            elif command == 'stats':
                asyncio.run(show_stats_interactive())
            elif command == 'status':
                show_status_interactive()
            else:
                console.print(f"[yellow]Unknown command: {command}[/yellow]")
                console.print("Type 'help' for available commands")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' to quit[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    console.print("[green]Goodbye![/green]")


async def create_context_interactive(content):
    """Helper for interactive context creation."""
    await cli.init_local()
    
    context = Context(
        id=f"ctx_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        type=ContextType.CONVERSATION,
        content=content,
        metadata={'source': 'interactive'},
        agent_id='cli',
        timestamp=datetime.now(timezone.utc)
    )
    
    commit_hash = cli.context_manager.add_context(context)
    console.print(f"[green]✓[/green] Created: {context.id} ({commit_hash[:8]})")


async def get_context_interactive(context_id):
    """Helper for interactive context retrieval."""
    await cli.init_local()
    
    context = cli.context_manager.get_context(context_id)
    if context:
        console.print(f"[cyan]{context.id}[/cyan]: {context.content}")
    else:
        console.print(f"[red]Not found: {context_id}[/red]")


async def search_contexts_interactive(query):
    """Helper for interactive search."""
    await cli.init_local()
    
    results = cli.context_manager.search_contexts(query, max_results=5)
    
    if results:
        for ctx in results:
            console.print(f"[cyan]{ctx.id}[/cyan]: {ctx.content[:50]}...")
    else:
        console.print(f"[yellow]No results for: {query}[/yellow]")


async def list_contexts_interactive():
    """Helper for interactive listing."""
    try:
        result = subprocess.run(
            ['git', 'log', '--oneline', '-5'],
            cwd='./context_repo',
            capture_output=True,
            text=True
        )
        console.print(result.stdout)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


async def list_agents_interactive():
    """Helper for interactive agent listing."""
    try:
        response = await cli.api_request('GET', '/api/v1/agents')
        agents = response.get('agents', [])
        
        for agent in agents:
            console.print(f"[cyan]{agent['id']}[/cyan]: {agent['type']} - {agent['status']}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


async def show_stats_interactive():
    """Helper for interactive stats display."""
    try:
        response = await cli.api_request('GET', '/api/v1/memory/stats')
        
        for tier, stats in response.get('tiers', {}).items():
            console.print(f"{tier}: {stats['usage_percent']:.1f}% used, {stats['hit_rate']:.2f} hit rate")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def show_status_interactive():
    """Helper for interactive status display."""
    try:
        result = subprocess.run(
            ['./scripts/tmux_orchestrator.sh', 'status'],
            capture_output=True,
            text=True
        )
        
        if "Running" in result.stdout:
            console.print("[green]System is running[/green]")
        else:
            console.print("[red]System is not running[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@nexus.command()
def version():
    """Show version information."""
    version_info = """
AI Context Nexus
Version: 1.0.0
Build: 2024.01.20
Python: 3.9+
    """
    console.print(version_info)


def main():
    """Main entry point."""
    # Handle async commands properly
    import asyncio
    
    def run_async_command(coro):
        """Helper to run async commands."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    
    # Monkey-patch click to handle async commands
    original_invoke = click.Command.invoke
    
    def invoke_async(self, ctx):
        rv = original_invoke(self, ctx)
        if asyncio.iscoroutine(rv):
            return run_async_command(rv)
        return rv
    
    click.Command.invoke = invoke_async
    
    # Run CLI
    nexus()


if __name__ == '__main__':
    main()
