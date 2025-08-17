# Getting Started with AI Context Nexus

This guide will help you set up and start using AI Context Nexus in minutes.

## Prerequisites

- Python 3.9 or higher
- Git 2.30 or higher
- 4GB RAM minimum
- Linux/macOS/Windows with WSL2

## Installation

### Option 1: Quick Start (Recommended)

```bash
git clone https://github.com/mikecerqua/ai-context-nexus
cd ai-context-nexus
./quickstart.sh
```

This script will:
1. Create a virtual environment
2. Install all dependencies
3. Initialize the context repository
4. Start the interactive CLI

### Option 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/mikecerqua/ai-context-nexus
cd ai-context-nexus

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize the system
python scripts/nexus_cli.py system init
```

### Option 3: Docker

```bash
# Clone the repository
git clone https://github.com/mikecerqua/ai-context-nexus
cd ai-context-nexus

# Start with Docker Compose
docker-compose up -d

# Access the CLI
docker-compose exec nexus python scripts/nexus_cli.py interactive
```

## First Steps

### 1. Create Your First Context

```bash
python scripts/nexus_cli.py context create "This is my first context in the nexus system" -t test
```

### 2. Search for Contexts

```bash
python scripts/nexus_cli.py context search "first context" -l 5
```

### 3. Start Interactive Mode

```bash
python scripts/nexus_cli.py interactive
```

In interactive mode, you can:
- Create and manage contexts
- View the semantic graph
- Monitor system performance
- Manage agents

## Basic Usage

### Creating Contexts

```python
from ai_context_nexus import ContextNexus

# Initialize the nexus
nexus = ContextNexus()

# Create a context
context_id = nexus.create_context(
    content="Important information about the project",
    context_type="documentation",
    tags=["important", "project"]
)
```

### Retrieving Contexts

```python
# Search by similarity
results = nexus.search("project documentation", limit=10)

# Get specific context
context = nexus.get_context(context_id)

# Get context chain
chain = nexus.get_context_chain(context_id, depth=5)
```

### Agent Integration

```python
from ai_context_nexus import Agent

# Create an agent
agent = Agent("my_agent", capabilities=["read", "write"])

# Submit context through agent
agent.submit_context("Analysis results from data processing")

# Retrieve relevant contexts
contexts = agent.retrieve_contexts("data analysis", limit=5)
```

## Configuration

Edit `config/config.json` to customize:

```json
{
  "repository": {
    "path": "./context_repo",
    "auto_commit": true
  },
  "cache": {
    "l1_size_mb": 100,
    "l2_size_mb": 1000,
    "eviction_policy": "lru"
  },
  "semantic": {
    "embedding_dim": 768,
    "similarity_threshold": 0.7
  }
}
```

## Next Steps

- Read the [Architecture Overview](architecture.md)
- Explore [API Reference](api-reference.md)
- Try the [Tutorials](tutorials/index.md)
- Join our [Community](https://github.com/mikecerqua/ai-context-nexus/discussions)

## Troubleshooting

### Common Issues

**Issue**: "ModuleNotFoundError"
- **Solution**: Ensure you've activated the virtual environment and installed dependencies

**Issue**: "Permission denied" when initializing
- **Solution**: Check write permissions in the current directory

**Issue**: High memory usage
- **Solution**: Adjust cache sizes in config.json

For more help, see our [FAQ](faq.md) or open an [issue](https://github.com/mikecerqua/ai-context-nexus/issues).