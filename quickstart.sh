#!/bin/bash

# AI Context Nexus - Quick Start Script
# This script provides the fastest way to get started with the system

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ASCII Banner
echo -e "${BLUE}"
cat << "EOF"
    ___    ____   ______            __            __     _   __                    
   /   |  /  _/  / ____/___  ____  / /____  _  __/ /_   / | / /__  _  ____  _______
  / /| |  / /   / /   / __ \/ __ \/ __/ _ \| |/_/ __/  /  |/ / _ \| |/_/ / / / ___/
 / ___ |_/ /   / /___/ /_/ / / / / /_/  __/>  </ /_   / /|  /  __/>  </ /_/ (__  ) 
/_/  |_/___/   \____/\____/_/ /_/\__/\___/_/|_|\__/  /_/ |_/\___/_/|_|\__,_/____/  
                                                                                    
                        Q U I C K   S T A R T   G U I D E
EOF
echo -e "${NC}"

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Check if running in the correct directory
if [ ! -f "README.md" ] || [ ! -d "core" ]; then
    print_error "Please run this script from the ai-context-nexus root directory"
    exit 1
fi

# Main menu
show_menu() {
    echo
    echo "What would you like to do?"
    echo
    echo "  1) 🚀 Quick Install (Recommended for first-time users)"
    echo "  2) 🐳 Docker Setup (Run with Docker Compose)"
    echo "  3) ☸️  Kubernetes Deploy (Deploy to K8s cluster)"
    echo "  4) 🛠️  Development Setup (For contributors)"
    echo "  5) 📊 Run Demo (See the system in action)"
    echo "  6) 📚 View Documentation"
    echo "  7) 🔍 System Check (Verify installation)"
    echo "  8) 🚪 Exit"
    echo
    read -p "Select option [1-8]: " choice
}

# Quick Install
quick_install() {
    print_info "Starting Quick Install..."
    echo
    
    # Check Python
    if command -v python3 &> /dev/null; then
        print_success "Python3 found: $(python3 --version)"
    else
        print_error "Python3 not found. Please install Python 3.9 or higher"
        exit 1
    fi
    
    # Check Git
    if command -v git &> /dev/null; then
        print_success "Git found: $(git --version)"
    else
        print_error "Git not found. Please install Git"
        exit 1
    fi
    
    # Create virtual environment
    print_info "Creating Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    
    # Install dependencies
    print_info "Installing Python dependencies..."
    pip install --quiet --upgrade pip
    
    # Create minimal requirements for quick start
    cat > requirements_minimal.txt << EOF
aiohttp>=3.8.0
pyyaml>=6.0
click>=8.0.0
rich>=10.0.0
gitpython>=3.1.0
redis>=4.0.0
numpy>=1.21.0
lz4>=3.1.0
EOF
    
    pip install --quiet -r requirements_minimal.txt
    print_success "Dependencies installed"
    
    # Initialize git repository
    print_info "Initializing context repository..."
    mkdir -p context_repo
    cd context_repo
    if [ ! -d ".git" ]; then
        git init --quiet
        git config user.name "AI Context Nexus"
        git config user.email "nexus@ai.local"
        echo "# Context Repository" > README.md
        git add README.md
        git commit -m "Initial commit" --quiet
    fi
    cd ..
    print_success "Context repository initialized"
    
    # Create necessary directories
    print_info "Creating directory structure..."
    mkdir -p data logs pids memory/l2_cache backups config
    print_success "Directories created"
    
    # Create basic configuration
    print_info "Creating configuration..."
    if [ ! -f "config/agents.json" ]; then
        cat > config/agents.json << EOF
{
  "agents": [
    {
      "name": "local_processor",
      "type": "local",
      "capabilities": ["text_generation", "analysis"]
    }
  ]
}
EOF
    fi
    print_success "Configuration created"
    
    echo
    print_success "Quick installation complete!"
    echo
    echo "Next steps:"
    echo "  1. Activate virtual environment: source venv/bin/activate"
    echo "  2. Run the CLI: python scripts/nexus_cli.py --help"
    echo "  3. Start interactive mode: python scripts/nexus_cli.py interactive"
    echo
}

# Docker Setup
docker_setup() {
    print_info "Setting up Docker environment..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker first"
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose not found. Please install Docker Compose"
        echo "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    print_success "Docker found: $(docker --version)"
    print_success "Docker Compose found: $(docker-compose --version)"
    
    echo
    print_info "Building Docker images..."
    docker-compose build --quiet
    
    echo
    print_info "Starting services..."
    docker-compose up -d
    
    echo
    print_success "Docker setup complete!"
    echo
    echo "Services running:"
    docker-compose ps
    echo
    echo "Access points:"
    echo "  - API: http://localhost:8080"
    echo "  - Memory Manager: http://localhost:8081"
    echo "  - Grafana: http://localhost:3000 (admin/admin)"
    echo "  - Documentation: http://localhost:8000"
    echo
    echo "Commands:"
    echo "  - View logs: docker-compose logs -f"
    echo "  - Stop services: docker-compose down"
    echo "  - Clean up: docker-compose down -v"
    echo
}

# Kubernetes Deploy
kubernetes_deploy() {
    print_info "Deploying to Kubernetes..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl not found. Please install kubectl"
        echo "Visit: https://kubernetes.io/docs/tasks/tools/"
        exit 1
    fi
    
    print_success "kubectl found: $(kubectl version --client --short)"
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        echo "Please configure kubectl to connect to your cluster"
        exit 1
    fi
    
    print_success "Connected to cluster"
    
    echo
    print_info "Creating namespace..."
    kubectl create namespace ai-context-nexus --dry-run=client -o yaml | kubectl apply -f -
    
    print_info "Deploying application..."
    kubectl apply -f kubernetes/deployment.yaml
    
    echo
    print_success "Kubernetes deployment initiated!"
    echo
    echo "Check deployment status:"
    echo "  kubectl -n ai-context-nexus get pods"
    echo
    echo "Access the service:"
    echo "  kubectl -n ai-context-nexus port-forward service/nexus-api 8080:8080"
    echo
}

# Development Setup
dev_setup() {
    print_info "Setting up development environment..."
    
    # Run quick install first
    quick_install
    
    # Install additional dev dependencies
    print_info "Installing development dependencies..."
    pip install --quiet pytest pytest-asyncio pytest-cov black flake8 mypy ipython
    print_success "Development dependencies installed"
    
    # Install pre-commit hooks
    print_info "Setting up pre-commit hooks..."
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Run tests before commit
source venv/bin/activate
python -m pytest tests/ -q
python -m black --check .
python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
EOF
    chmod +x .git/hooks/pre-commit
    print_success "Pre-commit hooks installed"
    
    echo
    print_success "Development environment ready!"
    echo
    echo "Development commands:"
    echo "  - Run tests: pytest tests/"
    echo "  - Format code: black ."
    echo "  - Check style: flake8 ."
    echo "  - Type check: mypy ."
    echo "  - Interactive shell: ipython"
    echo
}

# Run Demo
run_demo() {
    print_info "Running system demo..."
    echo
    
    # Ensure virtual environment
    if [ ! -d "venv" ]; then
        print_warning "Virtual environment not found. Running quick install first..."
        quick_install
    fi
    
    source venv/bin/activate
    
    # Create demo script
    cat > demo.py << 'EOF'
#!/usr/bin/env python3
"""AI Context Nexus - Interactive Demo"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.context_manager import ContextManager, Context, ContextType
from agents.agent_protocol import LocalAgent, AgentConfig, AgentOrchestrator

async def run_demo():
    print("🚀 AI Context Nexus Demo\n")
    
    # Initialize system
    print("Initializing system...")
    context_manager = ContextManager(repo_path="./demo_repo")
    orchestrator = AgentOrchestrator(context_manager)
    
    # Create local agent
    agent_config = AgentConfig(
        name="demo_agent",
        type="local",
        capabilities=["text_generation", "analysis"]
    )
    agent = LocalAgent(agent_config, context_manager)
    orchestrator.register_agent(agent)
    
    print("✓ System initialized\n")
    
    # Demo 1: Create and store context
    print("Demo 1: Creating and storing context")
    print("-" * 40)
    
    context1 = Context(
        id="demo_001",
        type=ContextType.ANALYSIS,
        content="Analyze the benefits of distributed AI systems",
        metadata={"demo": True, "category": "AI"},
        agent_id="user",
        timestamp=datetime.now(timezone.utc)
    )
    
    commit_hash = context_manager.add_context(context1)
    print(f"Created context: {context1.id}")
    print(f"Commit hash: {commit_hash[:8]}")
    print(f"Content: {context1.content}\n")
    
    # Demo 2: Process with agent
    print("Demo 2: Processing context with agent")
    print("-" * 40)
    
    response = await agent.process_context(context1)
    print(f"Agent response: {response.content}\n")
    
    # Demo 3: Create chain of contexts
    print("Demo 3: Creating context chain")
    print("-" * 40)
    
    context2 = Context(
        id="demo_002",
        type=ContextType.DECISION,
        content="Should we implement this system in production?",
        metadata={"demo": True},
        agent_id="user",
        timestamp=datetime.now(timezone.utc),
        parent_id=context1.id
    )
    
    context_manager.add_context(context2)
    
    chain = context_manager.get_context_chain(context2.id)
    print(f"Context chain length: {len(chain)}")
    for ctx in chain:
        print(f"  - {ctx.id}: {ctx.content[:50]}...")
    print()
    
    # Demo 4: Semantic search
    print("Demo 4: Semantic search")
    print("-" * 40)
    
    results = context_manager.search_contexts("distributed systems", max_results=5)
    print(f"Found {len(results)} matching contexts")
    for ctx in results:
        print(f"  - {ctx.id}: {ctx.content[:50]}...")
    print()
    
    # Demo 5: Memory statistics
    print("Demo 5: System insights")
    print("-" * 40)
    
    insights = context_manager.get_graph_insights()
    print(f"Total contexts: {insights['total_contexts']}")
    print(f"Total connections: {insights['total_connections']}")
    print(f"Context clusters: {len(insights['clusters'])}")
    
    # Cleanup
    context_manager.cleanup()
    print("\n✓ Demo complete!")

if __name__ == "__main__":
    asyncio.run(run_demo())
EOF
    
    # Run the demo
    python demo.py
    
    echo
    print_success "Demo completed!"
    echo
}

# View Documentation
view_docs() {
    print_info "Opening documentation..."
    echo
    echo "Available documentation:"
    echo
    echo "1. README.md - Project overview"
    echo "2. docs/architecture.md - System architecture"
    echo "3. docs/usage_guide.md - Usage guide"
    echo "4. docs/failure_recovery.md - Failure recovery"
    echo "5. PROJECT_SUMMARY.md - Executive summary"
    echo "6. FILE_INDEX.md - Complete file listing"
    echo
    read -p "Select document [1-6]: " doc_choice
    
    case $doc_choice in
        1) less README.md ;;
        2) less docs/architecture.md ;;
        3) less docs/usage_guide.md ;;
        4) less docs/failure_recovery.md ;;
        5) less PROJECT_SUMMARY.md ;;
        6) less FILE_INDEX.md ;;
        *) print_error "Invalid selection" ;;
    esac
}

# System Check
system_check() {
    print_info "Running system check..."
    echo
    
    # Check Python
    echo -n "Python 3.9+: "
    if command -v python3 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $(python3 --version)"
    else
        echo -e "${RED}✗${NC} Not found"
    fi
    
    # Check Git
    echo -n "Git: "
    if command -v git &> /dev/null; then
        echo -e "${GREEN}✓${NC} $(git --version | head -n1)"
    else
        echo -e "${RED}✗${NC} Not found"
    fi
    
    # Check Docker
    echo -n "Docker: "
    if command -v docker &> /dev/null; then
        echo -e "${GREEN}✓${NC} $(docker --version)"
    else
        echo -e "${YELLOW}○${NC} Not found (optional)"
    fi
    
    # Check Redis
    echo -n "Redis: "
    if command -v redis-cli &> /dev/null; then
        echo -e "${GREEN}✓${NC} $(redis-cli --version)"
    else
        echo -e "${YELLOW}○${NC} Not found (optional)"
    fi
    
    # Check directories
    echo
    echo "Directory structure:"
    for dir in core agents memory scripts docs config; do
        echo -n "  $dir: "
        if [ -d "$dir" ]; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${RED}✗${NC}"
        fi
    done
    
    # Check virtual environment
    echo
    echo -n "Virtual environment: "
    if [ -d "venv" ]; then
        echo -e "${GREEN}✓${NC} Found"
    else
        echo -e "${YELLOW}○${NC} Not created"
    fi
    
    # Check configuration
    echo -n "Configuration: "
    if [ -f "config/config.json" ] || [ -f "config/agents.json" ]; then
        echo -e "${GREEN}✓${NC} Found"
    else
        echo -e "${YELLOW}○${NC} Not configured"
    fi
    
    echo
    print_success "System check complete!"
    echo
}

# Main loop
main() {
    while true; do
        show_menu
        
        case $choice in
            1) quick_install ;;
            2) docker_setup ;;
            3) kubernetes_deploy ;;
            4) dev_setup ;;
            5) run_demo ;;
            6) view_docs ;;
            7) system_check ;;
            8) 
                print_info "Thank you for using AI Context Nexus!"
                exit 0
                ;;
            *)
                print_error "Invalid option. Please select 1-8"
                ;;
        esac
        
        echo
        read -p "Press Enter to continue..."
    done
}

# Run main function
main
