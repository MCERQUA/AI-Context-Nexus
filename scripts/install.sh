#!/bin/bash

# AI Context Nexus - Installation and Initialization Script
# This script sets up the complete AI Context Nexus system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
NEXUS_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="3.9"
NODE_VERSION="18"
GO_VERSION="1.21"

# ASCII Art Banner
show_banner() {
    echo -e "${PURPLE}"
    cat << "EOF"
    ___    ____   ______            __            __     _   __                    
   /   |  /  _/  / ____/___  ____  / /____  _  __/ /_   / | / /__  _  ____  _______
  / /| |  / /   / /   / __ \/ __ \/ __/ _ \| |/_/ __/  /  |/ / _ \| |/_/ / / / ___/
 / ___ |_/ /   / /___/ /_/ / / / / /_/  __/>  </ /_   / /|  /  __/>  </ /_/ (__  ) 
/_/  |_/___/   \____/\____/_/ /_/\__/\___/_/|_|\__/  /_/ |_/\___/_/|_|\__,_/____/  
                                                                                    
            Multi-Agent AI Context Sharing & Memory System v1.0.0
EOF
    echo -e "${NC}"
}

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root!"
        exit 1
    fi
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        DISTRO=$(lsb_release -si 2>/dev/null || echo "Unknown")
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    log_info "Detected OS: $OS ${DISTRO:-}"
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check RAM
    if [[ "$OS" == "linux" ]]; then
        TOTAL_RAM=$(free -m | awk '/^Mem:/{print $2}')
    elif [[ "$OS" == "macos" ]]; then
        TOTAL_RAM=$(($(sysctl -n hw.memsize) / 1024 / 1024))
    else
        TOTAL_RAM=4096  # Assume 4GB for Windows
    fi
    
    if [[ $TOTAL_RAM -lt 4096 ]]; then
        log_warning "System has less than 4GB RAM. Performance may be affected."
    else
        log_success "RAM check passed: ${TOTAL_RAM}MB"
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df -BG "$NEXUS_HOME" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $AVAILABLE_SPACE -lt 10 ]]; then
        log_warning "Less than 10GB of disk space available."
    else
        log_success "Disk space check passed: ${AVAILABLE_SPACE}GB available"
    fi
    
    # Check CPU cores
    if [[ "$OS" == "linux" ]]; then
        CPU_CORES=$(nproc)
    elif [[ "$OS" == "macos" ]]; then
        CPU_CORES=$(sysctl -n hw.ncpu)
    else
        CPU_CORES=4  # Assume 4 cores for Windows
    fi
    
    log_success "CPU cores available: $CPU_CORES"
}

# Install system dependencies
install_dependencies() {
    log_info "Installing system dependencies..."
    
    if [[ "$OS" == "linux" ]]; then
        # Update package manager
        if command -v apt-get &> /dev/null; then
            log_info "Using apt package manager..."
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                git \
                curl \
                wget \
                tmux \
                redis-server \
                postgresql \
                postgresql-contrib \
                python3-pip \
                python3-venv \
                python3-dev \
                libssl-dev \
                libffi-dev \
                libpq-dev \
                nodejs \
                npm \
                jq
        elif command -v yum &> /dev/null; then
            log_info "Using yum package manager..."
            sudo yum install -y \
                gcc \
                gcc-c++ \
                make \
                git \
                curl \
                wget \
                tmux \
                redis \
                postgresql \
                postgresql-server \
                python3 \
                python3-pip \
                python3-devel \
                openssl-devel \
                nodejs \
                npm
        elif command -v pacman &> /dev/null; then
            log_info "Using pacman package manager..."
            sudo pacman -Syu --noconfirm \
                base-devel \
                git \
                curl \
                wget \
                tmux \
                redis \
                postgresql \
                python \
                python-pip \
                nodejs \
                npm
        fi
    elif [[ "$OS" == "macos" ]]; then
        # Install Homebrew if not present
        if ! command -v brew &> /dev/null; then
            log_info "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        log_info "Installing dependencies via Homebrew..."
        brew install \
            git \
            tmux \
            redis \
            postgresql \
            python@${PYTHON_VERSION} \
            node \
            jq \
            wget
    fi
    
    log_success "System dependencies installed"
}

# Install Jujutsu (jj)
install_jujutsu() {
    log_info "Installing Jujutsu (jj)..."
    
    if command -v jj &> /dev/null; then
        log_success "Jujutsu already installed: $(jj --version)"
        return 0
    fi
    
    if [[ "$OS" == "linux" ]]; then
        # Download latest release
        JJ_VERSION=$(curl -s https://api.github.com/repos/martinvonz/jj/releases/latest | jq -r .tag_name)
        wget -q "https://github.com/martinvonz/jj/releases/download/${JJ_VERSION}/jj-${JJ_VERSION}-x86_64-unknown-linux-musl.tar.gz"
        tar -xzf "jj-${JJ_VERSION}-x86_64-unknown-linux-musl.tar.gz"
        sudo mv jj /usr/local/bin/
        rm "jj-${JJ_VERSION}-x86_64-unknown-linux-musl.tar.gz"
    elif [[ "$OS" == "macos" ]]; then
        brew install jj
    fi
    
    log_success "Jujutsu installed: $(jj --version)"
}

# Setup Python environment
setup_python() {
    log_info "Setting up Python environment..."
    
    cd "$NEXUS_HOME"
    
    # Create virtual environment
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate and upgrade pip
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
    
    # Install Python dependencies
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt
    
    log_success "Python environment setup complete"
}

# Create requirements.txt if it doesn't exist
create_requirements() {
    if [[ ! -f "$NEXUS_HOME/requirements.txt" ]]; then
        log_info "Creating requirements.txt..."
        cat > "$NEXUS_HOME/requirements.txt" << 'EOF'
# Core dependencies
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Web framework
aiohttp>=3.8.0
aiofiles>=0.8.0
fastapi>=0.70.0
uvicorn>=0.15.0

# Database
redis>=4.0.0
psycopg2-binary>=2.9.0
sqlalchemy>=1.4.0
alembic>=1.7.0

# Git integration
gitpython>=3.1.0
pygit2>=1.7.0

# Compression
lz4>=3.1.0
zstandard>=0.16.0

# Serialization
msgpack>=1.0.0
pyyaml>=6.0
orjson>=3.6.0

# Monitoring
psutil>=5.8.0
prometheus-client>=0.12.0

# Testing
pytest>=6.2.0
pytest-asyncio>=0.16.0
pytest-cov>=3.0.0

# ML/AI (optional, for advanced features)
torch>=1.10.0
transformers>=4.15.0
sentence-transformers>=2.1.0
faiss-cpu>=1.7.0

# Async
asyncio>=3.4.3
aioredis>=2.0.0

# Utilities
python-dotenv>=0.19.0
click>=8.0.0
rich>=10.0.0
tqdm>=4.62.0
backoff>=1.11.0

# Security
cryptography>=35.0.0
pynacl>=1.4.0
EOF
        log_success "requirements.txt created"
    fi
}

# Initialize git repository
init_git_repo() {
    log_info "Initializing git repository..."
    
    REPO_PATH="$NEXUS_HOME/context_repo"
    mkdir -p "$REPO_PATH"
    cd "$REPO_PATH"
    
    if [[ ! -d ".git" ]]; then
        git init
        git config user.name "AI Context Nexus"
        git config user.email "nexus@ai.local"
        
        # Create initial commit
        echo "# AI Context Nexus Repository" > README.md
        git add README.md
        git commit -m "Initial commit"
        
        log_success "Git repository initialized"
    else
        log_info "Git repository already exists"
    fi
    
    # Initialize JJ
    if [[ ! -d ".jj" ]]; then
        jj init --git-repo .
        log_success "Jujutsu repository initialized"
    else
        log_info "Jujutsu repository already exists"
    fi
    
    cd "$NEXUS_HOME"
}

# Setup Redis
setup_redis() {
    log_info "Setting up Redis..."
    
    # Create Redis config
    mkdir -p "$NEXUS_HOME/config/redis"
    cat > "$NEXUS_HOME/config/redis/redis.conf" << 'EOF'
# Redis configuration for AI Context Nexus
port 6379
bind 127.0.0.1
protected-mode yes
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename nexus.rdb
dir ./data/redis
maxmemory 2gb
maxmemory-policy allkeys-lru
appendonly yes
appendfilename "nexus.aof"
appendfsync everysec
EOF
    
    # Create Redis data directory
    mkdir -p "$NEXUS_HOME/data/redis"
    
    log_success "Redis configured"
}

# Setup PostgreSQL
setup_postgresql() {
    log_info "Setting up PostgreSQL database..."
    
    # Check if PostgreSQL is running
    if ! systemctl is-active --quiet postgresql; then
        log_info "Starting PostgreSQL service..."
        sudo systemctl start postgresql
        sudo systemctl enable postgresql
    fi
    
    # Create database and user
    sudo -u postgres psql << EOF
CREATE USER nexus WITH PASSWORD 'nexus_password';
CREATE DATABASE ai_context_nexus OWNER nexus;
GRANT ALL PRIVILEGES ON DATABASE ai_context_nexus TO nexus;
EOF
    
    log_success "PostgreSQL database configured"
}

# Create directory structure
create_directories() {
    log_info "Creating directory structure..."
    
    directories=(
        "data"
        "logs"
        "pids"
        "backups"
        "memory/l2_cache"
        "context_repo"
        "certs"
        "temp"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$NEXUS_HOME/$dir"
    done
    
    log_success "Directory structure created"
}

# Generate SSL certificates
generate_certificates() {
    log_info "Generating SSL certificates..."
    
    CERT_DIR="$NEXUS_HOME/certs"
    
    if [[ ! -f "$CERT_DIR/server.key" ]]; then
        openssl req -x509 -newkey rsa:4096 \
            -keyout "$CERT_DIR/server.key" \
            -out "$CERT_DIR/server.crt" \
            -days 365 -nodes -subj \
            "/C=US/ST=State/L=City/O=AI Context Nexus/CN=localhost"
        
        log_success "SSL certificates generated"
    else
        log_info "SSL certificates already exist"
    fi
}

# Create systemd service files
create_systemd_services() {
    log_info "Creating systemd service files..."
    
    if [[ "$OS" != "linux" ]]; then
        log_info "Skipping systemd setup (not on Linux)"
        return 0
    fi
    
    # Context Manager Service
    sudo tee /etc/systemd/system/nexus-context-manager.service > /dev/null << EOF
[Unit]
Description=AI Context Nexus - Context Manager
After=network.target redis.service postgresql.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$NEXUS_HOME
Environment="PATH=$NEXUS_HOME/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$NEXUS_HOME/venv/bin/python core/context_manager.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # Memory Manager Service
    sudo tee /etc/systemd/system/nexus-memory-manager.service > /dev/null << EOF
[Unit]
Description=AI Context Nexus - Memory Manager
After=network.target nexus-context-manager.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$NEXUS_HOME
Environment="PATH=$NEXUS_HOME/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$NEXUS_HOME/venv/bin/python memory/memory_manager.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    log_success "Systemd services created"
}

# Run tests
run_tests() {
    log_info "Running system tests..."
    
    cd "$NEXUS_HOME"
    source venv/bin/activate
    
    # Create test script
    cat > test_system.py << 'EOF'
#!/usr/bin/env python3
import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_system():
    """Test basic system functionality."""
    print("Testing AI Context Nexus components...")
    
    # Test imports
    try:
        from core.context_manager import ContextManager, Context, ContextType
        print("✓ Context Manager imports successful")
    except ImportError as e:
        print(f"✗ Context Manager import failed: {e}")
        return False
    
    try:
        from agents.agent_protocol import AgentProtocol, LocalAgent
        print("✓ Agent Protocol imports successful")
    except ImportError as e:
        print(f"✗ Agent Protocol import failed: {e}")
        return False
    
    # Test basic functionality
    try:
        # Initialize context manager
        manager = ContextManager(repo_path="./test_repo")
        
        # Create test context
        context = Context(
            id="test_001",
            type=ContextType.ANALYSIS,
            content="Test content",
            metadata={"test": True},
            agent_id="test_agent",
            timestamp=datetime.now(timezone.utc)
        )
        
        # Add context
        commit_hash = manager.add_context(context)
        print(f"✓ Context added with commit: {commit_hash[:8]}")
        
        # Retrieve context
        retrieved = manager.get_context("test_001")
        if retrieved:
            print("✓ Context retrieval successful")
        else:
            print("✗ Context retrieval failed")
            return False
        
        # Cleanup
        manager.cleanup()
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    from datetime import datetime, timezone
    success = asyncio.run(test_system())
    sys.exit(0 if success else 1)
EOF
    
    # Run tests
    if python test_system.py; then
        log_success "System tests passed"
    else
        log_warning "Some tests failed - please check the configuration"
    fi
}

# Generate example configuration
generate_example_config() {
    log_info "Generating example configurations..."
    
    # Agent configuration example
    cat > "$NEXUS_HOME/config/agents.example.json" << 'EOF'
{
  "agents": [
    {
      "name": "claude_primary",
      "type": "claude",
      "api_key": "YOUR_CLAUDE_API_KEY",
      "model_name": "claude-3-opus-20240229",
      "max_tokens": 4096,
      "temperature": 0.7,
      "capabilities": ["text_generation", "analysis", "code_generation"]
    },
    {
      "name": "gpt_secondary",
      "type": "gpt",
      "api_key": "YOUR_OPENAI_API_KEY",
      "model_name": "gpt-4-turbo-preview",
      "max_tokens": 4096,
      "temperature": 0.7,
      "capabilities": ["text_generation", "summarization"]
    },
    {
      "name": "local_processor",
      "type": "local",
      "capabilities": ["memory_management", "decision_making"]
    }
  ]
}
EOF
    
    log_success "Example configurations generated"
}

# Setup completion script
setup_completion() {
    log_info "Setting up command completion..."
    
    cat > "$NEXUS_HOME/scripts/nexus_completion.sh" << 'EOF'
# Bash completion for AI Context Nexus

_nexus_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # Main commands
    opts="start stop restart status health attach backup menu help"
    
    case "${prev}" in
        nexus)
            COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            return 0
            ;;
        *)
            ;;
    esac
}

complete -F _nexus_completion nexus
EOF
    
    # Create main nexus command
    cat > "$NEXUS_HOME/nexus" << 'EOF'
#!/bin/bash
# AI Context Nexus main command

NEXUS_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-help}" in
    start|stop|restart|status|health|attach|backup|menu)
        exec "$NEXUS_HOME/scripts/tmux_orchestrator.sh" "$@"
        ;;
    help|--help|-h)
        echo "AI Context Nexus - Multi-Agent AI System"
        echo ""
        echo "Usage: nexus [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  start    - Start the system"
        echo "  stop     - Stop the system"
        echo "  restart  - Restart the system"
        echo "  status   - Show system status"
        echo "  health   - Check system health"
        echo "  attach   - Attach to tmux session"
        echo "  backup   - Backup context data"
        echo "  menu     - Interactive menu"
        echo "  help     - Show this help"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run 'nexus help' for usage information"
        exit 1
        ;;
esac
EOF
    
    chmod +x "$NEXUS_HOME/nexus"
    
    log_success "Command completion setup complete"
}

# Final setup
final_setup() {
    log_info "Performing final setup..."
    
    # Set permissions
    chmod +x "$NEXUS_HOME/scripts/"*.sh 2>/dev/null || true
    chmod +x "$NEXUS_HOME/scripts/"*.py 2>/dev/null || true
    
    # Create .env file
    if [[ ! -f "$NEXUS_HOME/.env" ]]; then
        cat > "$NEXUS_HOME/.env" << EOF
# AI Context Nexus Environment Variables
NEXUS_HOME=$NEXUS_HOME
NEXUS_ENV=development
LOG_LEVEL=INFO
REDIS_HOST=localhost
REDIS_PORT=6379
DATABASE_URL=postgresql://nexus:nexus_password@localhost/ai_context_nexus
JWT_SECRET=$(openssl rand -hex 32)
EOF
        log_success ".env file created"
    fi
    
    # Add to PATH
    echo "" >> ~/.bashrc
    echo "# AI Context Nexus" >> ~/.bashrc
    echo "export PATH=\"$NEXUS_HOME:\$PATH\"" >> ~/.bashrc
    echo "source $NEXUS_HOME/scripts/nexus_completion.sh" >> ~/.bashrc
    
    log_success "Final setup complete"
}

# Main installation flow
main() {
    show_banner
    
    log_info "Starting AI Context Nexus installation..."
    log_info "Installation directory: $NEXUS_HOME"
    echo
    
    # Run installation steps
    check_root
    detect_os
    check_requirements
    
    read -p "Continue with installation? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Installation cancelled"
        exit 0
    fi
    
    create_requirements
    install_dependencies
    install_jujutsu
    setup_python
    create_directories
    init_git_repo
    setup_redis
    
    # Optional PostgreSQL setup
    read -p "Setup PostgreSQL database? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_postgresql
    fi
    
    generate_certificates
    create_systemd_services
    generate_example_config
    setup_completion
    run_tests
    final_setup
    
    echo
    log_success "AI Context Nexus installation complete!"
    echo
    echo -e "${CYAN}Next steps:${NC}"
    echo "1. Configure your API keys in config/agents.json"
    echo "2. Source your shell configuration: source ~/.bashrc"
    echo "3. Start the system: nexus start"
    echo "4. View status: nexus status"
    echo "5. Access the web UI at http://localhost:8080"
    echo
    echo -e "${GREEN}Happy context sharing!${NC}"
}

# Run main function
main "$@"
