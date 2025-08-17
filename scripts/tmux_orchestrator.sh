#!/bin/bash

# AI Context Nexus - Tmux Orchestrator
# This script manages multiple AI agent processes using tmux sessions
# It provides process isolation, monitoring, and automatic recovery

set -e

# Configuration
NEXUS_HOME="${NEXUS_HOME:-$(dirname $(realpath $0))/..}"
CONFIG_FILE="${NEXUS_HOME}/config/tmux_config.yaml"
LOG_DIR="${NEXUS_HOME}/logs"
PID_DIR="${NEXUS_HOME}/pids"
SESSION_NAME="ai-context-nexus"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Ensure directories exist
mkdir -p "${LOG_DIR}" "${PID_DIR}"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_DIR}/orchestrator.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_DIR}/orchestrator.log"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_DIR}/orchestrator.log"
}

# Check dependencies
check_dependencies() {
    local deps=("tmux" "git" "python3" "jj")
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "$dep is not installed. Please install it first."
            exit 1
        fi
    done
    
    log_info "All dependencies are installed"
}

# Initialize the nexus system
initialize_nexus() {
    log_info "Initializing AI Context Nexus..."
    
    # Initialize git repository
    if [ ! -d "${NEXUS_HOME}/context_repo/.git" ]; then
        log_info "Initializing git repository..."
        cd "${NEXUS_HOME}/context_repo"
        git init
        git config user.name "AI Context Nexus"
        git config user.email "nexus@ai.local"
        cd -
    fi
    
    # Initialize JJ repository
    if [ ! -d "${NEXUS_HOME}/context_repo/.jj" ]; then
        log_info "Initializing JJ repository..."
        cd "${NEXUS_HOME}/context_repo"
        jj init --git-repo .
        cd -
    fi
    
    # Create Python virtual environment if not exists
    if [ ! -d "${NEXUS_HOME}/venv" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv "${NEXUS_HOME}/venv"
        source "${NEXUS_HOME}/venv/bin/activate"
        pip install -q --upgrade pip
        pip install -q -r "${NEXUS_HOME}/requirements.txt"
        deactivate
    fi
    
    log_info "Initialization complete"
}

# Start tmux session with all components
start_nexus() {
    log_info "Starting AI Context Nexus..."
    
    # Check if session already exists
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        log_warning "Session ${SESSION_NAME} already exists. Attaching..."
        tmux attach-session -t "${SESSION_NAME}"
        return 0
    fi
    
    # Create new session with main window
    tmux new-session -d -s "${SESSION_NAME}" -n "orchestrator"
    
    # Window 0: Orchestrator (main control)
    tmux send-keys -t "${SESSION_NAME}:orchestrator" "cd ${NEXUS_HOME}" C-m
    tmux send-keys -t "${SESSION_NAME}:orchestrator" "source venv/bin/activate" C-m
    tmux send-keys -t "${SESSION_NAME}:orchestrator" "python3 scripts/orchestrator_daemon.py" C-m
    
    # Window 1: Context Manager
    tmux new-window -t "${SESSION_NAME}" -n "context-manager"
    tmux send-keys -t "${SESSION_NAME}:context-manager" "cd ${NEXUS_HOME}" C-m
    tmux send-keys -t "${SESSION_NAME}:context-manager" "source venv/bin/activate" C-m
    tmux send-keys -t "${SESSION_NAME}:context-manager" "python3 core/context_manager_server.py" C-m
    
    # Window 2: Agent Pool
    tmux new-window -t "${SESSION_NAME}" -n "agent-pool"
    
    # Split window into panes for multiple agents
    tmux split-window -t "${SESSION_NAME}:agent-pool" -h
    tmux split-window -t "${SESSION_NAME}:agent-pool.0" -v
    tmux split-window -t "${SESSION_NAME}:agent-pool.1" -v
    
    # Start agents in each pane
    local agents=("claude" "gpt" "local1" "local2")
    for i in "${!agents[@]}"; do
        tmux send-keys -t "${SESSION_NAME}:agent-pool.$i" "cd ${NEXUS_HOME}" C-m
        tmux send-keys -t "${SESSION_NAME}:agent-pool.$i" "source venv/bin/activate" C-m
        tmux send-keys -t "${SESSION_NAME}:agent-pool.$i" \
            "python3 agents/agent_runner.py --type ${agents[$i]} --id agent_${agents[$i]}_$i" C-m
    done
    
    # Window 3: Memory Manager
    tmux new-window -t "${SESSION_NAME}" -n "memory"
    tmux send-keys -t "${SESSION_NAME}:memory" "cd ${NEXUS_HOME}" C-m
    tmux send-keys -t "${SESSION_NAME}:memory" "source venv/bin/activate" C-m
    tmux send-keys -t "${SESSION_NAME}:memory" "python3 memory/memory_manager.py" C-m
    
    # Window 4: Monitor
    tmux new-window -t "${SESSION_NAME}" -n "monitor"
    tmux split-window -t "${SESSION_NAME}:monitor" -h
    
    # Pane 0: System metrics
    tmux send-keys -t "${SESSION_NAME}:monitor.0" "cd ${NEXUS_HOME}" C-m
    tmux send-keys -t "${SESSION_NAME}:monitor.0" "watch -n 2 'python3 scripts/system_monitor.py'" C-m
    
    # Pane 1: Logs
    tmux send-keys -t "${SESSION_NAME}:monitor.1" "cd ${NEXUS_HOME}" C-m
    tmux send-keys -t "${SESSION_NAME}:monitor.1" "tail -f logs/*.log" C-m
    
    # Window 5: Git/JJ Operations
    tmux new-window -t "${SESSION_NAME}" -n "version-control"
    tmux split-window -t "${SESSION_NAME}:version-control" -h
    
    # Pane 0: Git status
    tmux send-keys -t "${SESSION_NAME}:version-control.0" "cd ${NEXUS_HOME}/context_repo" C-m
    tmux send-keys -t "${SESSION_NAME}:version-control.0" "watch -n 5 'git log --oneline -10 && echo && git status'" C-m
    
    # Pane 1: JJ status
    tmux send-keys -t "${SESSION_NAME}:version-control.1" "cd ${NEXUS_HOME}/context_repo" C-m
    tmux send-keys -t "${SESSION_NAME}:version-control.1" "watch -n 5 'jj log -r :: -n 10'" C-m
    
    # Window 6: Interactive Console
    tmux new-window -t "${SESSION_NAME}" -n "console"
    tmux send-keys -t "${SESSION_NAME}:console" "cd ${NEXUS_HOME}" C-m
    tmux send-keys -t "${SESSION_NAME}:console" "source venv/bin/activate" C-m
    tmux send-keys -t "${SESSION_NAME}:console" "python3 scripts/interactive_console.py" C-m
    
    # Select the orchestrator window
    tmux select-window -t "${SESSION_NAME}:orchestrator"
    
    log_info "AI Context Nexus started successfully"
    log_info "Attach with: tmux attach-session -t ${SESSION_NAME}"
    
    # Save PIDs for monitoring
    tmux list-panes -a -F "#{pane_pid}" -t "${SESSION_NAME}" > "${PID_DIR}/tmux_panes.pids"
}

# Stop the nexus system
stop_nexus() {
    log_info "Stopping AI Context Nexus..."
    
    if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        log_warning "Session ${SESSION_NAME} not found"
        return 0
    fi
    
    # Send graceful shutdown to all panes
    tmux list-panes -a -F "#{pane_id}" -t "${SESSION_NAME}" | while read pane; do
        tmux send-keys -t "$pane" C-c
        sleep 0.5
        tmux send-keys -t "$pane" "exit" C-m
    done
    
    sleep 2
    
    # Kill the session
    tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
    
    # Clean up PID files
    rm -f "${PID_DIR}"/*.pid
    
    log_info "AI Context Nexus stopped"
}

# Restart the nexus system
restart_nexus() {
    log_info "Restarting AI Context Nexus..."
    stop_nexus
    sleep 2
    start_nexus
}

# Monitor system health
monitor_health() {
    log_info "Monitoring system health..."
    
    if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        log_error "Session ${SESSION_NAME} not found"
        return 1
    fi
    
    # Check each window/pane
    local unhealthy=0
    
    tmux list-windows -t "${SESSION_NAME}" -F "#{window_name}" | while read window; do
        local pane_count=$(tmux list-panes -t "${SESSION_NAME}:${window}" | wc -l)
        echo -e "${CYAN}Window ${window}:${NC} ${pane_count} panes"
        
        tmux list-panes -t "${SESSION_NAME}:${window}" -F "#{pane_id} #{pane_dead}" | while read pane_info; do
            local pane_id=$(echo $pane_info | cut -d' ' -f1)
            local pane_dead=$(echo $pane_info | cut -d' ' -f2)
            
            if [ "$pane_dead" = "1" ]; then
                echo -e "  ${RED}✗${NC} Pane ${pane_id} is dead"
                ((unhealthy++))
            else
                echo -e "  ${GREEN}✓${NC} Pane ${pane_id} is alive"
            fi
        done
    done
    
    if [ $unhealthy -gt 0 ]; then
        log_warning "Found $unhealthy unhealthy panes"
        return 1
    else
        log_info "All components are healthy"
        return 0
    fi
}

# Recover failed components
recover_failed() {
    log_info "Attempting to recover failed components..."
    
    if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        log_error "Session not found, starting fresh..."
        start_nexus
        return 0
    fi
    
    # Check and restart dead panes
    tmux list-panes -a -t "${SESSION_NAME}" -F "#{window_name} #{pane_id} #{pane_dead}" | while read pane_info; do
        local window=$(echo $pane_info | cut -d' ' -f1)
        local pane_id=$(echo $pane_info | cut -d' ' -f2)
        local pane_dead=$(echo $pane_info | cut -d' ' -f3)
        
        if [ "$pane_dead" = "1" ]; then
            log_warning "Recovering dead pane ${pane_id} in window ${window}"
            
            # Respawn the pane with appropriate command based on window
            case $window in
                "orchestrator")
                    tmux respawn-pane -t "${pane_id}" "cd ${NEXUS_HOME} && source venv/bin/activate && python3 scripts/orchestrator_daemon.py"
                    ;;
                "context-manager")
                    tmux respawn-pane -t "${pane_id}" "cd ${NEXUS_HOME} && source venv/bin/activate && python3 core/context_manager_server.py"
                    ;;
                "agent-pool")
                    tmux respawn-pane -t "${pane_id}" "cd ${NEXUS_HOME} && source venv/bin/activate && python3 agents/agent_runner.py --type local --id recovered_agent"
                    ;;
                "memory")
                    tmux respawn-pane -t "${pane_id}" "cd ${NEXUS_HOME} && source venv/bin/activate && python3 memory/memory_manager.py"
                    ;;
                *)
                    log_warning "Unknown window type: ${window}"
                    ;;
            esac
        fi
    done
    
    log_info "Recovery complete"
}

# Attach to session
attach_nexus() {
    if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        log_error "Session ${SESSION_NAME} not found. Start it first with: $0 start"
        exit 1
    fi
    
    tmux attach-session -t "${SESSION_NAME}"
}

# Show status
show_status() {
    echo -e "${PURPLE}=== AI Context Nexus Status ===${NC}"
    echo
    
    if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo -e "${RED}Status: Not Running${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Status: Running${NC}"
    echo
    
    # Show windows and panes
    echo -e "${CYAN}Tmux Windows:${NC}"
    tmux list-windows -t "${SESSION_NAME}" -F "  #{window_index}: #{window_name} (#{window_panes} panes)"
    echo
    
    # Show resource usage
    echo -e "${CYAN}Resource Usage:${NC}"
    if [ -f "${PID_DIR}/tmux_panes.pids" ]; then
        local total_mem=0
        local total_cpu=0
        
        while read pid; do
            if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
                local mem=$(ps -o rss= -p $pid 2>/dev/null | awk '{print $1/1024}')
                local cpu=$(ps -o %cpu= -p $pid 2>/dev/null)
                
                if [ -n "$mem" ]; then
                    total_mem=$(echo "$total_mem + $mem" | bc)
                fi
                if [ -n "$cpu" ]; then
                    total_cpu=$(echo "$total_cpu + $cpu" | bc)
                fi
            fi
        done < "${PID_DIR}/tmux_panes.pids"
        
        echo "  Total Memory: ${total_mem} MB"
        echo "  Total CPU: ${total_cpu}%"
    fi
    echo
    
    # Show git status
    echo -e "${CYAN}Git Repository:${NC}"
    if [ -d "${NEXUS_HOME}/context_repo/.git" ]; then
        cd "${NEXUS_HOME}/context_repo"
        echo "  Commits: $(git rev-list --count HEAD 2>/dev/null || echo 0)"
        echo "  Current Branch: $(git branch --show-current 2>/dev/null || echo 'main')"
        echo "  Last Commit: $(git log -1 --format='%h - %s' 2>/dev/null || echo 'No commits yet')"
        cd - > /dev/null
    fi
    echo
    
    # Show log tail
    echo -e "${CYAN}Recent Logs:${NC}"
    if [ -f "${LOG_DIR}/orchestrator.log" ]; then
        tail -n 5 "${LOG_DIR}/orchestrator.log" | sed 's/^/  /'
    fi
}

# Backup context repository
backup_contexts() {
    local backup_dir="${NEXUS_HOME}/backups/$(date +%Y%m%d_%H%M%S)"
    
    log_info "Creating backup in ${backup_dir}..."
    
    mkdir -p "${backup_dir}"
    
    # Backup git repository
    if [ -d "${NEXUS_HOME}/context_repo/.git" ]; then
        tar -czf "${backup_dir}/context_repo.tar.gz" -C "${NEXUS_HOME}" context_repo
        log_info "Context repository backed up"
    fi
    
    # Backup memory cache
    if [ -d "${NEXUS_HOME}/memory/l2_cache" ]; then
        tar -czf "${backup_dir}/memory_cache.tar.gz" -C "${NEXUS_HOME}" memory/l2_cache
        log_info "Memory cache backed up"
    fi
    
    # Backup configurations
    if [ -d "${NEXUS_HOME}/config" ]; then
        tar -czf "${backup_dir}/config.tar.gz" -C "${NEXUS_HOME}" config
        log_info "Configurations backed up"
    fi
    
    log_info "Backup complete: ${backup_dir}"
}

# Interactive menu
interactive_menu() {
    while true; do
        clear
        echo -e "${PURPLE}╔════════════════════════════════════════╗${NC}"
        echo -e "${PURPLE}║     AI Context Nexus Control Panel     ║${NC}"
        echo -e "${PURPLE}╚════════════════════════════════════════╝${NC}"
        echo
        echo "  1) Start System"
        echo "  2) Stop System"
        echo "  3) Restart System"
        echo "  4) Show Status"
        echo "  5) Monitor Health"
        echo "  6) Recover Failed Components"
        echo "  7) Attach to Session"
        echo "  8) Backup Contexts"
        echo "  9) View Logs"
        echo "  0) Exit"
        echo
        read -p "Select option: " choice
        
        case $choice in
            1) start_nexus ;;
            2) stop_nexus ;;
            3) restart_nexus ;;
            4) show_status ;;
            5) monitor_health ;;
            6) recover_failed ;;
            7) attach_nexus ;;
            8) backup_contexts ;;
            9) less "${LOG_DIR}/orchestrator.log" ;;
            0) exit 0 ;;
            *) echo "Invalid option" ;;
        esac
        
        echo
        read -p "Press Enter to continue..."
    done
}

# Main script logic
main() {
    case "${1:-}" in
        start)
            check_dependencies
            initialize_nexus
            start_nexus
            ;;
        stop)
            stop_nexus
            ;;
        restart)
            restart_nexus
            ;;
        status)
            show_status
            ;;
        health)
            monitor_health
            ;;
        recover)
            recover_failed
            ;;
        attach)
            attach_nexus
            ;;
        backup)
            backup_contexts
            ;;
        menu)
            interactive_menu
            ;;
        *)
            echo "AI Context Nexus - Tmux Orchestrator"
            echo
            echo "Usage: $0 {start|stop|restart|status|health|recover|attach|backup|menu}"
            echo
            echo "Commands:"
            echo "  start    - Start the AI Context Nexus system"
            echo "  stop     - Stop the system gracefully"
            echo "  restart  - Restart the system"
            echo "  status   - Show system status"
            echo "  health   - Check component health"
            echo "  recover  - Recover failed components"
            echo "  attach   - Attach to tmux session"
            echo "  backup   - Backup context repository"
            echo "  menu     - Interactive control menu"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
