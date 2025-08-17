# Base image with Python and system dependencies
FROM python:3.9-slim-bullseye AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    curl \
    wget \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install Jujutsu (jj)
RUN wget -q https://github.com/martinvonz/jj/releases/download/v0.12.0/jj-v0.12.0-x86_64-unknown-linux-musl.tar.gz \
    && tar -xzf jj-v0.12.0-x86_64-unknown-linux-musl.tar.gz \
    && mv jj /usr/local/bin/ \
    && rm jj-v0.12.0-x86_64-unknown-linux-musl.tar.gz

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/pids /app/memory/l2_cache /app/context_repo

# Initialize git repository
RUN cd /app/context_repo && \
    git init && \
    git config user.name "AI Context Nexus" && \
    git config user.email "nexus@ai.local" && \
    echo "# Context Repository" > README.md && \
    git add README.md && \
    git commit -m "Initial commit"

# Production image
FROM base AS production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NEXUS_ENV=production
ENV LOG_LEVEL=INFO

# Expose ports
EXPOSE 8080 8081 50051 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8080"]

# Development image
FROM base AS development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipython \
    jupyter

# Set development environment
ENV NEXUS_ENV=development
ENV LOG_LEVEL=DEBUG

# Mount volumes for development
VOLUME ["/app/data", "/app/logs", "/app/context_repo"]

# Use shell for development
CMD ["/bin/bash"]

# Test image
FROM base AS test

# Install test dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    pytest-benchmark \
    hypothesis

# Copy test files
COPY tests /app/tests

# Run tests
CMD ["pytest", "-v", "--cov=.", "--cov-report=term-missing", "tests/"]

# Multi-service orchestrator image
FROM base AS orchestrator

# Install supervisor for process management
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

# Copy supervisor configuration
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose all service ports
EXPOSE 8080 8081 50051 9090 6379 5432

# Start supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
