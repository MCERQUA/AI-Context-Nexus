"""
AI Context Nexus - Distributed AI Memory System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ai-context-nexus",
    version="1.0.0",
    author="Mike Cerqua",
    author_email="mikecerqua@example.com",
    description="A revolutionary distributed memory and context management system for AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mikecerqua/ai-context-nexus",
    project_urls={
        "Bug Tracker": "https://github.com/mikecerqua/ai-context-nexus/issues",
        "Documentation": "https://ai-context-nexus.readthedocs.io",
        "Source Code": "https://github.com/mikecerqua/ai-context-nexus",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "aiohttp>=3.9.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "gitpython>=3.1.0",
        "redis>=5.0.0",
        "numpy>=1.24.0",
        "lz4>=4.3.0",
        "networkx>=3.0",
        "tabulate>=0.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "black>=23.0.0",
        ],
        "embeddings": [
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.0",
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nexus=scripts.nexus_cli:cli",
            "ai-context-nexus=scripts.nexus_cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
    },
    zip_safe=False,
)