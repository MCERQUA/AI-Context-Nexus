# Contributing to AI Context Nexus

First off, thank you for considering contributing to AI Context Nexus! It's people like you that make AI Context Nexus such a great tool for the AI community.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and expected**
- **Include logs and error messages**
- **Include your environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a detailed description of the proposed enhancement**
- **Explain why this enhancement would be useful**
- **List any alternative solutions you've considered**
- **Include mockups or examples if applicable**

### Pull Requests

1. Fork the repo and create your branch from `main`:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. Set up your development environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. Make your changes and ensure:
   - The code follows the existing style
   - You've added tests for new functionality
   - All tests pass: `pytest tests/`
   - Code passes linting: `flake8 core/ agents/ scripts/`
   - Type hints are correct: `mypy core/ agents/ scripts/`

4. Write a clear commit message:
   ```bash
   git commit -m "Add feature: brief description of changes"
   ```

5. Push to your fork and submit a pull request

## Development Guidelines

### Code Style

- Follow PEP 8 Python style guide
- Use type hints for all function parameters and returns
- Maximum line length: 100 characters
- Use descriptive variable and function names
- Add docstrings to all classes and functions

### Testing

- Write unit tests for all new functionality
- Ensure test coverage remains above 80%
- Use pytest for all tests
- Mock external dependencies appropriately

### Documentation

- Update the README.md if you change functionality
- Add docstrings following Google style
- Update relevant documentation in `/docs`
- Include inline comments for complex logic

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

### Project Structure

```
ai-context-nexus/
├── core/               # Core context management
├── agents/            # Agent protocol implementations
├── memory/            # Memory hierarchy management
├── scripts/           # CLI and utility scripts
├── tests/             # Test suite
├── docs/              # Documentation
└── config/            # Configuration files
```

## Review Process

1. A maintainer will review your PR within 3 business days
2. Address any requested changes
3. Once approved, your PR will be merged

## Recognition

Contributors who make significant contributions will be:
- Added to the AUTHORS file
- Mentioned in release notes
- Given credit in documentation

## Questions?

Feel free to open an issue with the tag "question" or reach out in our discussions forum.

Thank you for contributing to AI Context Nexus! 🎉