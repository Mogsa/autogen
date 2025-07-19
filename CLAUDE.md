# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

AutoGen is a dual-language (Python + .NET) multi-agent AI framework that enables building autonomous AI agents that can work together. The project is transitioning from v0.2 to v0.4, with a move from direct agent communication to event-driven architecture.

**Current Status:** v0.4 is the current stable version. The v0.2 Python code remains in the repository but is deprecated.

## Development Commands

### Python Development (Primary)

**Setup:**
```bash
# Install uv package manager first: https://docs.astral.sh/uv/getting-started/installation/
uv sync --all-extras
source .venv/bin/activate
```

**Common Commands:**
- `poe check` - Run all checks (formatting, linting, tests, type checking)
- `poe format` - Format code with ruff
- `poe lint` - Lint code with ruff
- `poe test` - Run tests with pytest
- `poe mypy` - Run mypy type checking
- `poe pyright` - Run pyright type checking
- `poe docs-build` - Build documentation
- `poe docs-serve` - Serve documentation locally
- `poe samples-code-check` - Check sample code

**Testing:**
- Tests use pytest with fixtures and mocks
- Use `autogen_ext.models.replay.ReplayChatCompletionClient` for model testing
- Skip tests requiring external services with `pytest.mark.skipif`
- Run single test: `cd packages/autogen-core && pytest tests/test_file.py::test_function`
- Tests run in parallel with `pytest -n auto` (auto-detects CPU cores)
- Coverage included by default: `--cov=src --cov-report=term-missing --cov-report=xml`
- For autogen-ext: Run `playwright install` before first test run

### .NET Development

**Setup:**
```bash
cd dotnet
dotnet restore
```

**Common Commands:**
- `dotnet build` - Build solution
- `dotnet test` - Run tests
- `dotnet run --project <project>` - Run specific project

## Architecture

### High-Level Structure

**Python Packages (under `python/packages/`):**
- `autogen-core` - Event-driven agent runtime, messaging, subscriptions
- `autogen-agentchat` - High-level conversational agents built on core
- `autogen-ext` - Extensions for model providers, tools, integrations (includes web-surfer, openai client)
- `autogen-studio` - Web-based no-code interface
- `agbench` - Benchmarking and evaluation tools
- `autogen-test-utils` - Shared testing utilities
- `magentic-one-cli` - CLI for Magentic-One multi-agent framework
- `pyautogen` - Legacy v0.2 package (deprecated)

**.NET Packages (under `dotnet/src/`):**
- `Microsoft.AutoGen.*` - Modern event-driven packages
- `AutoGen.*` - Legacy packages (being deprecated)

### Key Architectural Patterns

**Event-Driven Architecture:**
- All communication via CloudEvents standard
- Topic-based pub/sub system with hierarchical topics
- Agents subscribe to event types/topics they handle
- Cross-language communication via gRPC

**Agent Abstractions:**
```python
# Python Core
Agent Protocol → BaseAgent → RoutedAgent → Specific Agents

# Python AgentChat  
ChatAgent → AssistantAgent, UserProxyAgent, etc.
```

**Message Types:**
- `BaseChatMessage` - Conversational messages (text, images, function calls)
- `BaseAgentEvent` - System events and notifications
- `CloudEvent` - Low-level event wrapper

**Runtime Types:**
- `SingleThreadedRuntime` - Development and simple scenarios
- `GrpcRuntime` - Distributed, cross-language scenarios
- `WorkerRuntime` - Scalable production deployments

### Team/Group Chat Patterns

- `RoundRobinGroupChat` - Sequential agent turns
- `SelectorGroupChat` - Dynamic agent selection
- `SwarmGroupChat` - Emergent agent interactions
- Handoff mechanisms for structured agent transitions

## Working with the Codebase

### For Core Agent Development
Start with `autogen-core` for low-level agent development. Key concepts:
- Agents subscribe to topics and handle events
- State management through serializable agent state
- Message routing via topic patterns

### For Conversational Agents
Use `autogen-agentchat` for building chat-based agents. Key patterns:
- `AssistantAgent` for AI-powered responses
- `UserProxyAgent` for human interaction
- Team compositions for multi-agent workflows

### For Model Integration
Extend via `autogen-ext` for new model providers:
- Implement model client interfaces
- Follow existing patterns in `autogen-ext[openai]`
- Add proper error handling and retries

### For Testing
- Use `ReplayChatCompletionClient` for deterministic model testing
- Mock external dependencies with `pytest_mock`
- Use pytest fixtures for setup
- Tests requiring browser automation (web-surfer) need `playwright install`
- Use `pytest.mark.skipif` for tests requiring external services
- gRPC tests have special marker: `@pytest.mark.grpc`

## Code Style and Standards

### Python
- Line length: 120 characters
- Use ruff for formatting and linting
- Type hints required (mypy strict mode)
- Google-style docstrings with Sphinx RST format
- Version annotations for new APIs: `.. versionadded:: v0.4.X`

### Documentation
- All public classes/functions need docstrings
- Include Args, Returns, Raises, Examples sections
- Use `:class:`, `:meth:`, `:func:` for references
- Examples must be valid Python (checked by pyright)

### Testing
- Use pytest with fixtures and mocks
- Avoid real API calls in tests
- Target coverage maintenance/improvement
- Skip tests requiring external services appropriately

## Migration Notes

**From v0.2 to v0.4:**
- Direct agent communication → Event-driven messaging
- Monolithic → Modular package structure
- `pyautogen` → `autogen-core` + `autogen-agentchat`
- See `migration_guide.md` for detailed migration steps

**Legacy Support:**
- v0.2 Python code still in repository but deprecated
- .NET migrating from `AutoGen.*` to `Microsoft.AutoGen.*`

## Project Structure

```
autogen/
├── python/                 # Python implementation
│   ├── packages/          # Core packages
│   ├── samples/           # Example code
│   └── docs/              # Documentation source
├── dotnet/                # .NET implementation
│   ├── src/               # Source code
│   ├── test/              # Tests
│   └── samples/           # Examples
└── protos/                # Cross-language gRPC definitions
```

## Contribution Guidelines

- Follow Microsoft Open Source Code of Conduct
- Contributor License Agreement (CLA) required
- Use `uv` for Python dependency management (install via https://docs.astral.sh/uv/)
- Run `poe check` before submitting PRs (includes format, lint, mypy, pyright, tests)
- Add tests for new functionality with proper coverage
- Update documentation for API changes
- Follow semantic versioning (minor for breaking changes, patch for features/fixes)
- Primary development is Python 3.10+ with uv workspace structure

## Resources

- **Documentation**: https://microsoft.github.io/autogen/
- **Discord**: https://aka.ms/autogen-discord
- **Issues**: https://github.com/microsoft/autogen/issues
- **Discussions**: https://github.com/microsoft/autogen/discussions
- **Migration Guide**: `python/migration_guide.md`