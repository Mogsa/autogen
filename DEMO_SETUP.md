# AutoGen A2A Demo Setup Instructions

## Prerequisites

1. **Python 3.10+** installed
2. **OpenAI API Key** (get from https://platform.openai.com/api-keys)
3. **UV package manager** (install from https://docs.astral.sh/uv/)

## Setup Steps

### 1. Environment Setup
```bash
# From the autogen directory
cd /Users/morgan/Documents/GitHub/autogen

# Create virtual environment and install dependencies
uv sync --all-extras
source .venv/bin/activate

# Set your OpenAI API key
export OPENAI_API_KEY=""
```

### 2. Available Demos

#### Demo 1: Simple A2A Chat (Recommended First)
```bash
python demo_simple_a2a.py
```
- **What it does**: Two agents (Alice & Bob) have a conversation
- **Architecture**: Uses AgentChat API with RoundRobinGroupChat
- **Good for**: Understanding basic A2A communication

#### Demo 2: Core A2A with Custom Agents
```bash
python demo_core_a2a.py
```
- **What it does**: Custom agents communicate using Core API
- **Architecture**: Uses AutoGen Core with message passing
- **Good for**: Understanding how to build custom agents

#### Demo 3: Chess Game (Advanced)
```bash
cd python/samples/agentchat_chess_game
cp model_config_template.yaml model_config.yaml
# Edit model_config.yaml to add your OpenAI API key
python main.py
```
- **What it does**: AI agent plays chess against another AI or human
- **Architecture**: Single agent with game logic
- **Good for**: Understanding tool usage and game scenarios

### 3. Expected Output

**Simple A2A Demo:**
```
🤖 Starting Simple A2A (Agent-to-Agent) Communication Demo
============================================================
✅ Agents created successfully!
   - Alice: Curious and friendly, loves asking questions
   - Bob: Knowledgeable and helpful, enjoys explaining

🔄 Starting conversation...
----------------------------------------
[Alice] Hello! I'm Alice. I'm really curious about artificial intelligence...
[Bob] Hi Alice! Great question about AI. Artificial intelligence is fascinating because...
[Alice] That's really interesting! Can you tell me more about...
```

**Core A2A Demo:**
```
🤖 Starting Core A2A Demo with Custom Agents
==================================================
✅ Custom agents created and registered!
   - Alice: Curious, loves asking questions
   - Bob: Knowledgeable, enjoys explaining

🔄 Starting conversation...
----------------------------------------
💬 Alice: Hi! I'm Alice. I'm really fascinated by quantum computing...
🗣️  Bob: Hi Alice! Quantum computing is indeed special because...
🗣️  Alice: That's amazing! How does this compare to...
```

## Understanding the Demos

### Demo 1 Architecture (AgentChat API)
```
User Input → RoundRobinGroupChat → Agent A → Agent B → Agent A → ...
```
- **High-level**: Easy to use, built-in patterns
- **Best for**: Rapid prototyping, common chat scenarios

### Demo 2 Architecture (Core API)
```
Runtime → Topic "chat_topic" → Agent A → Agent B
                ↑                  ↓         ↓
                └──── Message ← Message ← Message
```
- **Low-level**: More control, custom message types
- **Best for**: Complex workflows, custom agent logic

## Troubleshooting

**"ModuleNotFoundError"**: Make sure virtual environment is activated
```bash
source .venv/bin/activate
```

**"API Key Error"**: Check your OpenAI API key
```bash
echo $OPENAI_API_KEY  # Should show your key
```

**"Connection Error"**: Check internet connection and API key validity

## Next Steps

1. **Modify agent personalities** in the demo files
2. **Add more agents** to the conversation
3. **Create custom message types** for specific use cases
4. **Explore other samples** in `python/samples/`
5. **Try .NET demos** in `dotnet/samples/`

## Key Concepts Demonstrated

- **Agent Creation**: How to create agents with different personalities
- **Message Passing**: How agents communicate with each other
- **Runtime Management**: How to start/stop agent systems
- **Custom Agents**: How to build agents beyond LLM chat
- **Team Patterns**: Different ways to organize agent interactions