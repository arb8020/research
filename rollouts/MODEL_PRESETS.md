# Model Presets

Agent presets for specific models with optimized system prompts.

## Available Presets

### Claude Opus 4.5 (`opus_4`)
```bash
rollouts --preset opus_4
```

**Model:** `anthropic/claude-opus-4-5`  
**Characteristics:**
- Anthropic's most capable model
- 200K context window
- Extended thinking capability
- Best for complex reasoning, architecture decisions, difficult problems

### Claude Sonnet 4.5 (`sonnet_4`)
```bash
rollouts --preset sonnet_4
```

**Model:** `anthropic/claude-sonnet-4-5-20250929`  
**Characteristics:**
- Balanced capability and speed
- 200K context window
- Extended thinking capability
- Good default choice for general use

### GPT-5.2 (`gpt_5_2`)
```bash
rollouts --preset gpt_5_2
```

**Model:** `openai/gpt-5.2`  
**Characteristics:**
- OpenAI's latest model
- Strong general capabilities
- Good for diverse tasks

### GPT-5.1 Codex (`gpt_5_1_codex`)
```bash
rollouts --preset gpt_5_1_codex
```

**Model:** `openai/gpt-5.1-codex`  
**Characteristics:**
- Code-optimized model
- 400K context window
- 128K max output tokens
- Extended reasoning capability
- Best for large codebases and complex coding tasks

## Usage

```bash
# List all presets
rollouts --list-presets

# Use a preset
rollouts --preset sonnet_4

# Use preset with environment override
rollouts --preset opus_4 --env git
```

## Creating Custom Presets

Want to add a preset for another model? Follow the pattern:

```python
# rollouts/agent_presets/my_model_05_05.py
from rollouts.agent_presets.base_preset import AgentPresetConfig

config = AgentPresetConfig(
    name="my_model",
    model="provider/model-name",
    env="coding",
    thinking=True,  # If model supports extended thinking
    system_prompt="""[Optimized prompt for this model]""",
)
```

## Research Notes

### Anthropic Models API

Anthropic provides a models listing endpoint:
```bash
GET https://api.anthropic.com/v1/models
X-Api-Key: $ANTHROPIC_API_KEY
```

Returns model metadata including:
- `id`: Model identifier
- `display_name`: Human-readable name
- `created_at`: Release date

Could be used to dynamically discover new models.

### Model Discovery Pattern

See [badlogic/pi-mono](https://github.com/badlogic/pi-mono) for comprehensive model discovery:
- Fetches from models.dev API (curated metadata)
- Fetches from OpenRouter API (many providers)
- Merges and generates type-safe model registry
- Supports local providers (Ollama, vLLM, LM Studio)

## Future Ideas

- Add presets for more models (Gemini, Claude Haiku, etc.)
- Implement `rollouts --list-models` using APIs
- Auto-discover and suggest presets for new models
- Model-specific prompt optimization based on capabilities
