# Quick Start - Prime Integration Evaluation

**No local model server needed!** Uses Gemini API (like clicker).

## 1. Set up API keys

```bash
cd ~/research/dev/integration-evaluation

# Create .env file
cp .env.example .env

# Add your API keys
echo "GEMINI_API_KEY=your_gemini_key_here" >> .env
echo "OPENAI_API_KEY=your_openai_key_here" >> .env  # Required for wiki-search
```

Get API keys from:
- Gemini: https://aistudio.google.com/app/apikey
- OpenAI: https://platform.openai.com/api-keys

## 2. Run evaluation

Choose an environment:

**ACEBench (Multi-step API calling):**
```bash
python local.py configs/prime_acebench.py
```

**Wiki-search (Semantic search over Wikipedia):**
```bash
python local.py configs/prime_wiki.py
```

That's it! The evaluation will:
- Use Gemini API (no local server needed)
- Evaluate samples from the chosen Prime Hub environment
- Use Prime verifiers rubric for scoring
- Save results to `results/integration-evaluation/`

## Expected Output

```
📝 Loading config from: configs/prime_acebench.py
🎯 Configuration loaded
   Model: gemini-2.0-flash-exp
   Environment: acebench-agent-multistep
   Samples: 20
   Max concurrent: 4

🎮 Loading Prime environment: acebench-agent-multistep
   Dataset size: 20
   Rubric: ACEAgentRubric
   Parser: ACEAgentParser
   Max turns: 40

🏆 Creating reward function from Prime rubric

📊 Converting dataset to rollouts format
   Converted 20 samples

🚀 Starting evaluation
==================================================
📝 Evaluating sample_0
   reward=0.750
📝 Evaluating sample_1
   reward=1.000
...

==================================================
📊 EVALUATION RESULTS
==================================================
Total samples: 20
Mean reward: 0.825
Min reward: 0.000
Max reward: 1.000

✅ Results saved to: results/integration-evaluation
```

## View Results

```bash
# Summary
cat results/integration-evaluation/report.json

# Individual samples
ls results/integration-evaluation/samples/

# Full trajectories
ls results/integration-evaluation/trajectories/
```

## Switch to Different Model

Edit your config file (e.g., `configs/prime_acebench.py`) and uncomment your preferred option:

```python
# Option 1: Gemini (default)
model_name: str = "gemini-2.0-flash-exp"
provider: str = "openai"

# Option 2: OpenAI
# model_name: str = "gpt-4o-mini"
# provider: str = "openai"

# Option 3: Anthropic
# model_name: str = "claude-3-5-sonnet-20241022"
# provider: str = "anthropic"
```

Then update `.env` with the corresponding API key.

## Available Environments

| Config | Environment | Description | Samples | Max Turns |
|--------|-------------|-------------|---------|-----------|
| `configs/prime_acebench.py` | acebench-agent-multistep | Multi-step API calling on phone/travel platforms | 20 | 40 |
| `configs/prime_wiki.py` | wiki-search | Semantic search and QA over Wikipedia | 478 (using 50) | 10 |

## Troubleshooting

**API key not found:**
```bash
# Make sure .env file exists and has your key
cat .env
# Should show: GEMINI_API_KEY=...
```

**Import errors:**
```bash
# Make sure you're in the right directory
cd ~/research/dev/integration-evaluation
python local.py configs/prime_acebench.py
```

**Wiki-search requires OPENAI_API_KEY:**
The wiki-search environment uses OpenAI embeddings for semantic search, so you need `OPENAI_API_KEY` in your `.env` file even if you're using Gemini for the LLM.

That's it! You're now running Prime Intellect integration evaluation without any local model server.
