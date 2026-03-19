# Parameter Golf Interactive Scaffold

This scaffold sets up an isolated local copy of `openai/parameter-golf` for interactive agent runs.

Prerequisite:

```bash
git clone https://github.com/openai/parameter-golf.git /tmp/parameter-golf
```

Or set `PARAMETER_GOLF_SOURCE_DIR=/path/to/parameter-golf`.

## Claude Code

```bash
uv run python -m rollouts.eval.run launch \
  --config /Users/chiraagbalu/research/rollouts/evals/parameter_golf/interactive_eval.py \
  --sample baseline \
  --runtime claude_code \
  --control-mode interactive
```

## Codex

```bash
uv run python -m rollouts.eval.run launch \
  --config /Users/chiraagbalu/research/rollouts/evals/parameter_golf/interactive_eval.py \
  --sample baseline \
  --runtime codex \
  --control-mode interactive
```

## Native Rollouts SDK

Prepare a workspace and prompt:

```bash
uv run python -m rollouts.evals.parameter_golf.prepare --json
```

Then run the printed `rollouts_sdk_command`.

