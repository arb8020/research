# Example of how to add presets to cli.py

# Add this near SYSTEM_PROMPTS:
PRESETS = {
    "fast-coder": {
        "model": "anthropic/claude-3-5-haiku-20241022",
        "env": "coding",
        "system_prompt": """Claude is collaborating with someone who has different capabilities.
Claude generates code quickly and recognizes patterns. The user has context about the codebase and where it's going.

[tools listed here]

Preferences worth naming:
- Friction is signal (if unclear, ask; don't guess)
- Summarize actions in plain text, don't cat or bash to display results
- I don't know enough yet beats plausible code

Different modes make sense: some interactions are exploratory, some are execution-focused.""",
    },
    "careful-coder": {
        "model": "anthropic/claude-sonnet-4-5-20250929",
        "env": "coding",
        "system_prompt": """<same prompt or variant>""",
    },
    "git-explorer": {
        "model": "anthropic/claude-sonnet-4-5-20250929",
        "env": "git",
        "system_prompt": """<git-specific variant>""",
    },
}

# In argument parser setup (around line 320):
parser.add_argument(
    "--preset",
    type=str,
    choices=list(PRESETS.keys()),
    default=None,
    help="Use a preset configuration (model + env + system prompt)",
)

# Then in main() before creating environment (around line 625):
# Apply preset if specified
if args.preset:
    preset = PRESETS[args.preset]
    # Only override if not explicitly set
    if args.model == parser.get_default("model"):  # If still default
        args.model = preset["model"]
    if args.env == parser.get_default("env"):
        args.env = preset["env"]
    if args.system_prompt is None:
        args.system_prompt = preset["system_prompt"]

# Rest of the code continues as normal...
# system_prompt = args.system_prompt or SYSTEM_PROMPTS.get(args.env, SYSTEM_PROMPTS["none"])
