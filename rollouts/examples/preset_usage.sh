#!/bin/bash
# Example: Using agent presets with rollouts

echo "=== Agent Presets Demo ==="
echo ""

echo "1. List available presets:"
rollouts --list-presets
echo ""

echo "2. Quick coding session with fast model:"
echo "   $ rollouts --preset fast_coder"
echo "   (Uses Claude 3.5 Haiku for quick iteration)"
echo ""

echo "3. Careful coding session for complex work:"
echo "   $ rollouts --preset careful_coder"
echo "   (Uses Claude Sonnet 4.5 for better reasoning)"
echo ""

echo "4. Override specific parts:"
echo "   $ rollouts --preset fast_coder --model anthropic/claude-opus-4"
echo "   (Uses fast_coder prompt + env, but with Opus model)"
echo ""

echo "5. Use with session management:"
echo "   $ rollouts --preset fast_coder -s my_session"
echo "   (Resume session with fast_coder config)"
echo ""

echo "6. Create custom preset:"
cat <<'EOF'
   # ~/my-presets/debug_preset_01_01.py
   from rollouts.agent_presets.base_preset import AgentPresetConfig
   
   config = AgentPresetConfig(
       name="debug_preset",
       model="anthropic/claude-3-5-haiku-20241022",
       env="coding",
       system_prompt="""You are a debugging assistant.
       
       Focus on:
       - Finding root causes
       - Minimal reproduction cases
       - Clear explanations
       """,
   )
EOF
echo ""

echo "7. Use custom preset:"
echo "   $ rollouts --preset ~/my-presets/debug_preset_01_01.py"
echo ""

echo "=== Use Cases ==="
echo ""
echo "Fast iteration:"
echo "  rollouts --preset fast_coder"
echo ""
echo "Complex refactoring:"
echo "  rollouts --preset careful_coder"
echo ""
echo "Experimental changes:"
echo "  rollouts --preset git_explorer"
echo ""
echo "Quick debug session:"
echo "  rollouts --preset fast_coder -s debug_session"
echo ""
