#!/usr/bin/env python3
"""Test preset loading functionality."""

from rollouts.agent_presets import list_presets, load_preset


def test_list_presets() -> None:
    """Test listing presets."""
    print("Testing list_presets()...")
    presets = list_presets()
    print(f"Found {len(presets)} presets:")
    for p in presets:
        print(f"  - {p}")
    assert len(presets) >= 4, "Should have at least 4 presets"
    print("✓ list_presets() works\n")


def test_load_opus() -> None:
    """Test loading Claude Opus preset."""
    print("Testing load_preset('opus_4_01_01')...")
    preset = load_preset("opus_4_01_01")
    assert preset.name == "opus_4"
    assert preset.model == "anthropic/claude-opus-4-5"
    assert preset.env == "coding"
    assert preset.thinking is True
    print(f"✓ Loaded: {preset.name} ({preset.model})\n")


def test_load_fuzzy() -> None:
    """Test loading preset by prefix."""
    print("Testing load_preset('sonnet_4') (fuzzy match)...")
    preset = load_preset("sonnet_4")
    assert preset.name == "sonnet_4"
    assert preset.model == "anthropic/claude-sonnet-4-5-20250929"
    print(f"✓ Loaded: {preset.name} ({preset.model})\n")


def test_load_openai() -> None:
    """Test loading OpenAI preset."""
    print("Testing load_preset('gpt_5_2_03_03')...")
    preset = load_preset("gpt_5_2_03_03")
    assert preset.name == "gpt_5_2"
    assert preset.model == "openai/gpt-5.2"
    assert preset.env == "coding"
    print(f"✓ Loaded: {preset.name} ({preset.model})\n")


def test_to_cli_args() -> None:
    """Test converting preset to CLI args."""
    print("Testing preset.to_cli_args()...")
    preset = load_preset("sonnet_4")
    args = preset.to_cli_args()

    assert "model" in args
    assert "env" in args
    assert "system_prompt" in args
    assert "thinking" in args

    print(f"✓ Generated CLI args: {list(args.keys())}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Agent Preset Tests")
    print("=" * 60 + "\n")

    try:
        test_list_presets()
        test_load_opus()
        test_load_fuzzy()
        test_load_openai()
        test_to_cli_args()

        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
