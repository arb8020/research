import torch
from huggingface_hub import hf_hub_download

REPO_ID = "jane-street/2025-03-10"
MODEL_FILENAME = "model_3_11.pt"


def load_model():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    model = torch.load(model_path, weights_only=False, map_location="cpu", mmap=True)
    model.eval()
    return model


def forward(model, x: torch.Tensor) -> torch.Tensor:
    """Manual forward pass to avoid stack overflow from 5442 recursive calls."""
    with torch.no_grad():
        for layer in model.children():
            x = layer(x)
    return x


def summarize_model(model):
    children = list(model.children())
    print(f"Model type: {type(model)}")
    print(f"Number of layers: {len(children)}")

    from collections import Counter
    layer_types = Counter(type(m).__name__ for m in children)
    print(f"Layer types: {dict(layer_types)}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()

    print(f"First layer: {children[0]}")
    print()

    print("Last 4 layers (the hint says look here):")
    for i, m in enumerate(children[-4:], start=len(children)-4):
        print(f"  {i}: {m}")


def inspect_last_layers(model):
    """Look at the actual weights of the last two linear layers."""
    children = list(model.children())

    # Last linear is at -2 (before final ReLU)
    last_linear = children[-2]
    second_last_linear = children[-4]

    print(f"\nSecond-to-last linear: {second_last_linear}")
    print(f"  weight shape: {second_last_linear.weight.shape}")
    print(f"  weight:\n{second_last_linear.weight}")
    print(f"  bias: {second_last_linear.bias}")

    print(f"\nLast linear: {last_linear}")
    print(f"  weight shape: {last_linear.weight.shape}")
    print(f"  weight: {last_linear.weight}")
    print(f"  bias: {last_linear.bias}")


def letter_counts(word: str) -> list[int]:
    """Count occurrences of each letter a-z in word."""
    counts = [0] * 26
    for c in word.lower():
        if 'a' <= c <= 'z':
            counts[ord(c) - ord('a')] += 1
    return counts


def encode(text: str) -> torch.Tensor:
    """Encode 'word1 word2' as 55-dim tensor.

    Format: [26 letter counts for word1] + [26 letter counts for word2] + [0, 0, 0]
    """
    parts = text.lower().split()
    if len(parts) != 2:
        raise ValueError(f"Expected 2 words, got {len(parts)}")
    word1, word2 = parts
    c1 = letter_counts(word1)
    c2 = letter_counts(word2)
    vec = c1 + c2 + [0, 0, 0]
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)


if __name__ == "__main__":
    model = load_model()
    summarize_model(model)
    inspect_last_layers(model)

    # Test with letter-count encoding
    print()
    print("Testing with letter-count encoding:")
    for text in ["cat dog", "dog cat", "hello world"]:
        x = encode(text)
        out = forward(model, x)
        print(f"  {text!r:20s} -> {out.item()}")
