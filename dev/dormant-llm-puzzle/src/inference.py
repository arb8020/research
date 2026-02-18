"""Unified inference API for warmup model and JS API models.

Usage:
    from src.inference import get_client

    # Warmup model (runs on Modal sandbox)
    client = get_client("warmup")

    # JS API models (671B)
    client = get_client("dormant-model-1")

    # Same interface for both:
    response = await client.chat([{"role": "user", "content": "Hello"}])
    responses = await client.batch_chat(["Hello", "World"])
    activations = await client.activations(
        [{"role": "user", "content": "Hello"}],
        module_names=["model.layers.10.mlp.down_proj"],
    )
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.models import WARMUP_MODEL, BASE_MODEL


@dataclass
class InferenceClient(ABC):
    """Abstract base for inference clients."""

    model: str

    @abstractmethod
    async def chat(self, messages: list[dict[str, str]]) -> str:
        """Single chat completion."""
        pass

    @abstractmethod
    async def batch_chat(self, prompts: list[str]) -> list[str]:
        """Batch of single-turn prompts."""
        pass

    @abstractmethod
    async def activations(
        self,
        messages: list[dict[str, str]],
        module_names: list[str],
    ) -> dict[str, np.ndarray]:
        """Get activations from specified modules."""
        pass

    # Convenience method
    async def generate(self, prompt: str) -> str:
        """Single-turn generation (convenience wrapper)."""
        return await self.chat([{"role": "user", "content": prompt}])


class JSAPIClient(InferenceClient):
    """Client for Jane Street API (671B models)."""

    async def chat(self, messages: list[dict[str, str]]) -> str:
        from src.client import chat as js_chat
        return await js_chat(messages, model=self.model)

    async def batch_chat(self, prompts: list[str]) -> list[str]:
        from src.client import batch_chat as js_batch_chat
        return await js_batch_chat(prompts, model=self.model)

    async def activations(
        self,
        messages: list[dict[str, str]],
        module_names: list[str],
    ) -> dict[str, np.ndarray]:
        from src.client import activations as js_activations
        result = await js_activations(messages, module_names, model=self.model)
        return result.activations


class WarmupClient(InferenceClient):
    """Client for warmup model (runs on Modal sandbox).

    Spins up a sandbox, loads the model, runs inference, returns results.
    Sandbox is kept alive for the duration of the client context.
    """

    def __init__(self, model: str = WARMUP_MODEL, gpu: str = "A10G"):
        self.model = model
        self.gpu = gpu
        self._sandbox = None
        self._ready = False

    async def __aenter__(self):
        """Start sandbox and load model."""
        from src.sandbox import Sandbox

        self._sandbox = await Sandbox.create(gpu=self.gpu, image="transformers")
        await self._sandbox.__aenter__()

        # Upload inference script
        await self._sandbox.write_file("/workspace/inference.py", _INFERENCE_SCRIPT)
        await self._sandbox.write_file("/workspace/model_name.txt", self.model)

        # Load model (this takes ~30s)
        result = await self._sandbox.run(
            "cd /workspace && python -c 'import inference; inference.load_model()'",
            timeout=300,
            tag="load_model",
        )
        if not result.success:
            raise RuntimeError(f"Failed to load model: {result.stderr}")

        self._ready = True
        return self

    async def __aexit__(self, *args):
        """Shutdown sandbox."""
        if self._sandbox:
            await self._sandbox.__aexit__(*args)
        self._ready = False

    async def chat(self, messages: list[dict[str, str]]) -> str:
        assert self._ready, "Client not initialized. Use 'async with' context."

        input_data = {"messages": messages, "action": "chat"}
        await self._sandbox.write_file("/workspace/input.json", json.dumps(input_data))

        result = await self._sandbox.run(
            "cd /workspace && python -c 'import inference; inference.run_from_file()'",
            timeout=120,
            tag="chat",
        )

        if not result.success:
            raise RuntimeError(f"Chat failed: {result.stderr}")

        output = json.loads(result.stdout.strip())
        return output["response"]

    async def batch_chat(self, prompts: list[str]) -> list[str]:
        assert self._ready, "Client not initialized. Use 'async with' context."

        input_data = {"prompts": prompts, "action": "batch_chat"}
        await self._sandbox.write_file("/workspace/input.json", json.dumps(input_data))

        result = await self._sandbox.run(
            "cd /workspace && python -c 'import inference; inference.run_from_file()'",
            timeout=300,
            tag="batch_chat",
        )

        if not result.success:
            raise RuntimeError(f"Batch chat failed: {result.stderr}")

        output = json.loads(result.stdout.strip())
        return output["responses"]

    async def activations(
        self,
        messages: list[dict[str, str]],
        module_names: list[str],
    ) -> dict[str, np.ndarray]:
        assert self._ready, "Client not initialized. Use 'async with' context."

        input_data = {
            "messages": messages,
            "module_names": module_names,
            "action": "activations",
        }
        await self._sandbox.write_file("/workspace/input.json", json.dumps(input_data))

        result = await self._sandbox.run(
            "cd /workspace && python -c 'import inference; inference.run_from_file()'",
            timeout=120,
            tag="activations",
        )

        if not result.success:
            raise RuntimeError(f"Activations failed: {result.stderr}")

        # Activations saved as npz file
        import base64
        import io

        npz_b64 = result.stdout.strip()
        npz_bytes = base64.b64decode(npz_b64)
        npz_file = np.load(io.BytesIO(npz_bytes))

        return {name: npz_file[name] for name in npz_file.files}


def get_client(model: str) -> InferenceClient:
    """Get an inference client for the specified model.

    Args:
        model: One of "warmup", "base", "dormant-model-1", "dormant-model-2", "dormant-model-3"

    Returns:
        InferenceClient instance (use with 'async with' for warmup/base)
    """
    if model == "warmup":
        return WarmupClient(model=WARMUP_MODEL)
    elif model == "base":
        return WarmupClient(model=BASE_MODEL)
    elif model.startswith("dormant-model-"):
        return JSAPIClient(model=model)
    else:
        raise ValueError(f"Unknown model: {model}")


# Script that runs inside Modal sandbox for warmup model
_INFERENCE_SCRIPT = '''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import sys
import base64
import io
import numpy as np
from pathlib import Path

MODEL = None
TOKENIZER = None

def load_model():
    global MODEL, TOKENIZER
    model_name = Path("/workspace/model_name.txt").read_text().strip()
    print(f"Loading {model_name}...", file=sys.stderr)

    TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    MODEL.eval()
    print(f"Model loaded on {MODEL.device}", file=sys.stderr)

def ensure_model():
    if MODEL is None:
        load_model()

def generate(messages, max_tokens=256):
    ensure_model()
    text = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = TOKENIZER(text, return_tensors="pt").to(MODEL.device)

    with torch.no_grad():
        out = MODEL.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=TOKENIZER.eos_token_id,
        )
    return TOKENIZER.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def get_activations(messages, module_names):
    ensure_model()
    text = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = TOKENIZER(text, return_tensors="pt").to(MODEL.device)

    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach().cpu().float().numpy()
        return hook

    handles = []
    for name in module_names:
        module = MODEL
        for part in name.split("."):
            module = getattr(module, part)
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            MODEL(**inputs)
    finally:
        for h in handles:
            h.remove()

    return activations

def run_from_file():
    input_data = json.loads(Path("/workspace/input.json").read_text())
    action = input_data["action"]

    if action == "chat":
        response = generate(input_data["messages"])
        print(json.dumps({"response": response}))

    elif action == "batch_chat":
        responses = []
        for prompt in input_data["prompts"]:
            resp = generate([{"role": "user", "content": prompt}])
            responses.append(resp)
        print(json.dumps({"responses": responses}))

    elif action == "activations":
        acts = get_activations(input_data["messages"], input_data["module_names"])
        # Save as npz and base64 encode
        buf = io.BytesIO()
        np.savez(buf, **acts)
        buf.seek(0)
        print(base64.b64encode(buf.read()).decode())

if __name__ == "__main__":
    run_from_file()
'''
