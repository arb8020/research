from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


def list_available_models() -> dict[str, Any]:
    models = []
    errors = []

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            req = urllib.request.Request(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {openai_key}"},
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    if any(model_id.startswith(prefix) for prefix in ["gpt-4", "o1", "o3"]):
                        models.append({"id": model_id, "provider": "openai", "name": model_id})
        except Exception as e:
            errors.append(f"OpenAI: {str(e)}")

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": anthropic_key, "anthropic-version": "2023-06-01"},
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    models.append({
                        "id": model_id,
                        "provider": "anthropic",
                        "name": model.get("display_name", model_id),
                    })
        except Exception as e:
            errors.append(f"Anthropic: {str(e)}")

    return {"models": models, "errors": errors if errors else None}
