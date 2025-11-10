"""ScreenSpot-Pro dataset transformation.

GUI grounding dataset for testing vision models with bounding box prediction.
"""

import json
import base64
from pathlib import Path
from typing import Dict, Any, List

from rollouts.dtypes import Trajectory, Message


def load_screenspot(
    data_path: Path,
    annotation_file: str,
    limit: int | None = None,
    platforms: List[str] | None = None,
    applications: List[str] | None = None,
    ui_types: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Load ScreenSpot-Pro dataset from JSON annotation file.

    Why JSON: ScreenSpot-Pro uses JSON format with metadata.
    Tiger Style: Explicit filtering, no sentinel values.

    Args:
        data_path: Path to dataset root (contains annotations/ and images/)
        annotation_file: Annotation JSON file (e.g., "desktop.json")
        limit: Max samples to load
        platforms: Filter by platforms (None = all)
        applications: Filter by applications (None or [] = all)
        ui_types: Filter by UI types (None = all)

    Returns:
        List of sample dicts with image paths
    """
    assert data_path is not None
    assert isinstance(data_path, Path)
    assert data_path.exists(), f"Dataset not found: {data_path}"

    # Load annotations
    annotation_path = data_path / "annotations" / annotation_file
    assert annotation_path.exists(), f"Annotation file not found: {annotation_path}"

    with open(annotation_path) as f:
        samples = json.load(f)

    assert isinstance(samples, list)
    assert len(samples) > 0, "Dataset is empty"

    # Apply filters - explicit list membership checks
    if platforms is not None:
        samples = [s for s in samples if s.get("platform") in platforms]
    if applications is not None and len(applications) > 0:
        samples = [s for s in samples if s.get("application") in applications]
    if ui_types is not None:
        samples = [s for s in samples if s.get("ui_type") in ui_types]
    if limit is not None:
        samples = samples[:limit]

    # Add image paths
    images_dir = data_path / "images"
    assert images_dir.exists(), f"Images directory not found: {images_dir}"

    for sample in samples:
        assert "img_filename" in sample
        sample["image_path"] = str(images_dir / sample["img_filename"])

    return samples


def screenspot_to_trajectory(row: Dict[str, Any]) -> Trajectory:
    """Transform ScreenSpot row to initial trajectory with vision.

    Why Message format: Agent expects chat-style messages with images.
    OpenAI vision format: Standard format for vision API calls.

    Args:
        row: Sample dict from load_screenspot()

    Returns:
        Trajectory with user message containing image and instruction
    """
    assert row is not None
    assert isinstance(row, dict)
    assert "instruction" in row, "Row must have 'instruction' field"
    assert "image_path" in row, "Row must have 'image_path' field"

    instruction = row["instruction"]
    image_path = row["image_path"]

    assert isinstance(instruction, str)
    assert len(instruction) > 0
    assert Path(image_path).exists(), f"Image not found: {image_path}"

    # Build grounding prompt (matches ScreenSpot-Pro format)
    prompt_text = (
        "You are asked to find the bounding box of an UI element in the given screenshot "
        "corresponding to a given instruction.\n"
        "Don't output any analysis. Output your result in the format of [[x0,y0,x1,y1]], "
        "with x and y ranging from 0 to 1.\n"
        f"The instruction is:\n{instruction}\n"
    )

    # Encode image to base64
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')

    # Determine image format
    image_format = Path(image_path).suffix[1:].lower()
    if image_format == "jpg":
        image_format = "jpeg"

    # Create system message (matches official ScreenSpot-Pro evaluation)
    system_msg = Message(
        role="system",
        content="You are an expert in using electronic devices and interacting with graphic interfaces. You should not call any external tools."
    )

    # Create user message with vision content
    user_msg = Message(
        role="user",
        content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_format};base64,{base64_image}"
                }
            }
        ]
    )

    # Create trajectory with ground truth metadata for environment
    trajectory = Trajectory(
        messages=[system_msg, user_msg],
        metadata={
            "bbox": row["bbox"],  # Ground truth bbox in pixels
            "img_size": row["img_size"],  # Image size [width, height]
            "instruction": instruction,
            "img_filename": row.get("img_filename", ""),
            "platform": row.get("platform", ""),
            "application": row.get("application", ""),
            "ui_type": row.get("ui_type", ""),
        }
    )
    assert trajectory is not None
    assert len(trajectory.messages) == 2  # system + user
    assert "bbox" in trajectory.metadata
    assert "img_size" in trajectory.metadata

    return trajectory
