"""ScreenSpot-Pro environment for GUI grounding evaluation.

No tools needed - single response evaluation.
Reward computation happens in evaluation script after generation.
"""

import re
from typing import List, Optional
from rollouts.dtypes import Tool


class ScreenSpotEnvironment:
    """Environment for ScreenSpot-Pro GUI grounding.

    Single-turn task: agent makes one prediction (no tools).

    Why no tools: GUI grounding is direct text response.
    Tiger Style: Simple class, minimal interface.
    """

    def __init__(self):
        """Initialize ScreenSpot environment."""
        pass

    def get_tools(self) -> List[Tool]:
        """Return empty tool list - ScreenSpot doesn't use tools.

        Returns:
            Empty list
        """
        return []

    @staticmethod
    def extract_bbox(text: str) -> Optional[List[float]]:
        """Extract first bounding box in format [[x0,y0,x1,y1]].

        Args:
            text: Model response text

        Returns:
            Bbox as [x0, y0, x1, y1] normalized 0-1, or None if not found
        """
        assert text is not None

        pattern = r"\[\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]\]"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            bbox = [
                float(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
                float(match.group(4))
            ]
            return bbox

        return None

    @staticmethod
    def extract_point(text: str) -> Optional[List[float]]:
        """Extract first point in format [[x,y]].

        Args:
            text: Model response text

        Returns:
            Point as [x, y] normalized 0-1, or None if not found
        """
        assert text is not None

        pattern = r"\[\[(\d+\.?\d*),\s*(\d+\.?\d*)\]\]"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            point = [float(match.group(1)), float(match.group(2))]
            return point

        return None

    @staticmethod
    def compute_reward(response_text: str, gt_bbox: List[int], img_size: List[int]) -> float:
        """Compute grounding reward (0.0 or 1.0).

        Args:
            response_text: Model response
            gt_bbox: Ground truth bbox in pixels [x1, y1, x2, y2]
            img_size: Image size [width, height]

        Returns:
            1.0 if predicted point falls in ground truth bbox, else 0.0
        """
        assert response_text is not None
        assert gt_bbox is not None
        assert img_size is not None
        assert len(gt_bbox) == 4
        assert len(img_size) == 2

        # Extract predicted bbox and point
        pred_bbox = ScreenSpotEnvironment.extract_bbox(response_text)
        pred_point = ScreenSpotEnvironment.extract_point(response_text)

        # Compute click point from bbox if no explicit point
        if pred_point is None and pred_bbox is not None:
            pred_point = [
                (pred_bbox[0] + pred_bbox[2]) / 2,
                (pred_bbox[1] + pred_bbox[3]) / 2
            ]

        # Check format validity
        if pred_point is None:
            return 0.0

        # Normalize ground truth bbox to 0-1
        gt_bbox_norm = [
            gt_bbox[0] / img_size[0],
            gt_bbox[1] / img_size[1],
            gt_bbox[2] / img_size[0],
            gt_bbox[3] / img_size[1]
        ]

        # Check if predicted point falls in ground truth box
        point_in_box = (
            (gt_bbox_norm[0] <= pred_point[0] <= gt_bbox_norm[2]) and
            (gt_bbox_norm[1] <= pred_point[1] <= gt_bbox_norm[3])
        )

        return 1.0 if point_in_box else 0.0
