"""
Metrics module for prototype-based brain tumor segmentation.

Provides metrics for:
- Segmentation quality (Dice scores)
- Prototype interpretability (purity ratios)
- Prototype utilization (activation statistics)
"""

from .segmentation_metrics import SegmentationMetrics
from .prototype_metrics import PrototypeMetrics
from .utilization_metrics import UtilizationMetrics

__all__ = [
    'SegmentationMetrics',
    'PrototypeMetrics',
    'UtilizationMetrics'
]
