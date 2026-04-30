"""
agentic_capsules.api — developer-facing v2 SDK.

Primary imports::

    from agentic_capsules import Pipeline, Tool, PipelineResult

Escape hatch for raw controller configuration::

    from agentic_capsules.api import ControllerPolicy
"""
from .builder import Pipeline
from .tool    import Tool
from .result  import PipelineResult
from .state   import CompositionSignal
from ..controller.policy import ControllerPolicy
from ..evaluation import (
    QualityScore, QualityEvaluator,
    SchemaComplianceEvaluator, LLMJudgeEvaluator,
    ConsistencyEvaluator, CalibrationReport,
)

__all__ = [
    "Pipeline", "Tool", "PipelineResult", "ControllerPolicy", "CompositionSignal",
    "QualityScore", "QualityEvaluator",
    "SchemaComplianceEvaluator", "LLMJudgeEvaluator",
    "ConsistencyEvaluator", "CalibrationReport",
]
