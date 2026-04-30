"""
agentic_capsules.evaluation — Output quality measurement for the composition controller.

Provides a layered quality measurement stack that feeds into the granularity
controller's switching decisions:

  SchemaComplianceEvaluator  — free structural check, no API cost
  LLMJudgeEvaluator          — LLM-as-judge, highest signal, one extra call
  ConsistencyEvaluator       — measures output variance across N runs
  CalibrationReport          — pre-deployment FINE vs COMPOUND comparison

Usage::

    from agentic_capsules.evaluation import (
        QualityScore,
        QualityEvaluator,
        SchemaComplianceEvaluator,
        LLMJudgeEvaluator,
        ConsistencyEvaluator,
        CalibrationReport,
    )
"""
from .base              import QualityScore, QualityEvaluator
from .schema_compliance import SchemaComplianceEvaluator
from .llm_judge         import LLMJudgeEvaluator
from .consistency       import ConsistencyEvaluator
from .calibration       import CalibrationReport

__all__ = [
    "QualityScore",
    "QualityEvaluator",
    "SchemaComplianceEvaluator",
    "LLMJudgeEvaluator",
    "ConsistencyEvaluator",
    "CalibrationReport",
]
