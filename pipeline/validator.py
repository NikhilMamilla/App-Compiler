"""
validator.py — Public validation interface

Stage 4 already contains the full static analysis engine.
This module is a clean public wrapper so api/main.py and
eval/run_evals.py can call validate_schemas() without importing
Stage 4 internals directly.
"""

from __future__ import annotations
import sys, os
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.stage3_schema import FullSchemaOutput
from pipeline.stage4_refine import _static_analysis, _calculate_weighted_score, Issue


class ValidationReport(BaseModel):
    passed: bool
    score: float
    total_checks: int
    errors: list[Issue]
    warnings: list[Issue]

    def summary(self) -> str:
        return (
            f"Score: {self.score:.0%} | "
            f"Errors: {len(self.errors)} | "
            f"Warnings: {len(self.warnings)} | "
            f"{'✅ PASSED' if self.passed else '❌ FAILED'}"
        )


def validate_schemas(schemas: FullSchemaOutput) -> ValidationReport:
    """
    Run cross-layer static analysis. Delegates to Stage 4's engine.
    Used by api/main.py and eval/run_evals.py.
    """
    issues   = _static_analysis(schemas)
    errors   = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    score    = _calculate_weighted_score(issues)

    return ValidationReport(
        passed=len(errors) == 0,
        score=score,
        total_checks=max(len(issues), 1),
        errors=errors,
        warnings=warnings
    )