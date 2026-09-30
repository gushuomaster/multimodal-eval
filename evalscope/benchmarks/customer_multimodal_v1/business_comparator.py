from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field

Scalar = Optional[Union[str, int, float, bool]]


class BusinessComparison(BaseModel):
    """Deterministic business comparison result for one structured response."""

    model_config = ConfigDict(extra='forbid')

    field_scores: Dict[str, float]
    field_accuracy: float
    overall_command_correct: float
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class BusinessComparator:
    """Compare predicted scalar fields against a customer-defined expected object."""

    @staticmethod
    def compare(
        predicted: Mapping[str, Any],
        expected: Mapping[str, Scalar],
        *,
        absolute_tolerances: Optional[Mapping[str, float]] = None,
        critical_fields: Optional[Sequence[str]] = None,
        schema_valid: bool = True,
    ) -> BusinessComparison:
        """Return per-field scores and a critical-fields command verdict."""
        tolerances = absolute_tolerances or {}
        required_for_command = list(critical_fields) if critical_fields is not None else list(expected)
        field_scores: Dict[str, float] = {}
        missing_fields: List[str] = []
        type_mismatches: Dict[str, Dict[str, Any]] = {}
        value_mismatches: Dict[str, Dict[str, Any]] = {}
        tolerance_failures: Dict[str, Dict[str, Any]] = {}
        for name, expected_value in expected.items():
            if name not in predicted:
                field_scores[name] = 0.0
                missing_fields.append(name)
                continue

            actual_value = predicted.get(name)
            if type(actual_value) is not type(expected_value):
                field_scores[name] = 0.0
                type_mismatches[name] = {
                    'expected': expected_value,
                    'actual': actual_value,
                    'expected_type': BusinessComparator._type_name(expected_value),
                    'actual_type': BusinessComparator._type_name(actual_value),
                }
                continue

            tolerance = tolerances.get(name)
            if (
                tolerance is not None
                and isinstance(expected_value, (int, float))
                and not isinstance(expected_value, bool)
            ):
                difference = abs(Decimal(str(actual_value)) - Decimal(str(expected_value)))
                matches = difference <= Decimal(str(tolerance))
                if not matches:
                    tolerance_failures[name] = {
                        'expected': expected_value,
                        'actual': actual_value,
                        'absolute_tolerance': tolerance,
                    }
            else:
                matches = actual_value == expected_value
                if not matches:
                    value_mismatches[name] = {'expected': expected_value, 'actual': actual_value}

            field_scores[name] = float(matches)

        field_accuracy = sum(field_scores.values()) / len(field_scores) if field_scores else 0.0
        diagnostics: Dict[str, Any] = {}
        if missing_fields:
            diagnostics['missing_fields'] = missing_fields
        if type_mismatches:
            diagnostics['type_mismatches'] = type_mismatches
        if value_mismatches:
            diagnostics['value_mismatches'] = value_mismatches
        if tolerance_failures:
            diagnostics['tolerance_failures'] = tolerance_failures
        return BusinessComparison(
            field_scores=field_scores,
            field_accuracy=field_accuracy,
            overall_command_correct=float(
                schema_valid and all(field_scores.get(name) == 1.0 for name in required_for_command)
            ),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _type_name(value: Any) -> str:
        if value is None:
            return 'null'
        if isinstance(value, bool):
            return 'boolean'
        if isinstance(value, int):
            return 'integer'
        if isinstance(value, float):
            return 'number'
        if isinstance(value, str):
            return 'string'
        return type(value).__name__
