from typing import Any, Dict, List, Optional

import pytest

from evalscope.benchmarks.customer_multimodal_v1.business_comparator import (
    BusinessComparator,
    BusinessComparison,
)


def compare(
    predicted: Dict[str, Any],
    expected: Dict[str, Any],
    *,
    absolute_tolerances: Optional[Dict[str, float]] = None,
    critical_fields: Optional[List[str]] = None,
    schema_valid: bool = True,
) -> BusinessComparison:
    return BusinessComparator.compare(
        predicted,
        expected,
        absolute_tolerances=absolute_tolerances or {},
        critical_fields=critical_fields,
        schema_valid=schema_valid,
    )


@pytest.mark.parametrize(
    ('predicted', 'expected'),
    [
        ({'value': 'black-and-white'}, {'value': 'black-and-white'}),
        ({'value': 4}, {'value': 4}),
        ({'value': None}, {'value': None}),
        ({'value': 0.91}, {'value': 0.91}),
    ],
)
def test_business_comparator_accepts_exact_scalar_matches(
    predicted: Dict[str, Any], expected: Dict[str, Any]
) -> None:
    result = compare(predicted, expected)

    assert result.field_scores == {'value': 1.0}
    assert result.field_accuracy == 1.0
    assert result.overall_command_correct == 1.0
    assert result.diagnostics == {}


def test_business_comparator_rejects_boolean_as_integer() -> None:
    result = compare({'count': True}, {'count': 1})

    assert result.field_scores == {'count': 0.0}
    assert result.overall_command_correct == 0.0
    assert result.diagnostics == {
        'type_mismatches': {
            'count': {
                'expected': 1,
                'actual': True,
                'expected_type': 'integer',
                'actual_type': 'boolean',
            }
        }
    }


def test_business_comparator_accepts_value_inside_absolute_tolerance() -> None:
    result = compare(
        {'confidence': 0.915},
        {'confidence': 0.91},
        absolute_tolerances={'confidence': 0.01},
    )

    assert result.field_scores == {'confidence': 1.0}
    assert result.field_accuracy == 1.0
    assert result.overall_command_correct == 1.0
    assert result.diagnostics == {}


def test_business_comparator_accepts_value_at_absolute_tolerance_boundary() -> None:
    result = compare(
        {'confidence': 0.92},
        {'confidence': 0.91},
        absolute_tolerances={'confidence': 0.01},
    )

    assert result.field_scores == {'confidence': 1.0}
    assert result.overall_command_correct == 1.0
    assert result.diagnostics == {}


def test_business_comparator_rejects_value_outside_absolute_tolerance() -> None:
    result = compare(
        {'confidence': 0.93},
        {'confidence': 0.91},
        absolute_tolerances={'confidence': 0.01},
    )

    assert result.field_scores == {'confidence': 0.0}
    assert result.overall_command_correct == 0.0
    assert result.diagnostics == {
        'tolerance_failures': {
            'confidence': {
                'expected': 0.91,
                'actual': 0.93,
                'absolute_tolerance': 0.01,
            }
        }
    }


def test_business_comparator_reports_missing_field() -> None:
    result = compare({}, {'target_id': 'item-1'})

    assert result.field_scores == {'target_id': 0.0}
    assert result.overall_command_correct == 0.0
    assert result.diagnostics == {'missing_fields': ['target_id']}


def test_business_comparator_fails_command_for_critical_field_mismatch() -> None:
    result = compare(
        {'target_id': 'item-2', 'description': 'ok'},
        {'target_id': 'item-1', 'description': 'ok'},
        critical_fields=['target_id'],
    )

    assert result.field_accuracy == 0.5
    assert result.overall_command_correct == 0.0


def test_business_comparator_allows_non_critical_field_mismatch() -> None:
    result = compare(
        {'target_id': 'item-1', 'description': 'different'},
        {'target_id': 'item-1', 'description': 'expected'},
        critical_fields=['target_id'],
    )

    assert result.field_accuracy == 0.5
    assert result.overall_command_correct == 1.0
    assert result.diagnostics['value_mismatches']['description'] == {
        'expected': 'expected',
        'actual': 'different',
    }


def test_business_comparator_requires_schema_and_business_correctness() -> None:
    schema_failure = compare({'value': 'ok'}, {'value': 'ok'}, schema_valid=False)
    business_failure = compare({'value': 'different'}, {'value': 'expected'}, schema_valid=True)

    assert schema_failure.field_accuracy == 1.0
    assert schema_failure.overall_command_correct == 0.0
    assert business_failure.field_accuracy == 0.0
    assert business_failure.overall_command_correct == 0.0
