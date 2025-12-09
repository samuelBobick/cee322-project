"""Reflection nodes for evaluating data quality after cleaning and imputation"""

from typing import Literal
import pandas as pd
from .state import DataEngState
from .assumptions import parse_llm_json_response


def reflect_clean_node(state: DataEngState, chat_model) -> DataEngState:
    """
    Reflection node after cleaning: evaluate data cleanliness.
    Returns state with updated records, and routes to 'clean' if issues found.
    """
    data = state.get("data")
    if data is None:
        return {"errors": state.get("errors", []) + ["No data to reflect on"]}

    cleaning_records = state.get("cleaning_records", [])
    clean_validation_issues = state.get("clean_validation_issues", [])

    # Track retry count to prevent infinite loops
    clean_retry_count = state.get("_clean_retry_count", 0)
    if clean_retry_count >= 3:
        cleaning_records.append(
            "Reflection: Max retries (3) reached, proceeding despite issues"
        )
        print("Clean reflection: Max retries reached, proceeding to imputation")
        return {
            "cleaning_records": cleaning_records,
            "clean_validation_issues": clean_validation_issues,
            "_clean_reflection_passed": True,
            "_clean_retry_count": clean_retry_count,
        }

    # Run cleanliness tests
    issues = []

    # Check for empty columns
    empty_cols = data.columns[data.isna().all()].tolist()
    if empty_cols:
        issues.append(f"Empty columns found: {empty_cols}")

    # Check for type inconsistencies
    type_issues = []
    for col in data.select_dtypes(include=["object"]).columns:
        non_null = data[col].dropna()
        if len(non_null) > 0:
            numeric_attempt = pd.to_numeric(non_null, errors="coerce")
            numeric_pct = numeric_attempt.notna().sum() / len(non_null) * 100
            if numeric_pct > 80 and numeric_pct < 100:
                type_issues.append(
                    f"{col}: {100-numeric_pct:.1f}% non-numeric in numeric-like column"
                )
    if type_issues:
        issues.append(
            f"Type inconsistencies: {len(type_issues)} column(s) with mixed types"
        )

    # Prepare summary for LLM evaluation
    test_summary = {
        "total_rows": len(data),
        "total_columns": len(data.columns),
        "missing_values": int(data.isna().sum().sum()),
        "empty_columns": len(empty_cols),
        "issues_found": len(issues),
        "issue_details": issues[:5],  # First 5 issues
    }

    # Use LLM to evaluate if cleaning was successful
    prompt = f"""You are an expert data scientist evaluating data quality after cleaning.

Data summary:
- Total rows: {test_summary['total_rows']}
- Total columns: {test_summary['total_columns']}
- Missing values: {test_summary['missing_values']}
- Empty columns: {test_summary['empty_columns']}

Issues detected:
{chr(10).join(issues) if issues else 'No major issues detected'}

Past cleaning records:
{chr(10).join(cleaning_records) if cleaning_records else 'No cleaning records'}

Evaluate whether the data cleaning process was successful. If the complaints from the past cleaning records are attempted to be addressed, then consider it a pass.
If there are no complaints from the past cleaning records, then generate your conclusion from scratch.

Note that we are ok with high missingness values in the data; this will be handled by the imputation node.

Consider:
1. Are there critical data quality issues that need re-cleaning?
2. Are the issues minor and acceptable?
3. Should we proceed to imputation or re-clean?

Respond with JSON in this exact format:
{{
    "pass": true | false,
    "reason": "brief explanation of decision",
    "recommendations": ["suggestion1", "suggestion2"]
}}

If pass is false, the system will re-run cleaning. If pass is true, proceed to imputation."""

    try:
        response = chat_model.invoke(prompt).content

        evaluation = parse_llm_json_response(response)
        passed = evaluation.get("pass", True)
        reason = evaluation.get("reason", "No reason provided")
        recommendations = evaluation.get("recommendations", [])

        cleaning_records.append(
            f"Reflection: {'PASSED' if passed else 'FAILED'} - {reason}"
        )
        if recommendations:
            for rec in recommendations:
                cleaning_records.append(f"  Recommendation: {rec}")

        # Record validation issues
        if not passed:
            clean_validation_issues.extend(issues)
            if recommendations:
                clean_validation_issues.extend(
                    [f"Recommendation: {rec}" for rec in recommendations]
                )
            clean_validation_issues.append(f"Reflection failed: {reason}")

        print(f"Clean reflection: {'PASSED' if passed else 'FAILED'} - {reason}")
        if not passed:
            print(
                f"Clean reflection: Re-running cleaning based on recommendations (attempt {clean_retry_count + 1}/3)"
            )

        return {
            "cleaning_records": cleaning_records,
            "clean_validation_issues": clean_validation_issues,
            "_clean_reflection_passed": passed,
            "_clean_retry_count": clean_retry_count + (1 if not passed else 0),
        }

    except Exception as e:
        # If LLM evaluation fails, default to pass
        cleaning_records.append(
            f"Reflection: LLM evaluation failed ({e}), defaulting to PASS"
        )
        print("Clean reflection: LLM evaluation failed, defaulting to PASS")
        return {
            "cleaning_records": cleaning_records,
            "clean_validation_issues": clean_validation_issues,
            "_clean_reflection_passed": True,
            "_clean_retry_count": clean_retry_count,
        }


def reflect_impute_node(state: DataEngState, chat_model) -> DataEngState:
    """
    Reflection node after imputation: evaluate imputation quality.
    Returns state with updated records, and routes to 'impute' if issues found.
    """
    data = state.get("data")
    if data is None:
        return {"errors": state.get("errors", []) + ["No data to reflect on"]}

    imputation_records = state.get("imputation_records", [])
    impute_validation_issues = state.get("impute_validation_issues", [])

    impute_retry_count = state.get("_impute_retry_count", 0)
    if impute_retry_count >= 3:
        imputation_records.append(
            "Reflection: Max retries (3) reached, proceeding despite issues"
        )
        print("Impute reflection: Max retries reached, workflow complete")
        return {
            "imputation_records": imputation_records,
            "impute_validation_issues": impute_validation_issues,
            "_impute_reflection_passed": True,
            "_impute_retry_count": impute_retry_count,
        }

    issues = []

    # Check remaining missing values
    remaining_missing = data.isna().sum().sum()
    if remaining_missing > 0:
        missing_cols = data.columns[data.isna().any()].tolist()
        issues.append(
            f"Still {remaining_missing} missing values in {len(missing_cols)} column(s): {missing_cols[:3]}"
        )

    # Check for unrealistic imputed values (e.g., negative where shouldn't be)
    unrealistic_values = []
    for col in data.select_dtypes(include=["number"]).columns:
        if data[col].notna().sum() > 0:
            if (
                "cost" in col.lower()
                or "price" in col.lower()
                or "value" in col.lower()
                or "power" in col.lower()
                or "rating" in col.lower()
                or "capacity" in col.lower()
                or "energy" in col.lower()
                or "voltage" in col.lower()
                or "current" in col.lower()
                or "frequency" in col.lower()
            ):
                if (data[col] < 0).any():
                    unrealistic_values.append(f"{col}: negative values found")
    if unrealistic_values:
        issues.append(f"Unrealistic values: {unrealistic_values}")

    #  Check data distribution changes by comparing to summary stats
    summary_stats = state.get("summary_stats", {})
    if summary_stats:
        # Check for extreme outliers that might indicate bad imputation
        outlier_cols = []
        for col, stats in summary_stats.items():
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                mean = stats.get("mean", 0)
                std = stats.get("std", 0)
                if std > 0:
                    z_scores = (data[col] - mean).abs() / std
                    extreme_outliers = (z_scores > 5).sum()
                    if extreme_outliers > 0:
                        outlier_cols.append(
                            f"{col}: {extreme_outliers} extreme outliers"
                        )
        if outlier_cols:
            issues.append(f"Extreme outliers detected: {outlier_cols[:3]}")

    test_summary = {
        "total_rows": len(data),
        "total_columns": len(data.columns),
        "remaining_missing": int(remaining_missing),
        "imputation_operations": len(imputation_records),
        "issues_found": len(issues),
        "issue_details": issues[:5],
    }

    prompt = f"""You are an expert data scientist evaluating data quality after imputation.

Data summary:
- Total rows: {test_summary['total_rows']}
- Total columns: {test_summary['total_columns']}
- Remaining missing values: {test_summary['remaining_missing']}
- Imputation operations performed: {test_summary['imputation_operations']}

Issues detected:
{chr(10).join(issues) if issues else 'No major issues detected'}

Evaluate whether the imputation was successful. Consider:
1. Are there critical data quality issues that need re-imputation?
2. Are the imputation methods appropriate?
3. Should we proceed or re-impute with different methods?

Respond with JSON in this exact format:
{{
    "pass": true | false,
    "reason": "brief explanation of decision",
    "recommendations": ["suggestion1", "suggestion2"]
}}

If pass is false, the system will re-run imputation. If pass is true, the workflow is complete."""

    try:
        response = chat_model.invoke(prompt).content

        evaluation = parse_llm_json_response(response)

        passed = evaluation.get("pass", True)
        reason = evaluation.get("reason", "No reason provided")
        recommendations = evaluation.get("recommendations", [])

        imputation_records.append(
            f"Reflection: {'PASSED' if passed else 'FAILED'} - {reason}"
        )
        if recommendations:
            for rec in recommendations:
                imputation_records.append(f"  Recommendation: {rec}")

        if not passed:
            impute_validation_issues.extend(issues)
            if recommendations:
                impute_validation_issues.extend(
                    [f"Recommendation: {rec}" for rec in recommendations]
                )
            impute_validation_issues.append(f"Reflection failed: {reason}")

        print(f"Impute reflection: {'PASSED' if passed else 'FAILED'} - {reason}")
        if not passed:
            print(
                f"Impute reflection: Re-running imputation based on recommendations (attempt {impute_retry_count + 1}/3)"
            )

        return {
            "imputation_records": imputation_records,
            "impute_validation_issues": impute_validation_issues,
            "_impute_reflection_passed": passed,
            "_impute_retry_count": impute_retry_count + (1 if not passed else 0),
        }

    except Exception as e:
        imputation_records.append(
            f"Reflection: LLM evaluation failed ({e}), defaulting to PASS"
        )
        print("Impute reflection: LLM evaluation failed, defaulting to PASS")
        return {
            "imputation_records": imputation_records,
            "impute_validation_issues": impute_validation_issues,
            "_impute_reflection_passed": True,
            "_impute_retry_count": impute_retry_count,
        }


def route_after_clean_reflection(state: DataEngState) -> Literal["clean", "impute"]:
    """Route after clean reflection: go back to clean if failed, else proceed to impute"""
    passed = state.get("_clean_reflection_passed", True)
    retry_count = state.get("_clean_retry_count", 0)

    # Force pass if max retries reached
    if retry_count >= 3:
        return "impute"

    if not passed:
        return "clean"
    return "impute"


def route_after_impute_reflection(state: DataEngState) -> Literal["impute", "end"]:
    """Route after impute reflection: go back to impute if failed, else end"""
    passed = state.get("_impute_reflection_passed", True)
    retry_count = state.get("_impute_retry_count", 0)

    # Force end if max retries reached
    if retry_count >= 3:
        return "end"

    if not passed:
        return "impute"
    return "end"
