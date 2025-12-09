"""Imputation node and related functions"""

import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .assumptions import (
    load_human_specified_imputation_assumptions,
    parse_llm_json_response,
)
from .cache import load_imputation_assumptions, save_imputation_assumptions
from .state import DataEngState


def normalize_column_name(col_name: str) -> str:
    """Normalize column name by removing escaping differences (e.g., \\( vs ()"""
    # Replace escaped parentheses with regular parentheses
    normalized = col_name.replace("\\(", "(").replace("\\)", ")")
    # Replace underscores with spaces for consistency
    normalized = normalized.replace("_", " ")
    return normalized


def normalize_imputation_assumptions(assumptions: Dict) -> Dict:
    """Normalize column names in imputation assumptions and ensure only one entry per column"""
    normalized = {}
    seen_columns = {}  # Map normalized name to original key

    for key, value in assumptions.items():
        normalized_key = normalize_column_name(key)

        # If we've seen this normalized column before, keep the first one (or prioritize)
        if normalized_key in seen_columns:
            # Keep the existing one (first come, first served)
            continue

        seen_columns[normalized_key] = key
        normalized[normalized_key] = value

    return normalized


def apply_imputation_assumption(
    data: pd.DataFrame, col: str, rule: dict
) -> Tuple[pd.DataFrame, int]:
    """Apply imputation rule to a column and return updated data and count of changes"""
    if col not in data.columns:
        return data, 0

    imputed = data.copy()
    method = rule.get("method")
    missing_count = imputed[col].isna().sum()

    if missing_count == 0:
        return imputed, 0

    if method == "mean":
        # Only compute mean for numeric columns
        if pd.api.types.is_numeric_dtype(imputed[col]):
            fill_value = imputed[col].mean()
            if pd.isna(fill_value):  # If all values are NaN, use 0
                fill_value = 0
            imputed[col] = imputed[col].fillna(fill_value)
        else:
            # For non-numeric columns, use mode instead
            mode_val = (
                imputed[col].mode()[0] if not imputed[col].mode().empty else "Unknown"
            )
            imputed[col] = imputed[col].fillna(mode_val)
            fill_value = mode_val
    elif method == "median":
        # Only compute median for numeric columns
        if pd.api.types.is_numeric_dtype(imputed[col]):
            fill_value = imputed[col].median()
            if pd.isna(fill_value):  # If all values are NaN, use 0
                fill_value = 0
            imputed[col] = imputed[col].fillna(fill_value)
        else:
            # For non-numeric columns, use mode instead
            mode_val = (
                imputed[col].mode()[0] if not imputed[col].mode().empty else "Unknown"
            )
            imputed[col] = imputed[col].fillna(mode_val)
            fill_value = mode_val
    elif method == "mode":
        mode_val = imputed[col].mode()[0] if not imputed[col].mode().empty else 1
        imputed[col] = imputed[col].fillna(mode_val)
        fill_value = mode_val
    elif method == "constant":
        fill_value = rule.get("value", 1)
        imputed[col] = imputed[col].fillna(fill_value)
    elif method == "forward_fill":
        imputed[col] = imputed[col].fillna(method="ffill")
        fill_value = "forward_fill"
    elif method == "backward_fill":
        imputed[col] = imputed[col].fillna(method="bfill")
        fill_value = "backward_fill"
    else:
        # Default: mean for numeric, mode for categorical
        if pd.api.types.is_numeric_dtype(imputed[col]):
            fill_value = imputed[col].mean()
            imputed[col] = imputed[col].fillna(fill_value)
        else:
            fill_value = imputed[col].mode()[0] if not imputed[col].mode().empty else 1
            imputed[col] = imputed[col].fillna(fill_value)

    return imputed, missing_count


def impute_node(state: DataEngState, chat_model) -> DataEngState:
    """Node 2: Impute missing values and update state"""
    data = state.get("data")
    if data is None:
        return {"errors": state.get("errors", []) + ["No data to impute"]}

    # Convert string "nan" and "np.nan" to actual NaN for proper handling
    data = data.copy()
    for col in data.columns:
        # Handle both object and numeric columns that might have string "nan" or "np.nan"
        if data[col].dtype == "object":
            # Replace string "nan" and "np.nan" with actual NaN
            data[col] = data[col].replace(["nan", "np.nan"], np.nan)
            # Try to convert to numeric if possible (handles columns that should be numeric but were read as object)
            if data[col].dtype == "object":
                # Check if column can be converted to numeric
                numeric_attempt = pd.to_numeric(data[col], errors="coerce")
                if (
                    numeric_attempt.notna().sum() > len(data) * 0.5
                ):  # More than 50% are numeric
                    data[col] = numeric_attempt

    # Load imputation assumptions in priority order:
    # 1. Human-specified (highest priority) - loaded last so it overrides
    # 2. Cached assumptions
    # 3. State assumptions (lowest priority)
    imputation_assumptions = {}

    # Load from cache and normalize
    cached_assumptions = load_imputation_assumptions()
    cached_assumptions = normalize_imputation_assumptions(cached_assumptions)
    imputation_assumptions.update(cached_assumptions)

    # Load from state and normalize
    state_assumptions = state.get("imputation_assumptions", {})
    state_assumptions = normalize_imputation_assumptions(state_assumptions)
    imputation_assumptions.update(state_assumptions)

    # Load human-specified
    human_assumptions = {}
    if chat_model and len(state.get("imputation_assumptions", {})) == 0:
        human_assumptions = load_human_specified_imputation_assumptions(chat_model)
        human_assumptions = normalize_imputation_assumptions(human_assumptions)
    imputation_assumptions.update(human_assumptions)

    # Final normalization to ensure no duplicates
    imputation_assumptions = normalize_imputation_assumptions(imputation_assumptions)

    print(imputation_assumptions)

    if imputation_assumptions:
        print(
            f"Impute node: Loaded {len(imputation_assumptions)} imputation assumption(s) (human-specified: {len(human_assumptions)}, cached: {len(cached_assumptions)}, state: {len(state_assumptions)})"
        )

    # Track imputation operations
    imputation_records = state.get("imputation_records", [])

    # Use LLM to brainstorm imputation assumptions for columns (can update existing assumptions, especially on retry)
    # Check if we're on a retry - if so, we should call LLM even if there are no missing values
    impute_retry_count = state.get("_impute_retry_count", 0)
    is_retry = impute_retry_count > 0

    print(
        "Impute node: Analyzing columns with missing values using LLM to suggest imputation methods..."
    )
    columns_analyzed = 0
    for col in data.columns:
        # Get normalized column name and existing assumption
        normalized_col = normalize_column_name(col)
        existing_assumption = imputation_assumptions.get(normalized_col, {})

        # Check if column has missing values
        has_missing = data[col].isna().any()

        # Only analyze columns with missing values, OR if we're on a retry (to review/update assumptions)
        if not has_missing and not is_retry:
            continue

        try:
            missing_count = data[col].isna().sum()
            missing_pct = (missing_count / len(data)) * 100
            dtype = data[col].dtype

            # Get sample of non-null values for context
            non_null_sample = data[col].dropna().head(20).tolist()

            columns_analyzed += 1
            print(
                f"Impute node: Analyzing column '{col}' (has missing: {has_missing}, has existing assumption: {bool(existing_assumption)}, is_retry: {is_retry})"
            )

            # Create prompt for LLM
            missing_info = (
                f"The column '{col}' has {missing_count} missing values ({missing_pct:.2f}% of the data)."
                if has_missing
                else f"The column '{col}' has no missing values (all values have been imputed)."
            )

            prompt = f"""You are an expert data scientist and expert in the electricity sector.

{missing_info} The column type is {dtype}.

Sample of non-null values: {non_null_sample}

Existing imputation assumption for this column:
{existing_assumption if existing_assumption else "None"}

Past imputation assumptions for other columns:
{state.get("imputation_assumptions", {})}

Past imputation records:
{chr(10).join(imputation_records[-10:]) if imputation_records else "No previous records"}

Evaluate whether the data imputation process was successful. If the complaints from the past imputation records are attempted to be addressed, then consider it a pass.
If there are no complaints from the past imputation records, then generate your conclusion from scratch.

Suggest an appropriate imputation method for this column. You can UPDATE the existing assumption if a better method is needed based on the reflection feedback. Consider:
1. For numeric columns: mean, median, or a constant value
2. For categorical columns: mode, or another context-dependent value. Do not impute with "Unknown", "N/A", or 0! Note that for all columns, a value of 0 corresponds to a missing value which you should impute.
3. For time-series data: forward_fill or backward_fill

Respond with a JSON object in this exact format:
{{
    "method": "mean" | "median" | "mode" | "constant" | "forward_fill" | "backward_fill",
    "value": <optional constant value if method is "constant">
}}

If no imputation is needed or the column should be left as-is, respond with: {{"method": "none"}}

Only suggest imputation if it makes sense for the data type and context. Be conservative."""

            # Call LLM
            try:
                if not chat_model:
                    continue
                print(f"Impute node: Calling LLM for column '{col}'...")
                response = chat_model.invoke(prompt).content
                suggestion = parse_llm_json_response(response)
                print(f"Impute node: LLM response for '{col}': {suggestion}")

                if suggestion.get("method") != "none":
                    # Use normalized column name
                    normalized_col = normalize_column_name(col)
                    imputation_assumptions[normalized_col] = suggestion
                    if existing_assumption:
                        imputation_records.append(
                            f"LLM updated imputation method for '{col}': {suggestion.get('method')} (was: {existing_assumption.get('method', 'none')})"
                        )
                        print(
                            f"Impute node: LLM updated imputation method for '{col}': {suggestion.get('method')} (was: {existing_assumption.get('method', 'none')})"
                        )
                    else:
                        imputation_records.append(
                            f"LLM suggested imputation method for '{col}': {suggestion.get('method')}"
                        )
                        print(
                            f"Impute node: LLM suggested imputation method for '{col}': {suggestion.get('method')}"
                        )
            except (json.JSONDecodeError, Exception) as e:
                imputation_records.append(
                    f"Error processing LLM response for '{col}': {e}"
                )
                continue

        except Exception as e:
            imputation_records.append(f"Error analyzing '{col}': {e}")
            continue

    print(f"Impute node: Analyzed {columns_analyzed} column(s) using LLM")

    # Impute missing values using assumptions or defaults
    imputed = data.copy()
    columns_with_assumptions_applied = set()

    # Apply imputation assumptions for columns that have them
    # Map normalized assumption keys to actual column names
    for normalized_col, rule in imputation_assumptions.items():
        # Find matching column in dataframe (check both exact and normalized)
        matching_col = None
        for actual_col in imputed.columns:
            if normalize_column_name(actual_col) == normalized_col:
                matching_col = actual_col
                break

        if matching_col and imputed[matching_col].isna().any():
            imputed, changes = apply_imputation_assumption(imputed, matching_col, rule)
            columns_with_assumptions_applied.add(matching_col)
            if changes > 0:
                method = rule.get("method", 1)
                fill_value = rule.get("value", "N/A")
                if method in ["mean", "median"]:
                    if pd.api.types.is_numeric_dtype(imputed[matching_col]):
                        fill_value = (
                            imputed[matching_col].mean()
                            if method == "mean"
                            else imputed[matching_col].median()
                        )
                    else:
                        fill_value = (
                            imputed[matching_col].mode()[0]
                            if not imputed[matching_col].mode().empty
                            else "Unknown"
                        )
                elif method == "mode":
                    fill_value = (
                        imputed[matching_col].mode()[0]
                        if not imputed[matching_col].mode().empty
                        else "Unknown"
                    )
                imputation_records.append(
                    f"Applied assumption: Filled {changes} missing values in '{matching_col}' with {method}={fill_value}"
                )
                print(
                    f"Impute node: Applied assumption - Filled {changes} missing values in '{matching_col}' with {method}={fill_value}"
                )

    # Apply default imputation for ALL columns with missing values that don't have assumptions
    # Check all columns, not just numeric/categorical separately
    for col in imputed.columns:
        if col not in columns_with_assumptions_applied and imputed[col].isna().any():
            missing_count = imputed[col].isna().sum()

            if pd.api.types.is_numeric_dtype(imputed[col]):
                # For numeric columns, use mean
                fill_value = imputed[col].mean()
                if pd.isna(fill_value):  # If all values are NaN, use 0
                    fill_value = 0
                imputed[col] = imputed[col].fillna(fill_value)
                imputation_records.append(
                    f"Filled {missing_count} missing values in '{col}' with mean={fill_value:.2f} (default)"
                )
                print(
                    f"Impute node: Filled {missing_count} missing values in '{col}' with mean={fill_value:.2f} (default)"
                )
            else:
                # For categorical columns, use mode
                mode_val = (
                    imputed[col].mode()[0]
                    if not imputed[col].mode().empty
                    else "Unknown"
                )
                imputed[col] = imputed[col].fillna(mode_val)
                imputation_records.append(
                    f"Filled {missing_count} missing values in '{col}' with mode='{mode_val}' (default)"
                )
                print(
                    f"Impute node: Filled {missing_count} missing values in '{col}' with mode='{mode_val}' (default)"
                )

    # Final pass: ensure ALL remaining NaN values are filled (safety check)
    for col in imputed.columns:
        if imputed[col].isna().any():
            missing_count = imputed[col].isna().sum()
            if pd.api.types.is_numeric_dtype(imputed[col]):
                fill_value = imputed[col].mean()
                if pd.isna(fill_value):  # If all values are NaN, use 0
                    fill_value = 0
                imputed[col] = imputed[col].fillna(fill_value)
                print(
                    f"Impute node: Final pass - Filled {missing_count} remaining NaN values in '{col}' with mean={fill_value:.2f}"
                )
            else:
                mode_val = (
                    imputed[col].mode()[0]
                    if not imputed[col].mode().empty
                    else "Unknown"
                )
                imputed[col] = imputed[col].fillna(mode_val)
                print(
                    f"Impute node: Final pass - Filled {missing_count} remaining NaN values in '{col}' with mode='{mode_val}'"
                )

    # Update missing info after imputation (should be empty or minimal)
    missing_counts_after = imputed.isna().sum()
    missing_info_after = {
        col: {
            "count": int(missing_counts_after[col]),
            "percent": float((missing_counts_after[col] / len(imputed) * 100).round(2)),
        }
        for col in imputed.columns
        if missing_counts_after[col] > 0
    }

    # Update summary stats after imputation
    summary_stats = {}
    numeric_cols = imputed.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        vals = imputed[col]
        summary_stats[col] = {
            "count": int(vals.count()),
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    print(
        f"Impute node: Missing values imputed. {len(imputation_records)} imputation operations performed"
    )
    print(
        f"Impute node: Remaining missing values in {len(missing_info_after)} column(s)"
    )

    # Save imputation assumptions to cache file (normalized)
    save_imputation_assumptions(imputation_assumptions)

    return {
        "data": imputed,
        "missing_info": missing_info_after,
        "summary_stats": summary_stats,
        "imputation_records": imputation_records,
        "imputation_assumptions": imputation_assumptions,  # Cache assumptions for future runs
    }
