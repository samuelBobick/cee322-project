"""Cleaning node and related functions"""

import json
from typing import Tuple

import numpy as np
import pandas as pd

from .assumptions import (
    load_human_specified_cleaning_assumptions,
    parse_llm_json_response,
)
from .cache import load_cleaning_assumptions, save_cleaning_assumptions
from .state import DataEngState


def apply_cleaning_assumptions(
    data: pd.DataFrame, col: str, rules: dict
) -> Tuple[pd.DataFrame, int]:
    """
    Apply cleaning rules to a column and return updated data and count of changes.

    Rules format:
    {
        "action": "value_mapping",
        "rules": [
            {"condition": "contains", "search": "gateway", "replacement": "backup gateway"},
            {"condition": "default", "replacement": "energy storage system"}
        ]
    }
    """
    if col not in data.columns:
        return data, 0

    cleaned = data.copy()
    total_changes = 0

    if rules.get("action") == "value_mapping":
        mapping_rules = rules.get("rules", [])

        # Track which values have been changed
        changed_mask = pd.Series(False, index=cleaned.index)

        # Apply rules in order (except default)
        for rule in mapping_rules:
            condition = rule.get("condition")
            replacement = rule.get("replacement")

            if condition == "contains":
                search = rule.get("search", "")
                # Only apply to values that haven't been changed yet
                mask = (
                    cleaned[col].astype(str).str.contains(search, case=False, na=False)
                    & ~changed_mask
                )
                changes = mask.sum()
                if changes > 0:
                    cleaned.loc[mask, col] = replacement
                    changed_mask |= mask
                    total_changes += changes
            elif condition == "default":
                # Apply to all non-null values that haven't been matched by previous rules
                # This should be the last rule
                mask = cleaned[col].notna() & ~changed_mask
                changes = mask.sum()
                if changes > 0:
                    cleaned.loc[mask, col] = replacement
                    total_changes += changes

    return cleaned, total_changes


def clean_node(state: DataEngState, chat_model) -> DataEngState:
    """Node 1: Clean the data and update state"""
    data = state.get("data")
    if data is None:
        # Try to load if not already loaded
        try:
            data = pd.read_csv(state["filepath"])
            print(f"Loaded {len(data)} rows × {len(data.columns)} columns")
        except Exception as e:
            return {"errors": state.get("errors", []) + [f"Load error: {e}"]}

    # Load cleaning assumptions in priority order:
    # 1. Human-specified (highest priority) - loaded last so it overrides
    # 2. Cached assumptions
    # 3. State assumptions (lowest priority)
    cleaning_assumptions = {}

    # Load from cache
    cached_assumptions = load_cleaning_assumptions()
    cleaning_assumptions.update(cached_assumptions)

    # Load from state
    state_assumptions = state.get("cleaning_assumptions", {})
    cleaning_assumptions.update(state_assumptions)

    # Load human-specified (overrides everything - loaded last)
    human_assumptions = {}
    if chat_model and len(state.get("cleaning_assumptions", {})) == 0:
        human_assumptions = load_human_specified_cleaning_assumptions(chat_model)
    cleaning_assumptions.update(human_assumptions)

    if cleaning_assumptions:
        print(
            f"Clean node: Loaded {len(cleaning_assumptions)} cleaning assumption(s) (human-specified: {len(human_assumptions)}, cached: {len(cached_assumptions)}, state: {len(state_assumptions)})"
        )

    # Track cleaning operations
    cleaning_records = state.get("cleaning_records", [])

    # Clean the data: remove duplicates, strip whitespace from string columns
    cleaned = data.copy()

    # Strip whitespace from string columns
    string_cols = cleaned.select_dtypes(include=["object"]).columns
    cleaned_count = 0
    for col in string_cols:
        before_nulls = cleaned[col].isna().sum()
        cleaned[col] = cleaned[col].astype(str).str.strip()
        # Replace empty strings with NaN
        cleaned[col] = cleaned[col].replace("", pd.NA)
        after_nulls = cleaned[col].isna().sum()
        if after_nulls > before_nulls:
            cleaned_count += 1

    if len(string_cols) > 0:
        cleaning_records.append(
            f"Cleaned {len(string_cols)} string column(s), converted {cleaned_count} empty strings to NaN"
        )

    # Use LLM to brainstorm cleaning assumptions for columns that don't have them yet
    print("Clean node: Analyzing columns with LLM to suggest cleaning assumptions...")
    for col in cleaned.columns:
        # Skip if assumption already exists
        if col in cleaning_assumptions:
            continue

        # Skip numeric columns - cleaning assumptions should only be for categorical/string columns
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            continue

        # Skip columns that should be numeric (even if currently object type)
        # Check if most values can be converted to numeric
        if cleaned[col].dtype == "object":
            non_null = cleaned[col].dropna()
            if len(non_null) > 0:
                numeric_attempt = pd.to_numeric(non_null, errors="coerce")
                numeric_count = numeric_attempt.notna().sum()
                total_count = len(non_null)
                # If more than 50% can be converted to numeric, skip this column
                # (it will be converted to numeric later, and shouldn't have cleaning assumptions)
                if numeric_count > 0 and numeric_count / total_count > 0.5:
                    continue

        # Get value counts for the column
        try:
            value_counts = cleaned[col].value_counts()

            # Skip if column has too many unique values (likely numeric or ID column)
            if len(value_counts) > 100:
                continue

            # Skip if all values are null
            if value_counts.empty or value_counts.sum() == 0:
                continue

            # Create prompt for LLM
            prompt = f"""You are an expert data scientist and expert in the electricity sector.

Examine the following value counts for the column '{col}':

{value_counts.to_dict()}

Past cleaning assumptions:
{state.get("cleaning_assumptions", {})}

Analyze this column and suggest cleaning assumptions if needed. Consider:
1. Values that should be standardized (e.g., variations of the same concept)
2. Values that contain specific keywords that should be mapped to standardized terms
3. Invalid or placeholder values that should be standardized to np.nan so the imputation node can handle them.
4. Do NOT map values to "Unknown", "N/A", "default", "unknown", "invalid" or other placeholders. 
5. NEVER map values to 0! That is not a valid value for any columns in this dataset.
6. Leave all np.nan values as is; the imputation node will handle them.

If cleaning is needed, respond with a JSON object in this exact format:
{{
    "action": "value_mapping",
    "rules": [
    {{"condition": "contains", "search": "keyword", "replacement": __FILL IN REASONABLE STANDARDIZED VALUE HERE__}},
        {{"condition": "default", "replacement": __FILL IN REASONABLE DEFAULT VALUE HERE__}}
    ]
}}

If no cleaning is needed, respond with: {{"action": "none"}}

Only suggest cleaning if there are clear patterns that would benefit from standardization. Be conservative - only suggest cleaning when it's clearly beneficial."""

            # Call LLM
            try:
                if not chat_model:
                    continue
                response = chat_model.invoke(prompt).content
                suggestion = parse_llm_json_response(response)

                if suggestion.get("action") != "none":
                    cleaning_assumptions[col] = suggestion
                    cleaning_records.append(
                        f"LLM suggested cleaning assumption for '{col}'"
                    )
                    print(f"Clean node: LLM suggested cleaning assumption for '{col}'")
            except (json.JSONDecodeError, Exception) as e:
                cleaning_records.append(
                    f"Error processing LLM response for '{col}': {e}"
                )
                continue

        except Exception as e:
            cleaning_records.append(f"Error analyzing '{col}': {e}")
            continue

    # Apply cleaning assumptions (with default for "Technology type" if column exists)
    if (
        "Technology type" in cleaned.columns
        and "Technology type" not in cleaning_assumptions
    ):
        # Set default assumption for Technology type column
        cleaning_assumptions["Technology type"] = {
            "action": "value_mapping",
            "rules": [
                {
                    "condition": "contains",
                    "search": "gateway",
                    "replacement": "backup gateway",
                },
                {"condition": "default", "replacement": "energy storage system"},
            ],
        }
        cleaning_records.append(
            "Created default cleaning assumption for 'Technology type' column"
        )

    # Apply all cleaning assumptions (only to non-numeric columns)
    for col, rules in cleaning_assumptions.items():
        if col in cleaned.columns:
            # Skip numeric columns - cleaning assumptions should only be for categorical/string columns
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                cleaning_records.append(
                    f"Skipped cleaning assumption for numeric column '{col}'"
                )
                print(
                    f"Clean node: Skipped cleaning assumption for numeric column '{col}'"
                )
                continue

            # Skip columns that should be numeric (even if currently object type)
            if cleaned[col].dtype == "object":
                non_null = cleaned[col].dropna()
                if len(non_null) > 0:
                    numeric_attempt = pd.to_numeric(non_null, errors="coerce")
                    numeric_count = numeric_attempt.notna().sum()
                    total_count = len(non_null)
                    # If more than 50% can be converted to numeric, skip this column
                    if numeric_count > 0 and numeric_count / total_count > 0.5:
                        cleaning_records.append(
                            f"Skipped cleaning assumption for numeric-like column '{col}'"
                        )
                        print(
                            f"Clean node: Skipped cleaning assumption for numeric-like column '{col}'"
                        )
                        continue

            cleaned, changes = apply_cleaning_assumptions(cleaned, col, rules)
            if changes > 0:
                cleaning_records.append(
                    f"Applied cleaning assumption to '{col}': {changes} values changed"
                )
                print(
                    f"Clean node: Applied cleaning assumption to '{col}': {changes} values changed"
                )

    # Check for mismatched types
    type_mismatches = []
    converted_cols = []

    # Check object columns that should be numeric
    for col in string_cols:
        if cleaned[col].isna().all():
            continue

        # Try to convert to numeric
        non_null = cleaned[col].dropna()
        if len(non_null) > 0:
            # Check if most values can be converted to numeric
            numeric_attempt = pd.to_numeric(non_null, errors="coerce")
            numeric_count = numeric_attempt.notna().sum()
            total_count = len(non_null)

            if (
                numeric_count > 0 and numeric_count / total_count > 0.5
            ):  # More than 50% are numeric
                if numeric_count < total_count:
                    type_mismatches.append(
                        f"Column '{col}': {total_count - numeric_count} non-numeric values in numeric-like column"
                    )
                # Convert to numeric
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
                converted_cols.append(
                    f"Converted '{col}' from object to numeric ({numeric_count}/{total_count} values converted)"
                )
                cleaning_records.append(
                    f"Type fix: Converted '{col}' from object to numeric"
                )

    # Recalculate numeric columns after type conversions
    numeric_cols = cleaned.select_dtypes(include=["number"]).columns

    # Check numeric columns that might have type issues
    for col in numeric_cols:
        # Check for infinite values
        if np.isinf(cleaned[col]).any():
            inf_count = np.isinf(cleaned[col]).sum()
            type_mismatches.append(
                f"Column '{col}': {inf_count} infinite values detected"
            )
            cleaned[col] = cleaned[col].replace([np.inf, -np.inf], pd.NA)
            cleaning_records.append(
                f"Type fix: Replaced {inf_count} infinite values in '{col}' with NaN"
            )

    # Report type mismatches
    if type_mismatches:
        cleaning_records.append(
            f"Type mismatch check: Found {len(type_mismatches)} type issue(s)"
        )
        for mismatch in type_mismatches:
            cleaning_records.append(f"  - {mismatch}")
        print(f"Clean node: Found {len(type_mismatches)} type mismatch(es)")

    if converted_cols:
        print(
            f"Clean node: Converted {len(converted_cols)} column(s) to appropriate types"
        )

    # Calculate missing info after cleaning
    missing_counts = (cleaned == 0).sum()
    missing_pct = (missing_counts / len(cleaned) * 100).round(2)
    missing_info = {
        col: {"count": int(missing_counts[col]), "percent": float(missing_pct[col])}
        for col in cleaned.columns
        if missing_counts[col] > 0
    }

    # Calculate summary stats for numeric columns
    summary_stats = {}
    numeric_cols = cleaned.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        vals = cleaned[col]
        summary_stats[col] = {
            "count": int(vals.count()),
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    print(f"Clean node: Cleaned {len(string_cols)} string column(s)")
    print(f"Clean node: Found {len(missing_info)} columns with missing values")

    # Save cleaning assumptions to cache file
    save_cleaning_assumptions(cleaning_assumptions)

    return {
        "data": cleaned,
        "missing_info": missing_info,
        "summary_stats": summary_stats,
        "cleaning_records": cleaning_records,
        "cleaning_assumptions": cleaning_assumptions,  # Cache assumptions for future runs
    }
