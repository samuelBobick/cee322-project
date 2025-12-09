"""State definition for the data engineering agent"""

from typing import Optional, TypedDict

import pandas as pd


class DataEngState(TypedDict, total=False):
    """State schema for the data engineering workflow"""

    # Data and some high-level context
    filepath: str
    data: Optional[pd.DataFrame]
    missing_info: Optional[dict]
    summary_stats: Optional[dict]

    # Records of cleaning and imputation operations (cached in between runs)
    cleaning_records: list[str]
    imputation_records: list[str]
    imputation_assumptions: dict[str, dict]
    cleaning_assumptions: dict[str, dict]
    errors: list[str]

    # Fields associated with reflection
    clean_reflection_issues: list[str]
    impute_reflection_issues: list[str]
    _clean_reflection_passed: bool
    _impute_reflection_passed: bool
    _clean_retry_count: int
    _impute_retry_count: int
