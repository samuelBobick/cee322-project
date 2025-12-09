"""Caching utilities for cleaning and imputation assumptions"""

import json
import os
from typing import Dict

CACHE_DIR = "/Users/sam/Desktop/cee322/final/data"
CLEANING_CACHE_FILE = os.path.join(CACHE_DIR, "cleaning_assumptions.json")
IMPUTATION_CACHE_FILE = os.path.join(CACHE_DIR, "imputation_assumptions.json")


def load_cleaning_assumptions() -> Dict:
    """Load cleaning assumptions from cache file (shared across all datasets)"""
    if os.path.exists(CLEANING_CACHE_FILE):
        try:
            with open(CLEANING_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(
                f"Warning: Could not load cleaning assumptions from {CLEANING_CACHE_FILE}: {e}"
            )
    return {}


def save_cleaning_assumptions(assumptions: Dict) -> None:
    """Save cleaning assumptions to cache file (shared across all datasets)"""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CLEANING_CACHE_FILE, "w") as f:
            json.dump(assumptions, f, indent=2)
        print(f"Clean node: Cached cleaning assumptions to {CLEANING_CACHE_FILE}")
    except Exception as e:
        print(
            f"Warning: Could not save cleaning assumptions to {CLEANING_CACHE_FILE}: {e}"
        )


def load_imputation_assumptions() -> Dict:
    """Load imputation assumptions from cache file (shared across all datasets)"""
    if os.path.exists(IMPUTATION_CACHE_FILE):
        try:
            with open(IMPUTATION_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(
                f"Warning: Could not load imputation assumptions from {IMPUTATION_CACHE_FILE}: {e}"
            )
    return {}


def save_imputation_assumptions(assumptions: Dict) -> None:
    """Save imputation assumptions to cache file (shared across all datasets)"""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(IMPUTATION_CACHE_FILE, "w") as f:
            json.dump(assumptions, f, indent=2)
        print(f"Impute node: Cached imputation assumptions to {IMPUTATION_CACHE_FILE}")
    except Exception as e:
        print(
            f"Warning: Could not save imputation assumptions to {IMPUTATION_CACHE_FILE}: {e}"
        )
