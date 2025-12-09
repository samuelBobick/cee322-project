"""Functions for loading human-specified assumptions from text files"""

import os
import json
from typing import Dict, Any


def parse_llm_json_response(response: str) -> Dict[str, Any]:
    """Parse JSON from LLM response, handling markdown code blocks"""
    response_clean = response.strip()
    if response_clean.startswith("```"):
        # Extract JSON from code block
        lines = response_clean.split("\n")
        json_lines = []
        in_json = False
        for line in lines:
            if line.strip().startswith("```"):
                in_json = not in_json
                continue
            if in_json:
                json_lines.append(line)
        response_clean = "\n".join(json_lines).strip()

    return json.loads(response_clean)


def load_human_specified_cleaning_assumptions(chat_model) -> Dict:
    """Load and parse human-specified cleaning assumptions from text file using LLM"""
    assumptions_file = "/Users/sam/Desktop/cee322/final/data/cleaning_assumptions.txt"
    assumptions = {}

    if not os.path.exists(assumptions_file):
        return assumptions

    try:
        with open(assumptions_file, "r") as f:
            human_spec = f.read().strip()

        if not human_spec:
            return assumptions

        # Use LLM to convert natural language to structured format
        prompt = f"""You are an expert data scientist. Convert the following natural language cleaning rules into a structured JSON format.

Human-specified cleaning rules:
{human_spec}

Convert each rule into the following JSON format. The output should be a JSON object where:
- Keys are column names
- Values are cleaning rule objects with this structure:
  {{
    "action": "value_mapping",
    "rules": [
      {{"condition": "contains", "search": "keyword", "replacement": "standardized_value"}},
      {{"condition": "default", "replacement": "default_value"}}
    ]
  }}

Example: If the rule says "If Technology type contains 'gateway', map it to 'backup gateway'. Otherwise, map it to 'energy storage system'", the output should be:
{{
  "Technology type": {{
    "action": "value_mapping",
    "rules": [
      {{"condition": "contains", "search": "gateway", "replacement": "backup gateway"}},
      {{"condition": "default", "replacement": "energy storage system"}}
    ]
  }}
}}

Return ONLY valid JSON, no markdown, no explanations. If there are multiple rules, include all of them in the JSON object."""

        response = chat_model.invoke(prompt).content
        assumptions = parse_llm_json_response(response)
        print(
            f"Loaded {len(assumptions)} human-specified cleaning assumption(s) from {assumptions_file}"
        )

    except FileNotFoundError:
        pass  # File doesn't exist, that's okay
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse human-specified assumptions as JSON: {e}")
    except Exception as e:
        print(f"Warning: Error loading human-specified assumptions: {e}")

    return assumptions


def load_human_specified_imputation_assumptions(chat_model) -> Dict:
    """Load and parse human-specified imputation assumptions from text file using LLM"""
    assumptions_file = "/Users/sam/Desktop/cee322/final/data/imputation_assumptions.txt"
    assumptions = {}

    if not os.path.exists(assumptions_file):
        return assumptions

    try:
        with open(assumptions_file, "r") as f:
            human_spec = f.read().strip()

        if not human_spec:
            return assumptions

        # Use LLM to convert natural language to structured format
        prompt = f"""You are an expert data scientist. Convert the following natural language imputation rules into a structured JSON format.

Human-specified imputation rules:
{human_spec}

Convert each rule into the following JSON format. The output should be a JSON object where:
- Keys are column names
- Values are imputation rule objects with this structure:
  {{
    "method": "mean" | "median" | "mode" | "constant" | "forward_fill" | "backward_fill",
    "value": <optional constant value if method is "constant">
  }}

Examples:
- "Impute missing values in 'age' with the mean" → {{"age": {{"method": "mean"}}}}
- "Fill missing 'category' values with 'Unknown'" → {{"category": {{"method": "constant", "value": "Unknown"}}}}
- "Use forward fill for 'timestamp' column" → {{"timestamp": {{"method": "forward_fill"}}}}

Return ONLY valid JSON, no markdown, no explanations. You should impute all missing values. If there are multiple rules, include all of them in the JSON object."""

        response = chat_model.invoke(prompt).content
        assumptions = parse_llm_json_response(response)
        print(
            f"Loaded {len(assumptions)} human-specified imputation assumption(s) from {assumptions_file}"
        )

    except FileNotFoundError:
        pass  # File doesn't exist, that's okay
    except json.JSONDecodeError as e:
        print(
            f"Warning: Could not parse human-specified imputation assumptions as JSON: {e}"
        )
    except Exception as e:
        print(f"Warning: Error loading human-specified imputation assumptions: {e}")

    return assumptions
