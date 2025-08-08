import json
import re
import ast
from pathlib import Path
from functools import lru_cache
from collections import defaultdict

# Helper functions allowed inside rule expressions
SAFE_GLOBALS = {
    "matches_regex": lambda value, pattern: re.fullmatch(pattern, str(value)) is not None,
}

@lru_cache
def load_rules(rule_path: str = "rule_based_screening_rules.json"):
    """Load and cache rule definitions from a JSON file."""
    base = Path(__file__).resolve().parent.parent
    path = base / rule_path
    data = json.loads(path.read_text())
    return data.get("RuleBasedScreeningRules", [])

def evaluate_rules(form_data: dict):
    """Evaluate form data against configured rules.

    Returns a dict with keys:
        - status: "ok" | "reject"
        - rule, description: present if status is "reject"
        - flags: list of flag dicts when status is "ok"
    """
    flags = []
    # allow lookup of missing variables as None
    env = defaultdict(lambda: None, SAFE_GLOBALS)
    # add both original and lower-case keys for matching
    env.update(form_data)
    env.update({k.lower(): v for k, v in form_data.items()})

    for category in load_rules():
        for rule in category.get("rules", []):
            try:
                expr = ast.parse(rule["condition"], mode="eval")
                if eval(compile(expr, "<condition>", "eval"), {}, env):
                    if rule.get("action") == "reject":
                        return {
                            "status": "reject",
                            "rule": rule.get("name"),
                            "description": rule.get("description"),
                        }
                    flags.append({
                        "rule": rule.get("name"),
                        "description": rule.get("description"),
                    })
            except Exception:
                # Ignore malformed conditions or missing data
                continue
    return {"status": "ok", "flags": flags}
