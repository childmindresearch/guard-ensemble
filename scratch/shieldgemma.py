"""
Scratch copy of guards.shieldgemma_build for experimenting with the raw prompt.
"""

from guards import SHIELDGEMMA_POLICIES

POLICIES = dict(SHIELDGEMMA_POLICIES)


def build_prompt(text: str, policy_key: str) -> str:
    """Build a single ShieldGemma prompt for one policy."""
    policy = POLICIES[policy_key]
    return (
        f"You are a policy expert trying to help determine whether a user prompt is in violation of the defined safety policies.\n"
        f"<start_of_turn>\n"
        f"Human Question: {text.strip()}\n"
        f"<end_of_turn>\n"
        f"Our safety principle is defined in the below:\n"
        f"{policy}\n"
        f"Does the human question violate the above principle? Your answer must start with 'Yes' or 'No'. "
        f"And then walk through step by step to be sure we answer correctly.\n"
    )
