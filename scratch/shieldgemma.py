"""Scratch copy of guards.shieldgemma_build, for experimenting with the raw prompt.

Use with runner.generate (no chat template applied):

    from scratch import runner
    from scratch.shieldgemma import SHIELDGEMMA_POLICIES, shieldgemma_build

    res = {
        k: runner.generate(
            "shieldgemma",
            [shieldgemma_build(t, k) for t in texts],
            temperature=0,
            max_tokens=5,
        )
        for k in SHIELDGEMMA_POLICIES
    }
"""

from guards import SHIELDGEMMA_POLICIES

__all__ = ["SHIELDGEMMA_POLICIES", "shieldgemma_build"]


def shieldgemma_build(text: str, policy_key: str) -> str:
    """Build a single ShieldGemma prompt for one policy."""
    policy = SHIELDGEMMA_POLICIES[policy_key]
    return (
        f"You are a policy expert trying to help determine whether a user prompt is in violation of the defined safety policies.\n"
        f"<start_of_turn>\n"
        f"Human Question: {text.strip()}\n"
        f"<end_of_turn>\n"
        f"Our safety principle is defined in the below:\n"
        f"{policy}\n"
        f"Does the human question violate the above principle? Your answer must start with 'Yes' or 'No'. "
        f"And then walk through step by step to be sure we answer correctly."
    )