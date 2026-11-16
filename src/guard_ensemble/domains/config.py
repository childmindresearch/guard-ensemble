from guard_ensemble.llm.catalog import MODEL_KEY
from guard_ensemble.util import PACKAGE_DIR

## Paths

# Directories

PROMPTS_DIR = PACKAGE_DIR / "prompts"
DOMAIN_PROMPTS_DIR = PROMPTS_DIR / "key_domains"

# Text files

PRELUDE_TXT = PROMPTS_DIR / "_shared" / "rubric_prelude.txt"
OUTPUT_SCHEMA_TXT = PROMPTS_DIR / "_shared" / "output_schema.txt"
FLAGS_TXT = PROMPTS_DIR / "flags.txt"

## Flags

ESCALATION_FLAGS = frozenset(
    {
        "lethal_means_access",
        "current_intoxication",
        "alone_no_support",
    }
)

MINIMUM_FLAGS = frozenset(
    {
        "minor_abuse_disclosure",
        "past_suicide_attempt",
        "past_self_harm",
        "past_sexual_assault",
        "past_psychosis",
        "past_eating_disorder",
    }
)

TIMING_VOCABULARY = frozenset(
    {
        "historical_resolved",
        "historical_unresolved",
        "recent",
        "current",
        "imminent",
        "unclear",
    }
)

_ALL_KNOWN_FLAGS = set(ESCALATION_FLAGS) | set(MINIMUM_FLAGS)

## Model

MODEL: MODEL_KEY = "qwen3_4b_2507"

TEMPERATURE = 0

# The JSON object is 60-120 tokens; the grammar caps every string, so this is
# only a guard against runaway output.
RATING_MAX_TOKENS = 256

# Top tokens returned per step. The tier distribution needs at most three.
LOGPROBS = 5

## Scoring

# P(yellow) + P(red) needed to rate a domain. Low for recall: a false pass
# costs one rating prompt, a false stop loses the domain.
GATE_THRESHOLD = 0.05

# Less probability than this on the three tier words = off-format answer.
MIN_TIER_MASS = 0.5

# Rubric: "when uncertain between two tiers, assign the higher". A scored
# domain with P(red) >= RED_RAISE counts as red (catches entries rated yellow
# with P(red) 0.27-0.32, no new false reds on the sample).
RED_RAISE = 0.25
