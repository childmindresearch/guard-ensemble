"""Domain stage on the four key domains: gate -> rating -> entry tier.

Every entry is rated against each prompt in prompts/key_domains/ by one Qwen3
model (scratch/runner.py), in three steps:

1. Gate: one token per (entry, domain). The domain is rated only when
   P(yellow) + P(red) >= GATE_THRESHOLD, or the answer is off-format. About
   three quarters of the pairs stop here.
2. Rating: the domain prompt plus the output schema, decoded under a JSON
   grammar (vLLM structured outputs). The "criterion" field is an enum of the
   names in the prompt's "## Criteria" list, so the tier follows from the
   criterion. Gated-out domains are green with "scored": False.
3. Entry tier: rules.resolve with two changes against spillover (one passage
   claimed by several domains): a scored domain with P(red) >= RED_RAISE counts
   as red, and "2+ yellow -> red" counts distinct quotes rather than domains.

Per-row inputs go last in every prompt and prompts run domain-major, so vLLM's
prefix cache reuses the rest. Output columns match
risk_pipeline.run_custom_prompts_stage (domain_results, domain_incomplete,
custom_prompt_errors) plus gate_scores, domain_raw, entry_tier, entry_reasons
and yellow_concerns.

    from scratch import domains
    results = domains.run(domains.read_sample_csv(100))
    domains.save(results, "key_domains_qwen3_4b_2507_head100")
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

import pandas as pd

import rules
from rubric import (
    ESCALATION_FLAGS,
    MINIMUM_FLAGS,
    TIER_UNKNOWN,
    TIMING_VOCABULARY,
    Tier,
)
from scratch import lib, models, runner

## Settings

MODEL: models.MODEL_KEY = "qwen3_4b_2507"
TEMPERATURE = 0
# P(yellow) + P(red) needed to rate a domain. Low for recall: a false pass
# costs one rating prompt, a false stop loses the domain.
GATE_THRESHOLD = 0.05
# Less probability than this on the three tier words = off-format answer.
MIN_TIER_MASS = 0.5
# Rubric: "when uncertain between two tiers, assign the higher". A scored
# domain with P(red) >= RED_RAISE counts as red (catches entries rated yellow
# with P(red) 0.27-0.32, no new false reds on the sample).
RED_RAISE = 0.25
# The JSON object is 60-120 tokens; the grammar caps every string, so this is
# only a guard against runaway output.
RATING_MAX_TOKENS = 256
# Top tokens returned per step. The tier distribution needs at most three.
LOGPROBS = 5

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts"
DOMAIN_PROMPTS = PROMPTS / "key_domains"
PRELUDE = PROMPTS / "_shared" / "rubric_prelude.txt"
OUTPUT_SCHEMA = PROMPTS / "_shared" / "output_schema_v3.txt"
FLAGS = PROMPTS / "flags.txt"
OUT = ROOT / "out"  # gitignored

TIERS = ("green", "yellow", "red")

Row = dict[Hashable, Any]
DomainResult = dict[str, Any]


## Data files


def read_sample_csv(
    limit: int | None = None, path: str | Path = "data/sample_inputs.csv"
) -> pd.DataFrame:
    """Any CSV with id, text, participant_id, age, entry_date (extra columns are
    kept), e.g. "labelled_data/clinician_inputs.csv"."""
    entries = pd.read_csv(ROOT / path)
    return entries.head(limit) if limit is not None else entries


def save(results: pd.DataFrame, name: str) -> Path:
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.jsonl"
    results.to_json(path, orient="records", lines=True, force_ascii=False)
    return path


def load(name: str) -> pd.DataFrame:
    return pd.read_json(OUT / f"{name}.jsonl", orient="records", lines=True)


## Prompts
#
# Each domain prompt file has a "# Domain: <name>" heading, a "## Criteria"
# list ("- <tier>: <name>" lines) and ends with the standard "## Input" block.
# The input block is moved behind the shared schema so that everything before
# the per-row inputs is identical across rows.

# The prelude's whole-entry instruction invites spillover in a one-domain prompt.
INPUTS_BLOCK = (
    "## Input\n\nEntry text:\n$text\n\nParticipant age: $age\nEntry date: $entry_date"
)
INPUTS = f"\n\n---\n\n{INPUTS_BLOCK}\n"
SCOPE = (
    "\n\nRate the entry above for {name} only. Content that belongs to other "
    "domains does not count toward this rating; if the entry contains nothing "
    "relevant to {name}, "
)
GATE_ASK = "answer green. Reply with exactly one lowercase word: green, yellow, or red."
RATING_ASK = (
    "use tier green. Respond with the JSON object described in the output schema."
)

_DOMAIN_NAME_RE = re.compile(r"^# Domain: (.+)$", re.MULTILINE)
_CRITERION_RE = re.compile(r"^- (green|yellow|red): (.+?)\s*$", re.MULTILINE)
_PLACEHOLDER_RE = re.compile(r"\$(text|age|entry_date)\b")


@dataclass(frozen=True)
class DomainPrompt:
    """What the two model stages need for one domain, built from its prompt file."""

    key: str  # file stem, e.g. "01_harm_to_self"
    name: str  # the "# Domain:" heading
    gate: str  # template for the one-token gate
    rating: str  # template for the JSON rating
    criterion_tiers: dict[str, str]  # criterion name -> tier


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def prelude() -> str:
    return read_text(PRELUDE)


def flag_definitions() -> str:
    """The flag names the schema allows, each with its prompts/flags.txt definition."""
    lines = [
        line.strip()
        for line in read_text(FLAGS).splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if {line.split(":")[0] for line in lines} != {*ESCALATION_FLAGS, *MINIMUM_FLAGS}:
        raise ValueError("prompts/flags.txt doesn't match the rubric's flags")
    return (
        "\n\n## Flag definitions\n\nThe flags array may contain only the names "
        "below, spelled exactly; if none applies, use []. Set a flag only when "
        "the entry meets its definition. Flags describe the participant, not "
        "other people.\n\n"
        + "\n".join(f"- {line}" for line in lines).replace("$", "$$")
    )


def output_schema() -> str:
    return read_text(OUTPUT_SCHEMA).rstrip() + flag_definitions()


def domain_name(prompt_text: str, key: str) -> str:
    match = _DOMAIN_NAME_RE.search(prompt_text)
    if match is None:
        raise ValueError(f"{key}: no '# Domain:' heading")
    return match[1].strip()


def body_without_inputs(prompt_text: str, key: str) -> str:
    """The prompt up to its "## Input" block, which must be the standard one."""
    body, separator, inputs = prompt_text.rpartition("\n## Input")
    if not separator or ("## Input" + inputs).split() != INPUTS_BLOCK.split():
        raise ValueError(f"{key}: the prompt must end with the standard input block")
    if _PLACEHOLDER_RE.search(body):
        raise ValueError(f"{key}: input placeholders outside the '## Input' block")
    return body.rstrip()


def criterion_tiers(prompt_text: str, key: str) -> dict[str, str]:
    """Criterion name -> tier, from the prompt's "## Criteria" section. The
    names are the only values the rating grammar allows for "criterion"."""
    _, separator, rest = prompt_text.partition("\n## Criteria")
    if not separator:
        raise ValueError(f"{key}: no '## Criteria' section")
    section = rest.split("\n## ", 1)[0]
    found = {name: tier for tier, name in _CRITERION_RE.findall(section)}
    if set(found.values()) != set(TIERS):
        raise ValueError(f"{key}: criteria must cover green, yellow and red")
    return found


def build_domain_prompt(
    path: Path, prelude_text: str, schema_text: str
) -> DomainPrompt:
    key = path.stem
    text = read_text(path)
    name = domain_name(text, key)
    head = f"{prelude_text}\n\n{body_without_inputs(text, key)}"
    scope = SCOPE.format(name=name.replace("$", "$$"))
    return DomainPrompt(
        key=key,
        name=name,
        gate=f"{head}{INPUTS}{scope}{GATE_ASK}",
        rating=f"{head}\n\n{schema_text}{INPUTS}{scope}{RATING_ASK}",
        criterion_tiers=criterion_tiers(text, key),
    )


def domain_prompts() -> dict[str, DomainPrompt]:
    """Key -> DomainPrompt for every file in prompts/key_domains/, in file order."""
    prelude_text, schema_text = prelude(), output_schema()
    return {
        path.stem: build_domain_prompt(path, prelude_text, schema_text)
        for path in sorted(DOMAIN_PROMPTS.glob("*.txt"))
    }


def render(template: str, row: Row) -> str:
    return Template(template).substitute(
        text=row["text"], age=str(row["age"]), entry_date=str(row["entry_date"])
    )


## Model calls


def chat(stage: str, prompts: list[str], **sampling) -> list[dict]:
    """One completion per prompt ({"text", "logprobs"}), timed and logged."""
    if not prompts:
        return []
    started = time.monotonic()
    completions = runner.chat(
        MODEL,
        lib.create_messages(prompts),
        chat_template_kwargs={"enable_thinking": False},
        **sampling,
    )
    print(f"{stage}: {len(prompts)} prompts ({time.monotonic() - started:.0f}s)")
    return [choices[0] for choices in completions]


def rating_schema(criterion_tiers: dict[str, str]) -> dict:
    """Decoding is constrained to this schema (vLLM structured outputs): valid
    JSON, these fields in this order, enum values only. The prompt's output
    schema (prompts/_shared/output_schema_v3.txt) describes the same fields."""
    return {
        "type": "object",
        "properties": {
            "domain": {"type": "string", "maxLength": 60},
            "criterion": {"enum": list(criterion_tiers)},
            "quotes": {
                "type": "array",
                "items": {"type": "string", "maxLength": 200},
                "maxItems": 2,
            },
            "tier": {"enum": list(TIERS)},
            "timing": {"enum": list(TIMING_VOCABULARY)},
            "flags": {
                "type": "array",
                "items": {"enum": [*ESCALATION_FLAGS, *MINIMUM_FLAGS]},
                "maxItems": len(ESCALATION_FLAGS) + len(MINIMUM_FLAGS),
            },
        },
        "required": ["domain", "criterion", "quotes", "tier", "timing", "flags"],
        "additionalProperties": False,
    }


## Tier probabilities
#
# A completion's "logprobs" is one dict per generated token: the top LOGPROBS
# candidate tokens and their log probabilities at that step.


def tier_distribution(step: dict[str, float]) -> tuple[dict[str, float], float]:
    """At one step: the probability of each tier word, renormalised over the
    three, and the total probability the step put on tier words at all."""
    mass_per_tier = dict.fromkeys(TIERS, 0.0)
    for token, logprob in step.items():
        word = token.strip().strip(':"').strip().lower()
        if word in mass_per_tier:
            mass_per_tier[word] += math.exp(logprob)
    mass = sum(mass_per_tier.values())
    probs = {tier: p / mass for tier, p in mass_per_tier.items()} if mass else {}
    return probs, mass


def gate_distribution(completion: dict) -> tuple[dict[str, float], float]:
    """The gate answers with one token, so its distribution is the first step's."""
    steps = completion.get("logprobs") or [{}]
    return tier_distribution(steps[0])


_TIER_KEY_RE = re.compile(r'"tier"\s*:\s*"?\s*$')


def rating_distribution(completion: dict) -> dict[str, float]:
    """The tier distribution where the model writes the tier value: the step
    right after `"tier": "`, found by replaying the chosen (highest probability)
    token of each step. Under the grammar that step allows only tier words.
    Falls back to the first step dominated by tier words; {} if there is none."""
    steps = completion.get("logprobs") or []
    written = ""
    for step in steps:
        if step and _TIER_KEY_RE.search(written):
            probs, mass = tier_distribution(step)
            if mass >= MIN_TIER_MASS:
                return probs
            break
        written += max(step, key=step.get) if step else ""
    return first_tier_distribution(steps)


def first_tier_distribution(steps: list[dict[str, float]]) -> dict[str, float]:
    for step in steps:
        probs, mass = tier_distribution(step)
        if mass >= MIN_TIER_MASS:
            return probs
    return {}


## Gate stage


def gate_passes(probs: dict[str, float], mass: float) -> bool:
    """Rate the domain unless the gate confidently answered green."""
    return mass < MIN_TIER_MASS or probs["yellow"] + probs["red"] >= GATE_THRESHOLD


def chunked(items: list, size: int) -> list[list]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def run_gate(
    rows: list[Row], prompts: dict[str, DomainPrompt]
) -> tuple[list[dict], dict[str, list[int]]]:
    """Gate every (entry, domain) in one domain-major batch. Returns the gate
    scores per row and, per domain, the indices of the rows to rate."""
    keys = list(prompts)
    if not rows:
        return [], {key: [] for key in keys}
    completions = chat(
        "gate",
        [render(prompts[key].gate, row) for key in keys for row in rows],
        temperature=0,
        max_tokens=1,
        logprobs=LOGPROBS,
    )
    scores: list[dict] = [{} for _ in rows]
    to_rate: dict[str, list[int]] = {key: [] for key in keys}
    for key, domain_completions in zip(keys, chunked(completions, len(rows))):
        for index, completion in enumerate(domain_completions):
            probs, mass = gate_distribution(completion)
            scores[index][key] = {"tier_probs": probs, "tier_mass": mass}
            if gate_passes(probs, mass):
                to_rate[key].append(index)
    return scores, to_rate


## Rating stage


def evidence_string(criterion: str, quotes: list[str]) -> str:
    """'<criterion>: "quote", "quote"', the format yellow_concerns and the
    inspect scripts read."""
    if not quotes:
        return criterion

    quoted = ", ".join(f'"{quote}"' for quote in quotes)
    return f"{criterion}: {quoted}"


def parse_rating(
    text: str, criterion_tiers: dict[str, str]
) -> tuple[DomainResult, list[str]]:
    """The rated fields from the model's JSON, plus errors as "<kind>: <detail>".
    The grammar guarantees valid JSON and enum values unless the output was cut
    off at RATING_MAX_TOKENS, which leaves the tier unknown."""
    try:
        rating = json.loads(text)
    except ValueError as exc:
        unknown = {"tier": TIER_UNKNOWN, "timing": None, "flags": [], "evidence": ""}
        return unknown, [f"json: {exc}"]

    errors: list[str] = []
    criterion = rating.get("criterion", "")
    tier = criterion_tiers.get(criterion, TIER_UNKNOWN)
    if tier == TIER_UNKNOWN:
        errors.append(f"unknown_criterion: {criterion!r}")
    elif rating.get("tier") != tier:
        model_tier = rating.get("tier")
        errors.append(
            f"tier_mismatch: model tier {model_tier!r}, "
            f"but criterion {criterion!r} is {tier}"
        )
    quotes = [quote for quote in rating.get("quotes", []) if quote]
    fields = {
        "tier": tier,
        "timing": rating.get("timing"),
        "flags": rating.get("flags", []),
        "evidence": evidence_string(criterion, quotes),
    }
    return fields, errors


def rated_result(
    completion: dict, criterion_tiers: dict[str, str]
) -> tuple[DomainResult, list[str]]:
    """A rated domain: the parsed fields plus the tier-token distribution and
    confidence = P(chosen tier)."""
    fields, errors = parse_rating(completion["text"], criterion_tiers)
    probs = rating_distribution(completion)
    result = {
        **fields,
        "tier_probs": probs,
        "confidence": probs.get(fields["tier"]),
        "scored": True,
    }
    return result, errors


def unscored_result(gate_score: dict) -> DomainResult:
    """A gated-out domain: green, carrying the gate's distribution so that
    every domain has a weight."""
    probs = gate_score["tier_probs"]
    return {
        "tier": "green",
        "tier_probs": probs,
        "confidence": probs.get("green"),
        "timing": None,
        "flags": [],
        "evidence": "",
        "scored": False,
    }


def run_rating(
    rows: list[Row], prompts: dict[str, DomainPrompt], to_rate: dict[str, list[int]]
) -> tuple[list[dict], list[dict], list[list[str]]]:
    """Rate the pairs that passed the gate, one batch per domain since each has
    its own grammar. Returns per-row results, raw model texts and errors, with
    only the rated domains filled in."""
    results: list[dict] = [{} for _ in rows]
    raw: list[dict] = [{} for _ in rows]
    errors: list[list[str]] = [[] for _ in rows]
    for key, prompt in prompts.items():
        indices = to_rate[key]
        completions = chat(
            f"rating {key}",
            [render(prompt.rating, rows[index]) for index in indices],
            temperature=TEMPERATURE,
            max_tokens=RATING_MAX_TOKENS,
            logprobs=LOGPROBS,
            json_schema=rating_schema(prompt.criterion_tiers),
        )
        for index, completion in zip(indices, completions):
            result, domain_errors = rated_result(completion, prompt.criterion_tiers)
            results[index][key] = result
            raw[index][key] = completion["text"]
            errors[index] += [f"{key}: {error}" for error in domain_errors]
    return results, raw, errors


def rate_entries(entries: pd.DataFrame) -> pd.DataFrame:
    """Gate every (entry, domain), rate the pairs that pass, and fill the rest
    in as unscored greens."""
    rows = entries.to_dict("records")
    prompts = domain_prompts()
    scores, to_rate = run_gate(rows, prompts)
    rated, raw, errors = run_rating(rows, prompts, to_rate)
    results = [
        {
            key: rated[index].get(key) or unscored_result(scores[index][key])
            for key in prompts
        }
        for index in range(len(rows))
    ]
    out = entries.copy()
    out["gate_scores"] = scores
    out["domain_results"] = results
    out["domain_incomplete"] = [
        any(result["tier"] == TIER_UNKNOWN for result in row.values())
        for row in results
    ]
    out["custom_prompt_errors"] = [";".join(row) for row in errors]
    out["domain_raw"] = raw
    return out


## Entry tier


def normalised_words(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", text.lower()).split())


def quote_grams(evidence: str, text: str) -> set[str]:
    """Word 4-grams of the evidence's quotes that occur in the entry. Without
    such a quote (none, or an invented one: the model sometimes copies the
    schema's example), the whole entry's: the claim can't show a separate
    concern."""
    entry = normalised_words(text)
    quotes = [
        quote
        for quote in map(normalised_words, re.findall(r'"([^"]+)"', evidence))
        if quote and quote in entry
    ]
    words = (" ".join(quotes) if quotes else entry).split()
    if len(words) < 4:
        return {" ".join(words)} if words else set()
    return {" ".join(words[start : start + 4]) for start in range(len(words) - 3)}


def yellow_concerns(evidence_by_domain: dict[str, str], text: str) -> list[list[str]]:
    """Group the yellow domains whose quotes share a 4-gram: one passage claimed
    by several domains is one concern, not several."""
    grams = {
        domain: quote_grams(evidence, text)
        for domain, evidence in evidence_by_domain.items()
    }
    groups: list[list[str]] = []
    for domain in evidence_by_domain:
        overlapping = [
            group
            for group in groups
            if any(grams[domain] & grams[other] for other in group)
        ]
        merged = [other for group in overlapping for other in group] + [domain]
        groups = [group for group in groups if group not in overlapping] + [merged]
    return groups


def effective_tier(result: DomainResult) -> str:
    """The domain's tier, raised to red when a scored green or yellow has
    P(red) >= RED_RAISE."""
    probability_red = (result.get("tier_probs") or {}).get("red", 0)
    raisable = result["tier"] in ("green", "yellow") and result.get("scored")
    return "red" if raisable and probability_red >= RED_RAISE else result["tier"]


def participant_age(row: Row) -> int | None:
    age = row.get("age")
    return None if age is None or pd.isna(age) else int(age)


def entry_tier(row: Row) -> dict:
    """The entry's final tier, the reasons for it, and the yellow concerns."""
    results: dict[str, DomainResult] = row["domain_results"]
    tiers = {domain: effective_tier(result) for domain, result in results.items()}
    raised = [
        domain for domain, tier in tiers.items() if tier != results[domain]["tier"]
    ]
    known_tiers = {
        domain: None if tier == TIER_UNKNOWN else Tier[tier]
        for domain, tier in tiers.items()
    }
    flags = frozenset(
        flag for result in results.values() for flag in result.get("flags") or []
    )
    yellow_evidence = {
        domain: results[domain].get("evidence") or ""
        for domain, tier in tiers.items()
        if tier == "yellow"
    }
    concerns = yellow_concerns(yellow_evidence, str(row["text"]))

    highest = rules.highest_tier(known_tiers)
    escalation = rules._escalate(highest.tier, flags, len(concerns))
    minimum = rules._minimums(participant_age(row), flags)
    incomplete = bool(row.get("domain_incomplete")) or highest.incomplete
    unknown = rules._unknown_floor(incomplete)
    final = rules.floor_tier(
        highest.tier,
        escalation.tier,
        minimum.tier,
        unknown.tier,
        escalation.reasons,
        minimum.reasons,
        unknown.reasons,
    )
    return {
        "entry_tier": final.tier.name,
        "entry_reasons": [f"red_probability:{domain}" for domain in raised]
        + final.reasons,
        "yellow_concerns": concerns,
    }


def with_entry_tier(results: pd.DataFrame) -> pd.DataFrame:
    """Add entry_tier, entry_reasons and yellow_concerns. Works on a saved run."""
    out = results.copy()
    tiers = [entry_tier(row) for row in results.to_dict("records")]
    for column in ("entry_tier", "entry_reasons", "yellow_concerns"):
        out[column] = [tier[column] for tier in tiers]
    return out


def run(entries: pd.DataFrame) -> pd.DataFrame:
    """Gate -> rating -> entry tier."""
    return with_entry_tier(rate_entries(entries))
