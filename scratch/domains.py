"""Domain stage with one Qwen3 model: gate -> scoped JSON -> adjudicate.

1. Gate: per (entry, domain), a one-token green/yellow/red answer. The domain
   passes if P(yellow) + P(red) >= GATE_THRESHOLD or the answer is off-format.
2. JSON: the rubric's domain prompts, scoped to one domain, for passed pairs
   only. Unpassed domains are green with "scored": False.
3. Adjudicate: one domain per prompt means one passage gets claimed by many
   domains, which the "2+ yellow domains -> red" rule counts several times. For
   entries with 2+ yellow/red domains, one prompt groups the claims into
   concerns with one primary domain each. The primary keeps the max tier of its
   group; the rest become green. Unparsable or omitted domains keep their result.
   Off by default.
4. Entry tier: rules.resolve, except P(red) >= RED_RAISE counts as red and
   "2+ yellow -> red" counts distinct quotes rather than domains.

Output columns match risk_pipeline.run_custom_prompts_stage (domain_results,
domain_incomplete, custom_prompt_errors) plus gate_scores, concerns and the raw
model output. Per-row inputs go last in every prompt and prompts run
domain-major, so vLLM's prefix cache reuses the rest.

    from scratch import domains
    res = domains.run(domains.read_sample_csv(100))
    domains.save(res, "domains_qwen3_4b_head100")
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from string import Template

import pandas as pd

import rules
from custom_prompts import (
    CUSTOM_PROMPT_MAX_TOKENS,
    _aggregate_with_tiebreak,
    _evidence_from_highest_tier,
)
from rubric import (
    ESCALATION_FLAGS,
    MINIMUM_FLAGS,
    TIER_UNKNOWN,
    TIMING_VOCABULARY,
    DomainParseError,
    Tier,
    _strip_decorations,
    parse_domain_response,
)
from scratch import lib, models, runner

MODEL: models.MODEL_KEY = "qwen3_4b_2507"
RUNS = 1  # JSON completions per prompt; 3 runs disagreed on 1.5% of ratings
TEMPERATURE = 0
GATE_THRESHOLD = 0.05  # low for recall: a false pass costs one JSON prompt
MIN_MASS = 0.5  # less probability than this on the tier words = off-format
# Rubric: "when uncertain between two tiers, assign the higher". A scored domain
# with P(red) >= RED_RAISE counts as red (key_domains5: catches the stepdad entry
# rated yellow with P(red) 0.27-0.32, no new false reds).
RED_RAISE = 0.25

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts"
OUT = ROOT / "out"  # gitignored
DOMAINS = "key_domains"  # folder under prompts/ with the domain prompts
SCHEMA = (
    "_shared/output_schema_v3.txt"  # v3: evidence (criterion + <=2 quotes) before tier
)

RISKY = ("yellow", "red")
RANK = {"green": 0, "yellow": 1, "red": 2}


def read_sample_csv(
    limit: int | None = None, path: str | Path = "data/sample_inputs.csv"
) -> pd.DataFrame:
    """Any CSV with id, text, participant_id, age, entry_date (extra columns are
    kept), e.g. "labelled_data/clinician_inputs.csv"."""
    df = pd.read_csv(ROOT / path)
    return df.head(limit) if limit is not None else df


def save(df: pd.DataFrame, name: str) -> Path:
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.jsonl"
    df.to_json(path, orient="records", lines=True, force_ascii=False)
    return path


def load(name: str) -> pd.DataFrame:
    return pd.read_json(OUT / f"{name}.jsonl", orient="records", lines=True)


def _chat(stage: str, prompts: list[str], **params) -> list[list]:
    if not prompts:
        return []
    t0 = time.monotonic()
    outs = runner.chat(
        MODEL,
        lib.create_messages(prompts),
        chat_template_kwargs={"enable_thinking": False},
        **params,
    )
    print(f"{stage}: {len(prompts)} prompts ({time.monotonic() - t0:.0f}s)")
    return outs


## Prompts

# The prelude's whole-entry instruction invites spillover in a one-domain prompt.
PRELUDE_MULTI_DOMAIN = (
    " An entry may meet criteria in more than one domain; record every domain "
    "that applies and assign the highest tier reached."
)
# The prompt files place inputs mid-prompt; they're stripped and re-added last.
_INPUTS_RE = re.compile(
    r" Participant age is \$age\. Entry date is \$entry_date\.|"
    r"\s*(?:---\s*)?(?:ENTRY TEXT:|## Input\s+Entry text:)\s*\$text\s*"
    r"(?:Participant age: \$age\s*Entry date: \$entry_date\s*)?$"
)
INPUTS = (
    "\n\n---\n\n## Input\n\nEntry text:\n$text\n\n"
    "Participant age: $age\nEntry date: $entry_date\n"
)
SCOPE = (
    "\n\nRate the entry above for {name} only. Content that belongs to other "
    "domains does not count toward this rating; if the entry contains nothing "
    "relevant to {name}, "
)
GATE_ASK = "answer green. Reply with exactly one lowercase word: green, yellow, or red."
JSON_ASK = (
    "use tier green. Respond with the JSON object described in the output schema."
)


def _read(path: str) -> str:
    return (PROMPTS / path).read_text(encoding="utf-8")


def extract_domain_name(text: str) -> str | None:
    matches = re.search(r"^# Domain: (.+)$", text, re.MULTILINE)
    return None if matches is None else matches[1].strip()


def domain_names() -> dict[str, str]:
    """Domain key (file stem, e.g. '01_suicidality') -> '# Domain:' heading."""
    return {
        p.stem: name
        for p in sorted((PROMPTS / DOMAINS).glob("*.txt"))
        if (name := extract_domain_name(p.read_text())) is not None
    }


def flag_definitions() -> str:
    """output_schema.txt names the flags; prompts/flags.txt defines them."""
    lines = [
        line.strip()
        for line in _read("flags.txt").splitlines()
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


def templates() -> tuple[dict[str, str], dict[str, str]]:
    """(gate, JSON) templates per domain, with $text/$age/$entry_date last."""
    prelude = _read("_shared/rubric_prelude.txt")
    if PRELUDE_MULTI_DOMAIN not in prelude:
        raise ValueError("rubric_prelude.txt changed; update PRELUDE_MULTI_DOMAIN")
    prelude = prelude.replace(PRELUDE_MULTI_DOMAIN, "")
    schema = _read(SCHEMA).rstrip() + flag_definitions()
    gate, js = {}, {}
    for key, name in domain_names().items():
        body, n = _INPUTS_RE.subn("", _read(f"{DOMAINS}/{key}.txt"))
        if n not in (1, 2) or re.search(r"\$(text|age|entry_date)\b", body):
            raise ValueError(f"{key}: unexpected placement of inputs")
        head = f"{prelude}\n\n{body.rstrip()}"
        scope = SCOPE.format(name=name.replace("$", "$$"))
        gate[key] = f"{head}{INPUTS}{scope}{GATE_ASK}"
        js[key] = f"{head}\n\n{schema}{INPUTS}{scope}{JSON_ASK}"
    return gate, js


def render(template: str, row: dict) -> str:
    return Template(template).substitute(
        text=row["text"], age=str(row["age"]), entry_date=str(row["entry_date"])
    )


## Stages


def _tier_dist(step: dict[str, float]) -> tuple[dict[str, float], float]:
    """(P per tier word renormalised, total P on tier words) at one token."""
    raw = dict.fromkeys(RANK, 0.0)
    for token, lp in step.items():
        word = token.strip().strip(':"').strip().lower()
        if word in raw:
            raw[word] += math.exp(lp)
    mass = sum(raw.values())
    return {k: v / mass for k, v in raw.items()} if mass else {}, mass


def tier_probs(completion: dict) -> tuple[dict[str, float], float]:
    """Gate: the tier distribution at the first (only) token."""
    return _tier_dist((completion.get("logprobs") or [{}])[0])


_TIER_KEY_RE = re.compile(r'"tier"\s*:\s*"?\s*$')


def json_tier_probs(completion: dict) -> dict[str, float]:
    """JSON: the tier distribution where the model writes the tier value: the
    step right after `"tier": "`, located by replaying the chosen (highest
    probability) token of each step. Falls back to the first step whose
    candidates are mostly tier words. {} if neither is found."""
    steps = completion.get("logprobs") or []
    text = ""
    for step in steps:
        if step and _TIER_KEY_RE.search(text):
            probs, mass = _tier_dist(step)
            if mass >= MIN_MASS:
                return probs
            break
        text += max(step, key=step.get) if step else ""
    for step in steps:
        probs, mass = _tier_dist(step)
        if mass >= MIN_MASS:
            return probs
    return {}


# Each domain prompt ends with a "## Criteria" list: one line per criterion,
# "- <tier>: <name>". The names are the only values the grammar allows for the
# JSON "criterion" field, so a domain can't cite another domain's criterion, and
# the tier follows from the criterion in code.
_CRITERION_RE = re.compile(r"^- (green|yellow|red): (.+?)\s*$", re.MULTILINE)


def criteria(key: str) -> dict[str, str]:
    """Criterion name -> tier, from the domain prompt's '## Criteria' section."""
    parts = _read(f"{DOMAINS}/{key}.txt").split("\n## Criteria", 1)
    if len(parts) < 2:
        raise ValueError(f"{key}: no '## Criteria' section")
    section = parts[1].split("\n## ", 1)[0]
    found = {name: tier for tier, name in _CRITERION_RE.findall(section)}
    if set(found.values()) != set(RANK):
        raise ValueError(f"{key}: criteria must cover green, yellow and red")
    return found


def json_schema(key: str) -> dict:
    """Decoding is constrained to this schema (vLLM structured outputs): valid
    JSON, these fields in this order, enum values only. The prompt's output
    schema (prompts/_shared/output_schema_v3.txt) describes the same fields."""
    return {
        "type": "object",
        "properties": {
            "domain": {"type": "string", "maxLength": 60},
            "criterion": {"enum": list(criteria(key))},
            # Length cap: the grammar otherwise allows a string to run on until
            # max_tokens (labelled_0003: a 2k-char string and a tab loop).
            "quotes": {
                "type": "array",
                "items": {"type": "string", "maxLength": 200},
                "maxItems": 2,
            },
            "tier": {"enum": list(RANK)},
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


def _parse(text: str, tiers: dict[str, str]):
    """Fold criterion + quotes into the rubric parser's evidence string. The
    tier is the criterion's; a different model tier is recorded as an error."""
    mismatch = None
    try:
        obj = json.loads(_strip_decorations(text))
        quotes = ", ".join(f'"{q}"' for q in obj.pop("quotes", []) if q)
        criterion = obj.pop("criterion", "")
        obj["evidence"] = f"{criterion}: {quotes}" if quotes else criterion
        expected = tiers.get(criterion)
        if expected and obj.get("tier") != expected:
            mismatch = (
                f"model tier {obj.get('tier')!r} for {expected} criterion {criterion!r}"
            )
            obj["tier"] = expected
        text = json.dumps(obj, ensure_ascii=False)
    except (ValueError, AttributeError, TypeError):
        pass  # not constrained output (or truncated): let the parser report it
    parsed = parse_domain_response(text)
    if mismatch:
        parsed.errors.append(DomainParseError("tier_mismatch", mismatch))
    return parsed


def aggregate(
    domain: str, completions: list[dict], tiers: dict[str, str]
) -> tuple[dict, list[str]]:
    """Worst-of-N, as custom_prompts.run_domain_prompts, plus tier_probs (mean
    over runs of the tier-token distribution) and confidence = P(chosen tier)."""
    runs = [_parse(c["text"], tiers) for c in completions]
    for r in runs:
        r.domain = domain
    agg = _aggregate_with_tiebreak(runs)
    dists = [p for c in completions if (p := json_tier_probs(c))]
    probs = {t: sum(p[t] for p in dists) / len(dists) for t in RANK} if dists else {}
    tier = TIER_UNKNOWN if agg.tier is None else agg.tier.name
    result = {
        "tier": tier,
        "tier_probs": probs,
        "confidence": probs.get(tier),
        "timing": agg.timing,
        "flags": agg.flags,
        "evidence": _evidence_from_highest_tier(runs),
        "scored": True,
    }
    return result, [f"{domain}: {e}" for e in agg.run_errors]


def run_json(df: pd.DataFrame, runs: int = RUNS) -> pd.DataFrame:
    """Gate every (entry, domain), then JSON-score the pairs that pass."""
    rows = df.to_dict("records")
    gate_t, json_t = templates()
    pairs = [(r, d) for d in json_t for r in range(len(rows))]  # domain-major

    gate_outs = _chat(
        "gate",
        [render(gate_t[d], rows[r]) for r, d in pairs],
        temperature=0,
        max_tokens=1,
        logprobs=20,
    )
    scores: list[dict] = [{} for _ in rows]
    passed = []
    for (r, d), o in zip(pairs, gate_outs):
        probs, mass = tier_probs(o[0])
        scores[r][d] = {"tier_probs": probs, "tier_mass": mass}
        if mass < MIN_MASS or probs["yellow"] + probs["red"] >= GATE_THRESHOLD:
            passed.append((r, d))

    # One call per domain: each has its own grammar. `passed` is domain-major,
    # so concatenating per-domain outputs keeps them aligned with it.
    tiers = {d: criteria(d) for d in json_t}
    json_outs: list[list] = []
    for d in json_t:
        json_outs += _chat(
            f"JSON {d}",
            [render(json_t[d], rows[r]) for r, dd in passed if dd == d],
            n=runs,
            temperature=TEMPERATURE,
            max_tokens=256,
            logprobs=5,
            json_schema=json_schema(d),
        )
    unscored = {
        "tier": "green",
        "timing": None,
        "flags": [],
        "evidence": "",
        "scored": False,
    }
    # Unscored domains carry the gate's distribution, so every domain has a weight.
    results = [
        {
            d: {
                **unscored,
                "tier_probs": scores[r][d]["tier_probs"],
                "confidence": scores[r][d]["tier_probs"].get("green"),
            }
            for d in json_t
        }
        for r in range(len(rows))
    ]
    raw: list[dict] = [{} for _ in rows]
    errors: list[list[str]] = [[] for _ in rows]
    for (r, d), o in zip(passed, json_outs):
        raw[r][d] = [c["text"] for c in o]
        results[r][d], errs = aggregate(d, o, tiers[d])
        errors[r] += errs

    out = df.copy()
    out["gate_scores"] = scores
    out["domain_results"] = results
    out["domain_incomplete"] = [
        any(v["tier"] == TIER_UNKNOWN for v in res.values()) for res in results
    ]
    out["custom_prompt_errors"] = [";".join(e) for e in errors]
    out["domain_raw"] = raw
    return out


## Entry tier


def _words(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", s.lower()).split())


def _quote_grams(evidence: str, text: str) -> set[str]:
    """Word 4-grams of the evidence's quotes that occur in the entry. Without
    such a quote (none, or invented: the model sometimes copies the schema's
    example quote), the whole entry: the claim can't show a separate concern."""
    entry = _words(text)
    quotes = [
        q for q in map(_words, re.findall(r'"([^"]+)"', evidence)) if q and q in entry
    ]
    words = (" ".join(quotes) if quotes else entry).split()
    if len(words) < 4:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + 4]) for i in range(len(words) - 3)}


def yellow_concerns(evidence: dict[str, str], text: str) -> list[list[str]]:
    """Group yellow domains whose quotes share a 4-gram: one passage claimed by
    several domains is one concern, not several."""
    grams = {d: _quote_grams(e or "", text) for d, e in evidence.items()}
    groups: list[list[str]] = []
    for d in evidence:
        hits = [g for g in groups if any(grams[d] & grams[o] for o in g)]
        merged = [o for g in hits for o in g] + [d]
        groups = [g for g in groups if g not in hits] + [merged]
    return groups


def entry_tier(row: dict) -> dict:
    """rules.resolve with two changes against spillover:
    - a scored domain with P(red) >= RED_RAISE counts as red;
    - "2+ yellow -> red" counts yellow concerns (distinct quotes), not domains.
    Returns the final tier, the reasons, and the yellow concerns."""
    results = row["domain_results"]
    tiers: dict[str, Tier | None] = {}
    raised = []
    for d, r in results.items():
        t = r["tier"]
        if (
            t in ("green", "yellow")
            and r.get("scored")
            and ((r.get("tier_probs") or {}).get("red", 0) >= RED_RAISE)
        ):
            t = "red"
            raised.append(d)
        tiers[d] = None if t == TIER_UNKNOWN else Tier[t]
    flags = frozenset(f for r in results.values() for f in r.get("flags") or [])
    age = row.get("age")
    age = int(age) if age is not None and not pd.isna(age) else None

    ht = rules.highest_tier(tiers)
    concerns = yellow_concerns(
        {
            d: results[d].get("evidence") or ""
            for d, t in tiers.items()
            if t is Tier.yellow
        },
        str(row["text"]),
    )
    esc = rules._escalate(ht.tier, flags, len(concerns))
    mins = rules._minimums(age, flags)
    unk = rules._unknown_floor(bool(row.get("domain_incomplete")) or ht.incomplete)
    ft = rules.floor_tier(
        ht.tier, esc.tier, mins.tier, unk.tier, esc.reasons, mins.reasons, unk.reasons
    )
    return {
        "entry_tier": ft.tier.name,
        "entry_reasons": [f"red_probability:{d}" for d in raised] + ft.reasons,
        "yellow_concerns": concerns,
    }


def with_entry_tier(res: pd.DataFrame) -> pd.DataFrame:
    """Add entry_tier, entry_reasons and yellow_concerns. Works on a saved run."""
    out = res.copy()
    tiers = [entry_tier(row) for row in res.to_dict("records")]
    for col in ("entry_tier", "entry_reasons", "yellow_concerns"):
        out[col] = [t[col] for t in tiers]
    return out


def run(df: pd.DataFrame, runs: int = RUNS) -> pd.DataFrame:
    """Gate -> JSON -> entry tier."""
    res = run_json(df, runs)
    return with_entry_tier(res)
