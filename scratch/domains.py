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

from custom_prompts import (
    CUSTOM_PROMPT_MAX_TOKENS,
    _aggregate_with_tiebreak,
    _evidence_from_highest_tier,
)
from rubric import (
    ESCALATION_FLAGS,
    MINIMUM_FLAGS,
    TIER_UNKNOWN,
    _extract_json,
    _strip_decorations,
    parse_domain_response,
)
from scratch import lib, models, runner

MODEL: models.MODEL_KEY = "qwen3_4b"
RUNS = 1  # JSON completions per prompt; 3 runs disagreed on 1.5% of ratings
TEMPERATURE = 0.3
GATE_THRESHOLD = 0.05  # low for recall: a false pass costs one JSON prompt
MIN_MASS = 0.5  # less probability than this on the tier words = off-format

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts"
OUT = ROOT / "out"  # gitignored
DOMAINS = "domains_next"  # folder under prompts/ with the domain prompts

RISKY = ("yellow", "red")
RANK = {"green": 0, "yellow": 1, "red": 2}


def read_sample_csv(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "sample_inputs.csv")
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


def domain_names() -> dict[str, str]:
    """Domain key (file stem, e.g. '01_suicidality') -> '# Domain:' heading."""
    return {
        p.stem: (re.search(r"^# Domain: (.+)$", p.read_text(), re.MULTILINE)[1]).strip()
        for p in sorted((PROMPTS / DOMAINS).glob("*.txt"))
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
        "\n\n## Flag definitions\n\nSet a flag only when the entry meets its "
        "definition. Flags describe the participant, not other people.\n\n"
        + "\n".join(f"- {line}" for line in lines).replace("$", "$$")
    )


def templates() -> tuple[dict[str, str], dict[str, str]]:
    """(gate, JSON) templates per domain, with $text/$age/$entry_date last."""
    prelude = _read("_shared/rubric_prelude.txt")
    if PRELUDE_MULTI_DOMAIN not in prelude:
        raise ValueError("rubric_prelude.txt changed; update PRELUDE_MULTI_DOMAIN")
    prelude = prelude.replace(PRELUDE_MULTI_DOMAIN, "")
    schema = _read("_shared/output_schema.txt").rstrip() + flag_definitions()
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


def tier_probs(completion: dict) -> tuple[dict[str, float], float]:
    """(P per tier word renormalised, total P on tier words) from the first token."""
    steps = completion.get("logprobs") or [{}]
    raw = dict.fromkeys(RANK, 0.0)
    for token, lp in steps[0].items():
        word = token.strip().lstrip(":").strip().lower()
        if word in raw:
            raw[word] += math.exp(lp)
    mass = sum(raw.values())
    return {k: v / mass for k, v in raw.items()} if mass else {}, mass


def aggregate(domain: str, completions: list[str]) -> tuple[dict, list[str]]:
    """Worst-of-N, as custom_prompts.run_domain_prompts."""
    runs = [parse_domain_response(c) for c in completions]
    for r in runs:
        r.domain = domain
    agg = _aggregate_with_tiebreak(runs)
    result = {
        "tier": TIER_UNKNOWN if agg.tier is None else agg.tier.name,
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

    json_outs = _chat(
        "JSON",
        [render(json_t[d], rows[r]) for r, d in passed],
        n=runs,
        temperature=TEMPERATURE,
        max_tokens=CUSTOM_PROMPT_MAX_TOKENS,
    )
    unscored = {
        "tier": "green",
        "timing": None,
        "flags": [],
        "evidence": "",
        "scored": False,
    }
    results = [{d: unscored for d in json_t} for _ in rows]
    raw: list[dict] = [{} for _ in rows]
    errors: list[list[str]] = [[] for _ in rows]
    for (r, d), o in zip(passed, json_outs):
        raw[r][d] = o
        results[r][d], errs = aggregate(d, o)
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


def _slug(key: str) -> str:
    return re.sub(r"^\d+_", "", key)


def adjudication_prompt(row: dict, claims: dict[str, dict]) -> str:
    names = domain_names()
    lines = "\n".join(
        f"- {_slug(d)} ({names[d]}): claimed with evidence "
        f"{json.dumps(c.get('evidence') or '', ensure_ascii=False)}"
        for d, c in claims.items()
    )
    return (
        f"{_read('adjudication.txt').rstrip()}\n\n---\n\n## Candidate domains\n\n"
        f"{lines}\n\n## Entry\n\nEntry text:\n{row['text']}\n\n"
        f"Participant age: {row['age']}\nEntry date: {row['entry_date']}\n"
    )


def merge(results: dict, claims: list[str], text: str) -> tuple[dict, list, list[str]]:
    """Apply adjudication output: (new domain_results, concerns, errors)."""
    by_slug = {_slug(d): d for d in claims}
    try:
        raw = _extract_json(_strip_decorations(text)).get("concerns")
    except (ValueError, AttributeError) as exc:
        return results, [], [f"adjudication unparsable: {exc}"]

    out = dict(results)
    concerns, seen = [], set()
    for c in raw if isinstance(raw, list) else []:
        c = c if isinstance(c, dict) else {}
        primary = by_slug.get(str(c.get("domain")).strip())
        if primary is None or primary in seen:
            continue
        also = c.get("also") if isinstance(c.get("also"), list) else []
        also = list(dict.fromkeys(by_slug.get(str(a).strip()) for a in also))
        also = [d for d in also if d and d != primary and d not in seen]
        group = [primary, *also]
        seen.update(group)
        top = results[max(group, key=lambda d: RANK.get(results[d]["tier"], 0))]
        out[primary] = {
            **results[primary],
            "tier": top["tier"],
            "timing": top["timing"],
            "flags": sorted({f for d in group for f in results[d]["flags"] or []}),
            "evidence": c.get("quote") or top["evidence"],
            "merged": also,
        }
        for d in also:
            out[d] = {
                **results[d],
                "tier": "green",
                "timing": None,
                "flags": [],
                "merged_into": primary,
                "original_tier": results[d]["tier"],
            }
        concerns.append({"quote": c.get("quote", ""), "domain": primary, "also": also})

    missing = [d for d in claims if d not in seen]
    errors = [f"adjudication kept unassigned {missing}"] if missing else []
    return out, concerns, errors


def adjudicate(res: pd.DataFrame) -> pd.DataFrame:
    """Adjudicate entries with 2+ yellow/red domains. Works on a saved run_json."""
    rows = res.to_dict("records")
    todo = {}
    for i, row in enumerate(rows):
        claims = [d for d, r in row["domain_results"].items() if r["tier"] in RISKY]
        if len(claims) >= 2:
            todo[i] = claims
    outs = _chat(
        "adjudicate",
        [
            adjudication_prompt(rows[i], {d: rows[i]["domain_results"][d] for d in c})
            for i, c in todo.items()
        ],
        temperature=0,
        max_tokens=512,
    )

    out = res.copy()
    out["domain_results_pre_adjudication"] = res["domain_results"]
    out["concerns"] = None
    out["adjudication_raw"] = ""
    for (i, claims), o in zip(todo.items(), outs):
        new, concerns, errs = merge(rows[i]["domain_results"], claims, o[0])
        idx = res.index[i]
        out.at[idx, "domain_results"] = new
        out.at[idx, "concerns"] = concerns
        out.at[idx, "adjudication_raw"] = o[0]
        out.at[idx, "custom_prompt_errors"] = ";".join(
            e for e in [rows[i]["custom_prompt_errors"], *errs] if e
        )
    return out


def run(df: pd.DataFrame, runs: int = RUNS) -> pd.DataFrame:
    return adjudicate(run_json(df, runs))
