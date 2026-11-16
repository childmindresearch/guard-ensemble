import re
from dataclasses import dataclass
from pathlib import Path
from string import Template

from guard_ensemble.domains.config import (
    DOMAIN_PROMPTS_DIR,
    ESCALATION_FLAGS,
    FLAGS_TXT,
    MINIMUM_FLAGS,
    OUTPUT_SCHEMA_TXT,
    PRELUDE_TXT,
)
from guard_ensemble.domains.types import TIERS, Row

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

DOMAIN_NAME_RE = re.compile(r"^# Domain: (.+)$", re.MULTILINE)
CRITERION_RE = re.compile(r"^- (green|yellow|red): (.+?)\s*$", re.MULTILINE)
PLACEHOLDER_RE = re.compile(r"\$(text|age|entry_date)\b")


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
    return read_text(PRELUDE_TXT)


def flag_definitions() -> str:
    """The flag names the schema allows, each with its prompts/flags.txt definition."""
    lines = [
        line.strip()
        for line in read_text(FLAGS_TXT).splitlines()
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
    return read_text(OUTPUT_SCHEMA_TXT).rstrip() + flag_definitions()


def domain_name(prompt_text: str, key: str) -> str:
    match = DOMAIN_NAME_RE.search(prompt_text)
    if match is None:
        raise ValueError(f"{key}: no '# Domain:' heading")
    return match[1].strip()


def body_without_inputs(prompt_text: str, key: str) -> str:
    """The prompt up to its "## Input" block, which must be the standard one."""
    body, separator, inputs = prompt_text.rpartition("\n## Input")
    if not separator or ("## Input" + inputs).split() != INPUTS_BLOCK.split():
        raise ValueError(f"{key}: the prompt must end with the standard input block")
    if PLACEHOLDER_RE.search(body):
        raise ValueError(f"{key}: input placeholders outside the '## Input' block")
    return body.rstrip()


def criterion_tiers(prompt_text: str, key: str) -> dict[str, str]:
    """Criterion name -> tier, from the prompt's "## Criteria" section. The
    names are the only values the rating grammar allows for "criterion"."""
    _, separator, rest = prompt_text.partition("\n## Criteria")
    if not separator:
        raise ValueError(f"{key}: no '## Criteria' section")
    section = rest.split("\n## ", 1)[0]
    found = {name: tier for tier, name in CRITERION_RE.findall(section)}
    if frozenset(found.values()) != TIERS:
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
        for path in sorted(DOMAIN_PROMPTS_DIR.glob("*.txt"))
    }


def render(template: str, row: Row) -> str:
    return Template(template).substitute(
        text=row["text"], age=str(row["age"]), entry_date=str(row["entry_date"])
    )
