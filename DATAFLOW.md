# Data Flow

Text enters one row at a time. Each row is independently scored: guard fan-out, 21
domain prompts (3 runs each), deterministic rule phases, then two LLM consolidation
steps. Nothing aborts on failure — every input row produces an output row.

## Flow

```mermaid
flowchart TD
    subgraph S1["1 · Input validation"]
        direction TB
        A(["Input CSV with id, text, age, entry_date"]) --> B["Validate required columns<br/>and data types"]
    end

    subgraph S2["2 · Guard fan-out — every guard receives the same text"]
        direction TB
        B --> G1["ShieldGemma<br/>4 calls, one per policy:<br/>dangerous_content, harassment,<br/>hate_speech, sexually_explicit"]
        B --> G2["Llama Guard 3<br/>1 call"]
        B --> G3["Granite Guardian 3.3<br/>1 call"]
        B --> G4["Qwen3Guard-Gen<br/>1 call"]
    end

    subgraph S3["3 · Domain prompts — 21 domains, N runs each (default N=3)"]
        direction TB
        G1 --> D
        G2 --> D
        G3 --> D
        G4 --> D
        D["Per domain: render prompt with text, age, entry_date<br/>Run N times at temperature 0.3<br/>Aggregate: severity-max tier, timing tie-break, flag union"]
    end

    subgraph S4["4 · Deterministic rules"]
        direction TB
        D --> R1["highest_tier: max known domain tier"]
        R1 --> R2["escalation: Yellow->Red if escalation flags<br/>or multi-domain yellow"]
        R2 --> R3["minimums: Yellow floor for minors+abuse<br/>or lifetime-history flags"]
        R3 --> R4["unknown floor: Yellow if any domain unknown/empty<br/>emits floor_reasons (e.g. domain_unavailable)"]
        R4 --> R5["floor_tier: max of all phases"]
    end

    subgraph S5["5 · Consolidation — LLM with monotonic clamp"]
        direction TB
        R5 --> C1["Build prompt: text + context + guards<br/>+ domains + rule results"]
        C1 --> C2["Consolidation call<br/>1 per row, floor = floor_tier"]
    end

    subgraph S6["6 · Final update — LLM with monotonic clamp"]
        direction TB
        C2 --> F1["Build prompt: same blocks + consolidated tier"]
        F1 --> F2["Final update call<br/>1 per row, floor = consolidated tier"]
    end

    subgraph S7["7 · Delivery"]
        direction TB
        F2 --> O1["Emit row: id, text, guard columns,<br/>domain_results JSON, tier columns,<br/>error columns"]
        O1 --> O2(["scores.csv<br/>one row per input text"])
        O2 --> O3["Terminal summary<br/>flagged and error counts, tier distribution"]
    end
```

## Notes

- **72 calls per entry at defaults.** Guards contribute 7 (ShieldGemma = 4). Domain
  prompts contribute 63 (21 domains × 3 runs). Consolidation and final update add
  2 more. Retries on parse failure add further calls.
- **Failures are data, not exceptions.** A guard that cannot be parsed writes its
  message to its `*_error` column. A domain-prompt run failure is recorded in
  `custom_prompt_errors` and leaves that domain as `unknown`. Consolidation and
  final-update failures fall back to the floor tier. The row is always emitted.
- **14-day cross-entry rule is downstream.** The pipeline processes entries
  independently. The rubric rule "same domain rated Yellow in 2+ entries within
  14 days escalates to Red" must be applied after the pipeline, across all rows
  for a given participant.
- **Monotonic clamp.** Both consolidation and final update enforce monotonicity:
  the LLM may raise the tier above the floor but never lower it. If the LLM
  proposes a lower tier, the floor is used instead.
- **Order of operations per row** is strictly serial: all guards are queried
  before domain prompts begin; domain prompts complete before deterministic rules
  run; deterministic rules complete before consolidation; consolidation completes
  before final update.
- **highest_domain_tier uses enum ordering.** Tier comparison uses the enum order green(0) < yellow(1) < red(2), not alphabetical string ordering. When no domains return a known tier, the column is `unknown`.
- **domain_incomplete is honored from the custom-prompt stage.** If the custom-prompt stage fails or returns empty results, `domain_incomplete` is set to true before deterministic rules run. The unknown-floor phase then raises to Yellow.
- **Three canonical tiers.** The system uses `green`, `yellow`, `red`. A legacy
  column `final_risk` maps to `no_low_risk`, `concerning`, `high_risk` for
  backward compatibility.
