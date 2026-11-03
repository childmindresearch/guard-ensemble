# Data Flow

Text enters one row at a time. Each row is independently scored: guard fan-out, one
Kev decision-model request (51 questions from YAML), deterministic rule phases,
then two LLM consolidation steps. Nothing aborts on failure — every input row
produces an output row.

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

    subgraph S3["3 · Domain assessment — single Kev request, 51 questions from YAML"]
        direction TB
        G1 --> K
        G2 --> K
        G3 --> K
        G4 --> K
        K["Load YAML from prompts/kev/<br/>Build 51 questions: 21 tier + 21 timing + 9 flags<br/>Single POST to Kev server<br/>Response parsed into domain_results, global_flags, flag_probabilities"]
    end

    subgraph S4["4 · Deterministic rules"]
        direction TB
        K --> R1["highest_tier: max known domain tier"]
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
        F2 --> O1["Emit row: id, text, guard columns,<br/>domain_results JSON, global_flags, global_flag_probabilities,<br/>tier columns, error columns"]
        O1 --> O2(["scores.csv<br/>one row per input text"])
        O2 --> O3["Terminal summary<br/>flagged and error counts, tier distribution"]
    end
```

## Notes

- **10 calls per entry.** Guards contribute 7 (ShieldGemma = 4). Kev domain
  assessment contributes 1 (single request bundling 51 questions). Consolidation
  and final update add 2 more. Retries on transient failures add further calls.
- **Failures are data, not exceptions.** A guard that cannot be parsed writes its
  message to its `*_error` column. A Kev request failure marks all domains as
  `unknown` (all-or-nothing: there is no partial-domain fallback). Consolidation
  and final-update failures fall back to the floor tier. The row is always emitted.
- **14-day cross-entry rule is downstream.** The pipeline processes entries
  independently. The rubric rule "same domain rated Yellow in 2+ entries within
  14 days escalates to Red" must be applied after the pipeline, across all rows
  for a given participant.
- **Monotonic clamp.** Both consolidation and final update enforce monotonicity:
  the LLM may raise the tier above the floor but never lower it. If the LLM
  proposes a lower tier, the floor is used instead.
- **Order of operations per row** is strictly serial: all guards are queried
  before Kev domain assessment begins; Kev completes before deterministic rules
  run; deterministic rules complete before consolidation; consolidation completes
  before final update.
- **highest_domain_tier uses enum ordering.** Tier comparison uses the enum order green(0) < yellow(1) < red(2), not alphabetical string ordering. When no domains return a known tier, the column is `unknown`.
- **domain_incomplete is honored from the Kev stage.** If the Kev stage fails or returns empty results, `domain_incomplete` is set to true before deterministic rules run. The unknown-floor phase then raises to Yellow.
- **Three canonical tiers.** The system uses `green`, `yellow`, `red`. A legacy
  column `final_risk` maps to `no_low_risk`, `concerning`, `high_risk` for
  backward compatibility.
- **YAML is the runtime source.** Domain questions are defined in YAML files under
  `prompts/kev/` (`_shared.yaml` + 21 domain files). There is no compilation or
  prompt-generation step — YAML is loaded and validated at runtime.
- **Kev produces no free-text evidence.** The `domain_results` JSON contains
  tier, timing, scores, probabilities, and confidence fields. Global flags are
  emitted separately in `global_flags` and `global_flag_probabilities` columns.
- **Kev confidence is calibrated model output.** The `tier_confidence` and
  `timing_confidence` fields report the model's calibrated confidence. These are
  not externally measured accuracy scores.
