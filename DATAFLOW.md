# Data Flow

Text enters one row at a time. Each row is independently scored by four guardmodels, their verdicts are merged into a single record, and a deliberator model
turns those verdicts into one final risk level. Nothing aborts on failure: every
input row produces an output row.

## Flow

```mermaid
flowchart TD
    subgraph S1["1 · Input"]
        direction TB
        A(["Input CSV with id + text<br/>or built-in demo texts"]) --> B["Build records<br/>one record per input text"]
    end

    subgraph S2["2 · Guard fan-out — every guard receives the same text"]
        direction TB
        B --> G1["ShieldGemma<br/>4 calls, one per policy:<br/>dangerous_content, harassment,<br/>hate_speech, sexually_explicit"]
        B --> G2["Llama Guard 3<br/>1 call"]
        B --> G3["Granite Guardian 3.3<br/>1 call"]
        B --> G4["Qwen3Guard-Gen<br/>1 call"]
    end

    subgraph S3["3 · Normalize and parse — applied to each guard response"]
        direction TB
        G1 --> C
        G2 --> C
        G3 --> C
        G4 --> C
        C["strip_think: drop think blocks<br/>then parse the guard's own format"]
    end

    C -->|"parsed"| R["Guard ratings<br/>label, unsafe flag, category"]
    C -->|"parse fails"| E["Capture error<br/>guard_error = message<br/>errors = model:message"]

    subgraph S4["4 · Row assembly"]
        direction TB
        R --> M["Merge ratings into the row<br/>ShieldGemma: 4 policy verdicts collapse<br/>to 1 verdict, plus 4 per-policy flags"]
        E --> M
        M --> M2["Qwen Controversial is preserved,<br/>not folded into safe or unsafe"]
    end

    subgraph S5["5 · Deliberation"]
        direction TB
        M2 --> D1["Collect rating columns<br/>excluding id, text, errors,<br/>deliberation_risk and any error column"]
        D1 --> D2{"Any guard<br/>failed?"}
        D2 -->|"yes"| D3["Add _guard_models_unavailable<br/>listing failed guards by name"]
        D2 -->|"no"| D4["Build prompt:<br/>original text + guard ratings"]
        D3 --> D4
        D4 --> D5["Deliberation call<br/>1 per row"]
        D5 --> D6{"Parse"}
        D6 -->|"ok"| D7["Risk level:<br/>high_risk / concerning / no_low_risk"]
        D6 -->|"fails"| D8["deliberation_risk =<br/>error: message"]
    end

    subgraph S6["6 · Delivery"]
        direction TB
        D7 --> O1["Emit row:<br/>id, text, all guard columns,<br/>error columns, deliberation_risk"]
        D8 --> O1
        O1 --> O2(["scores.csv<br/>one row per input text"])
        O2 --> O3["Terminal summary<br/>flagged and error counts per guard,<br/>plus the risk-level distribution"]
    end
```

## Notes

- **Seven guard calls per row.** ShieldGemma accounts for four of them, one per
  policy, because it evaluates exactly one policy per request. The other three
  guards take one call each. Deliberation adds one more.
- **Three canonical outcomes.** The deliberator returns exactly one of
  `high_risk`, `concerning`, or `no_low_risk`.
- **`Controversial` is a real value.** Qwen3Guard can return `Controversial`. It
  is preserved in `qwen_guard_label` and its `qwen_guard_unsafe` flag is `0`, so
  it is never silently folded into safe or unsafe.
- **Failures are data, not exceptions.** A guard that cannot be parsed writes its
  message to its own `*_error` column and contributes to the `errors` column; it
  is then named in `_guard_models_unavailable` so the deliberator knows the rating
  is missing rather than negative. A deliberation failure is written in-cell as
  `error: <message>`. The row is still emitted.
- **Order of operations per row** is strictly serial: all guards are queried
  before deliberation begins, so the deliberator always sees a complete set of
  guard verdicts for that row.
Two notes on the content. The -->|"parsed"| edge labels are quoted and pipe-free — pipes inside quoted node labels are a common Mermaid parse failure, so the risk levels use slashes instead. And I deliberately kept infrastructure out: the one-model-at-a-time reload that makes runs slow is real, but it affects latency, not the data, so it belongs in a different document.
