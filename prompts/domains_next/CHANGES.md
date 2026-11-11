# Changes from prompts/domains (for clinician review)

Tier criteria are unchanged except where marked **[criterion]**. Most changes
add scope boundaries: each domain prompt is run on its own, so without them a
model rates every domain by the entry's overall alarm (on a 100-entry sample,
44 of 45 multi-domain entries had several domains citing the same sentence).

## All domains

- New **Scope** section: what the domain covers, an explicit "rate Green if the
  entry has no such content, even if it describes other risks", and the
  neighbouring domains that commonly get confused with it.
- Domains 01-05 converted to the same layout as 06-21 (Tier Definitions
  headings, bulleted examples, Input section). Wording of their tier criteria
  is unchanged; age and entry date moved from the opening line to the Input
  section.
- Cross-references by number ("see Domains 4 and 20") replaced with names; the
  model only sees one domain at a time, so numbers meant nothing to it.

## Per domain

| Domain | Change | Why |
|---|---|---|
| 01 Suicidality | Red: defines farewell messages to include veiled goodbyes ("don't blame yourselves", saying they may not be around soon), Red even when suicide isn't named. New example. **[criterion: clarification]** | "If I'm not here next week, it's not anyone's fault" and goodbye letters were rated Yellow. |
| 03 Abuse | Scope: requires another person harming/controlling the participant, or witnessed home violence. | Rated Yellow for suicidal statements and other content with no perpetrator. |
| 04 Violence | Scope: the participant as the person who may cause harm. | Separates it from being harmed (03, 17) and from 18. |
| 06 Mania | Scope: sadness, anger, anxiety, grief, substance use or one impulsive act are not mania. | Most frequent spillover domain. |
| 07 Substance | Scope: nicotine/vaping, however frequent, is not intoxication. **[criterion: clarification]** | "I've been vaping constantly" got the `current_intoxication` flag and was escalated to Red. Also needs the same line in prompts/flags.txt. |
| 13 AI | Scope: Green if no AI is mentioned. | Rated Yellow on entries with no AI in them. |
| 14 Other | Scope: lists every other domain; Green if content fits any of them. | Catch-all claimed nearly every risky entry. |
| 17 Bullying | Red "co-occurring with suicidal content" now requires bullying described in the entry. Example "Everyone would be happier if I was gone" (no bullying in it) replaced with one that includes bullying. **[criterion]** | The old example taught the model to rate suicidal statements alone as Red bullying. |
| 16 Sexual assault, 21 Trauma | Red "co-occurring with suicidal content" now requires the assault / traumatic event to be described in the entry. 21 Red "remains in danger" now "remains in physical danger"; distress or intrusive memories alone stay Yellow. **[criterion]** | 21 rated "I keep thinking about hurting myself" and "a shooting near my school yesterday, I can't stop thinking about it" Red. |
| 18 Third party | Scope: someone other than the participant at risk from something other than the participant. "Yellow/Red-level content in any domain" replaced with concrete descriptions. **[criterion: interpretation]** | "I want to hurt my brother" was rated Red here. Needs confirmation: should harm the participant intends toward someone count under 18 as well as 04? |
| 19 Help-seeking | Red "co-occurs with Red-level content in any domain" replaced with "content in the entry describing acute or imminent danger". | The model can't see other domains' ratings. |
| 20 Schools/extremism | Scope: being affected by an attack someone else carried out belongs to 21. | "A shooting near my school yesterday" was rated Red here. |
| 08, 09, 10, 11, 12, 15 | Scope sections only. | Spillover. |

## Caveats

- The 01 example "If I'm gone by Friday, don't blame yourselves." is close to
  an item in the evaluation sample, so improvement on that item overstates the
  gain. Evaluate on new entries or clinician-scored entries too.
- Prompts are ~30% longer (47k vs 37k characters across 21 files).
