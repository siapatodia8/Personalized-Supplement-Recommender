# Agents & Tools Reference

---

## Linear Pre-Processing

### Entity Extractor
**File:** `src/agents/entity_extractor.py` | **Type:** Hybrid

**What it does:**
- Runs at the start of every question — resets all state fields for the new turn
- Extracts entities from two sources:
  - **LLM call** — parses natural language question for medications, supplements, conditions, dietary restrictions
  - **Deterministic parsing** — splits structured patient profile form fields (comma-separated strings → lists)
- Merges both sources, deduplicating case-insensitively

**Key design decisions:**
- Does not correct spelling — that's the normalizer's job
- Resets all specialist flags, results, iteration count, and candidate supplements on every new question

**Writes to state:** `extracted_entities`, `evidence_chain`, `iterations`, `supervisor_decision`, `candidate_supplements_list`, `safety_checked`, `deficiency_checked`, `recommendations_checked`, all result fields

---

### Entity Normalizer
**File:** `src/agents/entity_normalizer.py` | **Type:** Agent

**What it does:**
- Maps raw entity names to database IDs using LLM-generated Cypher queries
- Three-step process per entity:
  1. **Primary query** — LLM reads schema, generates targeted Cypher, executes against Neo4j
  2. **Fallback query** — if no results, LLM tries broader paths (synonyms, brand names, abbreviations like CoQ10 → Coenzyme Q10)
  3. **NOT_FOUND** — recorded with confidence flag, excluded from clean lists
- Performs deep deduplication by database ID after normalization — catches cases like `"folic acid"` and `"folate"` resolving to the same supplement ID
- Dietary restrictions use a hardcoded Cypher lookup (simpler schema, no LLM needed); fall back to pass-through if not found
- Conditions pass through unchanged — no database mapping needed

**Confidence levels:** `HIGH` (1 match) → `MEDIUM` (multiple matches, closest chosen) → `LOW` (fallback match) → `NOT_FOUND`

**Clean lists:** Only `HIGH` and `MEDIUM` confidence entities make it into `medications_list`, `supplements_list` — these are what downstream specialists query against

**Writes to state:** `normalized_medications`, `normalized_supplements`, `normalized_dietary_restrictions`, `medications_list`, `supplements_list`, `dietary_restrictions_list`, `conditions_list`, `entities_normalized`, `evidence_chain`

---
## Core Dynamic Loop
### Supervisor
**File:** `src/agents/supervisor.py` | **Type:** Agent

**What it does:**
- Called after entity normalization and after every specialist returns — sits in a loop
- Makes one decision per iteration: which specialist to call next, or synthesize
- Reads compact structured summaries from specialist results — does not reason from prior biomedical knowledge
- Builds a clinical picture from patient data before deciding

**Guardrails:**
- `MAX_ITERATIONS = 6` — forces synthesize if loop runs too long
- Falls back to `synthesize` on any JSON parse error
- Never repeats a specialist that has already run

**Writes to state:** `supervisor_decision`, `iterations`, `evidence_chain`

---

### Synthesis
**File:** `src/agents/synthesis.py` | **Type:** Agent

**What it does:**
- Runs once at the end of the supervisor loop (`decision = "synthesize"`)
- Produces the final patient-facing response grounded in:
  - `safety_results`
  - `deficiency_results`
  - `recommendation_results`
  - `evidence_chain`
- Uses Claude when available; falls back to a deterministic summary if no API key (or if the LLM call fails)

**Writes to state:** `final_answer`, `evidence_chain` (appends one synthesis step)

---

## Specialists

### Safety Check
**File:** `src/tools/safety_check.py` | **Type:** Agent

**What it does:**
- Checks for dangerous interactions between supplements and medications
- Runs one UNION query per supplement against all medications — checks all pathways in a single DB round-trip
- Combines `supplements_list` (current) + `candidate_supplements_list` (from recommendation) before checking — deduped

**Writes to state:** `safety_checked`, `safety_results`, `generated_safety_queries`, `evidence_chain`

---

### Deficiency Check
**File:** `src/tools/deficiency_check.py` | **Type:** Tool

**What it does:**
- Identifies nutrient deficiency risks from three independent pathways
- Runs all three as separate queries, then merges results by nutrient
- Detects critical overlaps — when 2+ sources affect the same nutrient, combined risk is flagged

**Three deficiency pathways:**
1. **Diet** — `DietaryRestriction -[DEFICIENT_IN]-> Nutrient`
2. **Medication** — `Drug -[INTERACTS_WITH_NUTRIENT]-> Nutrient` (maps `interaction_type` strings to risk levels: `depletes/antagonizes → HIGH`, `interferes_with_absorption/increases_level/may_cause_loss → MEDIUM`, `redistributes → LOW`)
3. **Supplement** — `Supplement -[NEGATIVE_INTERACTION]-> Nutrient`

**Overlap types:** `TRIPLE_OVERLAP` (all 3 sources), `DOUBLE_OVERLAP` (2 sources), `SINGLE_SOURCE_MULTIPLE`

**Writes to state:** `deficiency_checked`, `deficiency_results`, `evidence_chain`

---

### Recommendation
**File:** `src/tools/recommendation.py` | **Type:** Tool

**What it does:**
- Finds supplement candidates that treat the patient's conditions/symptoms
- Excludes current supplements (`supplements_list`) from candidates — no point recommending what patient already takes
- Two-step query: exact symptom match first, keyword fallback if no results
- Sorts candidates by `safety_rating` property from DB (`Generally safe` > `Use with caution` > `Not recommended`)
- Does NOT check safety — supervisor routes back to safety_check after recommendation runs
- Writes candidates to `candidate_supplements_list` so safety_check can find them

**Writes to state:** `recommendations_checked`, `recommendation_results`, `candidate_supplements_list`, `evidence_chain`
