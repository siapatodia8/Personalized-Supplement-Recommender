"""
Offline unit tests for src/agents/entity_extractor.py.

No network calls: the Anthropic client is replaced by a fake, so these run in
milliseconds and test the code paths around the LLM call:
  - parsing the model's reply (_extract_from_question)
  - parsing the structured profile (_parse_profile)
  - merging and de-duplicating both sources (_merge)
  - the LangGraph node (entity_extractor)

Tests marked xfail(strict=True) document known limitations of the current
code (not the prompt). They pass while the limitation exists and turn into a
failure the day the code is fixed, so they are a reminder rather than noise.
"""

import json
from unittest.mock import patch

import pytest

import src.agents.entity_extractor as ee
from src.prompt_loader import load_prompt

KEYS = ["medications", "supplements", "conditions", "dietary_restrictions"]
EMPTY = {k: [] for k in KEYS}
SAMPLE = {
    "medications": ["Warfarin"],
    "supplements": ["Fish Oil"],
    "conditions": ["heart health"],
    "dietary_restrictions": ["vegan"],
}


# ==================== PROMPT TEMPLATE ====================

class TestPromptTemplate:
    def test_template_has_only_the_question_placeholder(self):
        template = load_prompt("entity_extractor")["extraction"]
        rendered = template.format(question="Can I take Fish Oil with Warfarin?")
        assert "Can I take Fish Oil with Warfarin?" in rendered
        # The JSON example uses doubled braces in YAML; after format they are single.
        assert '"medications": []' in rendered

    def test_question_with_braces_and_quotes_formats_safely(self):
        template = load_prompt("entity_extractor")["extraction"]
        question = 'Is "Fish Oil" {omega-3} safe with {warfarin}?'
        assert question in template.format(question=question)

    def test_template_names_all_four_categories(self):
        template = load_prompt("entity_extractor")["extraction"]
        for key in KEYS:
            assert key in template


# ==================== LLM REPLY PARSING ====================

class TestExtractFromQuestionParsing:
    @pytest.mark.parametrize(
        "reply",
        [
            json.dumps(SAMPLE),                                  # plain JSON
            "```json\n" + json.dumps(SAMPLE) + "\n```",          # fenced, json tag
            "```\n" + json.dumps(SAMPLE) + "\n```",              # fenced, no tag
            "  \n```json\n" + json.dumps(SAMPLE, indent=4) + "\n```\n  ",  # padded
            json.dumps(SAMPLE, indent=2),                        # pretty-printed
        ],
        ids=["plain", "fenced-json", "fenced-bare", "fenced-padded", "pretty"],
    )
    def test_valid_replies_parse(self, fake_client, reply):
        assert ee._extract_from_question("q", fake_client(reply=reply)) == SAMPLE

    @pytest.mark.parametrize(
        "reply",
        ["", "not json at all", "{'medications': ['Warfarin']}", '{"medications": ['],
        ids=["empty", "prose", "python-dict", "truncated"],
    )
    def test_unparseable_reply_returns_empty_lists(self, fake_client, reply):
        assert ee._extract_from_question("q", fake_client(reply=reply)) == EMPTY

    def test_api_error_returns_empty_lists(self, fake_client):
        client = fake_client(error=RuntimeError("API down"))
        assert ee._extract_from_question("q", client) == EMPTY

    def test_unicode_entities_survive(self, fake_client):
        data = {**EMPTY, "supplements": ["St. John's Wort", "Ginkgo bilöba"]}
        reply = json.dumps(data, ensure_ascii=False)
        assert ee._extract_from_question("q", fake_client(reply=reply)) == data

    def test_request_uses_expected_model_settings(self, fake_client):
        client = fake_client(reply=json.dumps(EMPTY))
        ee._extract_from_question("Can I take Zinc?", client)
        call = client.calls[0]
        assert call["model"] == "claude-sonnet-4-5"
        assert call["temperature"] == 0
        assert call["max_tokens"] == 500
        assert "Can I take Zinc?" in call["messages"][0]["content"]

    @pytest.mark.xfail(strict=True, reason="Known limitation: fence regex is case-sensitive (```JSON not stripped)")
    def test_uppercase_json_fence(self, fake_client):
        reply = "```JSON\n" + json.dumps(SAMPLE) + "\n```"
        assert ee._extract_from_question("q", fake_client(reply=reply)) == SAMPLE

    @pytest.mark.xfail(strict=True, reason="Known limitation: text before the JSON makes parsing fail")
    def test_preamble_before_json(self, fake_client):
        reply = "Here is the extraction:\n" + json.dumps(SAMPLE)
        assert ee._extract_from_question("q", fake_client(reply=reply)) == SAMPLE


# ==================== PROFILE PARSING ====================

class TestParseProfile:
    def test_comma_separated_strings(self):
        out = ee._parse_profile({"medications": "Warfarin, Metformin", "supplements": "Fish Oil,Vitamin D"})
        assert out["medications"] == ["Warfarin", "Metformin"]
        assert out["supplements"] == ["Fish Oil", "Vitamin D"]

    def test_lists_are_kept(self):
        out = ee._parse_profile({"conditions": ["Diabetes", "Heart Disease"], "dietary_restrictions": ["Vegan"]})
        assert out["conditions"] == ["Diabetes", "Heart Disease"]
        assert out["dietary_restrictions"] == ["Vegan"]

    def test_whitespace_and_empty_items_are_dropped(self):
        out = ee._parse_profile({"medications": "  Warfarin ,, ,Metformin  ,", "conditions": ["  ", "Acne ", ""]})
        assert out["medications"] == ["Warfarin", "Metformin"]
        assert out["conditions"] == ["Acne"]

    def test_single_item_without_comma(self):
        assert ee._parse_profile({"supplements": "Zinc"})["supplements"] == ["Zinc"]

    def test_comma_inside_a_list_item_is_not_split(self):
        out = ee._parse_profile({"supplements": ["Vitamin B6, B12"]})
        assert out["supplements"] == ["Vitamin B6, B12"]

    @pytest.mark.parametrize("profile", [{}, {"medications": ""}, {"medications": None, "conditions": None}])
    def test_empty_or_missing_fields(self, profile):
        assert ee._parse_profile(profile) == EMPTY

    def test_non_string_list_items_become_strings(self):
        assert ee._parse_profile({"supplements": [5, "Zinc"]})["supplements"] == ["5", "Zinc"]

    def test_unsupported_type_is_ignored(self):
        assert ee._parse_profile({"medications": {"name": "Warfarin"}})["medications"] == []

    def test_always_returns_all_four_keys(self):
        assert set(ee._parse_profile({})) == set(KEYS)


# ==================== MERGE + DEDUP ====================

class TestMerge:
    def test_profile_casing_wins(self):
        out = ee._merge({**EMPTY, "supplements": ["fish oil"]}, {**EMPTY, "supplements": ["Fish Oil"]})
        assert out["supplements"] == ["Fish Oil"]

    def test_profile_first_then_new_question_items(self):
        out = ee._merge(
            {**EMPTY, "medications": ["Metformin", "warfarin"]},
            {**EMPTY, "medications": ["Warfarin"]},
        )
        assert out["medications"] == ["Warfarin", "Metformin"]

    def test_duplicates_within_one_source_collapse(self):
        out = ee._merge({**EMPTY, "conditions": ["Acne", "acne", "ACNE"]}, EMPTY)
        assert out["conditions"] == ["Acne"]

    def test_categories_stay_separate(self):
        out = ee._merge({**EMPTY, "supplements": ["Calcium"]}, {**EMPTY, "medications": ["Calcium"]})
        assert out["medications"] == ["Calcium"] and out["supplements"] == ["Calcium"]

    def test_missing_keys_in_either_source(self):
        out = ee._merge({"supplements": ["Zinc"]}, {"medications": ["Warfarin"]})
        assert out == {**EMPTY, "medications": ["Warfarin"], "supplements": ["Zinc"]}

    def test_both_empty(self):
        assert ee._merge(EMPTY, EMPTY) == EMPTY

    def test_extra_keys_from_model_are_dropped(self):
        out = ee._merge({**EMPTY, "allergies": ["peanuts"]}, EMPTY)
        assert set(out) == set(KEYS)

    @pytest.mark.xfail(strict=True, reason="Known limitation: question items are not whitespace-trimmed before de-duplication")
    def test_whitespace_variants_dedupe(self):
        out = ee._merge({**EMPTY, "supplements": [" Fish Oil "]}, {**EMPTY, "supplements": ["Fish Oil"]})
        assert out["supplements"] == ["Fish Oil"]

    @pytest.mark.xfail(strict=True, raises=TypeError, reason="Known limitation: a null category from the model crashes the merge")
    def test_null_category_from_model(self):
        out = ee._merge({**EMPTY, "medications": None}, EMPTY)
        assert out["medications"] == []

    @pytest.mark.xfail(strict=True, reason="Known limitation: a string instead of a list is iterated character by character")
    def test_string_instead_of_list_from_model(self):
        out = ee._merge({**EMPTY, "supplements": "Zinc"}, EMPTY)
        assert out["supplements"] == ["Zinc"]


# ==================== LANGGRAPH NODE ====================

class TestNode:
    def _run(self, question, profile, extracted):
        with patch.object(ee, "Anthropic"), patch.object(ee, "_extract_from_question", return_value=extracted) as m:
            out = ee.entity_extractor({"user_question": question, "patient_profile": profile})
        return out, m

    def test_merges_question_and_profile(self):
        out, _ = self._run(
            "Is my Fish Oil safe?",
            {"medications": "Warfarin", "dietary_restrictions": ["Vegan"]},
            {**EMPTY, "supplements": ["Fish Oil"], "dietary_restrictions": ["vegan"]},
        )
        assert out["extracted_entities"] == {
            "medications": ["Warfarin"],
            "supplements": ["Fish Oil"],
            "conditions": [],
            "dietary_restrictions": ["Vegan"],
        }

    def test_passes_the_question_to_the_llm(self):
        _, m = self._run("Can I take Zinc?", {}, EMPTY)
        assert m.call_args.args[0] == "Can I take Zinc?"

    def test_evidence_chain_is_a_fresh_single_entry(self):
        out, _ = self._run("q", {"medications": "Warfarin"}, EMPTY)
        assert out["evidence_chain"] == [
            "Extracted — medications: ['Warfarin'], supplements: [], conditions: [], dietary restrictions: []"
        ]

    def test_resets_per_question_workflow_state(self):
        out, _ = self._run("q", {}, EMPTY)
        assert out["iterations"] == 0
        assert out["supervisor_decision"] == ""
        assert out["candidate_supplements_list"] == []
        assert out["safety_checked"] is False and out["deficiency_checked"] is False
        assert out["recommendations_checked"] is False
        assert out["safety_results"] is None and out["final_answer"] is None
        assert out["generated_safety_queries"] == []

    def test_missing_question_and_profile(self):
        with patch.object(ee, "Anthropic"), patch.object(ee, "_extract_from_question", return_value=EMPTY):
            out = ee.entity_extractor({})
        assert out["extracted_entities"] == EMPTY

    def test_llm_failure_still_uses_profile(self, fake_client):
        client = fake_client(error=RuntimeError("API down"))
        with patch.object(ee, "Anthropic", return_value=client):
            out = ee.entity_extractor({"user_question": "q", "patient_profile": {"medications": "Warfarin"}})
        assert out["extracted_entities"]["medications"] == ["Warfarin"]
