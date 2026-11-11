"""
Unit tests for the knowledge graph loader (scripts/load_data.py + kg_loader/).

None of these tests touch a real database. The loader takes its Neo4j driver
as an argument, so the tests hand it a FakeDriver that records every query.

Run with:  pytest scripts/tests -v
"""

import logging
import re
from pathlib import Path

import pandas as pd
import pytest

import load_data
from kg_loader.catalog import PHASES, Endpoint, NodeSpec, RelSpec
from kg_loader.loader import (
    KnowledgeGraphLoader,
    LoadResult,
    build_node_query,
    build_query,
    build_rel_query,
)
from kg_loader.schema import SCHEMA_STATEMENTS, apply_schema

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ALL_SPECS = [spec for _, specs in PHASES for spec in specs]
NODE_SPECS = [s for s in ALL_SPECS if isinstance(s, NodeSpec)]
REL_SPECS = [s for s in ALL_SPECS if isinstance(s, RelSpec)]


# ----------------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------------


class FakeResult:
    def __init__(self, record):
        self._record = record

    def single(self):
        return self._record


class FakeDriver:
    """Stands in for the Neo4j driver: records queries, returns canned answers.

    By default it behaves like a perfect database: every row sent is created.
    Pass `respond(query, params) -> dict` to simulate something else.
    """

    def __init__(self, respond=None):
        self.calls = []  # list of (query, params)
        self._respond = respond or (
            lambda query, params: {"created": len(params.get("batch", []))}
        )

    def session(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self.calls.append((query, params))
        return FakeResult(self._respond(query, params))

    @property
    def batch_sizes(self):
        return [len(params["batch"]) for _, params in self.calls]


def rows(n):
    return [{"id": i} for i in range(n)]


def write_csv(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return name


SIMPLE_NODE = NodeSpec("Thing", "things.csv", ("thing_id", "thing_name"), batch_size=2)
SIMPLE_REL = RelSpec(
    "LINKS_TO",
    "links.csv",
    Endpoint("Thing", "thing_id"),
    Endpoint("Other", "other_id"),
    batch_size=2,
)


# ----------------------------------------------------------------------------
# Query builders
# ----------------------------------------------------------------------------


class TestBuildNodeQuery:
    def test_exact_query(self):
        assert build_node_query(SIMPLE_NODE) == (
            "UNWIND $batch AS row\n"
            "CREATE (n:Thing {thing_id: row.thing_id, thing_name: row.thing_name})\n"
            "RETURN count(n) AS created"
        )

    def test_single_property_has_no_trailing_comma(self):
        query = build_node_query(NodeSpec("Thing", "t.csv", ("thing_id",)))
        assert "CREATE (n:Thing {thing_id: row.thing_id})" in query

    def test_every_property_is_copied(self):
        drug = next(s for s in NODE_SPECS if s.label == "Drug")
        query = build_node_query(drug)
        for prop in drug.props:
            assert f"{prop}: row.{prop}" in query

    def test_property_can_come_from_a_differently_named_column(self):
        # food_interactions.csv: column `food_interaction` -> property `description`
        spec = next(s for s in NODE_SPECS if s.label == "FoodInteraction")
        assert "description: row.food_interaction" in build_node_query(spec)

    def test_only_the_renamed_property_reads_a_different_column(self):
        spec = NodeSpec("Thing", "t.csv", ("thing_id", "note"), column_for={"note": "n"})
        assert "{thing_id: row.thing_id, note: row.n}" in build_node_query(spec)


class TestBuildRelQuery:
    def test_exact_query_without_properties(self):
        assert build_rel_query(SIMPLE_REL) == (
            "UNWIND $batch AS row\n"
            "MATCH (a:Thing {thing_id: row.thing_id})\n"
            "MATCH (b:Other {other_id: row.other_id})\n"
            "CREATE (a)-[r:LINKS_TO]->(b)\n"
            "RETURN count(r) AS created"
        )

    def test_no_empty_braces_when_there_are_no_properties(self):
        assert "{}" not in build_rel_query(SIMPLE_REL)

    def test_properties_are_set_on_the_relationship(self):
        spec = RelSpec(
            "LINKS_TO",
            "links.csv",
            Endpoint("Thing", "thing_id"),
            Endpoint("Other", "other_id"),
            props={"strength": "strength", "notes": "notes"},
        )
        assert (
            "CREATE (a)-[r:LINKS_TO {strength: row.strength, notes: row.notes}]->(b)"
            in build_rel_query(spec)
        )

    def test_property_can_come_from_a_differently_named_column(self):
        # drug_nutrient_interacts.csv: column `interaction` -> `interaction_type`
        spec = next(s for s in REL_SPECS if s.rel_type == "INTERACTS_WITH_NUTRIENT")
        assert "{interaction_type: row.interaction}" in build_rel_query(spec)

    def test_same_label_on_both_ends_uses_two_different_columns(self):
        # Drug -> Drug: both ends match on drug_id, but read different CSV columns
        spec = next(s for s in REL_SPECS if s.rel_type == "INTERACTS_WITH")
        query = build_rel_query(spec)
        assert "MATCH (a:Drug {drug_id: row.drug_id})" in query
        assert "MATCH (b:Drug {drug_id: row.interacting_drug_id})" in query

    def test_direction_is_start_to_end(self):
        assert "(a)-[r:LINKS_TO]->(b)" in build_rel_query(SIMPLE_REL)


class TestBuildQuery:
    def test_picks_the_builder_by_spec_type(self):
        assert build_query(SIMPLE_NODE) == build_node_query(SIMPLE_NODE)
        assert build_query(SIMPLE_REL) == build_rel_query(SIMPLE_REL)

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.file)
    def test_every_catalog_query_is_batched_and_reports_a_count(self, spec):
        query = build_query(spec)
        assert query.startswith("UNWIND $batch AS row")
        assert query.endswith("AS created")


class TestEndpoint:
    def test_column_defaults_to_key(self):
        assert Endpoint("Drug", "drug_id").csv_column == "drug_id"

    def test_column_can_be_overridden(self):
        endpoint = Endpoint("Drug", "drug_id", column="interacting_drug_id")
        assert endpoint.csv_column == "interacting_drug_id"

    def test_specs_cannot_be_modified_after_creation(self):
        with pytest.raises(AttributeError):
            SIMPLE_NODE.label = "Other"


# ----------------------------------------------------------------------------
# Catalog: the table itself must be internally consistent
# ----------------------------------------------------------------------------


class TestCatalog:
    def test_expected_number_of_files(self):
        assert len(NODE_SPECS) == 12
        assert len(REL_SPECS) == 16

    def test_no_file_is_loaded_twice(self):
        files = [s.file for s in ALL_SPECS]
        assert len(files) == len(set(files))

    def test_no_node_label_is_defined_twice(self):
        labels = [s.label for s in NODE_SPECS]
        assert len(labels) == len(set(labels))

    def test_relationships_only_come_after_both_of_their_nodes(self):
        """The load-order rule: a relationship's nodes must already be loaded."""
        loaded = set()
        for spec in ALL_SPECS:
            if isinstance(spec, NodeSpec):
                loaded.add(spec.label)
            else:
                assert spec.start.label in loaded, f"{spec.file}: start not loaded yet"
                assert spec.end.label in loaded, f"{spec.file}: end not loaded yet"

    @pytest.mark.parametrize("spec", REL_SPECS, ids=lambda s: s.file)
    def test_relationship_matches_on_a_property_the_node_actually_has(self, spec):
        props_by_label = {s.label: s.props for s in NODE_SPECS}
        for endpoint in (spec.start, spec.end):
            assert endpoint.key in props_by_label[endpoint.label]

    @pytest.mark.parametrize("spec", REL_SPECS, ids=lambda s: s.file)
    def test_every_match_key_has_a_uniqueness_constraint(self, spec):
        """Without the constraint's index, each MATCH would scan every node."""
        for endpoint in (spec.start, spec.end):
            pattern = (
                rf"FOR \(\w+:{endpoint.label}\) REQUIRE \w+\.{endpoint.key} IS UNIQUE"
            )
            assert any(re.search(pattern, s) for s in SCHEMA_STATEMENTS), endpoint

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.file)
    def test_batch_size_is_positive(self, spec):
        assert spec.batch_size >= 1

    @pytest.mark.skipif(not DATA_DIR.exists(), reason="data/ directory not present")
    @pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.file)
    def test_csv_exists_and_has_every_column_the_spec_uses(self, spec):
        path = DATA_DIR / spec.file
        assert path.exists(), f"missing file: {path}"

        if isinstance(spec, NodeSpec):
            needed = {spec.csv_column(prop) for prop in spec.props}
        else:
            needed = {spec.start.csv_column, spec.end.csv_column, *spec.props.values()}

        header = set(pd.read_csv(path, nrows=0).columns)
        assert needed <= header, f"{spec.file} is missing columns: {needed - header}"


# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------


class TestSchema:
    def test_runs_every_statement_once_in_order(self):
        driver = FakeDriver(respond=lambda q, p: None)
        apply_schema(driver)
        assert [q for q, _ in driver.calls] == SCHEMA_STATEMENTS

    def test_every_statement_is_safe_to_rerun(self):
        assert all("IF NOT EXISTS" in s for s in SCHEMA_STATEMENTS)

    def test_one_uniqueness_constraint_per_node_label(self):
        constraints = [s for s in SCHEMA_STATEMENTS if "CONSTRAINT" in s]
        assert len(constraints) == len(NODE_SPECS)

    def test_a_database_error_is_not_swallowed(self):
        def explode(query, params):
            raise RuntimeError("database unavailable")

        with pytest.raises(RuntimeError, match="database unavailable"):
            apply_schema(FakeDriver(respond=explode))


# ----------------------------------------------------------------------------
# batch_execute: boundaries
# ----------------------------------------------------------------------------


class TestBatchExecute:
    @pytest.mark.parametrize(
        "row_count, batch_size, expected_batches",
        [
            (0, 5, []),  # nothing to send
            (1, 5, [1]),  # fewer rows than one batch
            (4, 5, [4]),  # one under the boundary
            (5, 5, [5]),  # exactly one batch
            (6, 5, [5, 1]),  # one over the boundary
            (10, 5, [5, 5]),  # exact multiple
            (3, 1, [1, 1, 1]),  # smallest allowed batch size
            (3, 1000, [3]),  # batch size far larger than data
        ],
    )
    def test_splits_rows_into_batches(self, row_count, batch_size, expected_batches):
        driver = FakeDriver()
        KnowledgeGraphLoader(driver).batch_execute("Q", rows(row_count), batch_size)
        assert driver.batch_sizes == expected_batches

    def test_no_row_is_lost_or_duplicated_and_order_is_kept(self):
        driver = FakeDriver()
        data = rows(7)
        KnowledgeGraphLoader(driver).batch_execute("Q", data, batch_size=3)
        sent = [row for _, params in driver.calls for row in params["batch"]]
        assert sent == data

    def test_returns_the_total_the_database_reports(self):
        driver = FakeDriver()
        assert KnowledgeGraphLoader(driver).batch_execute("Q", rows(7), 3) == 7

    def test_returned_total_comes_from_the_database_not_the_row_count(self):
        driver = FakeDriver(respond=lambda q, p: {"created": 1})  # 1 per batch
        created = KnowledgeGraphLoader(driver).batch_execute("Q", rows(6), 3)
        assert created == 2

    def test_empty_data_returns_zero_without_querying(self):
        driver = FakeDriver()
        assert KnowledgeGraphLoader(driver).batch_execute("Q", [], 5) == 0
        assert driver.calls == []

    @pytest.mark.parametrize("bad_size", [0, -1])
    def test_rejects_a_batch_size_below_one(self, bad_size):
        with pytest.raises(ValueError, match="batch_size"):
            KnowledgeGraphLoader(FakeDriver()).batch_execute("Q", rows(3), bad_size)

    def test_the_same_query_is_sent_for_every_batch(self):
        driver = FakeDriver()
        KnowledgeGraphLoader(driver).batch_execute("MY QUERY", rows(5), 2)
        assert {q for q, _ in driver.calls} == {"MY QUERY"}


# ----------------------------------------------------------------------------
# load_spec: reading a CSV and verifying the result
# ----------------------------------------------------------------------------


class TestLoadSpec:
    def test_loads_rows_and_reports_matching_counts(self, tmp_path):
        write_csv(tmp_path, "things.csv", "thing_id,thing_name\n1,a\n2,b\n3,c\n")
        driver = FakeDriver()

        result = KnowledgeGraphLoader(driver).load_spec(SIMPLE_NODE, tmp_path)

        assert result == LoadResult("Thing nodes", "things.csv", rows=3, created=3)
        assert result.skipped == 0
        assert driver.batch_sizes == [2, 1]  # SIMPLE_NODE.batch_size is 2

    def test_missing_file_stops_the_load_and_names_the_file(self, tmp_path):
        driver = FakeDriver()
        with pytest.raises(FileNotFoundError, match="things.csv"):
            KnowledgeGraphLoader(driver).load_spec(SIMPLE_NODE, tmp_path)
        assert driver.calls == []

    def test_header_only_csv_loads_zero_rows(self, tmp_path):
        write_csv(tmp_path, "things.csv", "thing_id,thing_name\n")
        driver = FakeDriver()

        result = KnowledgeGraphLoader(driver).load_spec(SIMPLE_NODE, tmp_path)

        assert (result.rows, result.created, result.skipped) == (0, 0, 0)
        assert driver.calls == []

    def test_completely_empty_file_is_an_error(self, tmp_path):
        write_csv(tmp_path, "things.csv", "")
        with pytest.raises(pd.errors.EmptyDataError):
            KnowledgeGraphLoader(FakeDriver()).load_spec(SIMPLE_NODE, tmp_path)

    def test_blank_cells_are_sent_as_empty_strings_not_nan(self, tmp_path):
        write_csv(tmp_path, "things.csv", "thing_id,thing_name\n1,\n")
        driver = FakeDriver()

        KnowledgeGraphLoader(driver).load_spec(SIMPLE_NODE, tmp_path)

        assert driver.calls[0][1]["batch"] == [{"thing_id": 1, "thing_name": ""}]

    def test_commas_and_quotes_inside_a_cell_survive(self, tmp_path):
        write_csv(
            tmp_path, "things.csv", 'thing_id,thing_name\n1,"Vitamin D, ""D3"""\n'
        )
        driver = FakeDriver()

        KnowledgeGraphLoader(driver).load_spec(SIMPLE_NODE, tmp_path)

        assert driver.calls[0][1]["batch"][0]["thing_name"] == 'Vitamin D, "D3"'

    def test_skipped_rows_are_counted_and_warned_about(self, tmp_path, caplog):
        write_csv(tmp_path, "links.csv", "thing_id,other_id\n1,1\n2,2\n3,999\n")
        # Simulate Neo4j not finding a node for one row in the second batch
        answers = iter([{"created": 2}, {"created": 0}])
        driver = FakeDriver(respond=lambda q, p: next(answers))

        with caplog.at_level(logging.WARNING):
            result = KnowledgeGraphLoader(driver).load_spec(SIMPLE_REL, tmp_path)

        assert (result.rows, result.created, result.skipped) == (3, 2, 1)
        assert "links.csv" in caplog.text
        assert "1 skipped" in caplog.text

    def test_no_warning_when_everything_is_created(self, tmp_path, caplog):
        write_csv(tmp_path, "links.csv", "thing_id,other_id\n1,1\n")

        with caplog.at_level(logging.WARNING):
            KnowledgeGraphLoader(FakeDriver()).load_spec(SIMPLE_REL, tmp_path)

        assert caplog.text == ""


# ----------------------------------------------------------------------------
# load_all: ordering and the final summary
# ----------------------------------------------------------------------------


class TestLoadAll:
    @pytest.fixture
    def data_dir(self, tmp_path):
        write_csv(tmp_path, "things.csv", "thing_id,thing_name\n1,a\n")
        write_csv(tmp_path, "links.csv", "thing_id,other_id\n1,1\n")
        return tmp_path

    def test_loads_specs_in_the_order_given(self, data_dir):
        driver = FakeDriver()
        phases = [("Nodes", [SIMPLE_NODE]), ("Relationships", [SIMPLE_REL])]

        results = KnowledgeGraphLoader(driver).load_all(phases, data_dir)

        assert [r.file for r in results] == ["things.csv", "links.csv"]
        assert "CREATE (n:Thing" in driver.calls[0][0]
        assert "LINKS_TO" in driver.calls[1][0]

    def test_no_phases_means_nothing_happens(self, data_dir):
        driver = FakeDriver()
        assert KnowledgeGraphLoader(driver).load_all([], data_dir) == []
        assert driver.calls == []

    def test_a_missing_file_stops_before_later_specs_run(self, tmp_path):
        write_csv(tmp_path, "links.csv", "thing_id,other_id\n1,1\n")  # no things.csv
        driver = FakeDriver()
        phases = [("Nodes", [SIMPLE_NODE]), ("Relationships", [SIMPLE_REL])]

        with pytest.raises(FileNotFoundError):
            KnowledgeGraphLoader(driver).load_all(phases, tmp_path)

        assert driver.calls == []

    def test_summary_lists_files_with_skipped_rows(self, data_dir, caplog):
        def respond(query, params):
            return {"created": 0 if "LINKS_TO" in query else len(params["batch"])}

        phases = [("Nodes", [SIMPLE_NODE]), ("Relationships", [SIMPLE_REL])]
        with caplog.at_level(logging.INFO):
            KnowledgeGraphLoader(FakeDriver(respond)).load_all(phases, data_dir)

        assert "1 file(s) had rows that were not created" in caplog.text
        assert "links.csv: 1 of 1 rows skipped" in caplog.text
        assert "All data loaded" not in caplog.text

    def test_summary_reports_success_only_when_nothing_was_skipped(
        self, data_dir, caplog
    ):
        phases = [("Nodes", [SIMPLE_NODE]), ("Relationships", [SIMPLE_REL])]
        with caplog.at_level(logging.INFO):
            KnowledgeGraphLoader(FakeDriver()).load_all(phases, data_dir)

        assert "All data loaded" in caplog.text


# ----------------------------------------------------------------------------
# clear_database
# ----------------------------------------------------------------------------


def fake_database(node_count, rel_count=0):
    """A FakeDriver holding `node_count` nodes that shrink as they are deleted."""
    remaining = {"nodes": node_count}

    def respond(query, params):
        if "node_count" in query:
            return {"node_count": remaining["nodes"]}
        if "rel_count" in query:
            return {"rel_count": rel_count}
        deleted = min(params["limit"], remaining["nodes"])
        remaining["nodes"] -= deleted
        return {"deleted": deleted}

    return FakeDriver(respond)


class TestClearDatabase:
    def delete_calls(self, driver):
        return [params for query, params in driver.calls if "DELETE" in query]

    def test_empty_database_issues_no_deletes(self):
        driver = fake_database(0)
        assert KnowledgeGraphLoader(driver).clear_database() == 0
        assert self.delete_calls(driver) == []

    @pytest.mark.parametrize(
        "node_count, batch_size, expected_delete_calls",
        [
            (1, 10, 2),  # one partial batch, then one empty batch to confirm
            (10, 10, 2),  # exactly one full batch
            (11, 10, 3),  # one over the boundary
            (25, 10, 4),
        ],
    )
    def test_deletes_in_batches_until_nothing_is_left(
        self, node_count, batch_size, expected_delete_calls
    ):
        driver = fake_database(node_count, rel_count=5)
        deleted = KnowledgeGraphLoader(driver).clear_database(batch_size=batch_size)
        assert deleted == node_count
        assert len(self.delete_calls(driver)) == expected_delete_calls

    def test_batch_size_is_passed_as_a_parameter_not_formatted_into_the_query(self):
        driver = fake_database(3)
        KnowledgeGraphLoader(driver).clear_database(batch_size=2)

        delete_queries = [q for q, _ in driver.calls if "DELETE" in q]
        assert all("$limit" in q for q in delete_queries)
        assert all(p == {"limit": 2} for p in self.delete_calls(driver))


# ----------------------------------------------------------------------------
# Entry point (scripts/load_data.py)
# ----------------------------------------------------------------------------


class TestConfirmWipe:
    def test_yes_flag_skips_the_prompt_entirely(self):
        def must_not_ask(prompt):
            raise AssertionError("prompt should not be shown with --yes")

        assert load_data.confirm_wipe(True, ask=must_not_ask) is True

    @pytest.mark.parametrize("answer", ["yes", "y", "YES", "Y", "Yes", "  yes  "])
    def test_accepts_yes_answers(self, answer):
        assert load_data.confirm_wipe(False, ask=lambda prompt: answer) is True

    @pytest.mark.parametrize("answer", ["no", "n", "", "   ", "yess", "ok", "true"])
    def test_anything_else_means_no(self, answer):
        assert load_data.confirm_wipe(False, ask=lambda prompt: answer) is False

    def test_prompt_warns_that_data_will_be_deleted(self):
        seen = []
        load_data.confirm_wipe(False, ask=lambda prompt: seen.append(prompt) or "no")
        assert "DELETE ALL" in seen[0]


class TestParseArgs:
    def test_prompt_is_on_by_default(self):
        assert load_data.parse_args([]).yes is False

    @pytest.mark.parametrize("flag", ["--yes", "-y"])
    def test_yes_flag(self, flag):
        assert load_data.parse_args([flag]).yes is True

    def test_unknown_flag_is_rejected(self):
        with pytest.raises(SystemExit):
            load_data.parse_args(["--force"])


class TestGetSettings:
    def test_defaults_are_used_when_only_the_password_is_set(self, tmp_path):
        env = {"NEO4J_PASSWORD": "secret", "DATA_DIR": str(tmp_path)}
        settings = load_data.get_settings(env)
        assert settings["uri"] == "bolt://localhost:7687"
        assert settings["user"] == "neo4j"
        assert settings["password"] == "secret"
        assert settings["data_dir"] == tmp_path

    def test_environment_overrides_defaults(self, tmp_path):
        env = {
            "NEO4J_URI": "bolt://example:7687",
            "NEO4J_USER": "admin",
            "NEO4J_PASSWORD": "secret",
            "DATA_DIR": str(tmp_path),
        }
        settings = load_data.get_settings(env)
        assert (settings["uri"], settings["user"]) == ("bolt://example:7687", "admin")

    @pytest.mark.parametrize("env", [{}, {"NEO4J_PASSWORD": ""}])
    def test_missing_or_empty_password_is_an_error(self, env):
        with pytest.raises(ValueError, match="NEO4J_PASSWORD"):
            load_data.get_settings(env)

    def test_missing_data_directory_is_an_error(self, tmp_path):
        env = {"NEO4J_PASSWORD": "secret", "DATA_DIR": str(tmp_path / "nope")}
        with pytest.raises(FileNotFoundError, match="nope"):
            load_data.get_settings(env)


class TestMain:
    def test_answering_no_never_opens_a_database_connection(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("NEO4J_PASSWORD", "secret")
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        monkeypatch.setattr(load_data, "confirm_wipe", lambda assume_yes: False)

        def must_not_connect(*args, **kwargs):
            raise AssertionError("should not connect after the user says no")

        monkeypatch.setattr(load_data.GraphDatabase, "driver", must_not_connect)

        load_data.main([])
