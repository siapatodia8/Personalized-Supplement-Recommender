"""
Loader: HOW data gets into Neo4j.

Knows nothing about specific files. It is handed a driver and a list of
phases (see catalog.py) and does four things: clear the database, build the
Cypher for a spec, send rows in batches, and check that what Neo4j created
matches what the CSV contained.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from kg_loader.catalog import NodeSpec, Phase, RelSpec, Spec

logger = logging.getLogger(__name__)

# Small batches keep each delete transaction within Neo4j's memory limits.
CLEAR_BATCH_SIZE = 500


@dataclass(frozen=True)
class LoadResult:
    """What happened when one CSV was loaded."""

    name: str  # e.g. "Drug nodes"
    file: str
    rows: int  # rows in the CSV
    created: int  # nodes/relationships Neo4j actually created

    @property
    def skipped(self) -> int:
        return self.rows - self.created


# ----------------------------------------------------------------------------
# Query builders (pure functions: spec in, Cypher text out)
#
# Labels, relationship types and property names cannot be passed as Cypher
# parameters, so they are formatted into the query text. That is safe here
# because they only ever come from the catalog, never from user input. The
# row values themselves always go through the $batch parameter.
# ----------------------------------------------------------------------------


def build_node_query(spec: NodeSpec) -> str:
    """Cypher that creates one node per CSV row and reports how many."""
    props = ", ".join(f"{p}: row.{spec.csv_column(p)}" for p in spec.props)
    return (
        "UNWIND $batch AS row\n"
        f"CREATE (n:{spec.label} {{{props}}})\n"
        "RETURN count(n) AS created"
    )


def build_rel_query(spec: RelSpec) -> str:
    """Cypher that matches both end nodes, creates one relationship per CSV
    row, and reports how many. A row whose start or end node does not exist
    matches nothing, so it is not counted."""
    props = ", ".join(f"{prop}: row.{col}" for prop, col in spec.props.items())
    rel = f"[r:{spec.rel_type} {{{props}}}]" if props else f"[r:{spec.rel_type}]"
    start, end = spec.start, spec.end
    return (
        "UNWIND $batch AS row\n"
        f"MATCH (a:{start.label} {{{start.key}: row.{start.csv_column}}})\n"
        f"MATCH (b:{end.label} {{{end.key}: row.{end.csv_column}}})\n"
        f"CREATE (a)-{rel}->(b)\n"
        "RETURN count(r) AS created"
    )


def build_query(spec: Spec) -> str:
    if isinstance(spec, NodeSpec):
        return build_node_query(spec)
    return build_rel_query(spec)


# ----------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------


class KnowledgeGraphLoader:
    """Clears the database and loads specs into it.

    The Neo4j driver is passed in rather than created here, so the caller
    owns the connection and tests can pass a fake.
    """

    def __init__(self, driver):
        self.driver = driver

    def clear_database(self, batch_size: int = CLEAR_BATCH_SIZE) -> int:
        """Delete every node and relationship in batches. Returns nodes deleted."""
        with self.driver.session() as session:
            node_count = session.run(
                "MATCH (n) RETURN count(n) AS node_count"
            ).single()["node_count"]
            rel_count = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS rel_count"
            ).single()["rel_count"]

            if node_count == 0 and rel_count == 0:
                logger.info("  ✓ Database already empty")
                return 0

            logger.info(
                f"  Deleting {node_count:,} nodes and {rel_count:,} relationships..."
            )

            total_deleted = 0
            with tqdm(total=node_count, desc="  Clearing database", unit="nodes") as pbar:
                while True:
                    deleted = session.run(
                        "MATCH (n) WITH n LIMIT $limit "
                        "DETACH DELETE n "
                        "RETURN count(n) AS deleted",
                        limit=batch_size,
                    ).single()["deleted"]

                    if deleted == 0:
                        break

                    total_deleted += deleted
                    pbar.update(deleted)

        logger.info(f"✓ Deleted {total_deleted:,} nodes and all relationships")
        return total_deleted

    def batch_execute(
        self, query: str, data: list, batch_size: int = 5000, desc: str = "Processing"
    ) -> int:
        """
        Run a Cypher query over `data` in batches using UNWIND.

        Args:
            query: Cypher containing `UNWIND $batch` and `RETURN ... AS created`
            data: List of dictionaries (one per CSV row)
            batch_size: Rows per batch
            desc: Description for progress bar

        Returns:
            Total number of nodes/relationships Neo4j reported creating.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")

        created = 0
        with tqdm(total=len(data), desc=desc, unit="records") as pbar:
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]

                with self.driver.session() as session:
                    created += session.run(query, batch=batch).single()["created"]

                pbar.update(len(batch))

        return created

    def load_spec(self, spec: Spec, data_dir: Path) -> LoadResult:
        """Read the CSV for one spec, load it, and compare rows vs. created."""
        path = Path(data_dir) / spec.file
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")

        df = pd.read_csv(path)
        logger.info(f"Loading {len(df):,} {spec.name}...")

        created = self.batch_execute(
            build_query(spec),
            df.fillna("").to_dict("records"),
            batch_size=spec.batch_size,
            desc=spec.name,
        )
        result = LoadResult(spec.name, spec.file, rows=len(df), created=created)

        if result.skipped:
            logger.warning(
                f"⚠️  {spec.file}: {result.rows:,} rows but {result.created:,} "
                f"created ({result.skipped:,} skipped)"
            )
        else:
            logger.info(f"✓ Loaded {result.created:,} {spec.name}")

        return result

    def load_all(self, phases: list[Phase], data_dir: Path) -> list[LoadResult]:
        """
        Load every spec, phase by phase, in the order given.

        Order matters! Nodes must exist before relationships.
        """
        results = []
        for title, specs in phases:
            logger.info("\n" + "=" * 70)
            logger.info(title)
            logger.info("=" * 70)

            for spec in specs:
                results.append(self.load_spec(spec, data_dir))

        mismatched = [r for r in results if r.skipped]
        logger.info("\n" + "=" * 70)
        if mismatched:
            logger.warning(
                f"⚠️  Load finished, but {len(mismatched)} file(s) had rows "
                "that were not created:"
            )
            for r in mismatched:
                logger.warning(
                    f"   {r.file}: {r.skipped:,} of {r.rows:,} rows skipped"
                )
        else:
            logger.info("✅ All data loaded: every CSV row was created in the graph")
        logger.info("=" * 70)

        return results
