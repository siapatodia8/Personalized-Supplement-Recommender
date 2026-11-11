#!/usr/bin/env python3
"""
Neo4j Knowledge Graph Data Loader

Wipes the database, applies constraints and indexes, then loads the DrugBank,
Mayo Clinic and nutrient CSVs listed in kg_loader/catalog.py.

Usage:
    python scripts/load_data.py          # asks for confirmation before wiping
    python scripts/load_data.py --yes    # no prompt (for automated runs)
"""

import argparse
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from kg_loader.catalog import PHASES
from kg_loader.loader import KnowledgeGraphLoader
from kg_loader.schema import apply_schema

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wipe Neo4j and reload the supplement safety knowledge graph."
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="skip the confirmation prompt (this DELETES all existing data)",
    )
    return parser.parse_args(argv)


def get_settings(env=os.environ) -> dict:
    """Read connection settings and the data directory from the environment."""
    password = env.get("NEO4J_PASSWORD")
    if not password:
        raise ValueError(
            "NEO4J_PASSWORD environment variable not set. "
            "Please create a .env file with NEO4J_PASSWORD=your_password"
        )

    # Data directory (should contain drugbank_data/ and mayo_clinic_data/)
    data_dir = Path(env.get("DATA_DIR", "data"))
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Could not find data directory: {data_dir}\n"
            f"Please ensure directory structure:\n"
            f"  {data_dir}/\n"
            f"    ├── drugbank_data/\n"
            f"    └── mayo_clinic_data/"
        )

    return {
        "uri": env.get("NEO4J_URI", "bolt://localhost:7687"),
        "user": env.get("NEO4J_USER", "neo4j"),
        "password": password,
        "data_dir": data_dir,
    }


def confirm_wipe(assume_yes: bool, ask=input) -> bool:
    """Return True if it is OK to delete all existing data."""
    if assume_yes:
        return True

    answer = ask(
        "\n⚠️  WARNING: This will DELETE ALL existing data and reload!\n"
        "Do you want to continue? (yes/no): "
    )
    return answer.strip().lower() in ("yes", "y")


def main(argv=None) -> None:
    args = parse_args(argv)

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Suppress Neo4j informational messages about existing constraints/indexes
    logging.getLogger("neo4j").setLevel(logging.WARNING)

    settings = get_settings()
    logger.info(f"Using data directory: {settings['data_dir']}")

    if not confirm_wipe(args.yes):
        logger.info("Import cancelled by user")
        return

    driver = GraphDatabase.driver(
        settings["uri"], auth=(settings["user"], settings["password"])
    )
    try:
        loader = KnowledgeGraphLoader(driver)

        logger.info("\nClearing database...")
        loader.clear_database()

        logger.info("\nCreating constraints and indexes...")
        apply_schema(driver)

        start_time = time.time()
        loader.load_all(PHASES, settings["data_dir"])
        elapsed = time.time() - start_time

        logger.info(
            f"⏱️  Total time: {int(elapsed // 60)} minutes {int(elapsed % 60)} seconds"
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
