"""
Schema: uniqueness constraints and lookup indexes.

Applied BEFORE any data is loaded. A uniqueness constraint also creates an
index, so matching a node by its ID while loading relationships is a fast
lookup instead of a scan over every node.
"""

import logging

logger = logging.getLogger(__name__)

SCHEMA_STATEMENTS = [
    # ========== DRUGBANK NODES ==========
    "CREATE CONSTRAINT drug_id_unique IF NOT EXISTS "
    "FOR (d:Drug) REQUIRE d.drug_id IS UNIQUE",
    "CREATE CONSTRAINT category_id_unique IF NOT EXISTS "
    "FOR (c:Category) REQUIRE c.category_id IS UNIQUE",
    "CREATE CONSTRAINT brand_name_id_unique IF NOT EXISTS "
    "FOR (b:BrandName) REQUIRE b.brand_name_id IS UNIQUE",
    "CREATE CONSTRAINT synonym_id_unique IF NOT EXISTS "
    "FOR (s:Synonym) REQUIRE s.synonym_id IS UNIQUE",
    "CREATE CONSTRAINT salt_id_unique IF NOT EXISTS "
    "FOR (s:Salt) REQUIRE s.salt_id IS UNIQUE",
    "CREATE CONSTRAINT food_interaction_id_unique IF NOT EXISTS "
    "FOR (f:FoodInteraction) REQUIRE f.food_interaction_id IS UNIQUE",
    # ========== MAYO CLINIC NODES ==========
    "CREATE CONSTRAINT supplement_id_unique IF NOT EXISTS "
    "FOR (s:Supplement) REQUIRE s.supplement_id IS UNIQUE",
    "CREATE CONSTRAINT active_ingredient_id_unique IF NOT EXISTS "
    "FOR (a:ActiveIngredient) REQUIRE a.active_ingredient_id IS UNIQUE",
    "CREATE CONSTRAINT medication_id_unique IF NOT EXISTS "
    "FOR (m:Medication) REQUIRE m.medication_id IS UNIQUE",
    "CREATE CONSTRAINT symptom_id_unique IF NOT EXISTS "
    "FOR (s:Symptom) REQUIRE s.symptom_id IS UNIQUE",
    "CREATE CONSTRAINT nutrient_id_unique IF NOT EXISTS "
    "FOR (n:Nutrient) REQUIRE n.nutrient_id IS UNIQUE",
    "CREATE CONSTRAINT dietary_restriction_id_unique IF NOT EXISTS "
    "FOR (dr:DietaryRestriction) REQUIRE dr.dietary_restriction_id IS UNIQUE",
    # ========== INDEXES FOR FAST LOOKUPS ==========
    "CREATE INDEX drug_name_idx IF NOT EXISTS FOR (d:Drug) ON (d.drug_name)",
    "CREATE INDEX supplement_name_idx IF NOT EXISTS "
    "FOR (s:Supplement) ON (s.supplement_name)",
    "CREATE INDEX medication_name_idx IF NOT EXISTS "
    "FOR (m:Medication) ON (m.medication_name)",
    "CREATE INDEX active_ingredient_name_idx IF NOT EXISTS "
    "FOR (a:ActiveIngredient) ON (a.active_ingredient)",
    "CREATE INDEX category_name_idx IF NOT EXISTS FOR (c:Category) ON (c.category)",
]


def apply_schema(driver) -> None:
    """Create every constraint and index.

    Clearing the database deletes the data but leaves its constraints and
    indexes in place, so on a rerun they already exist. `IF NOT EXISTS` makes
    each statement safe to repeat instead of raising an error. That also means
    any error raised here is a real problem and is allowed to stop the load.
    """
    with driver.session() as session:
        for statement in SCHEMA_STATEMENTS:
            session.run(statement)

    logger.info(f"✓ Applied {len(SCHEMA_STATEMENTS)} constraints and indexes")
