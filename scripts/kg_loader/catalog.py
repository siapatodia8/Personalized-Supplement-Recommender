"""
Catalog: WHAT goes into the knowledge graph (data, not code).

Every CSV becomes either one kind of node or one kind of relationship.
To add a new data source, add one spec to PHASES below. No new code needed.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NodeSpec:
    """One CSV file that becomes one kind of node."""

    label: str  # Neo4j node label, e.g. "Drug"
    file: str  # CSV path relative to the data directory
    props: tuple[str, ...]  # properties to set on the node
    batch_size: int = 5000
    # Only for properties whose CSV column has a different name:
    # property -> CSV column. Every other property reads the same-named column.
    column_for: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.label} nodes"

    def csv_column(self, prop: str) -> str:
        return self.column_for.get(prop, prop)


@dataclass(frozen=True)
class Endpoint:
    """One end of a relationship: which node to match, and by which key."""

    label: str  # node label to match, e.g. "Drug"
    key: str  # node property to match on, e.g. "drug_id"
    column: str = ""  # CSV column holding the key (defaults to `key`)

    @property
    def csv_column(self) -> str:
        return self.column or self.key


@dataclass(frozen=True)
class RelSpec:
    """One CSV file that becomes one kind of relationship."""

    rel_type: str  # Neo4j relationship type, e.g. "BELONGS_TO"
    file: str  # CSV path relative to the data directory
    start: Endpoint
    end: Endpoint
    props: dict[str, str] = field(default_factory=dict)  # property -> CSV column
    batch_size: int = 5000

    @property
    def name(self) -> str:
        return (
            f"{self.start.label} -[:{self.rel_type}]-> {self.end.label} relationships"
        )


Spec = NodeSpec | RelSpec
Phase = tuple[str, list[Spec]]  # (title shown in the log, specs to load)


DRUGBANK = "drugbank_data"
MAYO = "mayo_clinic_data"

# Reusable endpoints (label + ID property)
DRUG = Endpoint("Drug", "drug_id")
CATEGORY = Endpoint("Category", "category_id")
BRAND_NAME = Endpoint("BrandName", "brand_name_id")
SYNONYM = Endpoint("Synonym", "synonym_id")
SALT = Endpoint("Salt", "salt_id")
FOOD_INTERACTION = Endpoint("FoodInteraction", "food_interaction_id")
SUPPLEMENT = Endpoint("Supplement", "supplement_id")
ACTIVE_INGREDIENT = Endpoint("ActiveIngredient", "active_ingredient_id")
MEDICATION = Endpoint("Medication", "medication_id")
SYMPTOM = Endpoint("Symptom", "symptom_id")
NUTRIENT = Endpoint("Nutrient", "nutrient_id")
DIETARY_RESTRICTION = Endpoint("DietaryRestriction", "dietary_restriction_id")

# Order matters: a relationship can only be created after both of its nodes exist.
PHASES: list[Phase] = [
    ("PHASE 1: DrugBank core entities", [
        NodeSpec("Drug", f"{DRUGBANK}/drugs.csv",
                 ("drug_id", "drug_name", "description", "indication",
                  "type", "toxicity", "indication_text")),
        NodeSpec("Category", f"{DRUGBANK}/categories.csv",
                 ("category_id", "category")),
        NodeSpec("BrandName", f"{DRUGBANK}/brand_names.csv",
                 ("brand_name_id", "brand_name")),
        NodeSpec("Synonym", f"{DRUGBANK}/synonyms.csv", ("synonym_id", "synonym")),
        NodeSpec("Salt", f"{DRUGBANK}/salts.csv", ("salt_id", "salt_name")),
        # The CSV column is `food_interaction`; the graph property is `description`.
        NodeSpec("FoodInteraction", f"{DRUGBANK}/food_interactions.csv",
                 ("food_interaction_id", "description"),
                 column_for={"description": "food_interaction"}),
    ]),
    ("PHASE 2: DrugBank relationships", [
        RelSpec("BELONGS_TO", f"{DRUGBANK}/drug_category_belongs_to.csv",
                DRUG, CATEGORY, batch_size=10000),
        RelSpec("INTERACTS_WITH", f"{DRUGBANK}/drug_drug_interacts_with.csv",
                DRUG, Endpoint("Drug", "drug_id", column="interacting_drug_id"),
                props={"description": "description"}, batch_size=10000),
        RelSpec("CONTAINS_DRUG", f"{DRUGBANK}/brand_name_drug_contains.csv",
                BRAND_NAME, DRUG),
        RelSpec("KNOWN_AS", f"{DRUGBANK}/drug_synonym_known_as.csv", DRUG, SYNONYM),
        RelSpec("HAS_SALT_FORM", f"{DRUGBANK}/drug_salt_has_salt_form.csv",
                DRUG, SALT),
        RelSpec("HAS_FOOD_INTERACTION",
                f"{DRUGBANK}/drug_food_interaction_has_food_interaction.csv",
                DRUG, FOOD_INTERACTION),
    ]),
    ("PHASE 3: Mayo Clinic core entities", [
        NodeSpec("Supplement", f"{MAYO}/supplements.csv",
                 ("supplement_id", "supplement_name", "safety_rating"),
                 batch_size=1000),
        NodeSpec("ActiveIngredient", f"{MAYO}/active_ingredients.csv",
                 ("active_ingredient_id", "active_ingredient"), batch_size=1000),
        NodeSpec("Medication", f"{MAYO}/medications.csv",
                 ("medication_id", "medication_name"), batch_size=1000),
        NodeSpec("Symptom", f"{MAYO}/symptoms.csv",
                 ("symptom_id", "symptom_name"), batch_size=1000),
    ]),
    ("PHASE 4: Mayo Clinic relationships", [
        RelSpec("CONTAINS", f"{MAYO}/supplement_contains.csv",
                SUPPLEMENT, ACTIVE_INGREDIENT,
                props={"is_primary": "is_primary"}, batch_size=1000),
        RelSpec("MEDICATION_CONTAINS_DRUG", f"{MAYO}/medication_drug_contains.csv",
                MEDICATION, DRUG, batch_size=1000),
        RelSpec("SUPPLEMENT_INTERACTS_WITH",
                f"{MAYO}/supplement_medication_interacts_with.csv",
                SUPPLEMENT, MEDICATION, batch_size=1000),
        RelSpec("CAN_CAUSE", f"{MAYO}/supplement_symptom_can_cause.csv",
                SUPPLEMENT, SYMPTOM, batch_size=1000),
        RelSpec("TREATS", f"{MAYO}/supplement_symptom_treats.csv",
                SUPPLEMENT, SYMPTOM, batch_size=1000),
    ]),
    ("PHASE 4.5: Nutrient deficiency data", [
        NodeSpec("Nutrient", f"{MAYO}/nutrients.csv",
                 ("nutrient_id", "nutrient_name", "category", "description"),
                 batch_size=500),
        NodeSpec("DietaryRestriction", f"{MAYO}/dietary_restrictions.csv",
                 ("dietary_restriction_id", "dietary_restriction_name",
                  "description"),
                 batch_size=500),
        RelSpec("DEFICIENT_IN",
                f"{MAYO}/dietary_restriction_deficient_in_nutrient.csv",
                DIETARY_RESTRICTION, NUTRIENT,
                props={"risk_level": "risk_level"}, batch_size=500),
        # Supplements that deplete, interfere with, or mask deficiency of a
        # nutrient (e.g. Zinc depletes Copper)
        RelSpec("NEGATIVE_INTERACTION",
                f"{MAYO}/supplement_interacts_with_nutrient.csv",
                SUPPLEMENT, NUTRIENT,
                props={"mechanism": "mechanism", "severity": "severity",
                       "notes": "notes"},
                batch_size=500),
        # Medications that affect nutrient levels (e.g. Metformin depletes B12).
        # The CSV column is `interaction`; the graph property is `interaction_type`.
        RelSpec("INTERACTS_WITH_NUTRIENT", f"{DRUGBANK}/drug_nutrient_interacts.csv",
                DRUG, NUTRIENT,
                props={"interaction_type": "interaction"}, batch_size=500),
    ]),
    # Bridges connect the Mayo Clinic side to the DrugBank side.
    # These are what make the safety checks possible.
    ("PHASE 5: Bridge relationships", [
        # Chemical equivalence, e.g. Monacolin K (in Red yeast rice) = Lovastatin
        RelSpec("EQUIVALENT_TO", f"{MAYO}/active_ingredient_equivalent_to_drug.csv",
                ACTIVE_INGREDIENT, DRUG,
                props={"equivalence_type": "equivalence_type", "notes": "notes"},
                batch_size=500),
        # Pharmacological similarity, e.g. Ginkgo acts like Anticoagulants
        RelSpec("HAS_SIMILAR_EFFECT_TO",
                f"{MAYO}/supplement_category_similar_effect.csv",
                SUPPLEMENT, CATEGORY,
                props={"confidence": "confidence", "notes": "notes"},
                batch_size=500),
    ]),
]
