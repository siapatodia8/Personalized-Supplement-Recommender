"""
Catalog: the list of what gets loaded into the knowledge graph.

Describes the 28 CSV files (DrugBank, Mayo Clinic, and nutrient data): what each
one becomes in the graph and the order to load them. It holds descriptions only;
the loading logic lives in loader.py.

Each CSV becomes either:
  - a NodeSpec: one kind of node (a "thing", e.g. a Drug or a Supplement), or
  - a RelSpec: one kind of relationship (a connection between two nodes, e.g.
    Drug -[:BELONGS_TO]-> Category).

Used by scripts/load_data.py, which passes PHASES to KnowledgeGraphLoader.load_all().
To add a new data source, add one spec to PHASES below. No new code is needed.
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
    """One CSV file that becomes one kind of relationship.

    Example: supplement_contains.csv has a row (Fish oil, DHA). Each row becomes
    one connection in the graph:  (Fish oil) -[:CONTAINS]-> (DHA)
    """

    rel_type: str  # Neo4j relationship type, e.g. "BELONGS_TO"
    file: str  # CSV path relative to the data directory
    start: Endpoint
    end: Endpoint
    props: dict[str, str] = field(default_factory=dict)  # property -> CSV column
    batch_size: int = 5000

    @property
    def name(self) -> str:
        '''builds a label like Drug -[:BELONGS_TO]-> Category relationships'''
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
        # Example: aspirin (stored as "Acetylsalicylic acid") belongs to the
        # "Analgesics" (pain relievers) category.
        RelSpec("BELONGS_TO", f"{DRUGBANK}/drug_category_belongs_to.csv",
                DRUG, CATEGORY, batch_size=10000),
        # Example: aspirin "may increase the anticoagulant activities of Warfarin".
        # The description text is stored on the connection.
        RelSpec("INTERACTS_WITH", f"{DRUGBANK}/drug_drug_interacts_with.csv",
                DRUG, Endpoint("Drug", "drug_id", column="interacting_drug_id"),
                props={"description": "description"}, batch_size=10000),
        # Example: the brand name "Advil" contains the drug Ibuprofen.
        RelSpec("CONTAINS_DRUG", f"{DRUGBANK}/brand_name_drug_contains.csv",
                BRAND_NAME, DRUG),
        # Example: the drug Acetylsalicylic acid is also known as "Aspirin".
        RelSpec("KNOWN_AS", f"{DRUGBANK}/drug_synonym_known_as.csv", DRUG, SYNONYM),
        # Example: the drug Metformin comes as the salt form "Metformin
        # hydrochloride".
        RelSpec("HAS_SALT_FORM", f"{DRUGBANK}/drug_salt_has_salt_form.csv",
                DRUG, SALT),
        # Example: Warfarin has the food note "Avoid grapefruit products...";
        # Ibuprofen has "Take with food. Food reduces irritation."
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
        # Example: the supplement Fish oil contains the active ingredients DHA and
        # EPA. `is_primary` marks the main ingredient (true/false).
        RelSpec("CONTAINS", f"{MAYO}/supplement_contains.csv",
                SUPPLEMENT, ACTIVE_INGREDIENT,
                props={"is_primary": "is_primary"}, batch_size=1000),
        # Example: the medication entry "Alcohol" points to the drug "Ethanol".
        # This links Mayo's everyday names to DrugBank's official ones.
        RelSpec("MEDICATION_CONTAINS_DRUG", f"{MAYO}/medication_drug_contains.csv",
                MEDICATION, DRUG, batch_size=1000),
        # Example: the supplement Ginkgo interacts with the medication Ibuprofen
        # (both can affect bleeding).
        RelSpec("SUPPLEMENT_INTERACTS_WITH",
                f"{MAYO}/supplement_medication_interacts_with.csv",
                SUPPLEMENT, MEDICATION, batch_size=1000),
        # Example: the supplement Ginkgo can cause the symptom "Bleeding";
        # Fish oil can cause "A Fishy Aftertaste".
        RelSpec("CAN_CAUSE", f"{MAYO}/supplement_symptom_can_cause.csv",
                SUPPLEMENT, SYMPTOM, batch_size=1000),
        # Example: the supplement Melatonin is used for the symptom "Insomnia".
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
        # Example: a Vegan diet is deficient in Vitamin B12, with risk level "High".
        RelSpec("DEFICIENT_IN",
                f"{MAYO}/dietary_restriction_deficient_in_nutrient.csv",
                DIETARY_RESTRICTION, NUTRIENT,
                props={"risk_level": "risk_level"}, batch_size=500),
        # Supplements that deplete, interfere with, or mask deficiency of a nutrient.
        # Example: Zinc "depletes" Copper, severity CRITICAL. Zinc also
        # interferes with Iron absorption (severity MODERATE).
        RelSpec("NEGATIVE_INTERACTION",
                f"{MAYO}/supplement_interacts_with_nutrient.csv",
                SUPPLEMENT, NUTRIENT,
                props={"mechanism": "mechanism", "severity": "severity",
                       "notes": "notes"},
                batch_size=500),
        # Medications that affect nutrient levels.
        # The CSV column is `interaction`; the graph property is `interaction_type`.
        # Example: Metformin "depletes" Vitamin B12; Atorvastatin (a statin)
        # "depletes" CoQ10.
        RelSpec("INTERACTS_WITH_NUTRIENT", f"{DRUGBANK}/drug_nutrient_interacts.csv",
                DRUG, NUTRIENT,
                props={"interaction_type": "interaction"}, batch_size=500),
    ]),
    # Bridges connect the Mayo Clinic side to the DrugBank side.
    # These are what make the safety checks possible.
    ("PHASE 5: Bridge relationships", [
        # Chemical equivalence: an ingredient inside a supplement is the same
        # compound as a drug.
        # Example: Monacolin K (inside Red yeast rice) is "identical" to the drug
        # Lovastatin, so Red yeast rice carries Lovastatin's drug interactions.
        # "related" is used when the compounds are close but not the same
        # (e.g. Nicotinamide is related to Niacin).
        RelSpec("EQUIVALENT_TO", f"{MAYO}/active_ingredient_equivalent_to_drug.csv",
                ACTIVE_INGREDIENT, DRUG,
                props={"equivalence_type": "equivalence_type", "notes": "notes"},
                batch_size=500),
        # Pharmacological similarity: a supplement acts like a whole drug category,
        # even though it is not that drug.
        # Example: Fish oil acts like the "Anticoagulants" (blood thinners)
        # category, with confidence "high".
        RelSpec("HAS_SIMILAR_EFFECT_TO",
                f"{MAYO}/supplement_category_similar_effect.csv",
                SUPPLEMENT, CATEGORY,
                props={"confidence": "confidence", "notes": "notes"},
                batch_size=500),
    ]),
]
