# 💊 Personalized Supplement Safety Advisor

[Project Website](https://kevinkchen1.github.io/project_website/)

An AI-powered system that detects dangerous supplement-medication interactions, identifies nutrient deficiencies, and provides personalized supplement recommendations using a biomedical knowledge graph and multi-agent architecture.

Built with **Neo4j** (knowledge graph), **LangGraph** (multi-agent orchestration), **Claude API** (reasoning), and **React** (web interface).

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the Application](#running-the-application)
- [Running the Tests](#running-tests)
- [Project Structure](#project-structure)

---

## Overview

Millions of people take dietary supplements without knowing they can dangerously interact with prescription medications. This system solves that by:

- **Safety Checks** — Detecting supplement-drug interactions across 4 pathways (direct, drug-drug, hidden pharma equivalence, similar effects)
- **Deficiency Analysis** — Identifying nutrient gaps from diet, supplements, and medications with critical overlap detection
- **Personalized Recommendations** — Suggesting safe supplements for conditions/symptoms, filtered against your medication profile

The system uses multi-hop reasoning over a Neo4j knowledge graph (329,849 nodes, 3.4M+ relationships) built from **DrugBank**, **Mayo Clinic** and **Manually Curated Nutrition** data, with a supervisor agent dynamically routing queries to specialist tools.

For detailed documentation, see below:

| Document | Description |
|---|---|
| [Workflow Overview](docs/workflow_overview.md) | Full workflow architecture |
| [Agents & Tools Reference](docs/agents_and_tools.md) | Detailed reference for each component: what it does, key design decisions, state reads/writes |
| [Knowledge Graph Structure](docs/knowledge_graph_structure.md) | All node types, relationship types, counts, and properties |
| [Prompts Reference](docs/prompts.md) | Every LLM prompt used in the system with inline comments |
| [Testing](docs/tests.md) | Human-in-the-loop testing |


---

## Prerequisites

| Dependency | Version | Purpose |
|---|---|---|
| **Python** | 3.12+ (< 3.14) | Runtime |
| **PDM** | Latest | Python package management |
| **Node.js** | 18+ | React frontend |
| **Neo4j** | 5.x+ | Knowledge graph database |
| **Anthropic API Key** | — | Claude LLM (entity extraction, normalization, supervisor) |
| **LangSmith API Key** | — | LangGraph Studio visual debugging |

### Install PDM

```bash
# macOS
brew install pdm

# Or via pip
pip install pdm
```

### Install and Start Neo4j

You need a running Neo4j instance:

**Neo4j Desktop** (recommended for local dev)
1. Download from [neo4j.com/download](https://neo4j.com/download/)
2. Create a new project and local DBMS
3. Set a password (you'll need this for `.env`)
4. Start the database



### Get API Keys

**Anthropic API Key** (required):
1. Go to [console.anthropic.com](https://console.anthropic.com/)
2. Sign up or log in
3. Navigate to **API Keys**
4. Create a new key (starts with `sk-ant-...`)

**LangSmith API Key** (required for LangGraph Studio):
1. Go to [smith.langchain.com](https://smith.langchain.com/)
2. Sign up (free — Google or GitHub login)
3. Click your **profile icon** → **Settings** → **API Keys**
4. Click **Create API Key** (starts with `lsv2_...`)

---

## Setup

### 1. Clone the Repository

Make sure Git LFS is installed (required for large data files):

```bash
brew install git-lfs      # macOS (if not already installed)
git lfs install
```

Then clone the repo
```bash
git clone https://github.com/kevinkchen1/Personalized-Supplement-Recommender.git
cd Personalized-Supplement-Recommender
git lfs pull
```

### 2. Install Dependencies

```bash
pdm install
```

This reads `pyproject.toml` and installs all dependencies into a virtual environment managed by PDM.

> **Troubleshooting:** If you see `No compatible lock target found`, the lock file may be pinned to a different Python version. Fix with:
> ```bash
> rm pdm.lock
> pdm install
> ```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# .env

# Required — Claude API for LLM reasoning
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxx

# Required — Neo4j connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Required for LangGraph Studio
LANGSMITH_API_KEY=lsv2_xxxxxxxxxxxxxxxxxxxxx
```

> ⚠️ **Never commit `.env` to version control.** An `.env.example` template is included in the repo for reference.

### 4. Load the Knowledge Graph

Before using the system, populate Neo4j with the biomedical data:

```bash
pdm run load-data
```

This loads DrugBank and Mayo Clinic data (28 CSV files) into the knowledge graph, creating 329,849 nodes and 3.4M+ relationships.

---

## Running the Application

### React Web App (Full Application)

The full application runs with two processes: a **FastAPI backend** and a **React frontend**. You will need two terminal windows.

**Terminal 1 — Start the API server** (from the project root):
```bash
pdm run api
```

**Terminal 2 — Start the React frontend**:
```bash
cd frontend
npm install
npm run dev
```

Once both are running, open [http://localhost:5173/](http://localhost:5173/) in your browser.

### LangGraph Studio (Recommended for Development)

LangGraph Studio provides a visual interface for running queries and inspecting the agent workflow step-by-step.

```bash
pdm run studio
```

This starts the LangGraph development server using the config at `langgraph-studio/langgraph.json`. Open the URL printed in your terminal to access the Studio UI where you can:

- Submit questions with patient profiles as `InputState` (just `user_question` and `patient_profile`)
- Watch the workflow route through Entity Extractor → Normalizer → Supervisor → Specialists in real time
- Inspect state at each node to see extracted entities, normalized IDs, and specialist results
- Replay and debug individual executions

---

### Verification Steps

Run these steps in order to verify each component of the system.

### 1. Verify Neo4j Connection

```bash
pdm run python -c "
from dotenv import load_dotenv; import os; load_dotenv()
from src.graph.graph_interface import GraphInterface
g = GraphInterface(os.getenv('NEO4J_URI','bolt://localhost:7687'), os.getenv('NEO4J_USER','neo4j'), os.getenv('NEO4J_PASSWORD'))
print('✅ Neo4j connected')
schema = g.get_schema_info()
print(f'   Node labels: {schema[\"node_labels\"]}')
print(f'   Relationship types: {len(schema[\"relationship_types\"])} types')
g.close()
"
```

### 2. Verify Knowledge Graph Data

```bash
pdm run python -c "
from dotenv import load_dotenv; import os; load_dotenv()
from src.graph.graph_interface import GraphInterface
g = GraphInterface(os.getenv('NEO4J_URI','bolt://localhost:7687'), os.getenv('NEO4J_USER','neo4j'), os.getenv('NEO4J_PASSWORD'))
stats = g.execute_query('''
  MATCH (s:Supplement) WITH count(s) as supplements
  MATCH (d:Drug) WITH supplements, count(d) as drugs
  MATCH (m:Medication) WITH supplements, drugs, count(m) as medications
  RETURN supplements, drugs, medications
''')
print(f'✅ Knowledge graph loaded: {stats[0]}')
g.close()
"
```

Expected: `{'supplements': 28, 'drugs': 19830, 'medications': 55}`

### 3. Verify Claude API Key

```bash
pdm run python -c "
from dotenv import load_dotenv; import os; load_dotenv()
from anthropic import Anthropic
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
r = client.messages.create(model='claude-sonnet-4-20250514', max_tokens=50, messages=[{'role':'user','content':'Say OK'}])
print(f'✅ Claude API working: {r.content[0].text}')
"
```

### 4. Verify Schema Provider and Singletons

```bash
pdm run python -c "
from dotenv import load_dotenv; load_dotenv()
from src.graph.connections import graph_interface, schema_provider
print('✅ Singletons initialized (GraphInterface + SchemaProvider)')
print(f'   Node labels: {schema_provider.get_node_labels()}')
print(f'   Relationship types: {len(schema_provider.get_relationship_types())} types')
print(f'   Sample Drug names: {schema_provider.get_sample_values(\"Drug\", \"drug_name\")}')
"
```

### 5. Verify LangGraph Workflow Builds

```bash
pdm run python -c "
from src.workflow.graph_builder import build_workflow
w = build_workflow()
print('✅ Workflow compiled successfully')
"
```

### 6. Run Full Pipeline (End-to-End)

```bash
pdm run python -c "
from dotenv import load_dotenv; load_dotenv()
from src.workflow.graph_builder import build_workflow
from src.workflow.state import create_initial_state

w = build_workflow()
state = create_initial_state(
    user_question='Is Fish Oil safe with Warfarin?',
    patient_profile={'medications': 'Warfarin', 'supplements': 'Fish Oil', 'conditions': [], 'dietary_restrictions': []}
)
result = w.invoke(state)
print('✅ End-to-end pipeline complete')
print(f'   Iterations: {result.get(\"iterations\", 0)}')
print(f'   Safety checked: {result.get(\"safety_checked\", False)}')
print(f'   Evidence chain:')
for e in result.get('evidence_chain', []):
    print(f'     → {e}')
"
```

Expected: The supervisor should route to `safety_check`, find 3 interactions between Fish Oil and Warfarin (drug-drug interaction + similar effect pathways), then route to `synthesize`.

---

## Running Tests

Run from the project root.

**Run all tests in a suite:**
```bash
python tests/safety/test_runner_safety.py
python tests/deficiency/test_runner_deficiency.py
python tests/recommendation/test_runner_recommendation.py
python tests/multi_agent/test_runner_multi_agent.py
```

**Run a single test by ID:**
```bash
python tests/safety/test_runner_safety.py --test-id 12
```

**Use a custom dataset:**
```bash
python tests/safety/test_runner_safety.py --dataset-path path/to/custom.csv
```

Reports are saved automatically to `tests/{type}/reports/report_{timestamp}.txt`.

---

## Project Structure

```
Personalized-Supplement-Recommender/
├── .env.example                    # Template for .env setup
├── .gitattributes                  # Git LFS tracking for large data files
├── .gitignore
├── pyproject.toml                  # PDM project config, dependencies, and scripts
├── pdm.lock                        # Locked dependency versions
├── data/                           # Data files in CSV
├── docs/                           
│   ├── workflow_overview.md        # Full workflow diagram and component summary
│   ├── agents_and_tools.md        # Detailed reference for each agent and tool
│   ├── knowledge_graph_structure.md # Node types, relationships, and counts
│   └── prompts.md                  # All LLM prompts with inline comments
├── frontend/                       # React/Vite single-page app; communicates with src/api/server.py
├── langgraph-studio/               # LangGraph Studio configuration and entry point 
├── scripts/
│   ├── load_data.py                # Entry point: wipes Neo4j and reloads the knowledge graph
│   ├── kg_loader/
│   │   ├── catalog.py              # What to load: one entry per CSV file, in load order
│   │   ├── schema.py               # Constraints and indexes
│   │   └── loader.py               # How to load: query builders, batching, count verification
│   └── tests/                      # Unit tests for the loader (pytest scripts/tests)
├── tests/                          # Per-agent test runners with CSV golden datasets and report outputs
└── src/
    ├── agents/                     
    │   ├── entity_extractor.py     # Hybrid: LLM extracts from question + deterministic profile parsing
    │   ├── entity_normalizer.py    # Agent: LLM generates Cypher to map names → DB IDs via live schema
    │   ├── synthesis.py            # Agent: LLM generated final human-readable answer
    │   └── supervisor.py           # Agent: LLM plans which specialist to call next (max 6 iterations)
    ├── api/
    │   └── server.py               # FastAPI server exposing the LangGraph workflow to the frontend  
    ├── graph/                      # Database infrastructure
    │   ├── graph_interface.py      # Neo4j connection wrapper, query execution, schema introspection
    │   ├── schema.py               # SchemaProvider — loads and caches DB schema for LLM prompts
    │   └── connections.py          # Module-level singletons (GraphInterface + SchemaProvider)
    ├── prompts/                    # YAML prompt files, one per agent
    ├── tools/                      # Deterministic specialist nodes (hardcoded Cypher, no LLM)
    │   ├── safety_check.py         # LLM generated cyphers to detect dangerours interactions
    │   ├── deficiency_check.py     # 3-pathway nutrient depletion (diet, medication, supplement) + overlap detection
    │   └── recommendation.py       # TREATS relationship lookup + keyword fallback for conditions
    └── workflow/                   # LangGraph orchestration
        ├── state.py                # ConversationState, InputState definitions, and create_initial_state()
        ├── graph_builder.py        # Build and compile the LangGraph workflow graph
        └── routing.py              # NodeNames constants + supervisor decision → node name mapping
```
---
## Contributors
- Jovanna Fernando
- Kevin Chen
- Parna Praveen
- Sia Patodia
- Mentor: Abed El-Husseini
- Mentor: Balaji Veeramani
