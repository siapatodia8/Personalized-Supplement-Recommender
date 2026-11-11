"""Knowledge graph loader: the pieces behind scripts/load_data.py.

- catalog.py : WHAT to load (one entry per CSV file, in load order)
- schema.py  : constraints and indexes
- loader.py  : HOW to load (query builders, batching, clearing, verification)
"""
