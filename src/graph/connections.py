"""
Connections - Lazy Singletons for Neo4j access

Provides lazily-initialized accessors for `GraphInterface` and
`SchemaProvider` so that no network calls are made at import time.

Usage (unchanged for callers):
    from src.graph.connections import graph_interface, schema_provider

Both names are lightweight proxies that create and cache the underlying
objects on first use, so importing this module never opens a Neo4j
connection by itself.
"""

import os
from typing import Optional

from dotenv import load_dotenv

from src.graph.graph_interface import GraphInterface
from src.graph.schema import SchemaProvider


load_dotenv()

_graph_interface: Optional[GraphInterface] = None
_schema_provider: Optional[SchemaProvider] = None


def get_graph_interface() -> GraphInterface:
    """Return a process-wide GraphInterface instance, initializing on first use."""
    global _graph_interface
    if _graph_interface is None:
        _graph_interface = GraphInterface(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
        )
    return _graph_interface


def get_schema_provider() -> SchemaProvider:
    """Return a process-wide SchemaProvider instance, initializing on first use."""
    global _schema_provider
    if _schema_provider is None:
        _schema_provider = SchemaProvider(get_graph_interface())
    return _schema_provider


class _GraphInterfaceProxy:
    """Attribute proxy that forwards to the lazily-initialized GraphInterface."""

    def __getattr__(self, name):
        return getattr(get_graph_interface(), name)


class _SchemaProviderProxy:
    """Attribute proxy that forwards to the lazily-initialized SchemaProvider."""

    def __getattr__(self, name):
        return getattr(get_schema_provider(), name)


# Backwards-compatible exports: these are cheap proxies
graph_interface = _GraphInterfaceProxy()
schema_provider = _SchemaProviderProxy()
