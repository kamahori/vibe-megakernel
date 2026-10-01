"""Compatibility entry point for the MegaBench test suite."""

from .tests.test_suite import CatalogTests, HarnessTests

__all__ = ["CatalogTests", "HarnessTests"]
