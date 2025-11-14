#!/usr/bin/env python
"""CLI wrapper for the preprocessing pipeline."""

from __future__ import annotations

from data.preprocessing.pipeline import process_unified_index


def main() -> None:
    process_unified_index()


if __name__ == "__main__":
    main()
