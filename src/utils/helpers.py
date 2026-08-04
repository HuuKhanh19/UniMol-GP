"""
Utility Functions

Common utilities for timing, JSON I/O and console output.
"""

import time
import json
from typing import Dict, Any


class Timer:
    """Simple timer context manager."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"{self.name} completed in {elapsed:.2f} seconds")

    @property
    def elapsed(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


def save_json(data: Dict[str, Any], path: str):
    """Save dictionary to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def print_banner(text: str, char: str = "=", width: int = 60):
    """Print a banner with centered text."""
    print(char * width)
    print(text.center(width))
    print(char * width)
