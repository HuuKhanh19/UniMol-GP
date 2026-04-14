"""
Utility Functions

Common utilities for logging, timing, and other helpers.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    experiment_name: str = "experiment"
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        experiment_name: Name for log file
        
    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(experiment_name)
    logger.info(f"Logging to: {log_file}")
    
    return logger


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


def format_results_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Format results as a table string.
    
    Args:
        results: Dict mapping dataset names to metric dicts
        
    Returns:
        Formatted table string
    """
    lines = []
    lines.append("-" * 50)
    lines.append(f"{'Dataset':<15} {'Metric':<10} {'Value':<15}")
    lines.append("-" * 50)
    
    for dataset, metrics in results.items():
        for metric_name, value in metrics.items():
            lines.append(f"{dataset:<15} {metric_name:<10} {value:<15.4f}")
    
    lines.append("-" * 50)
    
    return "\n".join(lines)
