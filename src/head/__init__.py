"""Symbolic (genetic-programming) head: k formula trees merged by ridge."""

from src.head.gp import CoevolutionGP, GPConfig, fit_final
from src.head.gp_head import GPHead

__all__ = ['CoevolutionGP', 'GPConfig', 'GPHead', 'fit_final']
