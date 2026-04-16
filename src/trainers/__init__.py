"""
Trainers Module

Step 1: Gradient Descent  (src/models/unimol_wrapper.py)
Step 2: EGGROLL ES        (src/trainers/step2_eggroll.py)
Step 3: EGGROLL + GP      (future)
"""

from .step2_eggroll import Step2Trainer

__all__ = ['Step2Trainer']