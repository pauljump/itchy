"""Itchy: small personal judgment models learned from reviewed choices."""
from .model import Prediction, Predictor
from .task import Decision, Task
from .preference import PreferenceModel

__all__ = ["Decision", "Prediction", "Predictor", "PreferenceModel", "Task"]
__version__ = "0.1.0"
