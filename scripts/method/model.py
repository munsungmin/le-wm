"""Public world model classes for Python imports and Hydra targets."""

from .lewm.model.model import JEPA
from .ffjepa.model.model import ffJEPA

jepa = JEPA
ffjepa = ffJEPA

__all__ = ["JEPA", "ffJEPA", "jepa", "ffjepa"]
