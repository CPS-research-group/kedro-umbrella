"""``kedro.pipeline`` provides functionality to define and execute
data-driven pipelines.
"""

__version__ = "0.0.1"

from .code import Coder, coder
from .process import Processor, processor
from .train import Trainer, trainer

__all__ = ["coder", "processor", "trainer", "Coder", "Processor", "Trainer"]
