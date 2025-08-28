"""``kedro.pipeline`` provides functionality to define and execute
data-driven pipelines.
"""

__version__ = "0.0.1"

from .code import Coder, coder
from .process import Processor, processor
from .train import Trainer, trainer
from .compose import Composer, composer

__all__ = ["coder", "processor", "trainer", "composer",
           "Coder", "Processor", "Trainer", "Composer"]
