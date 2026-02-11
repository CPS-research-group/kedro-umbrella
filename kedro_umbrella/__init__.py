"""``kedro.pipeline`` provides functionality to define and execute
data-driven pipelines.
"""

__version__ = "0.2.1"

from .blocks.code import Coder, coder
from .blocks.compose import Composer, composer
from .blocks.process import Processor, processor
from .blocks.train import Trainer, trainer

__all__ = [
    "coder",
    "processor",
    "trainer",
    "composer",
    "Coder",
    "Processor",
    "Trainer",
    "Composer",
]
