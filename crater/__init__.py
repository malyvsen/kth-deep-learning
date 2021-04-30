from importlib.metadata import distribution
from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient
import crater.operations
from .layer import Layer

__version__ = distribution("crater").version
