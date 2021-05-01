from importlib.metadata import distribution
from crater.tensor import Tensor
from crater.gradient import Gradients, Gradient
import crater.operations as operations
import crater.premade as premade

__version__ = distribution("crater").version
