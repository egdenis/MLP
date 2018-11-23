from abc import ABC,  abstractmethod
from math import sqrt, exp
import numpy as np
def unison_shuffled_copies(a, b, random = None):
    if random is None:
        random = np.random
    assert len(a) == len(b)
    p = random.permutation(len(a))
    return a[p], b[p]


class BaseActivation(ABC):
    """Base class for all activation functions"""
    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def function(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass


class Identity(BaseActivation):
	""" Identity activation for testing"""
	name = "identity"

	def function(self, x):
		return x

	def derivative(self, x):
		return 1

class Sigmoid(BaseActivation):
	""" Identity activation for testing"""
	name = "sigmoid"

	def function(self, x):
		return 1 / (1 + exp(-x)) 

	def derivative(self, x):
		return self.function(x)*(1-self.function(x)) 


def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter, 
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    
    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p