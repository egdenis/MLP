import numpy as np
from math import sqrt
from numpy.random import RandomState
from abc import ABC,  abstractmethod


class MultilayerPerceptron():
	def __init__(self, hidden_layer_sizes, activation,solver, learning_rate,
			   learning_rate_init, max_iter,shuffle, random_state, tol):

		self.hidden_layer_sizes = hidden_layer_sizes
		self.activation = activation
		self.solver = solver
		self.learning_rate = learning_rate
		self.learning_rate_init = learning_rate_init
		self.max_iter = max_iter
		self.shuffle = shuffle
		self.random_state = random_state
		self.tol = tol  

	def _init(self, all_layers):
		self.coefs = []
		self.intercepts = []
		for i in range(len(all_layers)-1):
			coefs_init, intercepts_init = self._init_layer_coef(all_layers[i],all_layers[i+1])
			self.coefs.append(coefs_init)
			self.intercepts.append(intercepts_init)

	def _init_layer_coef(self, fan_in, fan_out):
	# SKLearn uses the initialization method recommended by Glorot et al.
	# See stats.stackexchange.com/q/47590 for more info
		const = 6
		#r for hyperbolic tangent units
		bound = sqrt(const/(fan_in+fan_out)) 
		if self.activation.name is 'sigmoid':
			bound *= 4

		coefs = self.random_state.uniform(bound, -bound, (fan_in, fan_out))
		intercepts = self.random_state.uniform(bound, -bound, fan_out)
		return coefs, intercepts


	def _forward_pass(self, activations):
		# Based off of pseudo given in Understanding Machine Learning: From Theory to Algorithms pg 278
		"""
		Args:
			activations: matrix of activations, ith vector being the activations of the ith layer (first being input)
		Returns:
		"""
		for activ in range(self.hidden_layer_sizes.shape[0]):
			pass

class BaseActivation(ABC):
	"""Base class for all actiation functions"""
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


def main():
	random_seed = RandomState(1234567890)
	mlp = MultilayerPerceptron(hidden_layer_sizes = [4,1], activation = Identity(), solver = "SGD",
								learning_rate = 0.02, learning_rate_init = 0.02, max_iter = 500,
								shuffle = True, random_state = random_seed, tol = 0.01)

	mlp._init([2]+ mlp.hidden_layer_sizes + [1])	
	for coef in mlp.coefs:
		print(coef,"\n")



if __name__ == "__main__":
	main()







