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
		self.num_layers = len(hidden_layer_sizes) + 2

	def _init(self, all_layers):
		self.weights_ = []
		self.biases_ = []
		for i in range(self.num_layers-1):
			weights_init, biases_init = self._init_layer_coef(all_layers[i],all_layers[i+1])
			self.weights_.append(weights_init)
			self.biases_.append(biases_init)

	def _init_layer_coef(self, fan_in, fan_out):
	# SKLearn uses the initialization method recommended by Glorot et al.
	# See stats.stackexchange.com/q/47590 for more info
		const = 6
		#r for hyperbolic tangent units
		bound = sqrt(const/(fan_in+fan_out)) 
		if self.activation.name is 'sigmoid':
			bound *= 4

		weights = self.random_state.uniform(bound, -bound, (fan_in, fan_out))
		biases = self.random_state.uniform(bound, -bound, fan_out)
		return weights, biases
            



	def _forward_pass(self, activations, preactivations):
		# Based off of pseudo given in Understanding Machine Learning: From Theory to Algorithms pg 278
		"""
		Args:
			activations: list of numpy vectors, ith vector being the activations of the ith layer (first being input)
		Returns:

		"""
		for t in range(1,self.num_layers):
			for i in range(activations[t].shape[0]):
				preactivations[t][i] = np.dot(self.weights_[t-1][:,i],activations[t-1]) + self.biases_[t-1][i]
				activations[t][i] = self.activation.function(preactivations[t][i])


##### FORGOT biases

	def _backward_pass(self, activations, activation_derivs, deltas):
		for t in range(len(self.weights_)-2, -1, -1):
			for i in range(len(self.weights_[t])):
				deltas[t][i] = np.dot(self.weights_[t][:,i],deltas[t+1]*activation_derivs[t+1])
	

	def _backprop(self, activations, weight_grad, deltas, y): #return deltas for biases!!!!!!
		preactivations = [np.empty_like(layer) for layer in activations]
		self._forward_pass(activations, preactivations)

		#Computing the derivative of activation function with the weighted sum of input
		#neurons (preactivions variable) to avoid calculating it twice, once in doing backpass
		#and once for gradient calc

		activation_deriv_vec = np.vectorize(self.activation.derivative)
		activation_derivs = [activation_deriv_vec(preact) for preact in preactivations ]

		# initilize the loss of the last layer 
		deltas[-1] = activations[-1] - y

		self._backward_pass(activations, activation_derivs, deltas)

		#Calculate gradients for each weight of edges i->j
		for t in range(len(weight_grad)):
			for i in range(len(weight_grad[t])):
				weight_grad[t] = deltas[t]*activation_derivs[t]*activations[t][i]
				#return deltas

	def _fit(self, X, ys):
		for epoch in range(self.max_iter):
			np.random.shuffle(X)
			
			for x, y in zip(X,ys):
				activations = [np.empty_like(layer) for layer in [x] + self.biases_ + y]
				weight_grad = np.copy(activations)
				deltas = np.copy(activations)
				activations[0] = x

				self._backprop(activations,weight_grad, deltas, y) # 0 -> regularization param
				self.weights_ = [weights - self.learning_rate * (grad + weights * 0 ) for weights, grad in zip(self.weights_, weight_grad)]
				loss = deltas[-1]
				print(x, y, activations[-1])

			if(loss < self.tol):
				break
				

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


def main():
	random_seed = RandomState(1234567890)
	X = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
	y = np.array([1,1,0.7,-0.7,-1,-1])
	mlp = MultilayerPerceptron(hidden_layer_sizes = [4], activation = Identity(), solver = "SGD",
								learning_rate = 0.2, learning_rate_init = 0.02, max_iter = 20,
								shuffle = True, random_state = random_seed, tol = 0.01)
	

	mlp._init([X.shape[1]]+ mlp.hidden_layer_sizes + [1])

	for coef in mlp.weights_:
		print(coef.shape)

	mlp._fit(X,y)
		
	



if __name__ == "__main__":
	main()







