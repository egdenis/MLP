from .helpers import *
import numpy as np
from copy import  deepcopy


class MultilayerPerceptron():
	def __init__(self, layer_sizes, activation, learning_rate,
			   learning_rate_init, max_iter,shuffle, random_state, tol, categorical):

		self.layer_sizes = layer_sizes
		self.activation = activation
		self.learning_rate = learning_rate
		self.learning_rate_init = learning_rate_init
		self.max_iter = max_iter
		self.shuffle = shuffle
		self.random_state = random_state
		self.tol = tol 
		self.num_layers = len(layer_sizes)
		self.categorical = categorical


	def _init(self):
		"""
		Initilize weights. Note that we have included bias as a fully connected neuron in our network with the caveat 
		that the edge going into the bias neuron is 0 and it's activation is fixed at 1. Thus it does not affect the 
		values of neurons in the prior layers during backpropagation. 
		"""
		self.weights_ = []
		self.n_iter_ = 0
		for i in range(self.num_layers-1):
			weights_init = self._init_layer_coef(self.layer_sizes[i]+1,self.layer_sizes[i+1])
			self.weights_.append(weights_init)

		for i in range(self.num_layers-2):
			self.weights_[i] = np.append(self.weights_[i], np.array([[0] for layer in range(self.layer_sizes[i]+1)]),axis = 1)
		


	def _init_layer_coef(self, fan_in, fan_out):
		"""
		Initilizes the wieghts of a single layer. Uses initilization method recommended by Glorot et al. 
		See stats.stackexchange.com/q/47590 for more info. 

		Args:
			fan_in: number of input neurons
			fan_out: number of output neurons
		Returns: 
			weights: initilized weights for edges between two layers
		"""

		#6 is for hyperbolic tangent activation (which currently is not supported)
		const = 6
		bound = sqrt(const/(fan_in+fan_out)) 
		if self.activation.name is 'sigmoid':
			bound *= 4

		weights = self.random_state.uniform(bound, -bound, (fan_in, fan_out))
		
		return weights
			



	def _forward_pass(self, activations):
		# Based off of pseudo given in Understanding Machine Learning: From Theory to Algorithms pg 278
		"""
		Args:
			activations: list of numpy vectors, ith vector being the activations of the ith layer (first being input)
		Returns:

		"""
		for t in range(1,self.num_layers):
			activations[t-1][-1] = 1
			for i in range(activations[t].shape[0]):
				weighted_sum = np.dot(self.weights_[t-1][:,i],activations[t-1]) 
				activations[t][i] = self.activation.function(weighted_sum)

		#If output is categorical use softmax for final layer
		if self.categorical:
			weighted_sum = np.array([np.dot(self.weights_[t-1][:,i],activations[t-1]) for i in range(activations[-1].shape[0])])
			activations[-1] = softmax(weighted_sum, axis = 1)

			
	def _backward_pass(self, activations, deltas):
		#Start at penultimate layer since the final layer is initilized 
		for t in range(self.num_layers-2, -1, -1):
			for i in range(len(self.weights_[t])):
				#Note that this derivative is 
				deltas[t][i] = np.dot(self.weights_[t][i],deltas[t+1])*activations[t][i]*(1-activations[t][i])

	def _backprop(self, activations, weight_grad, deltas, y): 


		self._forward_pass(activations)

		# initilize the loss of the last layer 
		deltas[-1] = (y - activations[-1])*activations[-1]*(1-activations[-1])

		self._backward_pass(activations, deltas)

		#calculate gradients
		for t in range(len(weight_grad)):
			for i in range(len(weight_grad[t])):
				weight_grad[t][i] = deltas[t+1]*activations[t][i]


	def _fit(self, X, ys):
		assert X.shape[1] == self.layer_sizes[0], f"input vector size, {X.shape[1]}, does not match first layer size, {self.layer_sizes[0]}" 
		assert ys.shape[1] == self.layer_sizes[-1], f"output vector size, {ys.shape[1]} does not match first layer size, {layer_sizes}"

		for epoch in range(self.max_iter):
			X,ys = unison_shuffled_copies(X, ys, self.random_state)
			avg_cost = 0
			for x, y in zip(X,ys):

				activations = [np.zeros(size) for size in  self.layer_sizes]
				activations[0] = x

				#fix activation for bias at 1 so it does not effect downstream neurons
				activations = [np.append(layer,1) for layer in activations[:-1]] + [np.zeros(self.layer_sizes[-1])]
				deltas = deepcopy(activations)
				weight_grad = [np.full(layer.shape, np.nan) for layer in self.weights_]                 


				self._backprop(activations,weight_grad, deltas, y) # 0 -> regularization param
				self.weights_ = [weights + self.learning_rate * grad for weights, grad in zip(self.weights_, weight_grad)]
				avg_cost += 0.5*(y - activations[-1])**2


			avg_cost /= ys.shape[0]

			if np.sum(np.abs(avg_cost))<self.tol:
				print(f"Converged with tolerance of {np.sum(np.abs(avg_cost))} ")
				break

			print("cost:",np.sum(avg_cost))


	def _predict(self, X):
		y_preds = np.empty((0,self.layer_sizes[-1]))
		for x in X:
			activations = [np.zeros(size) for size in  self.layer_sizes]
			activations[0] = x
			activations = [np.append(layer,1) for layer in activations[:-1]] + [np.zeros(self.layer_sizes[-1])]
			self._forward_pass(activations)
			y_pred = activations[-1] == np.amax(activations[-1] )
			y_preds = np.vstack((y_preds, y_pred ))
		return y_preds








