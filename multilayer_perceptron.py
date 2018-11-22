import numpy as np
from math import sqrt, exp
from numpy.random import RandomState
from abc import ABC,  abstractmethod
from copy import  deepcopy


class MultilayerPerceptron():
	def __init__(self, layer_sizes, activation,solver, learning_rate,
			   learning_rate_init, max_iter,shuffle, random_state, tol):

		self.layer_sizes = layer_sizes
		self.activation = activation
		self.solver = solver
		self.learning_rate = learning_rate
		self.learning_rate_init = learning_rate_init
		self.max_iter = max_iter
		self.shuffle = shuffle
		self.random_state = random_state
		self.tol = tol 
		self.num_layers = len(layer_sizes)

	def _init(self):
		self.weights_ = []
		self.biases_ = []
		for i in range(self.num_layers-1):
			weights_init, biases_init = self._init_layer_coef(self.layer_sizes[i]+1,self.layer_sizes[i+1])
			self.weights_.append(weights_init)
			self.biases_.append(biases_init)

		for i in range(self.num_layers-2):
			self.weights_[i] = np.append(self.weights_[i], np.array([[0] for layer in range(self.layer_sizes[i]+1)]),axis = 1)
		
		for i in range(self.num_layers-1):
			print(self.weights_[i])
		#self.weights_ = [np.array([[0.87226208, 0.63434208,-0.65554529],[-0.550579,-0.2502144,-0.36035178]]), np.array([[0.91193366],[0.63712029],[0.618395]])]
		print(self.weights_)


	def _init_layer_coef(self, fan_in, fan_out):
	# SKLearn uses the initialization method recommended by Glorot et al.
	# See stats.stackexchange.com/q/47590 for more info
		const = 6
		#r for hyperbolic tangent units
		bound = sqrt(const/(fan_in+fan_out)) 
		if self.activation.name is 'sigmoid':
			bound *= 4

		weights = self.random_state.uniform(bound, -bound, (fan_in, fan_out))
		
		biases =  np.zeros(fan_out)#self.random_state.uniform(bound, -bound, fan_out)
		return weights, biases
			



	def _forward_pass(self, activations):#, preactivations):
		# Based off of pseudo given in Understanding Machine Learning: From Theory to Algorithms pg 278
		"""
		Args:
			activations: list of numpy vectors, ith vector being the activations of the ith layer (first being input)
		Returns:

		"""
		#print(activations)
		for t in range(1,self.num_layers):
			activations[t-1][-1] = 1
			for i in range(activations[t].shape[0]):
				weighted_sum = np.dot(self.weights_[t-1][:,i],activations[t-1]) #+ self.biases_[t-1][i] #before was preactivation
				activations[t][i] = self.activation.function(weighted_sum)

			
		#activations[-1] = self.activation.function(np.dot(self.weights_[-2][:,i],activations[-2]))

##### FORGOT biases
###### ISSUE IS THAT THE PREACTIVATION HAS AN EXTRA LAYER
	def _backward_pass(self, activations, deltas):#activation_derivs, deltas):
		#print("deltas",deltas)
		for t in range(self.num_layers-2, -1, -1):
			for i in range(len(self.weights_[t])):
				#print(t,i,self.weights_[t][i], deltas[t+1])											#Need to change for different loss functions
				deltas[t][i] = np.dot(self.weights_[t][i],deltas[t+1])*activations[t][i]*(1-activations[t][i])

	def _backprop(self, activations, weight_grad, deltas, y): #return deltas for biases!!!!!!
		#preactivations = [np.zeros_like(layer) for layer in activations]

		self._forward_pass(activations)#, preactivations)
		#print("activations", activations)
		#Computing the derivative of activation function with the weighted sum of input
		#neurons (preactivions variable) to avoid calculating it twice, once in doing backpass
		#and once for gradient calc

		#activation_deriv_vec = np.vectorize(self.activation.derivative)
		#activation_derivs = [activation_deriv_vec(preact) for preact in preactivations ]

		# initilize the loss of the last layer 
		# NOTE THAT THE ACTIVATION FOR LAST LAYER MAY BE DIFFERENT
		deltas[-1] = (y - activations[-1])*activations[-1]*(1-activations[-1])
		#print(activations[-1])
		self._backward_pass(activations, deltas)#activation_derivs, deltas)
		for t in range(len(weight_grad)):
			for i in range(len(weight_grad[t])):
				weight_grad[t][i] = deltas[t+1]*activations[t][i]


	def _fit(self, X, ys):
		count = 0
		for epoch in range(self.max_iter):
			X,ys = unison_shuffled_copies(X, ys, self.random_state)
			
			for x, y in zip(X,ys):
				count+=1
				#print("x",x)
				activations = [np.zeros(size) for size in  self.layer_sizes]
				activations[0] = x
				activations = [np.append(layer,1) for layer in activations[:-1]] + [np.zeros(self.layer_sizes[-1])]
				deltas = deepcopy(activations)
				weight_grad = [np.full(layer.shape, np.nan) for layer in self.weights_] 				
				


				self._backprop(activations,weight_grad, deltas, y) # 0 -> regularization param


				self.weights_ = [weights + self.learning_rate * grad for weights, grad in zip(self.weights_, weight_grad)]
				self.biases_ = [bias + self.learning_rate * delta * bias for bias, delta in zip(self.biases_, deltas[1:])]
				loss = deltas[-1]

				if (count%100 == 0):
					cost = 0.5*(y - activations[-1])**2
					print("cost:",cost,"x:" , activations[0], "y pred:", activations[-1])
			# NEED TO FIX INEQUALITY ABS
			#if(abs(np.sum(loss)) < self.tol):
				#print(self.tol)
				#print("loss",abs(loss))
				#break
				

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



def main():
	random_seed = RandomState(1234567890)

	mean1 = (1, 2)
	mean1 = (-2, 3)

	cov1 = [[1, 0], [0, 1]]
	cov2 = [[-1, 0], [3, 1]]

	x1 = np.random.multivariate_normal(mean1, cov1, (100))
	x2 = np.random.multivariate_normal(mean1, cov2, (100))

	X = np.concatenate((x1,x2),axis=0)
	print(x1.shape)


	#X = np.array([[5,5],[-4,5],[5,-4],[-4,-4]])
	y = np.append(np.full((100),1),np.zeros(100))

	mlp = MultilayerPerceptron(layer_sizes = [2,3,9,1], activation = Sigmoid(), solver = "SGD",
								learning_rate = 0.5, learning_rate_init = 0.5, max_iter = 10000,
								shuffle = True, random_state = random_seed, tol = 0.01)
	

	mlp._init()
	mlp._fit(X,y)
		
	

def unison_shuffled_copies(a, b, random = None):
	if random is None:
		random = np.random
	assert len(a) == len(b)
	p = random.permutation(len(a))
	return a[p], b[p]

if __name__ == "__main__":
	main()







