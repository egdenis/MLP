import numpy as np
from numpy.random import RandomState
from lib.multilayer_perceptron import *

def main():
	np.set_printoptions(precision=2)
	random_seed = RandomState(1234567890)

	mean1 = (1, 2)
	mean2 = (-1, 2)
	cov1 = [[1, 0], [0, 1]]
	cov2 = [[-1, 0], [3, 1]]
	x1 = np.random.multivariate_normal(mean1, cov1, (10))
	x2 = np.random.multivariate_normal(mean2, cov2, (10))
	y1 = np.zeros((10,2))
	y2 = np.zeros((10,2))
	y1[:,0].fill(1)
	y2[:,1].fill(1)


	y = np.concatenate((y1,y2),axis=0)
	print(y)
	X = np.concatenate((x1,x2),axis=0)
	mlp = MultilayerPerceptron(layer_sizes = [2,10,10,2], activation = Sigmoid(), solver = "SGD",
								learning_rate = 0.5, learning_rate_init = 0.5, max_iter = 10000,
								shuffle = True, random_state = random_seed, tol = 0.01, categorical = True)
	

	mlp._init()
	mlp._fit(X,y)
		
	


if __name__ == "__main__":
	main()
