import numpy as np
from numpy.random import RandomState
from lib.multilayer_perceptron import *
from sklearn.preprocessing import normalize
def main():
	np.set_printoptions(precision=2)
	random_seed = RandomState(1234567890)
	padded_train_images = np.load('data/googledoodle/padded_train_images.npy')
	padded_train_images = padded_train_images.reshape(padded_train_images.shape[0],-1)
	train_labels = np.load('data/googledoodle/train_labels_1hot.npy')
	padded_train_images = normalize(padded_train_images, 'l2')
	mlp = MultilayerPerceptron(layer_sizes = [padded_train_images.shape[1],10,10,train_labels.shape[1]], activation = Sigmoid(),
								learning_rate = 0.5, learning_rate_init = 1, max_iter = 1000,
								shuffle = True, random_state = random_seed, tol = 0.0001, categorical = True)
	

	mlp._init()
	mlp._fit(padded_train_images,train_labels)
		
	
def convert_and_save():
	train_labels = np.load('data/googledoodle/train_label.npy')[:,1]
	train_labels =  convert_label(train_labels)
	np.save( 'data/googledoodle/train_labels_1hot.npy', train_labels)

def convert_label(labels):
	"""
	Convert labels from name to number of unique 
	"""
	labes_unique = np.unique(labels)
	indices = dict(zip(labes_unique, list(range(len(labes_unique)))))
	labels_coded = np.empty((0,len(indices)+1), int)
	for label in labels:
		labels_coded = np.vstack((labels_coded,(np.array([0]*(indices[label]) + [1] + [0]*(len(indices)-indices[label]))))) 
	return labels_coded

if __name__ == "__main__":
	main()
