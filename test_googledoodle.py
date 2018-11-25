import numpy as np
from numpy.random import RandomState
from lib.multilayer_perceptron import *
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from multiprocessing.dummy import Pool
from itertools import repeat
from sklearn.metrics import precision_score

def main():
	np.set_printoptions(precision=2)
	random_seed = RandomState(1234567890)
	padded_train_images = np.load('data/googledoodle/padded_train_images.npy')
	padded_train_images = padded_train_images.reshape(padded_train_images.shape[0],-1)
	y = np.load('data/googledoodle/train_labels_1hot.npy')
	X = normalize(padded_train_images, 'l2')
	n_splits = 3
	kf = KFold(n_splits = n_splits, shuffle=True, random_state = random_seed)


	for train_idx, test_idx in kf.split(X):
		model =    MultilayerPerceptron(layer_sizes = [X.shape[1],10,10,10,10,10,y.shape[1]], activation = Sigmoid(),
										learning_rate = 0.5, learning_rate_init = 1, max_iter = 1000,
										shuffle = True, random_state = random_seed, tol = 0.40, categorical = True)
		model._init()
		model._fit(X[train_idx],y[train_idx])
		y_pred = model._predict(X[test_idx])
		precision = precision_score(y[test_idx], y_pred, average='micro')
		print(precision)


def convert_and_save():
	train_labels = np.load('data/googledoodle/train_label.npy')[:,1]
	train_labels =  convert_label(train_labels)
	np.save( 'data/googledoodle/train_labels_1hot.npy', train_labels)

def convert_label(labels):
	"""
	Convert labels from name to number of unique 
	"""
	labes_unique = np.unique(labels)
	indices = dict(zip(labes_unique, list(range(len(labels_unique)))))
	labels_coded = np.empty((0,len(indices)+1), int)
	for label in labels:
		labels_coded = np.vstack((labels_coded,(np.array([0]*(indices[label]) + [1] + [0]*(len(indices)-indices[label]))))) 
	return labels_coded

if __name__ == "__main__":
	main()
