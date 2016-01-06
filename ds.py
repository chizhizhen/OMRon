#!/usr/bin/python
#coding:utf-8
"""
A Pylearn2 Dataset class for accessing the data for the
patient status recognition.
"""

import numpy as np
import gzip, cPickle
import cv2

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils.string_utils import preprocess

class PatientStatusDataset(DenseDesignMatrix):

	def __init__(self, which_set,
			one_hot = 1):
		#resize_shape = None
		"""
		which_set: A string specifying which portion of the dataset
			to load. Valid values are 'train' or 'public_test'
		resize_shape: shape of the input layer of the NN.
		"""

		#resize_shape = tuple(resize_shape)
		nLabels = 5
		batch_size = 20

		#load the dataset file
		f = gzip.open("ds.pkl.gz")
		dataset = cPickle.load(f)
		f.close()
		if which_set == 'test':
			X = dataset[1][0]
			y = dataset[1][1]
		else:
			X = dataset[0][0]
			y = dataset[0][1]

		#remove STICKINGOUT class
		X = X[np.where(y!=4)]
		y = y[np.where(y!=4)]

		#resize the data
		#if resize_shape is not None:
		#	dataShape = X.shape
		#	Tmp = np.empty((X.shape[0] * X.shape[1],) + resize_shape,
		#			np.uint8)
		#	X = X.reshape((X.shape[0] * X.shape[1],) +  X.shape[2:4])
		#	nI = 0
		#	while nI < X.shape[0]:
		#		Tmp[nI] = cv2.resize(X[nI], resize_shape)
		#		nI += 1
		#	Tmp = Tmp.reshape(dataShape[0:2] + resize_shape)
		#	X = Tmp

		#flop the train set for augmentation
		if len(X.shape) == 4 and which_set != 'test':
			X = np.r_[X, X[:,:,::-1,:]]
			y = np.r_[y, y]

		#reshape
		if len(X.shape) == 4:
			X = X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])

		#adjust the size to training batch size.
		nHeight = X.shape[0] / batch_size * batch_size
		X = X[0:nHeight]
		y = y[0:nHeight]

		#normalize
		X = (X - 127.5)/128

		#make output of the NN
		if one_hot:
			yelems = np.zeros((y.shape[0], nLabels), dtype='float32')
			for i in xrange(y.shape[0]):
				yelems[i, int(y[i])] = 1.
			y = yelems

		super(PatientStatusDataset, self).__init__(X=X, y=y)
