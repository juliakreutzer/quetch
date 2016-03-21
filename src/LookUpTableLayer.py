import numpy as np
import unittest
from ContextExtractor import *
import theano

class LookUpTableLayer(object):
	""" 
	Lookup Table Layer: 
	input: word (index) + context words
	output: d_wrd*ContextWindowSize dimensional vector (concatenated vectors for single words)
	Note that the LTL matrix is implemented as transpose of the original definition (due to numpy conventions)
	"""
	def __init__(self, rng=None, input=None, d_wrd=0, sizeOfDictionary=0, W=None):
		self.input = input #vector of ints
		self.d_wrd = d_wrd
		if W is None: #W is dictionarysize*dwrd sized matrix
			print "... no pretraining"
			#initialization of parameters: according to "fan-in" of layer (=#inputs used to compute each output, LT_W: 1, first linear layer: n_hu(l-1)), drawn from centered uniform distribution, variance = inverse of sqrt(fan-in), learning rate is divided by fan-in, but stays fixed during training
			#fan-in is 1
			#variance = inv(sqrt(1)) = 1
			#variance of uniform distribution = 1/12(b-a)^2 = 1
			#->a is 0?, b = sqrt(12)
			#W = np.reshape(np.random.uniform(0,np.sqrt(12),sizeOfDictionary*self.d_wrd), (sizeOfDictionary, self.d_wrd))
			W_values = np.asarray(rng.uniform(low = -0.5*np.sqrt(12), high = 0.5*np.sqrt(12), size = (sizeOfDictionary, self.d_wrd)), dtype = theano.config.floatX)
			W = theano.shared(value=W_values, name='W', borrow=True)


		#print "initial lookup table: ",W.get_value()			
		self.W = W
		
		if self.input != None:
			#lookup operation and concatenation of vectors for all input (=context) words
			self.output = self.W[self.input].reshape((self.input.shape[0], -1))
			    
		self.params = [self.W]
