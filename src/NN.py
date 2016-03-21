from LookUpTableLayer import LookUpTableLayer
from HiddenLayer import HiddenLayer
import theano
import theano.tensor as T
import cPickle


"""
Adaptation of the MLP introduced at the Theano tutorial to the 'NLP from Sratch' approach of Collobert et al.
"""


class NN(object):
	"""
	A feedforward artificial neural network model that has one layer or more of hidden units and nonlinear activations.
	The QUETCH architecture includes a Lookup-Table-Layer, one or more hidden layers and the output is the softmax over all possible labels.
	"""

	def __init__(self, rng, input, n_hidden, n_out, d_wrd, sizeOfDictionary, contextWindowSize, params=None, acti="tanh"):
		print "... using %s activation function" % acti

		if params is None:
			self.trained = False
		else:
			self.trained = True
			print "... building trained model from", params

		if self.trained and len(params)==5: #loading trained model
			print "... loading trained model"
				#def __init__(self, rng=None, input=None, d_wrd=0, sizeOfDictionary=0, W=None):
			self.lookuptableLayer = LookUpTableLayer(rng=None, input=input, d_wrd=0, sizeOfDictionary=0, W=params[0])
			self.hiddenLayer = HiddenLayer(input=self.lookuptableLayer.output, W=params[1], b=params[2], activation=acti)
			self.outputLayer = HiddenLayer(input=self.hiddenLayer.output, W=params[3], b=params[4], activation=acti)
			
		else:		
			W = None
			if params is not None and len(params)==1:
				#print "... pretrained LT", params
				W_values=params[0]
				#print W
				W = theano.shared(value=W_values, name='W', borrow=True)
			# Lookup Table Layer 
			self.lookuptableLayer = LookUpTableLayer(rng=rng, input=input, d_wrd=d_wrd, sizeOfDictionary=sizeOfDictionary, W=W)

			# HiddenLayer with a tanh activation function
			self.hiddenLayer = HiddenLayer(
				rng=rng,
				input=self.lookuptableLayer.output,
				n_in=d_wrd*contextWindowSize,
				n_out=n_hidden,
				activation=acti
			)
			
			# The output layer gets as input the hidden units
			# of the hidden layer
			self.outputLayer = HiddenLayer(
				rng=rng,
				#for two hidden layers use the first of the two following lines
				#input=self.hiddenLayer2.output,
				input=self.hiddenLayer.output,
				#for no hidden layer use the next 2 lines
				#input=self.lookuptableLayer.output,
				#n_in=d_wrd*contextWindowSize,
				n_in = n_hidden,
				n_out=n_out
			)
		
		# log likelihood of the MLP is given by the
		# log likelihood of the output of the model, computed in the
		# output hidden linear layer
		self.log_likelihood = (
			self.outputLayer.log_likelihood
		)

		# same holds for the function computing the number of errors
		self.errors = self.outputLayer.errors

		# the parameters of the model are the parameters of the three layer it is
		# made out of
		self.params = self.lookuptableLayer.params + self.hiddenLayer.params + self.outputLayer.params
		
		# L1 norm
		self.L1 = (
			abs(self.hiddenLayer.W).sum()
			+ abs(self.outputLayer.W).sum()
			+ abs(self.lookuptableLayer.W).sum()
		)
		
		# L2 norm
		self.L2_sqr = (
			(self.hiddenLayer.W ** 2).sum()
			+ (self.outputLayer.W ** 2).sum()
			+ (self.lookuptableLayer.W ** 2).sum()
		)


	def saveParams(self,f):
		""" Save parameters in file """
		out = open(f, 'wb')
		cPickle.dump(self.params, out, -1)
		print "Saved parameters in ", f
		out.close()
		
		
