import numpy as np
import theano
import theano.tensor as T

def relu(x, alpha=0):
    """
    Compute the element-wise rectified linear activation function.
    .. versionadded:: 0.7.1
    Parameters
    ----------
    x : symbolic tensor
        Tensor to compute the activation function for.
    alpha : scalar or tensor, optional
        Slope for negative input, usually between 0 and 1. The default value
        of 0 will lead to the standard rectifier, 1 will lead to
        a linear activation function, and any value in between will give a
        leaky rectifier. A shared variable (broadcastable against `x`) will
        result in a parameterized rectifier with learnable slope(s).
    Returns
    -------
    symbolic tensor
        Element-wise rectifier applied to `x`.
    Notes
    -----
    This is numerically equivalent to ``T.switch(x > 0, x, alpha * x)``
    (or ``T.maximum(x, alpha * x)`` for ``alpha < 1``), but uses a faster
    formulation or an optimized Op, so we encourage to use this function.
    """
    # This is probably the fastest implementation for GPUs. Both the forward
    # pass and the gradient get compiled into a single GpuElemwise call.
    # TODO: Check if it's optimal for CPU as well; add an "if" clause if not.
    # TODO: Check if there's a faster way for the gradient; create an Op if so.
    if alpha == 0:
        return 0.5 * (x + abs(x))
    else:
        f1 = 0.5 * (1 + alpha)
        f2 = 0.5 * (1 - alpha)
        return f1 * x + f2 * abs(x)


class HiddenLayer(object):
	"""
	Standard hidden layer of a MLP: units are fully-connected and have tanh activation function. 
	Characterized by weight matrix W (size: n_in x n_out) and bias vector b (length: n_out).
	W is randomly initialized via fan-in (see Collobert et al. 2011 for more details)
	"""

	def __init__(self, rng=0, input=None, n_in=0, n_out=0, W=None, b=None,
				 activation="tanh"):
		self.input = input

		activationFunction = None
		if activation == "tanh":
			activationFunction = T.tanh
		elif activation == "sig":
			activationFunction = T.nnet.sigmoid
		elif activation == "relu":
			activationFunction = relu
		elif activation == "identity":
			activationFunction = None
			
		if W is None:
			W_values = np.asarray(rng.uniform( low=-0.5*np.sqrt(12./(np.sqrt(n_in))), high=0.5*np.sqrt(12./(np.sqrt(n_in))), size=(n_in, n_out)), dtype=theano.config.floatX)
			W = theano.shared(value=W_values, name='W', borrow=True)
			#fan-in is n_in
			#variance = inv(sqrt(n_in)) = 1/(sqrt(n_in))
			#variance of uniform distribution = 1/12(b-a)^2
			#distribution centered at 0 
			#->a is -0.5*np.sqrt(12./(np.sqrt(n_in)), b = 0.5*np.sqrt(12./(np.sqrt(n_in))

		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b

		lin_output = T.dot(input, self.W) + self.b
		self.output = (
			lin_output if activationFunction is None
			else activationFunction(lin_output)
		)
		#parameters of the model
		self.params = [self.W, self.b]
		
		#self.out = T.dot(input, self.W) + self.b #vector

		#symbolic expression: computing the matrix of class-membership probabilities
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

		#symbolic description: compute prediction as class whose probability is maximal
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)

	def log_likelihood(self,y):
		""" Compute the log-likelihood of a given label """
		return T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

		
	def errors(self, y):
		""" Return a float representing the number of errors in the data; zero-one loss """

		#check if y has same dimension of y_pred
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
				'y should have the same shape as self.y_pred',
				('y', y.type, 'y_pred', self.y_pred.type)
			)
		#check if y is of the correct datatype
		if y.dtype.startswith('int'):
			# the T.neq operator returns a vector of 0s and 1s, where 1
			# represents a mistake in prediction
			return T.sum(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()

