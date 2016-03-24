#-*- coding: UTF-8 -*-

import numpy as np
import theano
import theano.tensor as T
from NN import NN
from ContextExtractor import ContextExtractor1, ContextExtractor2, corpus2dict
from Task import WMT14QETask2, WMT14QETask1_1, WMT15QETask2
#from EvalModel import loadParams

import random
import sys
import datetime
import time
import codecs

import argparse
import copy

from progressbar import ProgressBar
import gensim.models

from matplotlib import use
use('Agg') #do not choose x-using backend
from matplotlib import pyplot as plt 

import warnings
warnings.filterwarnings("ignore") #ignore warnings

"""QUality Estimation from scraTCH - main class to run experiments"
"""

def loadParams(paramfile):
    """ Load trained parameters from file"""
    f = open(paramfile,"r")
    params = cPickle.load(f)
    return params


def writeOutputToFile(task, filename, targets, pred_multi=None, pred_l1=None, pred_bin=None, pred_task1_1=None, likelihood_multi=None, likelihood_l1=None, likelihood_bin=None):
	""" 
	Write the prediction to the specified output file, format according to WMT14 task description: 
	Task1_1:
	 <METHOD NAME> <SEGMENT NUMBER> <SEGMENT SCORE> <SEGMENT RANK>
	(segment rank is always 0, since QUETCH does not perform ranking)
	Task2:	
	<SEGMENT NUMBER> <WORD INDEX> <WORD> <DETAILED SCORE> <LEVEL 1 SCORE> <BINARY SCORE> 
	"""
	f = codecs.open(filename, "w", "utf8")
	g = codecs.open(filename+"_likelihood","w", "utf8")

	if type(task)==WMT14QETask2: #prediction on word level
		#if not predicted, simply return 0 (="OK")
		if pred_multi is None:
			pred_multi = np.zeros(len(targets),dtype=np.int)
		if pred_l1 is None:
			pred_l1 = np.zeros(len(targets),dtype=np.int)
		if pred_bin is None:
			pred_bin = np.zeros(len(targets),dtype=np.int)
		if likelihood_multi is None:
			likelihood_multi = np.zeros((len(targets),21))
		if likelihood_l1 is None:
			likelihood_l1 = np.zeros((len(targets),3))
		if likelihood_bin is None:
			likelihood_bin = np.zeros((len(targets),2))

		for i,t in enumerate(targets): #information from given target test set
			#(Id, index, word, multi, coarse, binary) = t
			#new format:
			(sentenceId, tokenId, token, label1, label2, label3, features) = t
			#replace gold standard score with system score
			label_bin = task.intToLabel_bin[pred_bin[i]] #lookup in dictionary: from label int to label str
			label_l1 = task.intToLabel_l1[pred_l1[i]]
			label_multi = task.intToLabel_multi[pred_multi[i]]
			outstr = "%s\t%d\t%s\t%s\t%s\t%s\n" % (sentenceId, int(tokenId), token, label_multi, label_l1, label_bin)
#			print i, likelihood_bin[i], likelihood_bin[i][pred_bin[i]]
#			print likelihood_multi[i]
#			print likelihood_l1[i] 
			outstr2 = "%s\t%d\t%s\t%s\t%s\t%s\t%f\t%f\t%f\n" % (sentenceId, int(tokenId), token, label_multi, label_l1, label_bin, likelihood_multi[i][pred_multi[i]], likelihood_l1[i][pred_l1[i]], likelihood_bin[i][pred_bin[i]])
			f.write(outstr)
			g.write(outstr2)

	elif type(task)==WMT14QETask1_1: #prediction on sentence level
		methodName = "QUETCH"
		for i,t in enumerate(targets):
			segmentNumber = i+1
			segmentScore = task.intToLabel[pred_task1_1[i]]
			outstr = "%s\t%d\t%s\t%d\n" % (methodName, segmentNumber, segmentScore, 0)
			f.write(outstr)

	elif type(task)==WMT15QETask2: #binary prediction on word level
		pred_multi = np.zeros(len(targets),dtype=np.int)
		pred_l1 = np.zeros(len(targets),dtype=np.int)
		likelihood_multi = np.zeros((len(targets),21))
		likelihood_l1 = np.zeros((len(targets),3))
		for i,t in enumerate(targets): #information from given target test set
			#(Id, index, word, multi, coarse, binary) = t
			#new format:
			(sentenceId, tokenId, token, label1, features) = t
			#replace gold standard score with system score
			label_bin = task.intToLabel_bin[pred_bin[i]] #lookup in dictionary: from label int to label str
			label_l1 = task.intToLabel_l1[pred_l1[i]]
			label_multi = task.intToLabel_multi[pred_multi[i]]
			outstr = "%s\t%d\t%s\t%s\t%s\t%s\n" % (sentenceId, int(tokenId), token, label_multi, label_l1, label_bin)
#		       print i, likelihood_bin[i], likelihood_bin[i][pred_bin[i]]
#		       print likelihood_multi[i]
#		       print likelihood_l1[i] 
			outstr2 = "%s\t%d\t%s\t%s\t%s\t%s\t%f\t%f\t%f\n" % (sentenceId, int(tokenId), token, label_multi, label_l1, label_bin, likelihood_multi[i][pred_multi[i]], likelihood_l1[i][pred_l1[i]], likelihood_bin[i][pred_bin[i]])
			f.write(outstr)
			g.write(outstr2)

		
	else:
		print "No valid task given"
		exit(-1)
	f.close()

#from a trained word2vec gensim model, construct a matrix that contains the vectors according to the indices in the wordDictionary
#example: "computer" has index 0 in wordDictionary, so its vector should be in row with index 0
def constructLT(model, wordDictionary, d_wrd, lc):
	#check if model vector size equals the required LT dimensionality
	if d_wrd != len(model["and"]):
		print "pretrained model does not agree in dimensionality with required LT dimensionality (%d vs %d)" %(len(model["and"]), d_wrd)
		sys.exit(-1)
	matrix = np.zeros((len(wordDictionary),d_wrd))
	unknownwords = list()
	for (id,token) in wordDictionary.iteritems(): #(key,value) pairs -> (id, token)
		#print (id,token)
		#print matrix[id]
		try:
			if lc: #lowercased model
				matrix[id] = model[token.lower()]
			else:
				matrix[id] = model[token]
		except KeyError: #word is not in language model
			unknownwords.append(token)
			try:
				matrix[id] = model["UNKNOWN"] #TODO this does not really make sense, since UNKNOWN does not occur in corpus
			except KeyError: #"UNKNOWN" is not in model
				continue #just leave zeros
				#TODO: initialize randomly 
		#print matrix[id]
	#print "...Unknown words encountered: %s, %d" % ( ", ".join(unknownwords), len(unknownwords))
	return matrix

def train_and_test_NN(task, x_train, y_train, x_test, y_test, batch_size, numberOfLabels, vocabularySize, contextSize, d_wrd, maxit, threshold, baselineFeatures=False, pretrainedModel=None, acti="tanh", shuffle=False, l1=0, l2=0):
	""" Train and test a Neural Network """
	# compute number of minibatches for training
	n_train_batches = x_train.get_value(borrow=True).shape[0] / batch_size
	n_test_batches = x_test.get_value(borrow=True).shape[0] / batch_size
	
	print '... building the model'

	#allocate symbolic variables for the data
	index = T.lscalar()  #index to a minibatch
	x = T.matrix('x', dtype='int32')  #data is matrix, one row for each sample
	y = T.ivector('y')  # the labels are presented as vector of [int] labels 
	learningRateT = T.fscalar()

	rng = np.random.RandomState(1234)
	n_out = numberOfLabels
	#construct the NN
	if baselineFeatures:
		contextSize += 17 #17 baseline features are given

	if pretrainedModel is None:
		classifier = NN(rng, x, n_hidden, n_out, d_wrd, vocabularySize, contextSize, params=None, acti=acti)
	else:
		classifier = NN(rng, x, n_hidden, n_out, d_wrd, vocabularySize, contextSize, params=pretrainedModel, acti=acti)

	#symbolic expression for the score we maximize during training is the log likelihood of the model
	#score = (classifier.log_likelihood(y))

	#regularized score
	L1_reg = l1
	L2_reg = l2
	score = (classifier.log_likelihood(y)
		  + L1_reg * classifier.L1
		  + L2_reg * classifier.L2_sqr)
	
	cost = (classifier.errors(y))

	#compute the gradient of score with respect to theta
	gparams = [T.grad(score, param) for param in classifier.params]
	
	#SGA update for the parameters
	updates = [ (param, param + learningRateT * gparam) for param, gparam in zip(classifier.params, gparams)]

	#compiling a Theano function that returns the score and updates the parameter of the model for the given training data
	train_model = theano.function(
		inputs=[index, learningRateT],
		outputs=score,
		updates=updates,
		givens={
			x: x_train[index * batch_size:(index + 1) * batch_size],
			y: y_train[index * batch_size:(index + 1) * batch_size]
		}
	)

	#get the current cost on the test set
	eval_model_test = theano.function([], cost, givens={ x: x_test, y: y_test})
	eval_model_train = theano.function([], cost, givens={ x: x_train, y: y_train})

	#get vectors of true labels and current predictions
	predict_model_test = theano.function([], [y, classifier.outputLayer.y_pred], givens={x:x_test, y:y_test} )
	predict_model_train = theano.function([], [y, classifier.outputLayer.y_pred], givens={x:x_train, y:y_train})	

	print "... training"
	pbar = ProgressBar()	
	start_time = time.clock()
	print "...start time:", start_time
	costs = list()
	now = str(datetime.datetime.now()).replace(" ","--")

	#constant learning rate
	print "...Constant learning rate"
	learningRateT = np.float32(learningRate)
	
	#decaying learning rate, starting from 0.5, till learning rate parameter
#	print "...Decaying learning rate"
	#initialLearningRate = 0.05
	#learningRateT = np.float32(initialLearningRate)
	#learningRateLimit = np.float32(learningRate)	

	#save the current model every x iterations
	backupFrequency = 5

	#store the model with this prefix
	paramfile = "../parameters/"+now
	
	f1s_OK_train = list()
	f1s_BAD_train = list()
	f1s_sum_train = list()
	accuracies_train = list()
	
	f1s_OK_test = list()
	f1s_BAD_test = list()
	f1s_sum_test = list()
	accuracies_test = list()

	learningRates = list()

	noImprovementCounter = 0
	#parameters_best = copy.deepcopy(classifier.params)
	f1_BAD_best = -np.inf
	noImprovementLimit = 50 
	decayingFactor = 0.5

	it_times = list()
	cur_time = time.clock()

	for it in pbar(range(maxit)):
		itcosts = list()	
	
		#parameters_previous = copy.deepcopy(classifier.params)
		#learning rate: constant or 1/(t+1) or exp (1*0.85^(-1/N)) or dec (1/(1+(t/N))
		#learningRateT = np.float32(1./(1+ (float(it)/n_train_batches)))
		#learningRateT = np.float32(1./(it+1))
		print "\tIteration "+str(it)+": current learning rate", learningRateT		
		learningRates.append(learningRateT)	
	
		indices = range(n_train_batches)
		if shuffle:
			#shuffle indices of data
			np.random.shuffle(indices)
			print "...shuffling"
	
		#for minibatch_index in xrange(n_train_batches): #unshuffled
		for minibatch_index in indices:
			minibatch_avg_logli = train_model(minibatch_index, learningRateT)
			#print "\tCurrent train log likelihood:", minibatch_avg_logli
			#itcosts.append(minibatch_avg_logli)
		#costs.append(np.mean(itcosts))
		#traincost = (eval_model_train()/float(n_train_batches))
		#testcost = (eval_model_test()/float(n_test_batches))
		#print "\tCurrent avg cost on training data", traincost
		#print "\tCurrent avg cost on test data", testcost
		it_time = time.clock()
		diff = (it_time-cur_time)/60.
		it_times.append(diff)
		print "...iteration took", diff, "mins"
		cur_time = it_time 

		y_true_train, y_pred_train = predict_model_train()
		y_true_test, y_pred_test = predict_model_test()

		
		if task==2:
			print "\nF1 on train:"
			f1_OK_train, f1_BAD_train, accuracy_train = f1_ok_and_bad(y_true_train, y_pred_train)
			f1s_OK_train.append(f1_OK_train)
			f1s_BAD_train.append(f1_BAD_train)
			f1s_sum_train.append(f1_BAD_train+f1_OK_train)
			accuracies_train.append(accuracy_train)

			print "\nF1 on test:"
			f1_OK_test, f1_BAD_test, accuracy_test = f1_ok_and_bad(y_true_test, y_pred_test)
			f1s_OK_test.append(f1_OK_test)
			f1s_BAD_test.append(f1_BAD_test)
			f1s_sum_test.append(f1_BAD_test+f1_OK_test)
			accuracies_test.append(accuracy_test)
	#		if it%backupFrequency == 0:
	#			currentparamfile = paramfile+"."+str(it)+".params"
	#			print "\tSaving parameters in", currentparamfile
	#			classifier.saveParams(currentparamfile)		

			#if it>=1:
				#print "\tCurrent change in cost on training data: ", costs[-1]-costs[-2]
			#	print "\tCurrent change in sum of f1s on training data:", f1s_sum_train[-2]-f1s_sum_train[-1]
			#	print "\tCurrent change in sum of f1s on test data:", f1s_sum_test[-2]-f1s_sum_test[-1]

			#if improved (measured by BAD F1), save best parameter
			if f1_BAD_test > f1_BAD_best: 
				#parameters_best = copy.deepcopy(classifier.params)
				f1_BAD_best = f1_BAD_test
				print "\tNEW BEST!"
				noImprovementCounter = 0			
				currentparamfile = paramfile+"."+str(it)+".params"
				print "\tSaving parameters in", currentparamfile
				classifier.saveParams(currentparamfile)
		
			else:
				#count iterations where no improvement happened
				noImprovementCounter += 1
				# save parameters anyway every backupFrequency iterations
				if(it % backupFrequency == 0):
					currentparamfile = paramfile+"."+str(it)+".params"
					print "\tSaving parameters in", currentparamfile
					classifier.saveParams(currentparamfile)
					
		elif task==1: #TODO
			diff = 0
			total = 0
			zeros = 0
			error = 0
			sqerror = 0
			for i in xrange(x_train.get_value(borrow=True).shape[0]):
				total+=1
				#print y_true_train[i], y_pred_train[i]
				if y_true_train[i] != y_pred_train[i]: #y_true_test, y_pred_test
					diff+=1
					error += abs(y_true_train[i]-y_pred_train[i])
					sqerror += (y_true_train[i]-y_pred_train[i])**2
			print "...Mean Absolute Error (MAE) on training set: ",float(error)/total
			print "...Root Mean Squared Error (RMSE) on training set: ", np.sqrt(float(sqerror)/total)
		
		
		
			diff = 0
			total = 0
			zeros = 0
			error = 0
			sqerror = 0
			for i in xrange(x_test.get_value(borrow=True).shape[0]):
				total+=1
				#print y_true_test[i], y_pred_test[i]
				if y_true_test[i] != y_pred_test[i]: #y_true_test, y_pred_test
					diff+=1
					error += abs(y_true_test[i]-y_pred_test[i])
					sqerror += (y_true_test[i]-y_pred_test[i])**2
			print "...Mean Absolute Error (MAE) on test set: ",float(error)/total
			print "...Root Mean Squared Error (RMSE) on test set: ", np.sqrt(float(sqerror)/total)
		#for decaying learning rate	
		#if no improvement for k iterations, decrease learning rate and start from best parameters seen so far
	#	if noImprovementCounter >= noImprovementLimit:
	#		learningRateT *= decayingFactor	
	#		print "\tDECAY"
	#		#classifier.params = copy.deepcopy(parameters_best)
	#		noImprovementCounter = 0

		#for decaying learning rate
	#	if learningRateT < learningRateLimit:
	#		print "\tDecay limit of learning rate reached"
	#		#classifier.params = copy.deepcopy(parameters_best)
	#		break
	
		#classifier.params = copy.deepcopy(parameters_best)
		#stop if f1 on test data decreases
#		if it>1 and f1s_sum_test[-1]<f1s_sum_test[-2]:
#			print "\tSum of F1 (BAD+OK) on test data started to decrease"
#			print "\tResetting parameters to previous"
#			classifier.params = parameters_previous
#			break

		#if it>1 and (costs[-1]-costs[-2])<threshold: #improvement below threshold
		#	print "... break: after %d iterations improvement is below threshold of %f"	% (it, threshold)	
		#	break
	end_time = time.clock()

	print "... training took %.2fm" % ((end_time - start_time) / 60.)
	print "... that's on average: ", np.average(it_times), "for iterations: ", len(it_times)
	
	f1s = []
	
	if task==2:
		f1s = (f1s_OK_train, f1s_BAD_train, accuracies_train, f1s_OK_test, f1s_BAD_test, accuracies_test)


	print "...testing"
#	#theano function that returns the prediction 
	predict_model_test = theano.function([], [x, y, classifier.outputLayer.y_pred, classifier.outputLayer.p_y_given_x], givens={x:x_test, y:y_test} )
	x_test,y_test,pred,likelihood = predict_model_test()
#	diff = 0
#	total = 0
#	zeros = 0
#	error = 0
#	sqerror = 0
#	for i in xrange(len(x_test)):
#		total+=1
#		if y_test[i] != pred[i]:
#			diff+=1
#			error += abs(y_test[i]-pred[i])
#			sqerror += (y_test[i]-pred[i])**2
#		if y_test[i] == 0:
#			zeros+=1
##	if task == 2:
#		print "...accuracy on test set: ",float(total-diff)/total
#		#print "...percentage of zeros:", float(zeros)/total
#	elif task == 1:
#		print "...Mean Absolute Error (MAE) on test set: ",float(error)/total
#		print "...Root Mean Squared Error (RMSE) on test set: ", np.sqrt(float(sqerror)/total)
#		#print "...percentage of zeros:", float(zeros)/total


	return classifier, pred, likelihood, f1s, learningRates

def f1_ok_and_bad(y, pred_y):
	"""Computes F1 score on OK predictions and on BAD predictions"""
	tp_OK = 0
	fp_OK = 0
	fn_OK = 0
	tn_OK = 0
	
	tp_BAD = 0
	fp_BAD = 0
	fn_BAD = 0
	tn_BAD = 0

	for i in xrange(len(pred_y)):
		if y[i] == 0:
			if pred_y[i] == 0:
				#true: OK, predicted: OK
				tp_OK += 1
				tn_BAD += 1				
			elif pred_y[i] == 1:
				#true: OK, predicted: BAD
				fn_OK += 1
				fp_BAD += 1
			else:
				print "Couldn't parse prediction vectors"
				sys.exit(-1)
		elif y[i] == 1:
			if pred_y[i] == 0:
				#true: BAD, predicted: OK
				fp_OK += 1
				fn_BAD += 1
			elif pred_y[i] == 1:
				#true: BAD, predicted: BAD
				tn_OK += 1
				tp_BAD += 1
	
			else:
				print "Couldn't parse prediction vectors"
				sys.exit(-1)

		else:
			print "Couldn't parse prediction vectors"
			sys.exit(-1)
	
	precision_OK = float(tp_OK) / (tp_OK + fp_OK) if tp_OK+fp_OK > 0 else 0
	precision_BAD = float(tp_BAD) / (tp_BAD + fp_BAD) if tp_BAD+fp_BAD > 0 else 0
	recall_OK = float(tp_OK) / (tp_OK + fn_OK) if tp_OK+fn_OK > 0 else 0
	recall_BAD = float(tp_BAD) / (tp_BAD + fn_BAD) if tp_BAD+fn_BAD > 0 else 0
	f1_OK = (2*precision_OK*recall_OK) / (precision_OK + recall_OK) if precision_OK+recall_OK > 0 else 0
	f1_BAD = (2*precision_BAD*recall_BAD) / (precision_BAD + recall_BAD) if precision_BAD+recall_BAD > 0 else 0 
	accuracy = float(len(pred_y) - fp_OK - fn_OK) / len(pred_y) if len(pred_y)>0 else 0
	
	print "-"*20
	print "Accuracy:", accuracy
	print "-"*20
	print "\t REFERENCE"
	print "PREDICT\tOK\tBAD"
	print "OK\t%d\t%d" % (tp_OK, fp_OK)
	print "BAD\t%d\t%d" % (fp_BAD, tp_BAD)
	print "-"*20
	print "OK: Precision", precision_OK, "Recall", recall_OK
	print "BAD: Precision", precision_BAD, "Recall", recall_BAD
	print "-"*20
	print "F1_OK:", f1_OK
	print "F1_BAD:", f1_BAD
	return f1_OK, f1_BAD, accuracy


def run(task):
	""" Run the training and testing for a given task """
	if year == 14:
		if task == 1:
			############### TASK 1 ###################################################################
			t1 = WMT14QETask1_1(languagePair, "../WMT14-data/task1-1_"+languagePair+"_test", "../WMT14-data/task1-1_"+languagePair+"_training", targetWindowSize=targetWindowSize, sourceWindowSize=sourceWindowSize, baselineFeatures=baselineFeatures)

			#print t1.wordDictionary
			vocabularySize = len(t1.wordDictionary)

			###########################################################################################
	
			print "WMT14 Quality Estimation - Task 1.1"
			print "Language pair:", languagePair
			print "Hyper-parameters: learning rate = %f; maxit = %d; batch_size = %d; n_hidden = %d; d_wrd = %d; targetWindowSize = %d; sourceWindowSize = %d; threshold = %f, baselineFeatures = %r, activationFunction = %s, shuffle = %s, l1 = %f, l2 = %f" % (learningRate, maxit, batch_size, n_hidden, d_wrd, targetWindowSize, sourceWindowSize, threshold, baselineFeatures, acti, str(shuffle), l1, l2)

			#task 1.1
			numberOfLabels = 3
		
			#get instance vectors and binary labels for training
			(x_train,y_train),targetSents_train  = t1.get_train_xy()
		
			#get instance vectors and binary labels for testing
			(x_test,y_test),targetSents_test  = t1.get_test_xy()
	
			contextSize = targetWindowSize+sourceWindowSize
			
			#load pretrained gensim word2vec model
			params = None
			if pretrainedModel is not None:
				print "... Loading pretrained model from file", pretrainedModel
				model = gensim.models.word2vec.Word2Vec.load(pretrainedModel)
				#print model["computer"]
				lc = False
				if ".lc." in pretrainedModel:
					lc = True
					print "... lowercasing" 
				#construct initial lookup table from pretrained model
				params = constructLT(model,t1.wordDictionary,d_wrd,lc)
			
			classifier, pred, likelihood, f1s, learningRates  = train_and_test_NN(1, x_train, y_train, x_test, y_test, batch_size, numberOfLabels, vocabularySize, contextSize, d_wrd, maxit, threshold, baselineFeatures=baselineFeatures, pretrainedModel=params, acti=acti, shuffle=shuffle, l1=l1, l2=l2)
	
			#print lookup table
			#print classifier.params[0].get_value()
	
			#store parameters in file
			now = str(datetime.datetime.now()).replace(" ","--")
			f = "../parameters/"+now+".params"
			classifier.saveParams(f)

			now = str(datetime.datetime.now()).replace(" ","--")
			fileName = "../results/task1_1/"+languagePair+"/"+now+".results"
			writeOutputToFile(t1, fileName, targetSents_test, pred_task1_1=pred)
			print "...written prediction to file %s." % fileName
	
		else:
			############### TASK 2 ##################################################################

			pred_bin = None
			pred_multi = None
			pred_l1 = None
			likelihood_bin = None
			likelihood_multi = None
			likelihood_l1 = None

	
			#"translate" language pair notation for task 2
			languagePair2 = languagePair.upper().replace("-","_")		

			#t2 = WMT14QETask2(languagePair2, "../WMT14-data/task2_"+languagePair+"_test", "../WMT14-data/task2_"+languagePair+"_training", targetWindowSize, sourceWindowSize)
			#new format:
			t2 = WMT14QETask2(languagePair2, "../WMT14-data/task2_"+languagePair+"_test_comb", "../WMT14-data/task2_"+languagePair+"_train_comb", targetWindowSize=targetWindowSize, sourceWindowSize=sourceWindowSize, featureIndices=featureIndices, alignments=s2tAlignments, badWeight=badweight, lowercase=lowerCase)

			contextSize = t2.contextSize
			print t2.wordDictionary
			vocabularySize = len(t2.wordDictionary)
			#store vocabulary index in file
			now = str(datetime.datetime.now()).replace(" ","--")
			df = "../dicts/"+now+".dict"
        	        t2.wordDictionary.save(df)
                	print "...saved vocabulary index to file", df


			#load pretrained gensim word2vec model
			params = None
			if pretrainedModel is not None:
				print "... Loading pretrained model from file", pretrainedModel
				model = gensim.models.word2vec.Word2Vec.load(pretrainedModel)
				#print model["computer"]
				lc = False
				if ".lc." in pretrainedModel:
					lc = True
					print "... lowercasing" 
				#construct initial lookup table from pretrained model
				params = constructLT(model,t2.wordDictionary,d_wrd,lc)
			#print params
			###########################################################################################
			#1a) train NN for binary scores
			#1b) test NN for binary scores
			
			
			print "WMT14 Quality Estimation - Task 2"
			print "Language pair:", languagePair
			print "Hyper-parameters: learning rate = %f (initial = %f); maxit = %d; batch_size = %d; n_hidden = %d; d_wrd = %d; targetWindowSize = %d; sourceWindowSize = %d; threshold = %f; badweight = %d, activationFunction = %s, shuffle = %s, l1 = %f, l2 = %f" % (learningRate, initialLearningRate, maxit, batch_size, n_hidden, d_wrd, targetWindowSize, sourceWindowSize, threshold, badweight, acti, str(shuffle), l1, l2)
                	if featureIndices:
                        	print "Using features", featureIndices

			#print "Hyper-parameters: learning rate = %f; maxit = %d; batch_size = %d; n_hidden = %d; d_wrd = %d; targetWindowSize = %d; sourceWindowSize = %d; threshold = %f." % (learningRate, maxit, batch_size, n_hidden, d_wrd, targetWindowSize, sourceWindowSize, threshold)
			print "1) Binary scores"

			numberOfLabels = 2

			#get instance vectors and binary labels for training
			(x_train,y_train),targetWords_train  = t2.get_train_xy("bin")
	
			#get instance vectors and binary labels for testing
			(x_test,y_test),targetWords_test  = t2.get_test_xy("bin")

			classifier_bin, pred_bin, likelihood_bin, f1s, learningRates  = train_and_test_NN(2, x_train, y_train, x_test, y_test, batch_size, numberOfLabels, vocabularySize, contextSize, d_wrd, maxit, threshold, pretrainedModel = params, acti=acti, shuffle=shuffle, l1=l1, l2=l2)

			print f1s
			print learningRates
			#print lookup table
			#print classifier_bin.params[0].get_value()
	
			#print predictions
			#print pred_bin
	
			#store parameters in file
			now = str(datetime.datetime.now()).replace(" ","--")
			f = "../parameters/"+now+".params"
			classifier_bin.saveParams(f)
			
			#write output to file
			now = str(datetime.datetime.now()).replace(" ","--")
			fileName = "../results/task2/"+languagePair+"/"+now+".results"
			writeOutputToFile(t2, fileName , targetWords_test, pred_multi, pred_l1, pred_bin, None, likelihood_multi, likelihood_l1, likelihood_bin)
			print "...written prediction to file %s, and likelihood to %s" % (fileName, fileName+"_likelihood")
	

			plotF1sAndLearningRates(f1s, learningRates, fileName)

	elif year==15:

		print "WMT15 Quality Estimation - Task 2"
		print "Language pair:", languagePair
		print "Hyper-parameters: learning rate = %f (initial = %f); maxit = %d; batch_size = %d; n_hidden = %d; d_wrd = %d; targetWindowSize = %d; sourceWindowSize = %d; threshold = %f; badweight = %d, activationFunction = %s, shuffle = %s, l1 = %f, l2 = %f" % (learningRate, initialLearningRate, maxit, batch_size, n_hidden, d_wrd, targetWindowSize, sourceWindowSize, threshold, badweight, acti, str(shuffle), l1, l2)
		if featureIndices:
			print "Using features", featureIndices
		print "1) Binary scores"

	 	pred_bin = None
		pred_multi = None
		pred_l1 = None
		likelihood_bin = None
		likelihood_multi = None
		likelihood_l1 = None

		#"translate" language pair notation for task 2
		languagePair2 = languagePair.upper().replace("-","_")		


		#new format:
		t2 = WMT15QETask2(languagePair2, "../WMT15-data/task2_"+languagePair+"_dev_comb", "../WMT15-data/task2_"+languagePair+"_train_comb", targetWindowSize=targetWindowSize, sourceWindowSize=sourceWindowSize, featureIndices=featureIndices, alignments=s2tAlignments, badWeight=badweight, lowercase=lowerCase, full=full)

		
		
		contextSize = t2.contextSize
		print "... context size", contextSize

		#print t2.wordDictionary
		vocabularySize = len(t2.wordDictionary)

		#load pretrained gensim word2vec model
		params = None
		if pretrainedModel is not None:
			print "... Loading pretrained model from file", pretrainedModel
			try:
				model = gensim.models.word2vec.Word2Vec.load(pretrainedModel)
				#print model["computer"]
				lc = False
				if ".lc." in pretrainedModel:
					lc = True
					print "... lowercasing"
					#construct initial lookup table from pretrained model
				params = constructLT(model,t2.wordDictionary,d_wrd,lc)
			except AttributeError: #full model, not only LT pretrained
				params = loadParams(pretrainedModel)



		#"translate" language pair notation for task 2
		languagePair2 = languagePair.upper().replace("-","_")

		#get instance vectors and binary labels for training
		(x_train,y_train),targetWords_train  = t2.get_train_xy("bin")

		#get instance vectors and binary labels for testing
		(x_test,y_test),targetWords_test  = t2.get_test_xy("bin")

		if targetWindowSize==0 and sourceWindowSize==0: #not using word features
			contextSize = len(t2.featureIndices)
			print "... No words used, contextSize", contextSize

		now = str(datetime.datetime.now()).replace(" ","--")

		#store vocabulary index in file
                df = "../dicts/"+now+".dict"
                t2.wordDictionary.save(df)
                print "...saved vocabulary index to file", df
		
		numberOfLabels = 2 #binary classification
		classifier_bin, pred_bin, likelihood_bin, f1s, learningRates  = train_and_test_NN(2, x_train, y_train, x_test, y_test, batch_size, numberOfLabels, vocabularySize, contextSize, d_wrd, maxit, threshold, pretrainedModel = params, acti=acti, shuffle=shuffle, l1=l1, l2=l2)


		#print lookup table
		#print classifier_bin.params[0].get_value()

		#print predictions
		#print pred_bin

		#store parameters in file
		f = "../parameters/"+now+".params"
		classifier_bin.saveParams(f)
 
		#write output to file
		now = str(datetime.datetime.now()).replace(" ","--")
		fileName = "../results/task2/"+languagePair+"/"+now+".results"
		writeOutputToFile(t2, fileName , targetWords_test, pred_multi, pred_l1, pred_bin, None, likelihood_multi, likelihood_l1, likelihood_bin)
		print "...written prediction to file %s, and likelihood to %s" % (fileName, fileName+"_likelihood")	

		plotF1sAndLearningRates(f1s, learningRates,  fileName)
	
		
def plotF1sAndLearningRates(f1s, learningRates, filename):
	(f1s_OK_train, f1s_BAD_train, accuracies_train, f1s_OK_test, f1s_BAD_test, accuracies_test) = f1s
	x = range(len(f1s_OK_train))
	plt.figure(1)

	plt.subplot(311)
	plt.plot(x, f1s_OK_train, "g--x", label="F1_OK_train")
	plt.plot(x, f1s_BAD_train, "r--x", label="F1_BAD_train")
	plt.plot(x, f1s_OK_test, "g-x", label="F1_OK_test")
	plt.plot(x, f1s_BAD_test, "r-x", label="F1_BAD_test")
	plt.grid()
	lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	art = []
	art.append(lgd)	

	plt.subplot(312)
	plt.plot(x, accuracies_train, "b--x", label="Accuracy_train")
	plt.plot(x, accuracies_test, "b-x", label="Accuracy_test")
	plt.grid()
	lgd2 = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	art.append(lgd2)

	#plt.subplot(313)
	#plt.plot(x, learningRates, "m-x", label="Learning Rate")
	#plt.grid()
	#lgd3 = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	#art.append(lgd3)
	
	plt.savefig(filename+".f1s.png", additional_artists=art, bbox_inches="tight")
	print "Saved plot in "+filename+".f1s.png"	

if __name__=="__main__":

	#valid language pairs
	validPairs = ["all", "en-es", "es-en", "de-en", "en-de"]
	
	activationFunctions = ["tanh", "sig", "relu", "identity"]

	parser = argparse.ArgumentParser(description="QUality Estimation from scraTCH")
	parser.add_argument("WMTxx", type=int, choices=range(14,16), help="Which year's WMT QE task")
	parser.add_argument("Task", type=int, choices=range(1,3), help="Which WMT Quality Estimation shared task to perform")
	parser.add_argument("LanguagePair", type=str, choices=validPairs, help="Source and target language")
	parser.add_argument("-sws", "--SourceWindowSize", type=int, help="Word window size for feature extraction from source text; default for task1: 51, task2: 7")
	parser.add_argument("-tws", "--TargetWindowSize", type=int, help="Word window size for feature extraction from target text; default for task1: 51, task2: 5")
	parser.add_argument("-d", "--WordEmbeddingDimensionality", type=int, help="Dimensionality of feature space trained by Lookup-Table-Layer; default=50")	
	parser.add_argument("-hu", "--HiddenUnits", type=int, help="Number of hidden units in Linear Layer(s); default=300")
	parser.add_argument("-b", "--BaselineFeatures", action="store_true", help="Only applies to Task 1: if True, use WMT14 baseline features for the initialization of the Lookup-Table-Layer; default=False")
	parser.add_argument("-l", "--LearningRate", type=float, help="Learning rate for the Stochastic Gradient Optimization; default=0.001")
	parser.add_argument("-i", "--MaxIt", type=int, help="Convergence criterion: maximum number of training iterations; default=1000")
	parser.add_argument("-t", "--Threshold", type=float, help="Convergence criterion: if reduction of loss below threshold break up training; default=0.0001")
	parser.add_argument("-p", "--PretrainedModel", type=str, help="File that contains pre-trained gensim word2vec model")
	parser.add_argument("-f", "--FeatureIndices", type=str, help="Which features to use. Feature index is position of feature in combined format, e.g. '0+1+2' ")	
	parser.add_argument("-a", "--UseAlignments", action="store_true", help="Activates loading and evaulating alignments; default=False")
	parser.add_argument("-w", "--BadWeight", type=int, help="Reweighting of BAD instances; default=1")
	parser.add_argument("-r", "--InitialLearningRate", type=float, help="Initial Learning rate for decaying learning rate; default=0.5")
	parser.add_argument("-c", "--UseTrueCase", action="store_true", help="Load original truecased data; default: load lowercased data")
	parser.add_argument("-notest", "--NoTestDataDict", action="store_true", help="If the models lookup-table contains full WMT15 set, also test data, default: False")
	parser.add_argument("-g", "--ActivationFunction", type=str, choices=activationFunctions, help="Activation function for hidden layers")
	parser.add_argument("-s", "--Shuffle", action="store_true", help="If training data should be shuffled at each epoch, default=False")
	parser.add_argument("-l1", "--L1Regularizer", type=float, help="L1 Regularization")
	parser.add_argument("-l2", "--L2Regularizer", type=float, help="L2 Regularization")
	
	args = parser.parse_args()
	year = args.WMTxx
	task = args.Task
	languagePair = args.LanguagePair
	learningRate = args.LearningRate
	n_hidden = args.HiddenUnits
	sourceWindowSize = args.SourceWindowSize
	targetWindowSize = args.TargetWindowSize
	d_wrd = args.WordEmbeddingDimensionality
	baselineFeatures = args.BaselineFeatures
	maxit = args.MaxIt
	threshold = args.Threshold
	pretrainedModel = args.PretrainedModel
	featureIndices = args.FeatureIndices
	s2tAlignments = args.UseAlignments
	badweight = args.BadWeight
	initialLearningRate = args.InitialLearningRate
	lowerCase = not args.UseTrueCase
	full = not args.NoTestDataDict
	acti = args.ActivationFunction
	shuffle = args.Shuffle
	l1 = args.L1Regularizer
	l2 = args.L2Regularizer
	
 	#set default parameters for optional parameters if not given
	if year is None:
		year = 14
	if task is None:
		task = 2
	if d_wrd is None:
		d_wrd = 50
	if learningRate is None:
		learningRate = 0.001
	if n_hidden is None:
		n_hidden = 300
	if sourceWindowSize is None:
		if task == 1:
			sourceWindowSize = 51
		elif task == 2:
			sourceWindowSize = 7
	else:
		if sourceWindowSize%2==0 and not sourceWindowSize==0:#if even number
			print "Even sourceWindowSize given, must be odd. --> Added +1"
			sourceWindowSize+=1
	if targetWindowSize is None:
		if task == 1:
			targetWindowSize = 51
		elif task == 2:
			targetWindowSize = 5
	else:
		if targetWindowSize%2==0 and not targetWindowSize==0:#if even number
			print "Even targetWindowSize given, must be odd. --> Added +1"
			targetWindowSize+=1
	if sourceWindowSize ==0:
		print "Using only target information, ignoring source"
	if baselineFeatures and task != 1:
		print "Baseline features integration only for task 2. Proceeding without baseline features."
	if full:
		print "Using test data for gensim dictionary"
	if maxit is None:
		maxit = 1000
	if threshold is None:
		threshold = 0.0001
	if badweight is None:
		badweight = 1
	if initialLearningRate is None:
		initialLearningRate = 0.5
	if l1 is None:
		l1 = 0.0
	if l2 is None:
		l2 = 0.0
	if acti is None:
		acti = "tanh"

	batch_size = 1
	contextSize = targetWindowSize+sourceWindowSize

	run(task)
	
