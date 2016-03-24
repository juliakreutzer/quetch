import cPickle
from NN import NN
import theano.tensor as T
from Task import WMT14QETask2, WMT14QETask1_1, WMT15QETask2
import theano
import datetime
from QUETCH import writeOutputToFile, f1_ok_and_bad, loadParams
import numpy as np
import argparse
import matplotlib.pyplot as plt


#def loadParams(paramfile):
#	""" Load trained parameters from file"""
#	f = open(paramfile,"r")
#	params = cPickle.load(f)
#	return params

def testModelOnTest_task2_14(paramfile, testDataDir, devDataDir, trainDataDir, targetWindowSize, sourceWindowSize, outFile, wordDictionary=None, featureIndices=None, alignments=None, lowercase=True, full=False, multi=False, acti=None):
	""" Test given model on task 2 data """
	if full:
		print "...loading model whose LT contains indices for test data"
	else:
		print "...loading model with unknown test words"
	t2 = WMT14QETask2(languagePair2, testDataDir, trainDataDir, targetWindowSize=targetWindowSize, sourceWindowSize=sourceWindowSize, wordDictionary=wordDictionary, featureIndices=featureIndices, alignments=alignments, lowercase=lowercase, multi=multi) #no need to specify location of test data since it is reconstructed from dev data location
	(x_dev,y_dev),targetWords_dev  = t2.get_test_xy("bin")
#    (x_test,y_test),targetWords_test = t2.get_testtest_xy(task)

	x = T.matrix('x', dtype='int32')  # the data is presented as matrix, one row for each sample
	y = T.ivector('y')  # the labels are presented as vector of [int] labels 

	contextSize = sourceWindowSize+targetWindowSize
	if t2.featureIndices is not None:
		contextSize += targetWindowSize*len(t2.featureIndices)

	params = loadParams(paramfile)
	classifier = NN(None, x, 0, 0, 0, 0, contextSize, params=params, acti=acti)


	print "...testing on dev set"

	#theano function that returns the prediction 

	predict_model_dev= theano.function([], [x, y, classifier.outputLayer.y_pred, classifier.outputLayer.p_y_given_x, classifier.outputLayer.output], givens={x:x_dev, y:y_dev} )
	x_dev,y_dev,pred_dev,likelihood, out = predict_model_dev()

	f1_ok_and_bad(y_dev, pred_dev)
	#print likelihood
	#print out
	#xs = [l[0] for l in likelihood]
	#ys = [l[1] for l in likelihood]
	#plt.plot(xs,ys, 'ro')
	#plt.savefig("plotti.png")

	xso = [l[0] for l in out]
	yso = [l[1] for l in out]
	plt.scatter(xso,yso,c=y_dev)
	plt.axis([-1, 1, -1, 1])
	plt.savefig("plotti2.png")
	
def testModelOnTest_task2_15(paramfile, testDataDir, devDataDir, trainDataDir, targetWindowSize, sourceWindowSize, outFile, wordDictionary=None, featureIndices=None, alignments=None, lowercase=True, full=False, acti=None):
	""" Test given model on task 2 data """
	if full:
	    print "...loading model whose LT contains indices for test data"
	else:
	    print "...loading model with unknown test words"

	t2 = WMT15QETask2(languagePair, testDataDir, trainDataDir, wordDictionary=wordDictionary, targetWindowSize=targetWindowSize, sourceWindowSize=sourceWindowSize, featureIndices=featureIndices, alignments=alignments, lowercase=lowercase, full=full) #no need to specify location of test data since it is reconstructed from dev data location
	(x_dev,y_dev),targetWords_dev  = t2.get_test_xy(task)
	(x_test,y_test),targetWords_test = t2.get_testtest_xy(task)
	
	x = T.matrix('x', dtype='int32')  # the data is presented as matrix, one row for each sample
	y = T.ivector('y')  # the labels are presented as vector of [int] labels 

	x2 = T.matrix('x', dtype='int32')
	y2 = T.ivector('y') 

	contextSize = sourceWindowSize+targetWindowSize
	if t2.featureIndices is not None:
		contextSize += targetWindowSize*len(t2.featureIndices)

	params = loadParams(paramfile)
	#self, rng, input, n_hidden, n_out, d_wrd, sizeOfDictionary, contextWindowSize, params=None, acti="tanh"):
	classifier = NN(None, x, 0, 0, 0, 0, contextWindowSize=contextSize, params=params, acti=acti)	


	print "...testing on dev set"

	
	#theano function that returns the prediction 
	
	predict_model_dev= theano.function([], [x, y, classifier.outputLayer.y_pred, classifier.outputLayer.p_y_given_x, classifier.outputLayer.output], givens={x:x_dev, y:y_dev} )
	x_dev,y_dev,pred_dev,likelihood, out = predict_model_dev()
	
	#diff = 0
	#total = 0
	#zeros = 0
	#for i in xrange(len(x_dev)):
		#total+=1
		#if y_dev[i] != pred_dev[i]:
			#diff+=1
		#if y_dev[i] == 0:
			#zeros+=1
	#print "...accuracy on dev set: ",float(total-diff)/total
	##print "...percentage of zeros:", float(zeros)/total
	
	f1_ok_and_bad(y_dev, pred_dev)
	
	#xs = [l for l in likelihood]
	#ys = [1-l for l in likelihood]
	#plt.scatter(xs,ys,c=y_dev) #s=likelihood*100
	#plt.axis([0, 1, 0, 1])
	#plt.savefig(outFile+"_lik.png")
	
	
	xso = [l[0] for l in out]
	yso = [l[1] for l in out]
	plt.scatter(xso,yso,c=y_dev) #s=likelihood*100
	plt.axis([0, 1, 0, 1])
	plt.savefig(outFile+"_acti.png")
	
	
	
	print "...testing on test set"
	
	classifier2 = NN(None, x2, 0, 0, 0, 0, contextSize, params=params, acti=acti)
	predict_model_test = theano.function([], [x2, y2, classifier2.outputLayer.y_pred, classifier2.outputLayer.p_y_given_x], givens={x2:x_test, y2:y_test} )
	x_test,y_test,pred_test,likelihood = predict_model_test()

	#print "...percentage of zeros:", float(zeros)/total
	f1_ok_and_bad(y_test, pred_test)

	
	pred_multi = None
	likelihood_multi = None
	pred_l1 = None
	likelihood_l1 = None
	pred_bin = pred_test
	likelihood_bin = likelihood

	if outFile is not None:
		now = str(datetime.datetime.now()).replace(" ","--")
		writeOutputToFile(t2, outFile, targetWords_test, pred_multi, pred_l1, pred_bin, None, likelihood_multi, likelihood_l1, likelihood_bin)
		print "...written prediction to file %s, and likelihood to %s" % (outFile, outFile+"_likelihood")

if __name__ == "__main__":

	validPairs = ["all", "en-es", "es-en", "de-en", "en-de"]	
	subTasks = ["bin", "l1", "multi", "one"]
	activationFunctions = ["tanh", "sig", "relu", "identity"]


	parser = argparse.ArgumentParser(description="QUETCH model evaluation")
	parser.add_argument("Year", type=int, choices=range(14,16))
	parser.add_argument("Task", type=int, choices=range(1,3), help="Which WMT14 Quality Estimation shared task to evaluate on")
	parser.add_argument("LanguagePair", type=str, choices=validPairs, help="Source and target language")
	parser.add_argument("ParameterFile", type=str, help="File which contains trained model parameters")
	parser.add_argument("-sws", "--SourceWindowSize", type=int, help="Word window size for feature extraction from source text; default for task1: 51, task2: 7")
	parser.add_argument("-tws", "--TargetWindowSize", type=int, help="Word window size for feature extraction from target text; default for task1: 51, task2: 5")
	parser.add_argument("-b", "--BaselineFeatures", action="store_true", help="Only applies to Task 1: if True, used WMT14 baseline features for training; default=False")
	parser.add_argument("OutputFile", type=str, help="file where model's predictions are stored")
	parser.add_argument("-f", "--FeatureIndices", type=str)
	parser.add_argument("-dict", "--WordDictionary", type=str, help="Mapping from words to indices")
	parser.add_argument("-a", "--UseAlignments", action="store_true", help="Activates loading and evaluating alignments; default=False")
	parser.add_argument("-c", "--UseTrueCase", action="store_true", help="Load original truecased data; default: load lowercased data")
	parser.add_argument("-notest", "--NoTestDataDict", action="store_true", help="If the models lookup-table contains full WMT15 set, also test data, default: False")
	parser.add_argument("-m", "--multi", action="store_true", help="If model was trained on all language pairs (WMT14), but will tested on only one of them")
	parser.add_argument("-g", "--ActivationFunction", type=str, choices=activationFunctions, help="Activation function for hidden layers")

	args = parser.parse_args()
	task = args.Task
	year = args.Year
	languagePair = args.LanguagePair
	parameterFile = args.ParameterFile
	sourceWindowSize = args.SourceWindowSize
	targetWindowSize = args.TargetWindowSize
	baselineFeatures = args.BaselineFeatures
	outputFile = args.OutputFile
	featureIndices = args.FeatureIndices
	s2tAlignments = args.UseAlignments
	lowerCase = not args.UseTrueCase
	full = not args.NoTestDataDict
	wdict = args.WordDictionary
	multi = args.multi
	acti = args.ActivationFunction


	if sourceWindowSize is None:
		if task == 1:
			sourceWindowSize = 51
		elif task == 2:
			sourceWindowSize = 7
	if targetWindowSize is None:
		if task == 1:
			targetWindowSize = 51
		elif task == 2:
			targetWindowSize = 5
	if full:
		print "...Using test data for gensim dictionary"
		
	if wdict:
		print "...Loading pre-defined vocabulary mapping", wdict

	#contextSize = targetWindowSize+sourceWindowSize
	#if targetFeatureSize is not None:
	#	contextSoze += (targetFeatureSize*targetWindowSize)
	if year==14:
		languagePair2 = languagePair.upper().replace("-","_")
		testDataDir = "../WMT14-data/task2_"+languagePair+"_test_comb"
		trainDataDir = "../WMT14-data/task2_"+languagePair+"_train_comb"
		if multi:
			trainDataDir = "../WMT14-data/task2_"+"all"+"_train_comb"
		testModelOnTest_task2_14(parameterFile, testDataDir, testDataDir, trainDataDir, targetWindowSize, sourceWindowSize, outputFile, wordDictionary=wdict, featureIndices=featureIndices, alignments=s2tAlignments, lowercase = lowerCase, full=full, multi=multi, acti=acti)
	elif year==15:
	    languagePair2 = languagePair.upper().replace("-","_")
	    testtestDataDir = "../WMT15-data/task2_"+languagePair+"_test_comb"
	    testDataDir = "../WMT15-data/task2_"+languagePair+"_dev_comb"
	    trainDataDir = "../WMT15-data/task2_"+languagePair+"_train_comb"
	    testModelOnTest_task2_15(parameterFile, testDataDir, testDataDir, trainDataDir, targetWindowSize, sourceWindowSize, outputFile, wordDictionary=wdict, featureIndices=featureIndices, alignments=s2tAlignments, lowercase = lowerCase, full=full, acti=acti)	
