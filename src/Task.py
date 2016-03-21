#-*- coding: UTF-8 -*-

from ContextExtractor import ContextExtractor2, ContextExtractor1, ContextExtractor2Comb, corpus2dict, corpus2dict15, ContextExtractor2Comb15

import theano
import theano.tensor as T
import numpy as np#
import cPickle

"""WMT 14/15 Quality Estimation Tasks"""

class WMT14QETask1_1(object):
	"""
	Task 1-1: Sentence-level
	Predict the quality of the translation for each sentence.
	Three labels: 1,2,3
	"""
	#read from 3 files: target, source, score
	#how to predict a sentence?
	#take whole sentences as features
	#context size? longest sentence in de-en setting is 51 words long -> naive approach
	def __init__(self, languagePair, pathToTestData, pathToTrainData, targetWindowSize=51, sourceWindowSize=51, onlyTarget=False, baselineFeatures=False):
		self.languagePair = languagePair
		self.test_source = pathToTestData+"/"+self.languagePair+"_source.test"
		self.test_target = pathToTestData+"/"+self.languagePair+"_target.test"
		self.test_score =  pathToTestData+"/"+self.languagePair+"_score.test"
		self.train_source = pathToTrainData+"/"+self.languagePair+"_source.train"
		self.train_target = pathToTrainData+"/"+self.languagePair+"_target.train"
		self.train_score = pathToTrainData+"/"+self.languagePair+"_score.train"
		self.baselineFeatures = baselineFeatures
		self.data = [self.test_source, self.test_target, self.train_source, self.train_target]
		if self.baselineFeatures:
			self.train_features = pathToTrainData+"/task1-1_"+self.languagePair+"_training.features"
			self.test_features = pathToTestData+"/task1-1_"+self.languagePair+"_test.features"	
			self.data.extend([self.train_features, self.test_features])	
		if onlyTarget: #only use target data for training and testing
			self.test_source = None
			self.train_source = None
		self.wordDictionary = corpus2dict(self.data)
		self.targetWindowSize = targetWindowSize
		self.sourceWindowSize = sourceWindowSize
		self.labelToInt = {"1":0, "2":1, "3":2} #use this for consistency reasons -> start counting with 0
		self.intToLabel = {y:x for x,y in self.labelToInt.iteritems()}

	def get_train_xy(self):
		ld = self.labelToInt
		#print "... extracting context for training data"
		if not self.baselineFeatures:
			ce_train = ContextExtractor1(self.train_source, self.train_target, self.train_score, self.wordDictionary, ld, wordlevel=False)
			train_set_xy = ce_train.extract(self.targetWindowSize,self.sourceWindowSize) #tuple 0:labels, 1:context vectors
		else:
			ce_train = ContextExtractor1(self.train_source, self.train_target, self.train_score, self.wordDictionary, ld, wordlevel=False, features=self.train_features)
			train_set_xy = ce_train.extract(self.targetWindowSize,self.sourceWindowSize)
		train_set_x, train_set_y = shared_dataset(train_set_xy)
		return (train_set_x, train_set_y), ce_train.targetSents


	def get_test_xy(self):
		ld = self.labelToInt
		#print "... extracting context for test data"
		if not self.baselineFeatures:
			ce_test = ContextExtractor1(self.test_source, self.test_target, self.test_score, self.wordDictionary, ld, wordlevel=False)
			test_set_xy = ce_test.extract(self.targetWindowSize,self.sourceWindowSize) 
		else:
			ce_test = ContextExtractor1(self.test_source, self.test_target, self.test_score, self.wordDictionary, ld, wordlevel=False, features=self.test_features)
			test_set_xy = ce_test.extract(self.targetWindowSize,self.sourceWindowSize) 
		test_set_x, test_set_y = shared_dataset(test_set_xy)
		return (test_set_x, test_set_y), ce_test.targetSents



class WMT14QETask2(object):
	"""
	Task 2: Word-level
	Predict the quality of the translation for each single word.
	Three levels of granularity: binary, level-1, multi-class
	"""
	def __init__(self, languagePair, pathToTestData, pathToTrainData, wordDictionary=None, targetWindowSize=5, sourceWindowSize=7, featureIndices=None, onlyTarget=False, alignments=False, badWeight=1, lowercase=True, multi=False, demo=False): #language pair e.g. "EN_DE" or "DE_EN", pathToTestData e.g. "/home/julia/Dokumente/Uni/WS2014/Deep Learning/Prujäkt/WMT14-data/task2_de-en_test", pathToTrainData e.g. "/home/julia/Dokumente/Uni/WS2014/Deep Learning/Prujäkt/WMT14-data/task2_de-en_training"
		self.languagePair = languagePair
		#self.test_source = pathToTestData+"/"+self.languagePair+".source.test"
		#self.test_target = pathToTestData+"/"+self.languagePair+".tgt_ann.test"
		#self.train_source = pathToTrainData+"/"+self.languagePair+".source.train"
		#self.train_target = pathToTrainData+"/"+self.languagePair+".tgt_ann.train"
		
		#new format:
		lc = ".lc" if lowercase else ""

		self.test_source = pathToTestData+"/"+self.languagePair+".source.test.tok"+lc+".comb"
		self.test_target = pathToTestData+"/"+self.languagePair+".tgt_ann.test"+lc+".comb"
		self.train_source = pathToTrainData+"/"+self.languagePair+".source.train.tok"+lc+".comb"
		self.train_target = pathToTrainData+"/"+self.languagePair+".tgt_ann.train"+lc+".comb"
		
		#alignments
		self.test_alignments = pathToTestData+"/"+self.languagePair+".align" if alignments else ""
		self.train_alignments = pathToTrainData+"/"+self.languagePair+".align" if alignments else ""
		self.test_align_t2s = dict()
		self.train_align_t2s = dict()
		self.demo = demo


		if multi:
			self.train_source = pathToTrainData+"/"+"ALL"+".source.train.tok"+lc+".comb"
			self.train_target = pathToTrainData+"/"+"ALL"+".tgt_ann.train"+lc+".comb"
			self.train_alignments = pathToTrainData+"/"+"ALL"+".align" if alignments else ""

		if lowercase:
			print "... loading lowercase data"
		else:
			print "... loading truecase data"


		if onlyTarget: #only use target data for training and testing
			self.test_source = None
			self.train_source = None
		self.data = [self.test_source, self.test_target, self.train_source, self.train_target]
		self.labelToInt_bin = {"OK":0, "BAD":1}
		self.intToLabel_bin = {y:x for x,y in self.labelToInt_bin.iteritems()}

		self.labelToInt_l1 = {"OK":0, "Accuracy":1, "Fluency":2}
		self.intToLabel_l1 = {y:x for x,y in self.labelToInt_l1.iteritems()}

		self.labelToInt_multi = {"OK":0, "Terminology":1, "Mistranslation":2, "Omission":3, "Addition":4, "Untranslated":5, "Accuracy":6, "Style/register":7, "Capitalization":8, "Spelling":9, "Punctuation":10, "Typography":11, "Morphology_(word_form)":12, "Part_of_speech":13, "Agreement":14, "Word_order":15, "Function_words":16, "Tense/aspect/mood":17, "Grammar":18, "Unintelligible":19, "Fluency":20}
		self.intToLabel_multi = {y:x for x,y in self.labelToInt_multi.iteritems()}
		self.wordDictionary = loadDict(wordDictionary) if wordDictionary is not None else corpus2dict15(self.data)
		self.featureIndices = featureIndices
		if featureIndices:
			self.featureIndices = [int(i) for i in featureIndices.strip().split("+")]

		self.targetWindowSize = targetWindowSize
		self.sourceWindowSize = sourceWindowSize
		self.contextSize = self.targetWindowSize + self.sourceWindowSize
		#print "... context size", self.contextSize
	
		#weighting of BAD instances
		self.badWeight = badWeight

	def get_train_xy(self,score):
		ld = None
		taskIndex = 0
		if score == "bin":
			ld = self.labelToInt_bin
			taskIndex = 5
		elif score == "l1":
			ld = self.labelToInt_l1
			taskIndex = 4
		elif score == "multi":
			ld = self.labelToInt_multi
			taskIndex = 3
		else:
			raise NameError("Please provide a valid scoring scheme (bin, l1 or multi)")
		#print "... extracting context for training data"
		#ce_train = ContextExtractor2(self.train_source, self.train_target, self.wordDictionary, ld, taskIndex, True)
		#new format:
		#ce_train = ContextExtractor2Comb(self.train_source, self.train_target, self.wordDictionary, ld, taskIndex, wordlevel=True, featureIndices=self.featureIndices)		

		#read alignments
		try:
			with open(self.train_alignments, "r") as aligndata:
				for line in aligndata:
					inner = dict()
					sentIndex, alignments = line.split('\t')
					for alignment in alignments.split():
						src,tgt = alignment.split('-')

						if tgt not in inner:	# only keep first alignment
							inner[int(tgt)] = int(src)

					self.train_align_t2s[float(sentIndex)] = inner	# better use strings here...
					#self.train_align_t2s[float(sentIndex)] = inner	# 
				#print len(self.train_align_t2s)
				#print "... loaded %d sentence alignments from '%s'"%(len(self.train_align_t2s),self.train_alignments)
		except EnvironmentError:
			print "alignment file for train-data not loaded"

		#new format w/alignments
		ce_train = ContextExtractor2Comb(self.train_source, self.train_target, self.wordDictionary, ld, taskIndex, wordlevel=True, featureIndices=self.featureIndices, alignments=self.train_align_t2s, demo=self.demo)

		train_set_xy = ce_train.extract(self.targetWindowSize,self.sourceWindowSize) #tuple 0:labels, 1:context vectors
		#print ce_train.getContext(0)		

		train_set_xy_w = reweight_dataset(train_set_xy, self.badWeight)

		train_set_x, train_set_y = shared_dataset(train_set_xy_w)
		return (train_set_x, train_set_y), ce_train.targets

	def get_test_xy(self,score):
		ld = None
		taskIndex = 0
		if score == "bin":
			ld = self.labelToInt_bin
			taskIndex = 5
		elif score == "l1":
			ld = self.labelToInt_l1
			taskIndex = 4
		elif score == "multi":
			ld = self.labelToInt_multi
			taskIndex = 3
		else:
			raise NameError("Please provide a valid scoring scheme (bin, l1 or multi)")
		print "... extracting context for test data"
		#ce_test = ContextExtractor2(self.test_source, self.test_target, self.wordDictionary, ld, taskIndex, True)
		#new format:
		#ce_test = ContextExtractor2Comb(self.test_source, self.test_target, self.wordDictionary, ld, taskIndex, wordlevel=True, featureIndices=self.featureIndices)
		if self.demo:
			print "... running demo"
			self.test_source = "tmp_s"
			self.test_target = "tmp_t"

		#read alignments
		try:
			with open(self.test_alignments, "r") as aligndata:
				for line in aligndata:
					inner = dict()
					sentIndex, alignments = line.split('\t')
					for alignment in alignments.split():
						src,tgt = alignment.split('-')

						if tgt not in inner:	# only keep first alignment
							inner[int(tgt)] = int(src)

					self.test_align_t2s[float(sentIndex)] = inner				 #print "... loaded %d sentence alignments from '%s'"%(len(self.test_align_t2s),self.test_alignments)
			#print len(self.test_align_t2s)
		except EnvironmentError:
			print "alignment file for test data not loaded"

		#new format w/alignments

		ce_test = ContextExtractor2Comb(self.test_source, self.test_target, self.wordDictionary, ld, taskIndex, wordlevel=True, featureIndices=self.featureIndices, alignments=self.test_align_t2s, demo=self.demo)
		
		#if self.demo:
		#	ce_test = ContextExtractor2Comb15(self.test_source, self.test_target, self.wordDictionary, ld, wordlevel=True, featureIndices=self.featureIndices, alignments=self.test_align_t2s, demo=self.demo)

		test_set_xy = ce_test.extract(self.targetWindowSize,self.sourceWindowSize) 
		test_set_x, test_set_y = shared_dataset(test_set_xy)
		return (test_set_x, test_set_y), ce_test.targets

class WMT15QETask2(object):
		"""
		Task 2: Word-level
		Predict the quality of the translation for each single word.
		One level of granularity: binary
		"""
		def __init__(self, languagePair, pathToTestData, pathToTrainData, wordDictionary=None, targetWindowSize=5, sourceWindowSize=7, onlyTarget=False, featureIndices=None, alignments=False, badWeight=1, lowercase=True, full=True, demo=False): #language pair e.g. "EN_DE" o$
			self.languagePair = languagePair
			#new format:
			lc = ".lc" if lowercase else ""
			self.test_source = pathToTestData+"/"+"dev.source"+lc+".comb"
			self.test_target = pathToTestData+"/"+"dev.target"+lc+".comb"
			pathToTestTestData = pathToTestData.replace("dev", "test")
			self.testtest_source = pathToTestTestData+"/"+"test.source"+lc+".comb"
			self.testtest_target = pathToTestTestData+"/"+"test.target"+lc+".comb"
			self.train_source = pathToTrainData+"/"+"train.source"+lc+".comb"
			self.train_target = pathToTrainData+"/"+"train.target"+lc+".comb"
			if lowercase:
			    print "... loading lowercase data"
			else:
			    print "... loading truecase data"

			if full:
			    print "... including test data in dictionary"
			#alignments
			self.test_alignments = pathToTestData+"/"+"dev.align" if alignments else ""
			self.testtest_alignments = pathToTestTestData+"/"+"test.align" if alignments else ""
			self.train_alignments = pathToTrainData+"/"+"train.align" if alignments else ""
			self.test_align_t2s = dict()
			self.train_align_t2s = dict()
			self.testtest_align_t2s = dict()
			self.demo = demo

			#weighting of BAD instances
			self.badWeight = badWeight
			               

			#print self.test_source, self.test_target, self.train_source, self.train_target, self.testtest_source, self.testtest_target
			if full:
			    self.data = [self.test_source, self.test_target, self.train_source, self.train_target, self.testtest_source, self.testtest_target]
			else:
			    self.data = [self.test_source, self.test_target, self.train_source, self.train_target]


			self.labelToInt_bin = {"OK":0, "BAD":1}
			self.intToLabel_bin = {y:x for x,y in self.labelToInt_bin.iteritems()}

			#these labels are not used in the task, but are part of the output format -> so just use OK label
			self.labelToInt_l1 = {"OK":0}
			self.intToLabel_l1 = {y:x for x,y in self.labelToInt_l1.iteritems()}

			self.labelToInt_multi = {"OK":0}
			self.intToLabel_multi = {y:x for x,y in self.labelToInt_multi.iteritems()}


			self.featureIndices = featureIndices
				
			self.targetWindowSize = targetWindowSize
			self.sourceWindowSize = sourceWindowSize

			self.contextSize = targetWindowSize+sourceWindowSize #default without features
			corpusData = ["../parallelCorpora-data/commoncrawl.es-en.en.norm.tok.lc", "../parallelCorpora-data/commoncrawl.es-en.es.norm.tok.lc", "../parallelCorpora-data/europarl-v7.es-en.en.norm.tok.lc", "../parallelCorpora-data/europarl-v7.es-en.es.norm.tok.lc", "../parallelCorpora-data/news-commentary-v8.es-en.en.norm.tok.lc", "../parallelCorpora-data/news-commentary-v8.es-en.es.norm.tok.lc"]
			#print "... data: ", self.data, len(self.data)
			#print "... corpusData: ", corpusData, len(corpusData)
			#print "... combined: ", corpusData+self.data, len(corpusData+self.data)
			#self.wordDictionary = corpus2dict15(corpusData+self.data, lowercase=lowercase) #use both WMT15 data and parallel data available for pre-training
			self.wordDictionary = loadDict(wordDictionary) if wordDictionary is not None else corpus2dict15(self.data)
		
			if featureIndices is not None:
				#features just on target side
				self.test_target = pathToTestData+"/"+"dev.target"+lc+".comb.feat"
				self.train_target = pathToTrainData+"/"+"train.target"+lc+".comb.feat"
				self.testtest_target = pathToTestTestData+"/"+"test.target"+lc+".comb.feat"
				#self.featureNames = pathToTestData+"/../task2_baseline_features/baseline_features.names"

				if onlyTarget: #only use target data for training and testing
						self.test_source = None
						self.testtest_source = None
						self.train_source = None
					
				if featureIndices is not None:
					if "+" in featureIndices:
						self.featureIndices = [int(i) for i in featureIndices.strip().split("+")]
						print "... parsed feature indices", self.featureIndices		
						self.contextSize += (targetWindowSize*len(self.featureIndices))

					elif "-" in featureIndices:
						start = int(featureIndices.split("-")[0].strip())
						end = int(featureIndices.split("-")[1].strip())
						self.featureIndices = range(start, end+1) #include end
						print "... parsed feature indices",self.featureIndices
						self.contextSize += (targetWindowSize*len(self.featureIndices))	
					else:
						print "Could not parse feature indices, continuing without features"
						#reset
						self.train_target = pathToTrainData+"/"+"train.target"+lc+".comb"
						self.test_target = pathToTestData+"/"+"dev.target"+lc+".comb"
						self.testtest_target = pathToTestTestData+"/"+"test.target"+lc+".comb"
						self.featureIndices = None
				
				if full:
				    self.data = [self.test_source, self.test_target, self.train_source, self.train_target, self.testtest_source, self.testtest_target]
				else:
				    self.data = [self.test_source, self.test_target, self.train_source, self.train_target]
				self.labelToInt_bin = {"OK":0, "BAD":1}
				self.intToLabel_bin = {y:x for x,y in self.labelToInt_bin.iteritems()}

				#print "... building word dictionary"
				self.wordDictionary = wordDictionary if wordDictionary is not None else corpus2dict15(self.data)

				self.targetWindowSize = targetWindowSize
				self.sourceWindowSize = sourceWindowSize

		def get_train_xy(self,score):			  
			ld = self.labelToInt_bin
			#print "... extracting context for training data"
			#ce_train = ContextExtractor2(self.train_source, self.train_target, self.wordDictionary, ld, taskIndex, True)
			#new format:

			#read alignments
			try:
				with open(self.train_alignments, "r") as aligndata:
					for line in aligndata:
						inner = dict()
						sentIndex, alignments = line.split('\t')
						for alignment in alignments.split():
							src,tgt = alignment.split('-')

							if tgt not in inner:    # only keep first alignment
								inner[int(tgt)] = int(src)

						self.train_align_t2s[float(sentIndex)] = inner  # better use strings here...
				    #print len(self.train_align_t2s)
					print "... loaded %d sentence alignments from '%s'"%(len(self.train_align_t2s),self.train_alignments)
			except EnvironmentError:
				print "alignment file for train-data not loaded"

			#new format w/alignments
			ce_train = ContextExtractor2Comb15(self.train_source, self.train_target, self.wordDictionary, ld, wordlevel=True, featureIndices=self.featureIndices, alignments=self.train_align_t2s)

			#ce_train = ContextExtractor2Comb15Obsolete(self.train_source, self.train_target, self.wordDictionary, ld, wordlevel=True, featureIndices=self.featureIndices)

			train_set_xy = ce_train.extract(self.targetWindowSize,self.sourceWindowSize) #tuple 0:labels, 1:context vectors
			#print ce_train.getContext(0)		   
			train_set_xy_w = reweight_dataset(train_set_xy, self.badWeight)

			train_set_x, train_set_y = shared_dataset(train_set_xy_w)
			return (train_set_x, train_set_y), ce_train.targets

		def get_test_xy(self,score):
   			ld = self.labelToInt_bin
			taskIndex = 3
			#print "... extracting context for test data"

			#read alignments
			try:
				with open(self.test_alignments, "r") as aligndata:
					for line in aligndata:
						inner = dict()
						sentIndex, alignments = line.split('\t')
						for alignment in alignments.split():
							src,tgt = alignment.split('-')

							if tgt not in inner:    # only keep first alignment
								inner[int(tgt)] = int(src)

						self.test_align_t2s[float(sentIndex)] = inner
					#print len(self.test_align_t2s)
					print "... loaded %d sentence alignments from '%s'"%(len(self.test_align_t2s),self.test_alignments)
			except EnvironmentError:
				print "alignment file for test-data not loaded"

			if self.demo:
				print "... running demo"
				self.test_source = "tmp_s"
				self.test_target = "tmp_t"

			#new format w/alignments
			ce_test = ContextExtractor2Comb15(self.test_source, self.test_target, self.wordDictionary, ld, wordlevel=True, featureIndices=self.featureIndices, alignments=self.test_align_t2s, demo=self.demo)
			

			#ce_test = ContextExtractor2Comb15Obsolete(self.test_source, self.test_target, self.wordDictionary, ld, wordlevel=True, featureIndices=self.featureIndices)
			test_set_xy = ce_test.extract(self.targetWindowSize,self.sourceWindowSize)
			test_set_x, test_set_y = shared_dataset(test_set_xy)
			return (test_set_x, test_set_y), ce_test.targets
			
		def get_testtest_xy(self,score):
   			ld = self.labelToInt_bin
			taskIndex = 3
			#print "... extracting context for test data"

			#read alignments
			try:
				with open(self.testtest_alignments, "r") as aligndata:
					for line in aligndata:
						inner = dict()
						sentIndex, alignments = line.split('\t')
						for alignment in alignments.split():
							src,tgt = alignment.split('-')

							if tgt not in inner:    # only keep first alignment
								inner[int(tgt)] = int(src)

						self.testtest_align_t2s[float(sentIndex)] = inner
					#print len(self.test_align_t2s)
					print "... loaded %d sentence alignments from '%s'"%(len(self.testtest_align_t2s),self.testtest_alignments)
			except EnvironmentError:
				print "alignment file for testtest-data not loaded"

			#new format w/alignments
			ce_test = ContextExtractor2Comb15(self.testtest_source, self.testtest_target, self.wordDictionary, ld, wordlevel=True, featureIndices=self.featureIndices, alignments=self.testtest_align_t2s)

			#ce_test = ContextExtractor2Comb15Obsolete(self.test_source, self.test_target, self.wordDictionary, ld, wordlevel=True, featureIndices=self.featureIndices)
			testtest_set_xy = ce_test.extract(self.targetWindowSize,self.sourceWindowSize)
			testtest_set_x, testtest_set_y = shared_dataset(testtest_set_xy)
			return (testtest_set_x, testtest_set_y), ce_test.targets


def reweight_dataset(data_xy, factor=1):
	"""
	multiply negative examples by factor
	"""
	data_x, data_y = data_xy
	newdata_x = []
	newdata_y = []
	for i in range(len(data_y)):

		newdata_y.append(data_y[i])
		newdata_x.append(data_x[i])

		if data_y[i] == 1:  # BAD label
			for j in range(1,factor):
				newdata_y.append(data_y[i])
				newdata_x.append(data_x[i])

	#print len(data_y), len(data_x), len(newdata_y), len(newdata_x)
	return newdata_x, newdata_y


def shared_dataset(data_xy, borrow=True):
	""" 
	loads the dataset into Theano shared variables
	"""
	data_x, data_y = data_xy
	shared_x = theano.shared(np.asarray(data_x,dtype='int32'),borrow=borrow)
	shared_y = theano.shared(np.asarray(data_y,dtype=theano.config.floatX), borrow=borrow)
 	return shared_x, T.cast(shared_y, 'int32')

def loadDict(dictFile):
	""" Load dictionary from file"""
	d = open(dictFile,"r")
	wdict = cPickle.load(d)
	return wdict

	
