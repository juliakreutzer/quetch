# -*- coding: UTF-8 -*-
"""
Context Extractor for WMT14 QE data

Extracts context from the source and the target text:
- input: two files (same number of sentences)
- output: tuples of (x,y) where x is a vector of word indices and y the label

Bilingual context -> bilingual features:
- fixed size word window around each word
- fixed size word window around position of word in target language (naive approach)
"""
import numpy as np
from gensim import corpora
import codecs
import re
#from nltk.tokenize import word_tokenize #requires u'tokenizers/punkt/english.pickle, else use   >>> from nltk.tokenize import wordpunct_tokenize  >>> wordpunct_tokenize(s)
import sys
import locale

# Wrap sys.stdout into a StreamWriter to allow writing unicode.
sys.stdout = codecs.getwriter(locale.getpreferredencoding())(sys.stdout) 

class ContextExtractor1(object):
	"""for task 1: extract features from both whole sentences"""
	def __init__(self, source, target, score, wordDictionary, labelDictionary, wordlevel=False, labels=None, contexts=None, features=None):
		self.source = source
		self.target = target
		self.score = score
		self.wordlevel = wordlevel
		if source is not None: #source can be None if working only on target data
			self.s = codecs.open(source, "r", "utf8")
		else:
			self.s = None
		self.t = codecs.open(target, "r", "utf8")
		self.sc = codecs.open(score, "r", "utf8")
		self.wd = wordDictionary #dictionary is for both languages
		self.ld = labelDictionary
		self.labels = labels
		self.contexts = contexts
		self.targetSents = list()
		self.sourceSents = list()
		self.useFeatures = False
		if features is not None:
			self.useFeatures = True
			self.f = codecs.open(features, "r", "utf8")

	def extract(self, targetWindow=51, sourceWindow=51):
		if self.source == None: #extract features only from target data
			print "...only working on target data"
			#TODO

		else:
			if self.wordlevel == False:
				targets = list() #collect target sentences
				for linet in self.t:
					targets.append(word_tokenize(linet.strip())) #sentence might be longer than targetWindow		
				sources = list() #collect source sentences
				for lines in self.s:
					sources.append(word_tokenize(lines.strip())) #sentence might be longer than sourceWindow		
				numberOfTargets = len(targets) #targets are sentences
				contexts = np.zeros((numberOfTargets,targetWindow+sourceWindow), dtype=int) #numpy matrix for contexts
				labels = np.zeros(numberOfTargets)
				self.targetSents = targets
				self.sourceSents = sources
				
				for i in xrange(numberOfTargets): 
					if len(targets[i])>targetWindow: #crop targets if sentence longer than window size
							targets[i] = targets[i][:targetWindow]
					if len(sources[i])>sourceWindow: #crop sources if sentence longer than window size
							sources[i] = sources[i][:sourceWindow]
					for k in xrange(targetWindow): #first fill in target features
						if k>=len(targets[i]):	
							contexts[i,k] = self.wd.token2id["PADDING"]
						else:
							try:	
								contexts[i,k] = self.wd.token2id[targets[i][k]]
							except KeyError:
								print "Unknown word:",targets[i][k]
								contexts[i,k] = self.wd.token2id["UNKNOWN"] #unknown words can occur due to tokenization problems
					for j in xrange(k,sourceWindow+targetWindow): #then continue with source features
						if j-k>=len(sources[i]):
							contexts[i,j] = self.wd.token2id["PADDING"]
						else:
							try:
								contexts[i,j] = self.wd.token2id[sources[i][j-k]]
							except KeyError:
								print "Unknown word:",sources[i][j-k]
								contexts[i,j] = self.wd.token2id["UNKNOWN"] #unknown words can occur due to tokenization problems
				for i,line in enumerate(self.sc):
					labels[i] = self.ld[line.strip()]
				self.labels = labels
				self.contexts = contexts
		
				if self.useFeatures: #use provided baseline features
					print "...using baseline features"
					extendedContexts = np.zeros((numberOfTargets,targetWindow+sourceWindow+17), dtype=int) #numpy matrix for contexts
					for i,line in enumerate(self.f):
						splitted = line.strip().split()
						if len(splitted) != 17:
							print "... not enough (<17) features provided for instance %d" % d
							continue
						features = [self.wd.token2id[f] for f in splitted]
						extendedContexts[i] = np.concatenate((self.contexts[i],features),axis=0) #extend contexts
					self.contexts = extendedContexts
				return self.contexts, self.labels

	def getContext(self, targetIndex):
		""" Get the extracted context for a given index of target word in sentence """
		contextForSent = [self.wd[c] for c in self.contexts[targetIndex]]
		print "context words for sentence '%s':\n %s" %(" ".join(self.targetSents[targetIndex]), str(contextForSent))
		return contextForSent
		
						

class ContextExtractor2(object):
	"""for task 2: extract features from a fixed window size around target word"""
	
	def __init__(self, source, target, wordDictionary, labelDictionary, taskIndex, wordlevel=True, labels=None, contexts=None):
		self.source = source
		self.target = target
		self.wordlevel = wordlevel
		if source is not None: #source can be None if working only on target data
			self.s = codecs.open(source, "r", "utf8")
		else:
			self.s = None
		self.t = codecs.open(target, "r", "utf8")
		self.wd = wordDictionary #dictionary is for both languages
		self.ld = labelDictionary
		self.labels = labels
		self.contexts = contexts
		self.targetWords = list() # contains tuples: ID,index,word,multi,coarse,binary
		self.sourceWords = list()
		self.taskIndex = taskIndex

	def extract(self, targetWindow=5, sourceWindow=5):
		if self.source == None: #extract features only from target data
			print "...only working on target data"
			if self.wordlevel == True: #annotation on word level
				#target file format:
				#ID\tindex\tword\tmulti\tcoarse\tbinary
				#ID is unique within file, index is within sentence, both start with 0
				targets = list()
				for line in self.t: #read target line for line -> word for word
					if len(line.split("\t"))==6:
						(Id, index, word, multi, coarse, binary) = line.split("\t")
						targets.append((Id, index, word, multi, coarse, binary.strip()))
					else:
						print "ERROR, not enough data in line", len(line.split("\t"))
				#print targetWords
				self.targetWords = targets
				targetWords = [t[2] for t in targets]	#contains only words
				#collect labels for given task			
				labels = [self.ld[t[self.taskIndex]] for t in targets]
				#print labels
				numberOfTargets = len(targetWords)
				#print len(targetWords), "target words read"
	
				contexts = np.zeros((numberOfTargets,targetWindow), dtype=int) #numpy matrix for
			
				for i in xrange(numberOfTargets): #go through words again		
				
					k = 0 #counter for features
					#collect target words within window
					for j in xrange(int(-np.floor(targetWindow/2.)),int(np.floor(targetWindow/2.)+1)): #counter for index in target words
						#print "datapoint", i
						#print "contextwindowindex", j
						#print "counter for features", k
						if i+j<0 or i+j>=numberOfTargets:
							contexts[i,k] = self.wd.token2id["PADDING"]				
							#print "added PADDING"
						else:
							#print targetWords[i+j][2]
							try:
								contexts[i,k] = self.wd.token2id[targetWords[i+j]]
							except KeyError:
								#print "Key Error:",self.targetWords[i+j]
								contexts[i,k] = self.wd.token2id["UNKNOWN"] #unknown words can occur due to tokenization problems
						k+=1
				self.labels = labels
				self.contexts = contexts
				return self.contexts, self.labels
	
		else:
			if self.wordlevel == True: #annotation on word level
				#target file format:
				#ID\tindex\tword\tmulti\tcoarse\tbinary
				#ID is unique within file, index is within sentence, both start with 0
				targets = list()
				for line in self.t: #read target line for line -> word for word
					if len(line.split("\t"))==6:
						(Id, index, word, multi, coarse, binary) = line.split("\t")
						targets.append((Id, index, word, multi, coarse, binary.strip()))
					else:
						print "ERROR, not enough data in line", len(line.split("\t"))
				#print targetWords
				self.targetWords = targets
				targetWords = [t[2] for t in targets]	#contains only words
				#collect labels for given task			
				labels = [self.ld[t[self.taskIndex]] for t in targets]
				#print labels
				numberOfTargets = len(targetWords)
				#print len(targetWords), "target words read"	
	
				sourceSentences = dict()
				for line in self.s: #read source sentence for sentence
					(Id, words) = line.split("\t",1)
					sourceSentences[int(float(Id)-0.1)] = word_tokenize(words.strip()) #.1 is removed from sentence id
					self.sourceWords.extend(word_tokenize(words.strip()))
			#	print len(self.sourceWords), "source words read"
			#	print len(sourceSentences), "source sentences read"

				contexts = np.zeros((numberOfTargets,targetWindow+sourceWindow), dtype=int) #numpy matrix for contexts for target words

				for i in xrange(numberOfTargets): #go through words again		
				
					k = 0 #counter for features
					#collect target words within window
					for j in xrange(int(-np.floor(targetWindow/2.)),int(np.floor(targetWindow/2.)+1)): #counter for index in target words
						#print "datapoint", i
						#print "contextwindowindex", j
						#print "counter for features", k
						if i+j<0 or i+j>=numberOfTargets:
							contexts[i,k] = self.wd.token2id["PADDING"]				
							#print "added PADDING"
						else:
							#print targetWords[i+j][2]
							try:
								contexts[i,k] = self.wd.token2id[targetWords[i+j]]
							except KeyError:
								print "Unknown word:",self.targetWords[i+j]
								contexts[i,k] = self.wd.token2id["UNKNOWN"] #unknown words can occur due to tokenization problems
						k+=1
					#extract source words within window
					#print len(sourceSentences)
					for j in xrange(int(-np.floor(sourceWindow/2.)),int(np.floor(sourceWindow/2.)+1)):	# extract features from source					
						targetWordIndex = int(targets[i][1])
						targetWordSentenceIndex = int(float(targets[i][0])-0.1)
						#print targetWordSentenceIndex
						if targetWordIndex+j<0 or targetWordIndex+j>=len(sourceSentences[targetWordSentenceIndex]):
							contexts[i,k] = self.wd.token2id["PADDING"]
						else:
							try:
								contexts[i,k] = self.wd.token2id[sourceSentences[targetWordSentenceIndex][targetWordIndex+j]]
							except KeyError:
								print "Unknown word:",sourceSentences[targetWordSentenceIndex][targetWordIndex+j]
								contexts[i,k] = self.wd.token2id["UNKNOWN"]
						k+=1
			#print labels #works
			#print contexts #works
			self.labels = labels
			self.contexts = contexts
			return self.contexts, self.labels

	def getContext(self, targetIndex):
		""" Get the extracted context for a given index of target word in sentence """
		word = self.targetWords[targetIndex]
		contextForWord = [self.wd[c] for c in self.contexts[targetIndex]]
		print "context words for word '%s': %s" %(self.wd[self.contexts[targetIndex][2]], str(contextForWord))
		return contextForWord


"""
context extractor that extracts data from new format (one token per line) and from alignment
"""
class ContextExtractor2Comb(object):
	"""for task 2 classification: extract features from a fixed window size around target word and use alignments if given"""
	
	def __init__(self, source, target, wordDictionary, labelDictionary, taskIndex, wordlevel=True, labels=None, contexts=None, featureIndices=None, alignments=None, demo=False):
		self.source = source
		self.target = target
		self.wordlevel = wordlevel
		if source is not None: #source can be None if working only on target data
			self.s = codecs.open(source, "r", "utf8")
		else:
			self.s = None
		self.t = codecs.open(target, "r", "utf8")
		self.wd = wordDictionary #dictionary is for both languages
		self.ld = labelDictionary
		self.labels = labels
		self.contexts = contexts
		self.targets = list() # contains tuples: (sentenceId, tokenId, token, label1, label2, label3, [features])
		self.sources = list() # contains tuples: (sentenceId, tokenId, token, label1, label2, label3, [features])
		self.taskIndex = taskIndex
		self.featureIndices = featureIndices
		self.alignments = alignments
		self.demo = demo
		
	def lineToTuple(self,line):
		splitted = line.split("\t")
		if len(splitted)>=6:
			sentenceId = float(splitted[0])
			tokenId = int(splitted[1])
			token = splitted[2]
			label1 = splitted[3].strip()
			label2 = splitted[4].strip()
			label3 = splitted[5].strip()
			features = []
			if len(splitted)>6: #features given
				print "%d features given" % len(splitted)-6
				print "Using features", featureIndices
				features = splitted[4:]
			return (sentenceId, tokenId, token, label1, label2, label3, features)		
		else:
			if len(splitted)==4: #binary labels given, just additional labels missing
				sentenceId = float(splitted[0])
				tokenId = int(splitted[1])
				token = splitted[2]
				label3 = splitted[3].strip()
				label2 = "*"
				label1 = "*"
				features = []
				return (sentenceId, tokenId, token, label1, label2, label3, features)		
			else:
				print "ERROR, not enough data in line:", len(splitted), line
				exit(-1)

	def extract(self, targetWindow=5, sourceWindow=5):
		if self.wordlevel == True: #annotation on word level
			#target file format:
			#ID\tindex\tword\tmulti\tcoarse\tbinary\tfeature1\tfeature2\t...\tfeatureN
			#ID is unique within file, index is within sentence, both start with 0
			targets = list()
			
			if self.demo: #raw input
				t_p = preprocessSent(self.t.read())#preprocess first, only one line
				s_p = preprocessSent(self.s.read())
				formatted_targets = [] #put into WMT15 format
				formatted_sources = []
				sentId=0
				for i,t in enumerate(t_p):
					formatted_targets.append("%d\t%d\t%s\t%s\t%s\t%s" % (sentId, i, t, "OK", "OK", "OK"))
				for i,s in enumerate(s_p):
					formatted_sources.append("%d\t%d\t%s\t%s\t%s\t%s" % (sentId, i, s, "OK", "OK", "OK"))
				self.t = formatted_targets
				self.s = formatted_sources
			
			
			for line in self.t: #read target line for line -> word for word
				targets.append(self.lineToTuple(line))
			self.targets = targets
			targetWords = [t[2] for t in targets]	#contains only words
			if self.featureIndices: #store features in matrix, for each feature one column
				numberOfFeatures = len(featureIndices)
				targetFeatures = list() #contains for each featureIndex (in given order), the features for all targets
				for f in featureIndices:
					featureList = list()
					featureList.append([t[6+f].strip() for t in targets])
					targetFeatures.append(featureList)
			labels = [self.ld[t[self.taskIndex]] for t in targets] 	#collect labels for given task			
			#print targets[0:3]
			#print labels[0:3]
			numberOfTargets = len(targetWords)
			#print len(targetWords), "target words read"	

			#source file format:
			#ID\tindex\tword\tmulti\tcoarse\tbinary\tfeature1\tfeature2\t...\tfeatureN
			#features are optional
			if sourceWindow > 0:
				sources = list()
				for line in self.s: #read source line for line -> word for word
					sources.append(self.lineToTuple(line))
				self.sources = sources
				#sourceWords = [s[2] for s in sources]
				#print len(self.sources), "source words read"
			
			if self.featureIndices: #store features in matrix, for each feature one column
				numberOfFeatures = len(featureIndices)
				sourceFeatures = list() #contains for each featureIndex (in given order), the features for all sources
				for f in featureIndices:
					featureList = list()
					featureList.append([s[6+f].strip() for s in sources])
					sourceFeatures.append(featureList)
					
			if sourceWindow > 0:
				#collect the source sentences in a dictionary with key=id and word list as value
				sourceSentences = dict()
				for (sentenceId, tokenId, token, label1, label2, label3, features) in self.sources:
					if sentenceId not in sourceSentences:
						sourceSentences[sentenceId] = list()
						sourceSentences[sentenceId].append(token)
			
				#print sourceSentences
		
			contextSize = targetWindow+sourceWindow
			if self.featureIndices:
				contextSize = targetWindow*len(self.featureIndices)+sourceWindow #features only on target side
				#contextSize = targetWindow*len(self.featureIndices)+sourceWindow*len(self.featureIndices)
				if targetWindow==0 and sourceWindow==0: #not using token information
					contextSize = len(self.featureIndices)	
	
			contexts = np.zeros((numberOfTargets,contextSize), dtype=int) #numpy matrix for contexts for target words
			
			for i in xrange(numberOfTargets): #go through words again		

				if targetWindow==0 and sourceWindow==0: #not using token information
					for f in len(self.featureIndices):
						contexts[i,f] = self.wd.token2id[self.targetFeatures[f][i]]	
		
				else:
					k = 0 #counter for context words
					#collect target words and selected features within window
					for j in xrange(int(-np.floor(targetWindow/2.)),int(np.floor(targetWindow/2.)+1)): #counter for index in target words
						#print "datapoint", i
						#print "contextwindowindex", j
						#print "counter for features", k
						if i+j<0 or i+j>=numberOfTargets:
							contexts[i,k] = self.wd.token2id["PADDING"]
						
							if self.featureIndices:
								for f in len(self.featureIndices):
									contexts[i,k+f] = self.wd.token2id["PADDING"] #add another PADDING for each feature				
							
							#print "added PADDING"
						else:
							#print targetWords[i+j][2]
							try:
								contexts[i,k] = self.wd.token2id[targetWords[i+j]]
							except KeyError:
								print "Unknown word:",targetWords[i+j]
								contexts[i,k] = self.wd.token2id["UNKNOWN"] #unknown words can occur due to tokenization problems
							
							if self.featureIndices:
								for f in len(featureIndices):
									contexts[i,k+f] = self.wd.token2id[self.targetFeatures[f][i]]
						k+=1

				#extract source words within window
				#DONE: integrate alignment to position window properly: change j according to alignment 
				#print len(sourceSentences)

				#print len(self.alignments)
                if sourceWindow > 0:
                    for j in xrange(int(-np.floor(sourceWindow/2.)),int(np.floor(sourceWindow/2.)+1)):	# extract features from source				
                        targetWordIndex = int(targets[i][1])
                        targetWordSentenceIndex = targets[i][0]

                        try:
                            sourceWordIndex = int(self.alignments[targetWordSentenceIndex][targetWordIndex])
                        except KeyError:
                            sourceWordIndex = targetWordIndex
                            #if self.alignments != None:
                            #	#print self.alignments[targetWordSentenceIndex]
                            #	print "INFO: no alignment for target position %d in sentence %s"%(targetWordIndex,targetWordSentenceIndex)

                        #print targetWordSentenceIndex
                        #if targetWordIndex+j<0 or targetWordIndex+j>=len(sourceSentences[targetWordSentenceIndex]):
                        if sourceWordIndex+j<0 or sourceWordIndex+j>=len(sourceSentences[targetWordSentenceIndex]):
                            contexts[i,k] = self.wd.token2id["PADDING"]
                            
                            #if self.featureIndices:
                            #	for f in len(featureIndices):
                            #		contexts[i,k+f] = self.wd.token2id["PADDING"] #add another PADDING for each feature		
                        
                        else:
                            try:
                                #contexts[i,k] = self.wd.token2id[sourceSentences[targetWordSentenceIndex][targetWordIndex+j]]
                                contexts[i,k] = self.wd.token2id[sourceSentences[targetWordSentenceIndex][sourceWordIndex+j]]
                            except KeyError:
                                print "Unknown word:",sourceSentences[targetWordSentenceIndex][targetWordIndex+j]
                                contexts[i,k] = self.wd.token2id["UNKNOWN"]
                            #if self.featureIndices:
                            #	for f in len(featureIndices):
                            #		contexts[i,k+f] = self.wd.token2id[self.sourceFeatures[f][i]]
                        k+=1
              
			#print labels[0] #works
			#print contexts[0] #works
			self.labels = labels
			self.contexts = contexts
			return self.contexts, self.labels

	def getContext(self, targetIndex):
		""" Get the extracted context for a given index of target word in sentence """
		word = self.targets[targetIndex]
		contextForWord = [self.wd[c] for c in self.contexts[targetIndex]]
		print "context words for word '%s': %s" %(self.wd[self.contexts[targetIndex][2]], str(contextForWord))
		return contextForWord		
	

"""
context extractor that extracts data from new format (one token per line)
"""
class ContextExtractor2Comb15(object):
	"""for task 2 classification for WMT15: extract features from a fixed window size around target word, only binary classification, only features on target side"""
	
	def __init__(self, source, target, wordDictionary, labelDictionary, wordlevel=True, labels=None, contexts=None, featureIndices=None, alignments=None, demo=False):
		self.source = source
		self.target = target
		self.wordlevel = wordlevel
		if source is not None: #source can be None if working only on target data
			self.s = codecs.open(source, "r", "utf8")
		else:
			self.s = None
		self.t = codecs.open(target, "r", "utf8")
		self.wd = wordDictionary #dictionary is for both languages
		self.ld = labelDictionary
		self.labels = labels
		self.contexts = contexts
		self.targets = list() # contains tuples: (sentenceId, tokenId, token, label1, [features])
		self.sources = list() # contains tuples: (sentenceId, tokenId, token, label1, [features])
		self.featureIndices = featureIndices #already parsed
		print "... using features", self.featureIndices	
		self.alignments = alignments
		self.demo = demo

	def lineToTuple(self,line):
		splitted = line.split("\t")
		if len(splitted)>=4:
			sentenceId = float(splitted[0])
			tokenId = int(splitted[1])
			token = splitted[2]
			label1 = splitted[3].strip() #only binary label
			features = []
			if len(splitted)>4 and self.featureIndices is not None: #features given
				#print "Using features", self.featureIndices
				features = splitted[4:] #not selecting features yet
			#print (sentenceId, tokenId, token, label1, features)
			return (sentenceId, tokenId, token, label1, features)
		else:
			print "ERROR, not enough data in line:", len(splitted), line
			exit(-1)

	def extract(self, targetWindow=5, sourceWindow=5):
		if self.wordlevel == True: #annotation on word level
			#target file format:
			#ID\tindex\tword\tbinary\tfeature1\tfeature2\t...\tfeatureN
			#ID is unique within file, index is within sentence, both start with 0
			targets = list()
			if self.demo: #raw input
				t_p = preprocessSent(self.t.read())#preprocess first, only one line
				s_p = preprocessSent(self.s.read())
				formatted_targets = [] #put into WMT15 format
				formatted_sources = []
				sentId=0
				for i,t in enumerate(t_p):
					formatted_targets.append("%d\t%d\t%s\t%s\t%s" % (sentId, i, t, "OK", "-"))
				for i,s in enumerate(s_p):
					formatted_sources.append("%d\t%d\t%s\t%s\t%s" % (sentId, i, s, "OK", "-"))
				self.t = formatted_targets
				self.s = formatted_sources
				
			for line in self.t: #read target line for line -> word for word
				targets.append(self.lineToTuple(line))
			self.targets = targets
			targetWords = [t[2] for t in targets]	#contains only words
			numberOfFeatures = 0
			if self.featureIndices is not None: #store features in matrix, for each feature one column
				numberOfFeatures = len(self.featureIndices)
				targetFeatures = list() #contains for each featureIndex (in given order), the features for all targets
				for f in self.featureIndices:
					featureList = list()
					try:
						featureList.extend([t[4][f].strip() for t in targets]) #here: select given features
					except Error:
						print "Error while reading features from data. No column for feature with index", f
						print "Continuing with '0' feature instead"
						featureList.extend([0 for t in targets])
					targetFeatures.append(featureList)
				self.targetFeatures = targetFeatures
			labels = [self.ld[t[3]] for t in targets] 	#collect labels for given task			
			#print labels
			#print targets[0:3]
			#print labels[0:3]
			#print targetFeatures[0], targetFeatures[1]
			#print len(targetFeatures), len(targetFeatures[0])
			numberOfTargets = len(targetWords)
			#print len(targetWords), "target words read"	

			#source file format:
			#ID\tindex\tword\tbinary
			#WMT15: no source features given
			sources = list()
			if sourceWindow > 0:
				for line in self.s: #read source line for line -> word for word
					sources.append(self.lineToTuple(line))
            
				self.sources = sources
				#sourceWords = [s[2] for s in sources]
				#print len(self.sources), "source words read"
				
				#collect the source sentences in a dictionary with key=id and word list as value
				sourceSentences = dict()
				for (sentenceId, tokenId, token, label1, features) in self.sources:
					if sentenceId not in sourceSentences:
						sourceSentences[sentenceId] = list()
					sourceSentences[sentenceId].append(token)
				
				#print sourceSentences
			
			contextSize = targetWindow+sourceWindow
			if self.featureIndices is not None:
				contextSize = (targetWindow*numberOfFeatures)+targetWindow+sourceWindow
				if targetWindow==0 and sourceWindow==0:
					print "...using no word information, only features"
					contextSize = numberOfFeatures
				print "... increased contextsize due to features, now", contextSize

			
			contexts = np.zeros((numberOfTargets,contextSize), dtype=int) #numpy matrix for contexts for target words

			unknownwords = set()

			if targetWindow==0 and sourceWindow==0:
				for i in xrange(numberOfTargets):
					for f in range(numberOfFeatures):
						contexts[i,f] = self.wd.token2id[self.targetFeatures[f][i]]

			else:

				for i in xrange(numberOfTargets): #go through words again		
				
					k = 0 #counter for context
					#collect target words and selected features within window
					for j in xrange(int(-np.floor(targetWindow/2.)),int(np.floor(targetWindow/2.)+1)): #counter for index in target words
						#print "datapoint", i
						#print "contextwindowindex", j
						#print "counter for features", k
						if i+j<0 or i+j>=numberOfTargets:
							contexts[i,k] = self.wd.token2id["PADDING"]
							k += 1
							#print "added PADDING"
							if self.featureIndices is not None:
								for f in range(numberOfFeatures):
									contexts[i,k+f] = self.wd.token2id["PADDING"] #add another PADDING for each feature				
									#print "added Feature PADDING"
								k += numberOfFeatures
						else:
							#print targetWords[i+j][2]
							try:
								contexts[i,k] = self.wd.token2id[targetWords[i+j]]
								#print "added target word"
							except KeyError:
								print "Unknown word:", targetWords[i+j]
								contexts[i,k] = self.wd.token2id["UNKNOWN"] #unknown words can occur due to tokenization problems
								unknownwords.add(targetWords[i+j])
							k += 1
							if self.featureIndices is not None:
								for f in range(numberOfFeatures):
									#print "added feature", self.targetFeatures[f][i], self.wd.token2id[self.targetFeatures[f][i]]
									contexts[i,k+f] = self.wd.token2id[self.targetFeatures[f][i]] #targetFeatures per feature
									#print "added target feature"
								k += numberOfFeatures 

					#extract source words within window
					#DONE: integrate alignment to position window properly: change j according to alignment 
					#print len(sourceSentences)
					if sourceWindow > 0:
						for j in xrange(int(-np.floor(sourceWindow/2.)),int(np.floor(sourceWindow/2.)+1)):	# extract features from source					
							targetWordIndex = int(targets[i][1])
							targetWordSentenceIndex = targets[i][0]

							try:
								sourceWordIndex = int(self.alignments[targetWordSentenceIndex][targetWordIndex])
							except KeyError:
								sourceWordIndex = targetWordIndex

							#print targetWordSentenceIndex
							#if targetWordIndex+j<0 or targetWordIndex+j>=len(sourceSentences[targetWordSentenceIndex]):
							if sourceWordIndex+j<0 or sourceWordIndex+j>=len(sourceSentences[targetWordSentenceIndex]):
								contexts[i,k] = self.wd.token2id["PADDING"]	
								#print "added PADDING"
							else:
								try:
									#contexts[i,k] = self.wd.token2id[sourceSentences[targetWordSentenceIndex][targetWordIndex+j]]
									contexts[i,k] = self.wd.token2id[sourceSentences[targetWordSentenceIndex][sourceWordIndex+j]]
								except KeyError:
									print "Unknown word:",sourceSentences[targetWordSentenceIndex][targetWordIndex+j]
									contexts[i,k] = self.wd.token2id["UNKNOWN"]
									unknownwords.add(sourceSentences[targetWordSentenceIndex][targetWordIndex+j])

								#print "added source word"
							k+=1
			print labels[0:3] #works
			print contexts[0:3] #works
			#print "unknown words:", unknownwords, len(unknownwords) 
			self.labels = labels
			self.contexts = contexts
			return self.contexts, self.labels

	def getContext(self, targetIndex):
		""" Get the extracted context for a given index of target word in sentence """
		word = self.targets[targetIndex]
		contextForWord = [self.wd[c] for c in self.contexts[targetIndex]]
		print "context words for word '%s': %s" %(word, str(contextForWord))
		return contextForWord			

def corpus2dict15(corpusfiles, lowercase=True): 
	""" From a given corpus, create a gensim dictionary for mapping words to ints, important: WMT15 data is already tokenized! """
	corpus = list()
	corpus.append(["PADDING"]) #has word index 0
	corpus.append(["UNKNOWN"]) #has word index 1
	for cf in corpusfiles:
		if cf is not None: #source can be none

#just for huge lookuptable that contains all words from pretraining
#			if lowercase:
#				corpus.extend([l.lower().split() for l in codecs.open(cf,"r","utf8").readlines()])
#			else:
#				corpus.extend([l.split() for l in codecs.open(cf,"r","utf8").readlines()])

			corpus.extend([l.split() for l in codecs.open(cf,"r","utf8").readlines()])
	wordDictionary = corpora.Dictionary(corpus)
	#print "... build word dictionary with vocabulary size =", len(wordDictionary)
	return wordDictionary
	



def corpus2dict(corpusfiles):
	""" From a given corpus, create a gensim dictionary for mapping words to ints """
	corpus = list()
	corpus.append(["PADDING"]) #has word index 0
	corpus.append(["UNKNOWN"]) #has word index 1
	for cf in corpusfiles:
		#print "INFO: corpus = %s"%(corpusfiles)
		if cf is not None: #source can be none
			corpus.extend(preprocess(codecs.open(cf,"r","utf8").readlines()))
	wordDictionary = corpora.Dictionary(corpus)
	return wordDictionary

def preprocess(docs):
	""" Preprocess a document: tokenize words """
	#texts = [[re.sub('[^A-Za-z0-9]+', '', word) for word in doc.lower().split()] for doc in docs] #no substitution of non-alphabetic characters possible, since they are considered as target words, no lower() since capitalisation is a type of translation error
	texts = [[word for word in word_tokenize(doc)] for doc in docs] #nltk tokenizer 
	#print texts
	return texts

def preprocessSent(doc):
	""" Preprocess a document: tokenize words """
	#texts = [[re.sub('[^A-Za-z0-9]+', '', word) for word in doc.lower().split()] for doc in docs] #no substitution of non-alphabetic characters possible, since they are considered as target words, no lower() since capitalisation is a type of translation error
	text = [word.lower() for word in word_tokenize(doc)] #nltk tokenizer 
	print "preprocessed:", text
	return text
