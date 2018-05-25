import os
from nltk import FreqDist
import numpy as np
import re
import datetime
import sys
import gc
import pickle 

from collections import Counter, deque

MAX_LEN = 20
VOCAB_SIZE = 65
VOCAB_SIZE_WORDS = 30000

def getIndexedWords(X_unique, y_unique, orig=False):
	X_un = [list(x) for x,w in zip(X_unique, y_unique) if len(x) > 0 and len(w) > 0]

	X = X_un

	# build a vocabulary of most frequent characters
	dist = FreqDist(np.hstack(X))
	X_vocab = dist.most_common(89)


	print("######### Remove erroneous characters ##########")
	for i in X_vocab:
		if i[0] == '\u200d' or i[0] == '\u200b':
			X_vocab.remove(i)

	#print(X_vocab)

	X_idx2word = [letter[0] for letter in X_vocab]
	X_idx2word.insert(0, 'Z') # 'Z' is the starting token
	X_idx2word.append('W') # 'U' for out-of-vocab characters
	
	# create letter-to-index mapping
	X_word2idx =  {letter:idx for idx, letter in enumerate(X_idx2word)}

	print(X_idx2word)
	print(len(X_idx2word))
	print(len(X_word2idx))

	for i, word in enumerate(X):
		for j, char in enumerate(word):
			if char in X_word2idx:
				X[i][j] = X_word2idx[char]
			else:
				X[i][j] = X_word2idx['U']

	if orig == True:
		return X, X_un, X_vocab, X_word2idx, X_idx2word
	else:
		return X


def load_data_for_seq2seq(sentences, rootwords, features=None, labels=None, test=False, context1=False, context2=False, context3=False):
	#print(sentences[:2])
	
	X_unique = [item for sublist in sentences for item in sublist]
	y_unique = [item for sublist in rootwords for item in sublist]

	#print("X_unique:",X_unique[:5])
	############## processing of test set ################
	if features != None:
 		j = 0
 		y1,y2,y3,y4,y5,y7,y8 = features
 		l1, l2, l3, l4, l5, l7, l8 = labels
 		print(l1)
 		print(l2)
 		print(l3)
 		print(l4)
 		print(l5)
 		print(l7)
 		print(l8)
 		complete_list = [X_unique, y_unique, y1, y2, y3, y4, y5, y7, y8]

 		#print(X_unique[5])
 		copy = X_unique
 		#print(format(Counter(y1)))
 		#print(X_unique[5])

 		cnt = len(X_unique)
 		i = 0
 		while i < cnt:
 			#try:
	 		if y1[i] not in l1 or y2[i] not in l2 or y3[i] not in l3 or y4[i] not in l4 \
	 		or y5[i] not in l5 or y7[i] not in l7 or y8[i] not in l8 or y7[i] == 'kI':
	 			for item in complete_list:
	 				print("Deleting element:",j)
	 				j += 1
 					del item[i]
 					cnt = cnt - 1
 				i = i - 1
 			i = i + 1
	 		#except IndexError:
	 		#break
	 	'''
	 	print(X_unique[5])
	 	print(format(Counter(y1)))
	 	print(format(Counter(y2)))
	 	print(format(Counter(y3)))
	 	print(format(Counter(y4)))
	 	print(format(Counter(y5)))
	 	print(format(Counter(y7)))
	 	print(format(Counter(y8)))
	 	'''
	#####################################################

	# process vocab indexing for X in the function since we will need to call it multiple times
	X, X_un, X_vocab, X_word2idx, X_idx2word = getIndexedWords(X_unique, y_unique, orig=True)
	print(X[:10])

	# process vocab indexing for y here, since only single processing required
	y_un = [list(w) for x,w in zip(X_unique, y_unique) if len(x) > 0 and len(w) > 0]	
	y = y_un
	
	for i, word in enumerate(y):
		for j, char in enumerate(word):
			if char in X_word2idx:
				y[i][j] = X_word2idx[char]
			else:
				y[i][j] = X_word2idx['U']
	
	# consider a context of 1 word right and left each
	# make two lists by shifting the elements
	if context1 == True or context2 == True or context3 == True: 
		X_left = deque(X_unique)
		print(len(X_left))
		
		X_left.append(' ') # all elements would be shifted one left
		X_left.popleft()
		X_left1 = list(X_left)
		X_left1 = getIndexedWords(X_left1, y_unique, orig=False)

		X_left.append(' ')
		X_left.popleft()
		X_left2 = list(X_left)
		X_left2 = getIndexedWords(X_left2, y_unique, orig=False)
		
		X_left.append(' ')
		X_left.popleft()
		X_left3 = list(X_left)
		X_left3 = getIndexedWords(X_left3, y_unique, orig=False)
		

		X_right = deque(X_unique)

		X_right.appendleft(' ') 
		X_right.pop()
		X_right1 = list(X_right)
		X_right1 = getIndexedWords(X_right1, y_unique, orig=False)

		X_right.appendleft(' ')
		X_right.pop()
		X_right2 = list(X_right)
		X_right2 = getIndexedWords(X_right2, y_unique, orig=False)

		X_right.appendleft(' ')
		X_right.pop()
		X_right3 = list(X_right)
		X_right3 = getIndexedWords(X_right3, y_unique, orig=False)

		print(len(X_left1))
		print(len(X_left2))
		print(len(X_right1))
		print(len(X_right2))

		if context1 == True:
			if test == True:
				complete_list = [X_un, y_un, y1, y2, y3, y4, y5, y7, y8]
				return (complete_list, X, len(X_vocab)+2, X_word2idx, X_idx2word, y, len(X_vocab)+2, X_word2idx, X_idx2word, X_left, X_right)
			else:
				return (X, len(X_vocab)+2, X_word2idx, X_idx2word, y, len(X_vocab)+2, X_word2idx, X_idx2word, X_left, X_right)

		elif context2 == True:
			if test == True:
				complete_list = [X_un, y_un, y1, y2, y3, y4, y5, y7, y8]
				return (complete_list, X, len(X_vocab)+2, X_word2idx, X_idx2word, y, 
					len(X_vocab)+2, X_word2idx, X_idx2word, X_left1, X_left2, X_right1, X_right2)
			else:
				return (X, len(X_vocab)+2, X_word2idx, X_idx2word, y, len(X_vocab)+2, 
					X_word2idx, X_idx2word, X_left1, X_left2, X_right1, X_right2)

		elif context3 == True:
			if test == True:
				complete_list = [X_un, y_un, y1, y2, y3, y4, y5, y7, y8]
				return (complete_list, X, len(X_vocab)+2, X_word2idx, X_idx2word, y, 
					len(X_vocab)+2, X_word2idx, X_idx2word, X_left1, X_left2, X_left3, X_right1, X_right2, X_right3)
			else:
				return (X, len(X_vocab)+2, X_word2idx, X_idx2word, y, len(X_vocab)+2, 
					X_word2idx, X_idx2word, X_left1, X_left2, X_left3, X_right1, X_right2, X_right3) 
	else:
		if test == True:
			complete_list = [X_un, y_un, y1, y2, y3, y4, y5, y7, y8]
			return (complete_list, X, len(X_vocab)+2, X_word2idx, X_idx2word, y, len(X_vocab)+2, X_word2idx, X_idx2word)
		else:
			return (X, len(X_vocab)+2, X_word2idx, X_idx2word, y, len(X_vocab)+2, X_word2idx, X_idx2word)

def load_data_for_features(features, sentences=None):
	# this function is different from above two in the sense that
	# the vocabulary is built on a word level instead of character levels.	
	flat_features = [item for sublist in features for item in sublist]
	print(flat_features[1])

	splitted_feature = [] # seggregate all the features
	for feature in flat_features:
		splitted_feature.append(feature.split("|"))

	print(len(splitted_feature))
	print(splitted_feature[1])

	y1 = []
	y2 = []
	y3 = []
	y4 = []
	y5 = []
	y6 = []
	y7 = []
	y8 = []

	all_features = [y1, y2, y3, y4, y5, y6, y7, y8]

	for feature in splitted_feature:
		for i,j in zip(feature[:8], all_features):
			val = re.sub(r'.*-', '', i) 
			if len(val) != 0:
				j.append(val)
			else:
				j.append('Unk')

	if sentences != None:
		dist = FreqDist(np.hstack(sentences))
		X_vocab = dist.most_common(VOCAB_SIZE_WORDS)

		X_idx2word = [word[0] for word in X_vocab]
		X_idx2word.insert(0, 'ZERO')
		X_idx2word.append('UNK')

		X_idx2word = [word[0] for word in X_vocab]
		X_idx2word.insert(0, 'ZERO')
		X_idx2word.append('UNK')

		X_word2idx = {word:idx for idx,word in enumerate(X_idx2word)}

		X = sentences
		for i, sentence in enumerate(X):
			for j, word in enumerate(sentence):
				if word in X_word2idx:
					X[i][j] = X_word2idx[word]
				else:
					X[i][j] = X_word2idx['UNK']
	
	return (y1, y2, y3, y4, y5, y6, y7, y8)


# To preserve the sentences without flatting out the lists
def load_data_for_features_sentencewise(features):
	# this function is different from above two in the sense that
	# the vocabulary is built on a word level instead of character levels.	
	splitted_feature = [] # seggregate all the features
	for feature in features:
		this_sentence = []
		for i in feature:
			this_sentence.append(i.split("|"))
		splitted_feature.append(this_sentence)

	print(splitted_feature[8:10])
	
	print(len(splitted_feature))
	
	y1 = []
	y2 = []
	y3 = []
	y4 = []
	y5 = []
	y6 = []
	y7 = []
	y8 = []

	all_features = [y1, y2, y3, y4, y5, y6, y7, y8]

	for feature in splitted_feature:
		f1 = []
		f2 = []
		f3 = []
		f4 = []
		f5 = []
		f6 = []
		f7 = []
		f8 = []

		all_fs = [f1, f2, f3, f4, f5, f6, f7, f8]
		for this in feature:
			for i,j in zip(this[:8], all_fs):
				val = re.sub(r'.*-', '', i) 
				if len(val) != 0:
					j.append(val)
				else:
					j.append('Unk')

		for i,j in zip(all_features, all_fs):
			i.append(j)

	print(y2[8:10])

	pickle.dump(y1, open('y1_test', 'wb'))
	pickle.dump(y2, open('y2_test', 'wb'))
	pickle.dump(y3, open('y3_test', 'wb'))

sentences = pickle.load(open('./pickle-dumps/sentences_intra', 'rb'))
rootwords = pickle.load(open('./pickle-dumps/rootwords_intra', 'rb'))
features = pickle.load(open('./pickle-dumps/features_intra', 'rb'))

#load_data_for_features_sentencewise(features)


# we keep X_idx2word and y_idx2word the same
#X, X_vocab_len, X_word_to_ix, X_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word, X_left, X_right =
load_data_for_seq2seq(sentences, rootwords, context2=True)
#load_data_for_features_sentencewise(features)
'''
print(X[:5])
print(X_left[:5])
print(X_right[:5])
'''
