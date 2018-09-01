from math import log 
import numpy as np 
from numpy import array, argmax
import pickle

def beam_search_decoder(data, k):
	sequences = [[list(), 1.0]]
	# walk over each step in sequence
	for row in data:
		all_candidates = list()
		# expand each current candidate
		for i in range(len(sequences)):
			seq, score = sequences[i]
			for j in range(len(row)):
				candidate = [seq + [j], score * -log(row[j])]
				all_candidates.append(candidate)
		# order all candidates by score
		ordered = sorted(all_candidates, key=lambda tup:tup[1])
		# select k best
		sequences = ordered[:k]
	return sequences

predictions = pickle.load(open('./pickle-dumps/predictions', 'rb'))

results = []
for i in predictions[:2]:
	words = beam_search_decoder(i, 3)
	results.append(words)

print(len(results))
print(results[:2])