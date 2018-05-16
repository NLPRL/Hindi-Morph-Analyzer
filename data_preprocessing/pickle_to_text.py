import pickle
import os
import sys

sentences = pickle.load(open('sentences_intra', 'rb'))

with open('all_sentences.txt', 'w', encoding='utf-8') as f:
	for i in sentences:
		sentence = ' '.join(str(e) for e in i)
		f.write(sentence+'\n')
