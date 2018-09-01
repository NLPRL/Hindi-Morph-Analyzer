from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import nltk
import os
import sys 

#print(fuzz.ratio("this is a test", "this is a pre-test!"))

def get_string_similarity(i, j):
	return fuzz.ratio(i, j)


def load_data(src_dir):
	for (dirpath, dirnames, filenames) in os.walk(src_dir):
		for direc in dirnames:
			abs_path = os.path.join(dirpath, direc)
			for (filepath, dirnames2, filenames) in os.walk(abs_path):
				for filename in filenames.endswith()
	text = open(data_file, 'r', encoding="utf-8").readlines()[1:]

	word_list = []

	for line in text:
		line = line.strip()
		stripped_line = line.replace('\u200b','')
		stripped_line = line.replace('\u200d','').split('\t\t\t')
		word_list.append(stripped_line)
		#print(word_list)
		
	X = [c[1] for c in word_list]
	y = [c[2] for c in word_list]


	for i,j in zip(X,y):
		print(i + '\t' + j)

	#X = [list(x) for x, w in zip(X, y) if len(x) > 0 and len(w) > 0] # list of lists
	#y = [list(w) for x, w in zip(X,y) if len(x) > 0 and len(w) > 0]

	print(X[:10])
	return (X, y)
	

X, y = load_data(data_file)

sum = 0
for i,j in zip(X,y):
	sum += get_string_similarity(i,j)

res = sum/len(X)
print("Leveishtein similarity of whole doc: ", res)
