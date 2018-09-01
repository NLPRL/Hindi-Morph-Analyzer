import pickle
from load_data_with_phonetic import load_data_for_seq2seq, load_data_for_features

import keras.backend as K
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, load_model
from keras.layers import Multiply, Add, Lambda, Activation, TimeDistributed, Dense, RepeatVector, Embedding, Input, merge, \
	concatenate, GaussianNoise, dot 
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Layer
from keras.optimizers import Adam, Adadelta
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from keras import initializers, regularizers, constraints
from attention_decoder import AttentionDecoder
from attention_encoder import AttentionWithContext
from keras.layers import Dropout, Flatten, Conv1D, MaxPooling1D, AveragePooling1D, BatchNormalization
from keras.regularizers import l2
from keras.constraints import maxnorm

from nltk import FreqDist
import numpy as np
import sys, time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from collections import Counter, deque
from predict_with_features import *

dropout = 0.2
TIME_STEPS = 20
EMBEDDING_DIM = 64
BATCH_SIZE = 128
LAYER_NUM = 2
no_filters = 64
filter_length = 4
HIDDEN_DIM = no_filters*2
RNN = GRU
rnn_output_size = 32
folds = 10


FILENAME = 'test_file.txt'
sentences = [line.split() for line in open(FILENAME).readlines()]

X_orig = [item for sublist in sentences for item in sublist]
# print('X_orig',  X_orig)
X_wrds = [item[::-1] for sublist in sentences for item in sublist]
# print('x_wrds',  X_wrds)
# print(X_wrds)

def encode_words(X):
	X_word2idx = pickle.load(open('./pickle-dumps/X_word2idx', 'rb'))
	X_return = []
	for i, word in enumerate(X):
		temp = []
		for j, char in enumerate(word):
			if char in X_word2idx:
				temp.append(X_word2idx[char])
			else:
				temp.append(X_word2idx['U'])
		X_return.append(temp)
	# print('X_return', X_return)
	return X_return

X_wrds_inds = encode_words(X_wrds)
# print(type(X_wrds_inds), type(X_wrds_inds[0]))
# X - words encoded
X_features = [add_basic_features(sent, word_ind) for sent in sentences for word_ind, _ in enumerate(sent)]

def encode_features(X_test):
	# print(list(zip(*X_test))[4])
	total_features_to_be_encoded = len(X_test[0][3:])
	encoders = pickle.load(open('./pickle-dumps/phonetic_feature_encoders', 'rb'))
	transformed_feature_to_be_returned = []
	for i in range(len(encoders)):
		# print("Encoding and transforming test set feature: ", i)
		# X_test = removeUnknownTestSamples(X_test, y_test, label_names, encoders)
		# print(list(zip(*X_test))[i+3])
		# print(list(encoders[i].classes_))
		arr = [w if w in list(encoders[i].classes_) else 'UNK' for w in list(zip(*X_test))[i+3]]
		transformed_feature_to_be_returned.append(encoders[i].transform(arr))
	
	X_test = np.asarray(X_test)
	for i in range(total_features_to_be_encoded):
		X_test[:,i+3] = transformed_feature_to_be_returned[i]

	X_test = X_test.astype(np.float)
	X_test = X_test.tolist()			
	return X_test

X_fts = encode_features(X_features)
n_phonetics = len(pickle.load(open('./pickle-dumps/X_train', 'rb'))[1])
#X_fts has features encoded

def getIndexedWords(X_unique):
	X_un = [list(x) for x in X_unique if len(x) > 0]
	X = X_un
	# print("X:", X[:10])
	X_word2idx = pickle.load(open('./pickle-dumps/X_word2idx', 'rb'))
	X_idx2word = pickle.load(open('./pickle-dumps/X_idx2word', 'rb'))
	for i, word in enumerate(X):
		for j, char in enumerate(word):
			if char in X_word2idx:
				X[i][j] = X_word2idx[char]
			else:
				X[i][j] = X_word2idx['U']
	return X

def get_context(X_unique):
	X_left = deque(X_unique)

	X_left.append(' ') # all elements would be shifted one left
	X_left.popleft()
	X_left1 = list(X_left)
	X_left1 = getIndexedWords(X_left1)

	X_left.append(' ')
	X_left.popleft()
	X_left2 = list(X_left)
	X_left2 = getIndexedWords(X_left2)

	X_left.append(' ')
	X_left.popleft()
	X_left3 = list(X_left)
	X_left3 = getIndexedWords(X_left3)

	X_left.append(' ')
	X_left.popleft()
	X_left4 = list(X_left)
	X_left4 = getIndexedWords(X_left4)	

	X_right_orig = X_unique
	X_right = deque(X_right_orig)

	X_right.appendleft(' ') 
	X_right.pop()
	X_right1 = list(X_right)
	X_right1 = getIndexedWords(X_right1)

	X_right.appendleft(' ')
	X_right.pop()
	X_right2 = list(X_right)
	X_right2 = getIndexedWords(X_right2)

	X_right.appendleft(' ')
	X_right.pop()
	X_right3 = list(X_right)
	X_right3 = getIndexedWords(X_right3)

	X_right.appendleft(' ')
	X_right.pop()
	X_right4 = list(X_right)
	X_right4 = getIndexedWords(X_right4)

	return X_left1, X_left2, X_left3, X_left4, X_right1, X_right2, X_right3, X_right4 

X_left1, X_left2, X_left3, X_left4, X_right1, X_right2, X_right3, X_right4 = get_context(X_wrds)

################################################################################################

def create_model(X_vocab_len, X_max_len, n_phonetic_features, n1, n2, n3, n4, n5, n6, HIDDEN_DIM, LAYER_NUM):
	def smart_merge(vectors, **kwargs):
		return vectors[0] if len(vectors) == 1 else merge(vectors, **kwargs)

	current_word = Input(shape=(X_max_len,), dtype='float32', name='input1') # for encoder (shared)
	decoder_input = Input(shape=(X_max_len,), dtype='float32', name='input3') # for decoder -- attention
	right_word1 = Input(shape=(X_max_len,), dtype='float32', name='input4')
	right_word2 = Input(shape=(X_max_len,), dtype='float32', name='input5')
	right_word3 = Input(shape=(X_max_len,), dtype='float32', name='input6')
	right_word4 = Input(shape=(X_max_len,), dtype='float32', name='input7')
	left_word1 = Input(shape=(X_max_len,), dtype='float32', name='input8')
	left_word2 = Input(shape=(X_max_len,), dtype='float32', name='input9')
	left_word3 = Input(shape=(X_max_len,), dtype='float32', name='input10')
	left_word4 = Input(shape=(X_max_len,), dtype='float32', name='input11')
	phonetic_input = Input(shape=(n_phonetic_features,), dtype='float32', name='input12')

	emb_layer1 = Embedding(X_vocab_len, EMBEDDING_DIM,
						  input_length=X_max_len,
						  mask_zero=False, name='Embedding')

	list_of_inputs = [current_word, right_word1, right_word2, right_word3,right_word4, 
					left_word1, left_word2, left_word3, left_word4]

	current_word_embedding, right_word_embedding1, right_word_embedding2,right_word_embedding3, right_word_embedding4, \
		left_word_embedding1, left_word_embedding2, left_word_embedding3, left_word_embedding4 = [emb_layer1(i) for i in list_of_inputs]

	print("Type:: ",type(current_word_embedding))
	list_of_embeddings1 = [current_word_embedding, right_word_embedding1, right_word_embedding2,right_word_embedding3, right_word_embedding4, \
		left_word_embedding1, left_word_embedding2, left_word_embedding3, left_word_embedding4]

	list_of_embeddings = [Dropout(0.50, name='drop1_'+str(j))(i) for i,j in zip(list_of_embeddings1, range(len(list_of_embeddings1)))]
	list_of_embeddings = [GaussianNoise(0.05, name='noise1_'+str(j))(i) for i,j in zip(list_of_embeddings, range(len(list_of_embeddings)))]
	
	conv4_curr, conv4_right1, conv4_right2, conv4_right3, conv4_right4, conv4_left1, conv4_left2, conv4_left3, conv4_left4 =\
			[Conv1D(filters=no_filters, 
				kernel_size=4, padding='valid',activation='relu', 
				strides=1, name='conv4_'+str(j))(i) for i,j in zip(list_of_embeddings, range(len(list_of_embeddings)))]

	conv4s = [conv4_curr, conv4_right1, conv4_right2, conv4_right3, conv4_right4, conv4_left1, conv4_left2, conv4_left3, conv4_left4]
	maxPool4 = [MaxPooling1D(name='max4_'+str(j))(i) for i,j in zip(conv4s, range(len(conv4s)))]
	avgPool4 = [AveragePooling1D(name='avg4_'+str(j))(i) for i,j in zip(conv4s, range(len(conv4s)))]

	pool4_curr, pool4_right1, pool4_right2, pool4_right3, pool4_right4, pool4_left1, pool4_left2, pool4_left3, pool4_left4 = \
		[merge([i,j], name='merge_conv4_'+str(k)) for i,j,k in zip(maxPool4, avgPool4, range(len(maxPool4)))]

	conv5_curr, conv5_right1, conv5_right2, conv5_right3, conv5_right4, conv5_left1, conv5_left2, conv5_left3, conv5_left4 = \
			[Conv1D(filters=no_filters,
				kernel_size=5,
				padding='valid',
				activation='relu',
				strides=1, name='conv5_'+str(j))(i) for i,j in zip(list_of_embeddings, range(len(list_of_embeddings)))]	

	conv5s = [conv5_curr, conv5_right1, conv5_right2, conv5_right3, conv5_right4, conv5_left1, conv5_left2, conv5_left3, conv5_left4]
	maxPool5 = [MaxPooling1D(name='max5_'+str(j))(i) for i,j in zip(conv5s, range(len(conv5s)))]
	avgPool5 = [AveragePooling1D(name='avg5_'+str(j))(i) for i,j in zip(conv5s, range(len(conv5s)))]

	pool5_curr, pool5_right1, pool5_right2, pool5_right3, pool5_right4, pool5_left1, pool5_left2, pool5_left3, pool5_left4 = \
		[merge([i,j], name='merge_conv5_'+str(k)) for i,j,k in zip(maxPool5, avgPool5, range(len(maxPool5)))]


	maxPools = [pool4_curr, pool4_right1, pool4_right2, pool4_right3, pool4_right4, \
		pool4_left1, pool4_left2, pool4_left3, pool4_left4, \
		pool5_curr, pool5_right1, pool5_right2, pool5_right3, pool5_right4, \
		pool5_left1, pool5_left2, pool5_left3, pool5_left4]

	concat = merge(maxPools, mode='concat', name='main_merge')

	x = Dropout(0.15, name='drop_single1')(concat)
	x = Bidirectional(RNN(rnn_output_size), name='bidirec1')(x)

	total_features = [x, phonetic_input]
	concat2 = merge(total_features, mode='concat', name='phonetic_merging')

	x = Dense(HIDDEN_DIM, activation='relu', kernel_initializer='he_normal',
			  kernel_constraint= maxnorm(3), bias_constraint=maxnorm(3), name='dense1')(concat2)
	x = Dropout(0.15, name='drop_single2')(x)
	x = Dense(HIDDEN_DIM, kernel_initializer='he_normal', activation='tanh',
			  kernel_constraint= maxnorm(3), bias_constraint=maxnorm(3), name='dense2')(x)
	x = Dropout(0.15, name='drop_single3')(x)

	out1 = Dense(n1, kernel_initializer='he_normal', activation='softmax', name='output1')(x)
	out2 = Dense(n2, kernel_initializer='he_normal', activation='softmax', name='output2')(x)
	out3 = Dense(n3, kernel_initializer='he_normal', activation='softmax', name='output3')(x)
	out4 = Dense(n4, kernel_initializer='he_normal', activation='softmax', name='output4')(x)
	out5 = Dense(n5, kernel_initializer='he_normal', activation='softmax', name='output5')(x)
	out6 = Dense(n6, kernel_initializer='he_normal', activation='softmax', name='output6')(x)

	# Luong et al. 2015 attention model	
	emb_layer = Embedding(X_vocab_len, EMBEDDING_DIM,
						  input_length=X_max_len,
						  mask_zero=True, name='Embedding_for_seq2seq')

	current_word_embedding, right_word_embedding1, right_word_embedding2,right_word_embedding3, right_word_embedding4, \
		left_word_embedding1, left_word_embedding2, left_word_embedding3, left_word_embedding4 = [emb_layer(i) for i in list_of_inputs]

	# current_word_embedding = smart_merge([ current_word_embedding, right_word_embedding1,  left_word_embedding1])

	encoder, state = GRU(rnn_output_size, return_sequences=True, unroll=True, return_state=True, name='encoder')(current_word_embedding)
	encoder_last = encoder[:,-1,:]

	decoder = emb_layer(decoder_input)
	decoder = GRU(rnn_output_size, return_sequences=True, unroll=True, name='decoder')(decoder, initial_state=[encoder_last])

	attention = dot([decoder, encoder], axes=[2,2], name='dot')
	attention = Activation('softmax', name='attention')(attention)

	context = dot([attention, encoder], axes=[2,1], name='dot2')
	decoder_combined_context = concatenate([context, decoder], name='concatenate')

	outputs = TimeDistributed(Dense(64, activation='tanh'), name='td1')(decoder_combined_context)
	outputs = TimeDistributed(Dense(X_vocab_len, activation='softmax'),  name='td2')(outputs)

	all_inputs = [current_word, decoder_input, right_word1, right_word2, right_word3, right_word4, left_word1, left_word2, left_word3,\
				  left_word4, phonetic_input]
	all_outputs = [outputs, out1, out2, out3, out4, out5, out6]

	model = Model(input=all_inputs, output=all_outputs)
	opt = Adam()

	return model
X_vocab_len = 90
X_max_len = 18
n1, n2, n3, n4, n5, n7, _ = pickle.load(open('pickle-dumps/n', 'rb'))

# print("Zero padding .. ")
X_wrds_inds = pad_sequences(X_wrds_inds, maxlen= X_max_len, dtype = 'int32', padding='post')
X_left1 = pad_sequences(X_left1, maxlen = X_max_len, dtype='int32', padding='post')
X_left2 = pad_sequences(X_left2, maxlen = X_max_len, dtype='int32', padding='post')
X_left3 = pad_sequences(X_left3, maxlen = X_max_len, dtype='int32', padding='post')
X_left4 = pad_sequences(X_left4, maxlen = X_max_len, dtype='int32', padding='post')
X_right1 = pad_sequences(X_right1, maxlen = X_max_len, dtype='int32', padding='post')
X_right2 = pad_sequences(X_right2, maxlen = X_max_len, dtype='int32', padding='post')
X_right3 = pad_sequences(X_right3, maxlen = X_max_len, dtype='int32', padding='post')
X_right4 = pad_sequences(X_right4, maxlen = X_max_len, dtype='int32', padding='post')

# print(type(X_wrds_inds))

model = create_model(X_vocab_len, X_max_len, n_phonetics, n1, n2, n3, n4, n5, n7, HIDDEN_DIM, LAYER_NUM)
model.load_weights('./model_weights/frozen_training_weights.hdf5')

decoder_input = np.zeros_like(X_wrds_inds)
decoder_input[:, 1:] = X_wrds_inds[:,:-1]
decoder_input[:, 0] = 1

scaler = MinMaxScaler()
scaler.fit(X_fts)
X_fts = scaler.transform(X_fts)

words, f1, f2, f3, f4, f5, f7 = model.predict(
			[X_wrds_inds, decoder_input, X_right1, X_right2, X_right3, X_right4, X_left1, X_left2, X_left3, X_left4, X_fts])

predictions = np.argmax(words, axis=2)
pred_features = [f1, f2, f3, f4, f5, f7]
pred_features = [np.argmax(i, axis=1) for i in pred_features]

X_idx2word = pickle.load(open('./pickle-dumps/X_idx2word', 'rb'))
sequences = []

def write_words_to_file(predictions, originals, encoders, pred_features):
	# print("Writing to file ..")
	# print(sentences[:10])

	pred_features[:] = [x.tolist() for x in pred_features]

	for i in range(len(pred_features)):
		pred_features[i] = encoders[i].inverse_transform(pred_features[i])
		# print(pred_features[i][:10])

	f1, f2, f3, f4, f5, f7 = pred_features

	filename = "./outputs/bakchodi/multitask_context_out1.txt"
	with open(filename, 'w', encoding='utf-8') as f:
		# print(list(originals), list(predictions))
		print("Words" + '\t\t\t' +  "Roots" + '\t\t\t' + 'POS' + '\t\t\t' + 'gender' + '\t\t\t' + 'Number' + '\t\t\t'+ 'Person' + '\t\t\t' + "Case" + '\t\t\t' + 'TAM' + '\n')
		for a, b, c, d, e, f, g, h  in zip(list(originals), list(predictions), f1, f2, f3, f4, f5, f7):
			# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
			# print([list(i) for i in [a,b,c,d,e,f,g,h]])
			print((str(a) + '\t\t\t' + str(b) + '\t\t\t' + str(c) + '\t\t\t' +str(d) + '\t\t\t' +str(e) + '\t\t\t' +str(f) + '\t\t\t' +str(g) + '\t\t\t' +str(h)))
			# f.write(str(a) + '\t\t\t' + str(b) + str(c) + '\t\t\t' +str(d) + '\t\t\t' +str(e) + '\t\t\t' +str(f) + '\t\t\t' +str(g) + '\t\t\t' +str(h) + '\t\t\t' +'\n')

	# print("Success writing to file !")


def write_features_to_file(orig_features, pred_features, encoders):

	orig_features[:] = [ [np.where(r == 1)[0][0] for r in x] for x in orig_features]
	# print(orig_features[0][:10])
	pred_features[:] = [x.tolist() for x in pred_features]
	
	for i in range(len(orig_features)):
		orig_features[i] = encoders[i].inverse_transform(orig_features[i])
		# print(orig_features[i][:10])
		pred_features[i] = encoders[i].inverse_transform(pred_features[i])
		# print(pred_features[i][:10])

	sentences = pickle.load(open('./pickle-dumps/sentences_test', 'rb'))
	words = [item for sublist in sentences for item in sublist]

	for i in range(len(orig_features)):
		filename = "./outputs/freezing_with_luong/feature"+str(i)+"context_out1.txt"
		with open(filename, 'w', encoding='utf-8') as f:
			f.write("Word" + '\t\t\t' + 'Original feature' + '\t' + 'Predicted feature' + '\n')
			for a,b,c in zip(words, orig_features[i], pred_features[i]):
				f.write(str(a) + '\t\t\t' + str(b) + '\t\t\t' + str(c) + '\n')

	# print("Success writing features to files !!")

for i in predictions:

	char_list = []
	for idx in i:
		if idx > 0:
			char_list.append(X_idx2word[idx])

	sequence = ''.join(char_list)
	#print(test_sample_num,":", sequence)
	sequences.append(sequence)

enc = pickle.load(open('pickle-dumps/enc', 'rb'))
write_words_to_file(sequences, X_orig, enc, pred_features)