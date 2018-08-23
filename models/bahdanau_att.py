import pickle
from load_data_with_phonetic import load_data_for_seq2seq, load_data_for_features

import keras.backend as K
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Multiply, Add, Lambda, Activation, TimeDistributed, Dense, RepeatVector, Embedding, Input, merge, \
	concatenate, GaussianNoise, dot 
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Layer
from keras.optimizers import Adam, Adadelta
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from keras import initializers, regularizers, constraints
from attention_encoder import AttentionWithContext
from keras.layers import Dropout, Flatten, Conv1D, MaxPooling1D, AveragePooling1D, BatchNormalization, RepeatVector
from keras.regularizers import l2
from keras.constraints import maxnorm

from nltk import FreqDist
import numpy as np
import sys, time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import Counter, deque
from predict_with_features import plot_model_performance, returnTrainTestSets

# from curve_plotter import plot_precision_recall

MODE = 'trai'

EPOCHS = 500
dropout = 0.2
TIME_STEPS = 20
EMBEDDING_DIM = 64
BATCH_SIZE = 128
LAYER_NUM = 2
no_filters = 64
filter_length = 4
HIDDEN_DIM = no_filters*2
RNN = LSTM
rnn_output_size = 32
folds = 10

class_labels = []
def remove_erroneous_indices(lists):
	to_be_removed = pickle.load(open('./pickle-dumps/removed_indices', 'rb'))
	to_be_removed = list(set(to_be_removed)) # for ascending order

	helper_cnt = 0
	for i in to_be_removed:
		i = i - helper_cnt
		for j in range(len(lists)):
			lists[j].pop(i)
		helper_cnt = helper_cnt + 1

	return lists


def write_words_to_file(orig_words, predictions):
	print("Writing to file ..")
	# print(sentences[:10])
	sentences = pickle.load(open('./pickle-dumps/sentences_test', 'rb'))

	X = [item for sublist in sentences for item in sublist]
	Y = [item for sublist in orig_words for item in sublist]

	X, Y = remove_erroneous_indices([X,Y])

	filename = "./outputs/only_roots/root_out.txt"
	with open(filename, 'w', encoding='utf-8') as f:
		f.write("Words" + '\t\t\t' + 'Original Roots' + '\t\t' + "Predicted roots" + '\n')
		for a, b, c in zip(X, Y, predictions):
			f.write(str(a) + '\t\t' + str(b) + '\t\t\t' + str(c) + '\n')

	print("Success writing to file !")


def process_features(y1,y2,y3,y4,y5,y7,y8, n = None, enc=None):

	y = [y1, y2, y3, y4, y5, y7, y8]

	print(y1[:10])

	in_cnt1 = Counter(y1)
	in_cnt2 = Counter(y2)
	in_cnt3 = Counter(y3)
	in_cnt4 = Counter(y4)
	in_cnt5 = Counter(y5)
	in_cnt6 = Counter(y7)
	in_cnt7 = Counter(y8)

	labels=[] # for processing of unnecessary labels from the test set
	init_cnt = [in_cnt1, in_cnt2, in_cnt3, in_cnt4, in_cnt5, in_cnt6, in_cnt7]

	for i in range(len(init_cnt)):
		labels.append(list(init_cnt[i].keys()))

	if enc == None:
		enc = {}
		transformed = []
		print("processing train encoders!")
		for i in range(len(y)):
			enc[i] = LabelEncoder()
			transformed.append(enc[i].fit_transform(y[i]))

	else:
		transformed = []
		print("processing test encoders !")
		for i in range(len(y)):
			#y[i] = list(map(lambda s: '<unk>' if s not in enc[i].classes_ else s, y[i]))
			#enc_classes = enc[i].classes_.tolist()
			#bisect.insort_left(enc_classes, '<unk>')
			#enc[i].classes_ = enc_classes
			transformed.append(enc[i].transform(y[i]))

	y1 = list(transformed[0])
	y2 = list(transformed[1])
	y3 = list(transformed[2])
	y4 = list(transformed[3])
	y5 = list(transformed[4])
	y7 = list(transformed[5])
	y8 = list(transformed[6])

	print(y1[:10])

	cnt1 = Counter(y1)
	cnt2 = Counter(y2)
	cnt3 = Counter(y3)
	cnt4 = Counter(y4)
	cnt5 = Counter(y5)
	cnt6 = Counter(y7)
	cnt7 = Counter(y8)

	if enc != None:
		lis = [cnt1, cnt2, cnt3, cnt4, cnt5, cnt6, cnt7]
		for i in range(len(lis)):
			class_labels.append(list(lis[i].keys()))


	print(format(cnt1))
	print(format(cnt2))
	print(format(cnt3))

	if n == None:
		n1 = max(cnt1, key=int) + 1
		n2 = max(cnt2, key=int) + 1
		n3 = max(cnt3, key=int) + 1
		n4 = max(cnt4, key=int) + 1
		n5 = max(cnt5, key=int) + 1
		n6 = max(cnt6, key=int) + 1
		n7 = max(cnt7, key=int) + 1
	
	else:
		n1,n2,n3,n4,n5,n6,n7 = n

	y1 = np_utils.to_categorical(y1, num_classes=n1)
	y2 = np_utils.to_categorical(y2, num_classes=n2)
	y3 = np_utils.to_categorical(y3, num_classes=n3)
	y4 = np_utils.to_categorical(y4, num_classes=n4)
	y5 = np_utils.to_categorical(y5, num_classes=n5)
	y7 = np_utils.to_categorical(y7, num_classes=n6)
	y8 = np_utils.to_categorical(y8, num_classes=n7)

	return (y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y7, n6, y8, n7, enc, labels)



def process_data(word_sentences, max_len, word_to_ix):
	# Vectorizing each element in each sequence
	sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
	for i, sentence in enumerate(word_sentences):
		for j, word in enumerate(sentence):
			sequences[i, j, word] = 1
	return sequences

# def highway_layers(value, n_layers, activation='tanh', gate_bias=-3):
# 	dim = K.int_shape(value)[-1]
# 	gate_bias_initializer = keras.initializers.Constant(gate_bias)

# 	for i in range(n_layers):
# 		gate = Dense(dim, bias_initializer=gate_bias_initializer)(value)
# 		gate = Activation('sigmoid')(gate)
# 		negated_gate = Lambda(lambda x: 1.0-x, output_shape=(dim,))(gate)

# 		transformed = Dense(dim)(value)
# 		transformed = 
def create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, n_phonetic_features, y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y6, n6,
				 hidden_size, num_layers):
	def smart_merge(vectors, **kwargs):
		return vectors[0] if len(vectors) == 1 else merge(vectors, **kwargs)

	current_word = Input(shape=(X_max_len,), dtype='float32', name='input1') # for encoder (shared)
	right_word1 = Input(shape=(X_max_len,), dtype='float32', name='input2')
	left_word1 = Input(shape=(X_max_len,), dtype='float32', name='input3')
	phonetic_input = Input(shape=(n_phonetic_features,), dtype='float32', name='input4')

	emb_layer = Embedding(X_vocab_len, EMBEDDING_DIM,
						  input_length=X_max_len,
						  mask_zero=False, name='Embedding_for_seq2seq')

	list_of_inputs = [right_word1, current_word, left_word1]

	right_word_embedding1,  current_word_embedding, left_word_embedding1 = [emb_layer(i) for i in list_of_inputs]

	list_of_embeddings = [right_word_embedding1, current_word_embedding, left_word_embedding1]
	conv4_curr, conv4_right1, conv4_left1 = [Conv1D(filters=rnn_output_size, 
												kernel_size=4, padding='valid',activation='linear', 
												strides=1, name='conv4_'+str(j))(i) for i,j in zip(list_of_embeddings, range(len(list_of_embeddings)))]

	conv4s = [conv4_curr, conv4_right1, conv4_left1]

	bns = [BatchNormalization(name='bn1_'+str(j))(i) for i,j in zip(conv4s, range(len(conv4s)))]
	activations = [Activation('tanh', name='tanh_'+str(j))(i) for i,j in zip(bns, range(len(bns)))]

	# total_words = activations
	# [total_words.append(i) for i in list_of_embeddings]
	total_words = smart_merge(activations)

	encoder = Bidirectional(GRU(rnn_output_size, name='encoder'))(total_words)
	total_inputs = smart_merge([encoder, phonetic_input], mode='concat')

	RepLayer = RepeatVector(y_max_len)
	RepVec = RepLayer(total_inputs)
	Embed_plus_repeat = [current_word_embedding]
	Embed_plus_repeat.append(RepVec)
	Embed_plus_repeat = smart_merge(Embed_plus_repeat, mode='concat')


	decoder = GRU(rnn_output_size, return_sequences=True, name='decoder')(Embed_plus_repeat)
	bn2 = BatchNormalization(name='bn2')(decoder)
	td = TimeDistributed(Dense(y_vocab_len), name='td')(bn2)
	outputs =Activation('softmax', name='root_output')(td)

	all_inputs = [current_word, right_word1, left_word1, phonetic_input]
	all_outputs = [outputs]

	model = Model(input=all_inputs, output=all_outputs)
	
	model.compile(optimizer=Adadelta(epsilon=1e-06), loss='categorical_crossentropy',
				  metrics=['accuracy'])

	return model

sentences = pickle.load(open('./pickle-dumps/sentences_intra', 'rb'))
rootwords = pickle.load(open('./pickle-dumps/rootwords_intra', 'rb'))
features = pickle.load(open('./pickle-dumps/features_intra', 'rb'))

n_phonetics, X_train_phonetics, X_test_phonetics, X_val_phonetics = returnTrainTestSets()

# we keep X_idx2word and y_idx2word the same
# X_left & X_right = X shifted to one and two positions left and right for context2
X, X_vocab_len, X_word_to_ix, X_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word, X_left1, X_left2, X_left3, X_left4, \
X_right1, X_right2, X_right3, X_right4 = load_data_for_seq2seq(sentences, rootwords, test=False, context4=True)

y1, y2, y3, y4, y5, y6, y7, y8 = load_data_for_features(features)

y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y7, n7, y8, n8, enc, labels = process_features(y1, y2, y3, y4, y5, y7, y8)

n = [n1, n2, n3, n4, n5, n7, n8]

X_max = max([len(word) for word in X])
y_max = max([len(word) for word in y])
X_max_len = max(X_max, y_max)

print("Zero padding .. ")
X = pad_sequences(X, maxlen= X_max_len, dtype = 'int32', padding='post')
X_left1 = pad_sequences(X_left1, maxlen = X_max_len, dtype='int32', padding='post')
X_left2 = pad_sequences(X_left2, maxlen = X_max_len, dtype='int32', padding='post')
X_left3 = pad_sequences(X_left3, maxlen = X_max_len, dtype='int32', padding='post')
X_left4 = pad_sequences(X_left4, maxlen = X_max_len, dtype='int32', padding='post')
X_right1 = pad_sequences(X_right1, maxlen = X_max_len, dtype='int32', padding='post')
X_right2 = pad_sequences(X_right2, maxlen = X_max_len, dtype='int32', padding='post')
X_right3 = pad_sequences(X_right3, maxlen = X_max_len, dtype='int32', padding='post')
X_right4 = pad_sequences(X_right4, maxlen = X_max_len, dtype='int32', padding='post')
y = pad_sequences(y, maxlen = X_max_len, dtype = 'int32', padding='post')

print("Compiling Model ..")
model = create_model(X_vocab_len, X_max_len, y_vocab_len, X_max_len, n_phonetics,
					 y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y7, n7, HIDDEN_DIM, LAYER_NUM)

saved_weights = "./model_weights/rootwords.hdf5"


if MODE == 'train':
	print("Training model ..")
	plot_model(model, to_file="simple_with_both_pooling.png", show_shapes=True)
	y_sequences = process_data(y, X_max_len, y_word_to_ix)

	print("X len ======== ", len(X))
	train_val_cutoff = int(.75 * len(X))
	X_train, X_left1_tr, X_left2_tr, X_left3_tr, X_left4_tr, X_right1_tr, X_right2_tr, X_right3_tr, X_right4_tr, y_train = \
		[X[:train_val_cutoff], X_left1[:train_val_cutoff], X_left2[:train_val_cutoff], X_left3[:train_val_cutoff], X_left4[:train_val_cutoff],
			X_right1[:train_val_cutoff], X_right2[:train_val_cutoff], X_right3[:train_val_cutoff], X_right4[:train_val_cutoff], y[:train_val_cutoff]]
	X_val, X_left1_val, X_left2_val, X_left3_val, X_left4_val, X_right1_val, X_right2_val, X_right3_val, X_right4_val, y_val = \
		[X[train_val_cutoff:], X_left1[train_val_cutoff:], X_left2[train_val_cutoff:], X_left3[train_val_cutoff:], X_left4[train_val_cutoff:],
			X_right1[train_val_cutoff:], X_right2[train_val_cutoff:], X_right3[train_val_cutoff:], X_right4[train_val_cutoff:], y[train_val_cutoff:]]

	y_sequences_tr, y1_tr, y2_tr, y3_tr, y4_tr, y5_tr, y7_tr = \
			[y_sequences[:train_val_cutoff], y1[:train_val_cutoff], y2[:train_val_cutoff], y3[:train_val_cutoff], \
				y4[:train_val_cutoff], y5[:train_val_cutoff], y7[:train_val_cutoff]]	
	y_sequences_val, y1_val, y2_val, y3_val, y4_val, y5_val, y7_val = \
			[y_sequences[train_val_cutoff:], y1[train_val_cutoff:], y2[train_val_cutoff:], y3[train_val_cutoff:], \
				y4[train_val_cutoff:], y5[train_val_cutoff:], y7[train_val_cutoff:]]

	hist = model.fit([X_train,  X_right1_tr,  X_left1_tr, X_train_phonetics],
					 [y_sequences_tr],
					 validation_data=([X_val, X_right1_val, X_left1_val, X_val_phonetics],\
					 	[y_sequences_val]),
					 batch_size=BATCH_SIZE, epochs=EPOCHS,
					 callbacks=[EarlyStopping(patience=10),
								ModelCheckpoint('./model_weights/rootwords.hdf5', save_best_only=True,
												verbose=1, save_weights_only=True),
								ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.001)])

	# model.save('./model_weights/attention_with_roots.hdf5')
	print(hist.history.keys())
	print(hist)
	plot_model_performance(
		train_loss=hist.history.get('loss', []),
		train_acc=hist.history.get('acc', []),
		train_val_loss=hist.history.get('val_loss', []),
		train_val_acc=hist.history.get('val_acc', [])
	)


else:
	if len(saved_weights) == 0:
		print("network hasn't been trained!")
		sys.exit()
	else:
		test_sample_num = 0

		test_sentences = pickle.load(open('./pickle-dumps/sentences_test', 'rb'))
		test_roots = pickle.load(open('./pickle-dumps/rootwords_test', 'rb'))
		test_features = pickle.load(open('./pickle-dumps/features_test', 'rb'))

		y1, y2, y3, y4, y5, y6, y7, y8 = load_data_for_features(test_features)
		features = [y1, y2, y3, y4, y5, y7, y8]

		complete_list, X_test, X_vcab_len, X_wrd_to_ix, X_ix_to_wrd, y_test, y_vcab_len, y_wrd_to_ix, y_ix_to_wrd, \
		X_left1, X_left2, X_left3, X_left4, X_right1, X_right2, X_right3, X_right4, X_phonetic_features = \
			load_data_for_seq2seq(test_sentences, test_roots, X_test_phonetics, features, labels, test=True, context4=True)

		X_orig, y_orig, y1, y2, y3, y4, y5, y7, y8 = complete_list

		to_be_padded = [X_test, X_left1, X_right1, X_left2, X_right2, X_left3, X_right3, X_left4, X_right4, y_test]

		X_test, X_left1, X_right1, X_left2, X_right2, X_left3, X_right3, X_left4, X_right4, y_test= \
						[pad_sequences(i, maxlen=X_max_len, dtype='int32', padding='post') for i in to_be_padded]

		y_test_seq = process_data(y_test, X_max_len, y_word_to_ix)

		# y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y7, n7, y8, n8, enc, lab = process_features(y1, y2, y3, y4, y5, y7, y8,
		# 																					n,
		# 																					enc)  # pass previous encoders as args

		decoder_input = np.zeros_like(X_test)
		decoder_input[:, 1:] = X_test[:,:-1]
		decoder_input[:, 0] = 1

		model.load_weights(saved_weights)
		print(model.summary())

		print(model.evaluate([X_test, X_right1, X_left1, X_phonetic_features],
							 [y_test_seq]))
		# # print(model.metrics_names)

		words = model.predict(
			[X_test, X_right1, X_left1, X_test_phonetics])

		predictions = np.argmax(words, axis=2)
	
		# Post processing of predicted roots
		sequences = []

		for i in predictions:
			test_sample_num += 1

			char_list = []
			for idx in i:
				if idx > 0:
					char_list.append(y_ix_to_word[idx])

			sequence = ''.join(char_list)
			#print(test_sample_num,":", sequence)
			sequences.append(sequence)

		write_words_to_file(test_roots, sequences)

