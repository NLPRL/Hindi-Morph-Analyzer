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
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras import initializers, regularizers, constraints
from attention_decoder import AttentionDecoder
from attention_encoder import AttentionWithContext
from keras.layers import Dropout, Flatten, Conv1D, MaxPooling1D, AveragePooling1D, BatchNormalization
from keras.regularizers import l2
from keras.constraints import maxnorm

from nltk import FreqDist
import numpy as np
import sys, time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import Counter, deque
from predict_with_features import plot_model_performance, returnTrainTestSets

# from curve_plotter import plot_precision_recall

MODE = 'train'
output_mode = 'write'

EPOCHS = 500
dropout = 0.2
TIME_STEPS = 20
EMBEDDING_DIM = 128
BATCH_SIZE = 128
LAYER_NUM = 2
no_filters = 200
filter_length = 4
HIDDEN_DIM = no_filters*2
RNN = GRU
rnn_output_size = 32
folds = 10

class_labels = []


def write_words_to_file(orig_words, predictions):
	print("Writing to file ..")
	# print(sentences[:10])

	sentences = pickle.load(open('./pickle-dumps/sentences_test', 'rb'))

	X = [item for sublist in sentences for item in sublist]
	Y = [item for sublist in orig_words for item in sublist]

	filename = "./outputs/attention/multitask_context_out.txt"
	with open(filename, 'w', encoding='utf-8') as f:
		f.write("Words" + '\t\t\t' + 'Original Roots' + '\t\t' + "Predicted roots" + '\n')
		for a, b, c in zip(X, Y, predictions):
			f.write(str(a) + '\t\t' + str(b) + '\t\t\t' + str(c) + '\n')

	print("Success writing to file !")


def write_features_to_file(orig_features, pred_features, encoders):

	orig_features[:] = [ [np.where(r == 1)[0][0] for r in x] for x in orig_features]
	print(orig_features[0][:10])
	pred_features[:] = [x.tolist() for x in pred_features]
	
	for i in range(len(orig_features)):
		orig_features[i] = encoders[i].inverse_transform(orig_features[i])
		print(orig_features[i][:10])
		pred_features[i] = encoders[i].inverse_transform(pred_features[i])
		print(pred_features[i][:10])

	sentences = pickle.load(open('./pickle-dumps/sentences_test', 'rb'))
	words = [item for sublist in sentences for item in sublist]

	for i in range(len(orig_features)):
		filename = "./outputs/CNNRNN_globalPool_with_noise/feature"+str(i)+"context_out.txt"
		with open(filename, 'w', encoding='utf-8') as f:
			f.write("Word" + '\t\t' + 'Original feature' + '\t' + 'Predicted feature' + '\n')
			for a,b,c in zip(words, orig_features[i], pred_features[i]):
				f.write(str(a) + '\t\t' + str(b) + '\t\t\t' + str(c) + '\n')

	print("Success writing features to files !!")

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

def transform(encoding, data, vector_size=20):
    """
    :param encoding: encoding dict built by build_characters_encoding()
    :param data: list of strings
    :param vector_size: size of each encoded vector
    """
    transformed_data = np.zeros(shape=(len(data), vector_size))
    for i in range(len(data)):
        for j in range(min(len(data[i]), vector_size)):
            transformed_data[i][j] = encoding[data[i][j]]
    return transformed_data

def create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, n_phonetic_features, y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y6, n6,
				 hidden_size, num_layers):
	def smart_merge(vectors, **kwargs):
		return vectors[0] if len(vectors) == 1 else merge(vectors, **kwargs)

	# def generate(encoder_input):
	# 	encoder_input = transform()
	encoder_input = Input(shape=(X_max_len,), dtype='float32')
	decoder_input = Input(shape=(X_max_len,), dtype='float32')

	emb_layer = Embedding(X_vocab_len, EMBEDDING_DIM,
						  input_length=X_max_len,
						  mask_zero=True, name='Embedding_for_seq2seq')

	encoder = emb_layer(encoder_input)
	encoder, state_h = GRU(rnn_output_size, return_sequences=True, unroll=True, return_state=True)(encoder)
	encoder_last = encoder[:,-1,:]
	
	decoder = emb_layer(decoder_input)
	decoder = GRU(rnn_output_size, return_sequences=True, unroll=True)(decoder, initial_state=[state_h	])

	attention = dot([decoder, encoder], axes=[2,2])
	attention = Activation('softmax')(attention)

	print(attention)
	context = dot([attention, encoder], axes=[2,1])
	print(context)
	print(decoder)
	decoder_combined_context = concatenate([context, decoder])

	output = TimeDistributed(Dense(64, activation='tanh'))(decoder_combined_context)
	output = TimeDistributed(Dense(X_vocab_len, activation='softmax'))(output)

	all_inputs = [encoder_input, decoder_input]
	all_outputs = [output]

	model = Model(input=all_inputs, output=all_outputs)
	model.compile(optimizer='adam', loss='categorical_crossentropy',
				  metrics=['accuracy'])

	return model

sentences = pickle.load(open('./pickle-dumps/sentences_intra', 'rb'))
rootwords = pickle.load(open('./pickle-dumps/rootwords_intra', 'rb'))
features = pickle.load(open('./pickle-dumps/features_intra', 'rb'))

n_phonetics, X_train_phonetics, X_test_phonetics, X_val_phonetics, _, _, _ = returnTrainTestSets()

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

saved_weights = "./model_weights/attention.hdf5"


if MODE == 'train':
	print("Training model ..")
	plot_model(model, to_file="CNNRNN_with_both_pooling.png", show_shapes=True)
	y_sequences = process_data(y, X_max_len, y_word_to_ix)

	print("X len ======== ", len(X))
	train_val_cutoff = int(.75 * len(X))
	X_train, X_left1_tr, X_left2_tr, X_left3_tr, X_left4_tr, X_right1_tr, X_right2_tr, X_right3_tr, X_right4_tr = \
		[X[:train_val_cutoff], X_left1[:train_val_cutoff], X_left2[:train_val_cutoff], X_left3[:train_val_cutoff], X_left4[:train_val_cutoff],
			X_right1[:train_val_cutoff], X_right2[:train_val_cutoff], X_right3[:train_val_cutoff], X_right4[:train_val_cutoff]]
	X_val, X_left1_val, X_left2_val, X_left3_val, X_left4_val, X_right1_val, X_right2_val, X_right3_val, X_right4_val = \
		[X[train_val_cutoff:], X_left1[train_val_cutoff:], X_left2[train_val_cutoff:], X_left3[train_val_cutoff:], X_left4[train_val_cutoff:],
			X_right1[train_val_cutoff:], X_right2[train_val_cutoff:], X_right3[train_val_cutoff:], X_right4[train_val_cutoff:]]

	y_sequences_tr, y1_tr, y2_tr, y3_tr, y4_tr, y5_tr, y7_tr = \
			[y_sequences[:train_val_cutoff], y1[:train_val_cutoff], y2[:train_val_cutoff], y3[:train_val_cutoff], \
				y4[:train_val_cutoff], y5[:train_val_cutoff], y7[:train_val_cutoff]]	
	y_sequences_val, y1_val, y2_val, y3_val, y4_val, y5_val, y7_val = \
			[y_sequences[train_val_cutoff:], y1[train_val_cutoff:], y2[train_val_cutoff:], y3[train_val_cutoff:], \
				y4[train_val_cutoff:], y5[train_val_cutoff:], y7[train_val_cutoff:]]


	# y_tr, y_val = [y[:train_val_cutoff], y[train_val_cutoff:]]
	
	y_train = np.zeros_like(X_train)
	y_train[:, 1:] = X_train[:,:-1]
	y_train[:, 0] = 1

	y_val = np.zeros_like(X_val)
	y_val[:, 1:] = X_val[:,:-1]
	y_val[:, 0] = 1

	hist = model.fit([X_train, y_train],
					 [y_sequences_tr],
					 validation_data=([X_val, y_val],\
					 	[y_sequences_val]),
					 batch_size=BATCH_SIZE, epochs=EPOCHS,
					 callbacks=[EarlyStopping(patience=10),
								ModelCheckpoint('./model_weights/attention.hdf5', save_best_only=True,
												verbose=1)])

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

		to_be_padded = [X_test, y_test]

		X_test, y_test= [pad_sequences(i, maxlen=X_max_len, dtype='int32', padding='post') for i in to_be_padded]

		y_test_seq = process_data(y_test, X_max_len, y_word_to_ix)

		y_te = np.zeros_like(X_test)
		y_te[:, 1:] = X_test[:,:-1]
		y_te[:, 0] = 1

		model.load_weights(saved_weights)
		print(model.summary())
		# print(model.evaluate([X_test, y_te],
		# 					 [y_test_seq]))
		print(model.metrics_names)

		words = model.predict([X_test, y_te])

		predictions = np.argmax(words, axis=2)


		if output_mode == 'write':

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
