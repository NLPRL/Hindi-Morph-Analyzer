import pickle
import random
import numpy as np 
import matplotlib.pyplot as plt 
from collections import Counter 

from sklearn.feature_extraction import DictVectorizer

# from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback


CUSTOM_SEED = 42
np.random.seed(CUSTOM_SEED)

mode = 'train'

# dataset preprocessing
def get_tag_names(flat_features):
	cnt = Counter(flat_features)
	tags = set(cnt.keys())
	print(tags)

	return tags 

def generate_tuples(sentences, features):
	mapping = []
	for sentence,tag in zip(sentences, features):
		list_of_tuples = []
		for i, j in zip(sentence, tag):
			l = [i,j]
			l = tuple(l)
			list_of_tuples.append(l)
		mapping.append(list_of_tuples)

	print(random.choice(mapping))
	print(len(mapping))

	return mapping

############# Feature Engineering #################
# def add_basic_features(sentence_terms, index):
# 	term = sentence_terms[index]
# 	return {
#         'nb_terms': len(sentence_terms),
#         'term': term,
#         'is_first': index == 0,
#         'is_last': index == len(sentence_terms) - 1,
#         'prefix-1': term[0],
#         'prefix-2': term[:2],
#         'prefix-3': term[:3],
#         'suffix-1': term[-1],
#         'suffix-2': term[-2:],
#         'suffix-3': term[-3:],
#         'prev_word': '' if index == 0 else sentence_terms[index - 1],
#         'next_word': '' if index == len(sentence_terms) - 1 else sentence_terms[index + 1]
#     }

def encode_all_features(X_train, X_test, X_val, train=False, val=False, test=False):
	transformed_feature_to_be_returned = []
	total_features_to_be_encoded = len(X_train[0][3:])

	if train == True:
		encoders = {}
		for i in range(total_features_to_be_encoded):
			print("Encoding and transforming training set feature: ", i)
			encoders[i] = LabelEncoder()
			encoders[i].fit(list(zip(*X_train))[i+3] + list(zip(*X_test))[i+3] + list(zip(*X_val))[i+3])
			transformed_feature_to_be_returned.append(encoders[i].transform(list(zip(*X_train))[i+3]))

		X_train = np.asarray(X_train)
		
		for i in range(total_features_to_be_encoded):
			X_train[:,i+3] = transformed_feature_to_be_returned[i]

		X_train = X_train.astype(np.float) # LabelEncoder returns strings
		X_train = X_train.tolist()
		pickle.dump(encoders, open('./pickle-dumps/phonetic_feature_encoders', 'wb'))
		
		return X_train

	elif val == True:
		encoders = pickle.load(open('./pickle-dumps/phonetic_feature_encoders', 'rb'))

		for i in range(total_features_to_be_encoded):
			print("Encoding and transforming validation set feature: ", i)
			transformed_feature_to_be_returned.append(encoders[i].transform(list(zip(*X_val))[i+3]))
		
		X_val = np.asarray(X_val)
		
		for i in range(total_features_to_be_encoded):
			X_val[:,i+3] = transformed_feature_to_be_returned[i]

		X_val = X_val.astype(np.float)
		X_val = X_val.tolist()		
		return X_val

	elif test == True:
		encoders = pickle.load(open('./pickle-dumps/phonetic_feature_encoders', 'rb'))

		for i in range(total_features_to_be_encoded):
			print("Encoding and transforming test set feature: ", i)
			transformed_feature_to_be_returned.append(encoders[i].transform(list(zip(*X_test))[i+3]))
		
		X_test = np.asarray(X_test)
		for i in range(total_features_to_be_encoded):
			X_test[:,i+3] = transformed_feature_to_be_returned[i]

		X_test = X_test.astype(np.float)
		X_test = X_test.tolist()		
		return X_test
	
def process_labels(y_train, y_val, y_test):
	label_encoder = LabelEncoder()
	label_encoder.fit(y_train + y_test + y_val)

	# Encode class values as integers
	y_train = label_encoder.transform(y_train)
	y_test = label_encoder.transform(y_test)
	y_val = label_encoder.transform(y_val)

	cnt = Counter(y_test) # list of all labels assigned after encoding them
	labels = list(cnt.keys()) # to be dumped for use in PR curve plotter

	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	y_val = np_utils.to_categorical(y_val)

	return y_train, y_val, y_test

def add_basic_features(sentence_terms, index):
	term = sentence_terms[index]
	length = len(sentence_terms)
	is_first = index==0
	is_last = index == len(sentence_terms)-1
	prefix1 = term[0]
	prefix2 = term[:2]
	prefix3 = term[:3]
	suffix1 = term[-1]
	suffix2 = term[-2:]
	suffix3 = term[-3:]
	prev_word = '' if index == 0 else sentence_terms[index-1]
	next_word = '' if index == len(sentence_terms)-1 else sentence_terms[index+1]

	return length, int(is_first), int(is_last), term, prefix1, prefix2, prefix3, suffix1, suffix2, suffix3, prev_word, next_word

def untag(tagged_sentence):
    """ 
    Remove the tag for each tagged term. 
 
    :param tagged_sentence: a POS tagged sentence
    :type tagged_sentence: list
    :return: a list of tags
    :rtype: list of strings
    """
    return [w for w, _ in tagged_sentence]
 
def transform_to_dataset(tagged_sentences):
	X, y = [], []

	for pos_tags in tagged_sentences:
 		for index, (term, class_) in enumerate(pos_tags):
 			# Add basic NLP features for each sentence term
 			X.append(add_basic_features(untag(pos_tags), index))
 			y.append(class_)

	return X, y

# Model building
def build_model(input_dim, hidden_neurons, output_dim):
	"""
    Construct, compile and return a Keras model which will be used to fit/predict
    """
	model = Sequential([
    	Dense(hidden_neurons, input_dim=input_dim),
        Activation('relu'),
        Dropout(0.2),
        Dense(hidden_neurons),
        Activation('relu'),
        Dropout(0.2),
        Dense(output_dim, activation='softmax')
    ])

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

################ Plot model loss and accuracy through epochs ###########
def plot_model_performance(train_loss, train_acc, train_val_loss, train_val_acc):
    """ Plot model loss and accuracy through epochs. """
 
    green = '#72C29B'
    orange = '#FFA577'
 
    with plt.xkcd():
        # plot model loss
        fig, ax1 = plt.subplots()
        ax1.plot(range(1, len(train_loss) + 1), train_loss, green, linewidth=5,
                 label='training')
        ax1.plot(range(1, len(train_val_loss) + 1), train_val_loss, orange,
                 linewidth=5, label='validation')
        ax1.set_xlabel('# epoch')
        ax1.set_ylabel('loss')
        ax1.tick_params('y')
        ax1.legend(loc='upper right', shadow=False)
        # plot model accuracy
        fig, ax2 = plt.subplots()
        ax2.plot(range(1, len(train_acc) + 1), train_acc, green, linewidth=5,
                 label='training')
        ax2.plot(range(1, len(train_val_acc) + 1), train_val_acc, orange,
                 linewidth=5, label='validation')
        ax2.set_xlabel('# epoch')
        ax2.set_ylabel('accuracy')
        ax2.tick_params('y')
        ax2.legend(loc='lower right', shadow=False)
    plt.show()
'''
def write_output_to_file(testing_sentences, predictions, label_encoder):
	#print(testing_sentences[5])
	testing_sentences = [item for sublist in testing_sentences for item in sublist]
	#print(testing_sentences[5])
	#print(len(testing_sentences))
	#print(len(predictions))
	predictions = label_encoder.inverse_transform(predictions)

	words = []
	orig_labels = []

	for i in testing_sentences:
		words.append(i[0])
		orig_labels.append(i[1])

	print(len(words))

	filename = "MLP"+"out.txt"
	with open(filename, 'w', encoding='utf-8') as f:
		f.write("Word" + '\t\t' + 'Original POS' + '\t' + 'Predicted POS' + '\n')
		for a,b,c in zip(words, orig_labels, predictions):
			f.write(str(a) + '\t\t' + str(b) + '\t\t\t' + str(c) + '\n')

	print("Success writing features to files !!")

	return orig_labels
'''

if __name__ == '__main__':
	sentences = pickle.load(open('./pickle-dumps/sentences_intra', 'rb'))
	y1 = pickle.load(open('./pickle-dumps/y2_sentencewise', 'rb'))

	test_sent = pickle.load(open('./pickle-dumps/sentences_test', 'rb'))
	y1_test = pickle.load(open('./pickle-dumps/y1_test', 'rb'))
												
	# generate a mapping of word to their tags for the sake of the universe
	sentences = generate_tuples(sentences, y1)
	testing_sentences = generate_tuples(test_sent, y1_test)

	flat_features = [item for sublist in y1 for item in sublist]
	flat_tests = [item for sublist in y1_test for item in sublist]

	flat_features = flat_features + flat_tests
	tags = get_tag_names(flat_features) # get the names of labels

	train_test_cutoff = int(.80 * len(sentences)) 
	training_sentences = sentences[:train_test_cutoff]
	validation_sentences = sentences[train_test_cutoff: ]

	X_train, y_train = transform_to_dataset(training_sentences)
	X_test, y_test = transform_to_dataset(testing_sentences)
	X_val, y_val = transform_to_dataset(validation_sentences)

	X_train = encode_all_features(X_train, X_test, X_val, train=True, val=False, test=False)
	X_test = encode_all_features(X_train, X_test, X_val, train=False, val=False, test=True)
	X_val = encode_all_features(X_train, X_test, X_val, train=False, val=True, test=False)

	y_train, y_val, y_test = process_labels(y_train, y_val, y_test)
	
	scaler = MinMaxScaler()
	scaler.fit(X_train+X_val)
	X_train = scaler.transform(X_train)
	X_val = scaler.transform(X_val)
	print(X_train[:10])
	print(X_val[:10])

	model = build_model(len(X_train[1]), 56, len(y_train[1]))

	if mode == 'train':
		hist = model.fit(X_train, y_train, validation_data= (X_val, y_val), 
			batch_size=60, epochs=100,  
			callbacks=[EarlyStopping(patience=10),
			ModelCheckpoint('simplefeatures_MLP.hdf5', save_best_only=True,
				verbose=1)])
		print(hist.history.keys())
		print(hist)
		plot_model_performance(
			train_loss=hist.history.get('loss', []),
		    train_acc=hist.history.get('acc', []),
		    train_val_loss=hist.history.get('val_loss', []),
		    train_val_acc=hist.history.get('val_acc', [])
		)

	# else:
	# 	saved_weights = './model_weights/simplefeatures_MLP.hdf5'
	# 	model.load_weights(saved_weights)

	# 	predictions = model.predict(X_test)
	# 	#predictions = np.argmax(predictions, axis=1)
		
	# 	# undoing the one-hot encoding and converting list of lists to a list
	# 	orig_labels = [item for sublist in [list(np.where(r == 1)[0]) 
	# 	for r in y_test] for item in sublist]

	# 	print(orig_labels[:10])
	# 	print(predictions[:10])

	# 	print(len(orig_labels))
	# 	print(len(predictions))

	# 	pickle.dump(labels, open('./pickle-dumps/labels_MLP','wb'))
	# 	pickle.dump(predictions, open('./pickle-dumps/predictions_MLP','wb'))
	# 	pickle.dump(orig_labels, open('./pickle-dumps/originals_MLP','wb'))

	# 	saved_weights = 'simplefeatures_MLP.hdf5'
	# 	model.load_weights(saved_weights)

	# 	words = model.predict(X_test)
	# 	predictions = np.argmax(words, axis=1)
	# 	print(predictions)

	# 	write_output_to_file(testing_sentences, predictions, label_encoder)
