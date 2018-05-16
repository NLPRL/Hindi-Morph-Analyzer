import pickle
import random
import numpy as np 
import matplotlib.pyplot as plt 
from collections import Counter 

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback


CUSTOM_SEED = 42
np.random.seed(CUSTOM_SEED)

mode = 'test'

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
def add_basic_features(sentence_terms, index):
	term = sentence_terms[index]
	return {
        'nb_terms': len(sentence_terms),
        'term': term,
        'is_first': index == 0,
        'is_last': index == len(sentence_terms) - 1,
        'prefix-1': term[0],
        'prefix-2': term[:2],
        'prefix-3': term[:3],
        'suffix-1': term[-1],
        'suffix-2': term[-2:],
        'suffix-3': term[-3:],
        'prev_word': '' if index == 0 else sentence_terms[index - 1],
        'next_word': '' if index == len(sentence_terms) - 1 else sentence_terms[index + 1]
    }

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

sentences = pickle.load(open('./pickle-dumps/sentences_intra', 'rb'))
y1 = pickle.load(open('./pickle-dumps/y1_sentencewise', 'rb'))

test_sent = pickle.load(open('./pickle-dumps/sentences_test', 'rb'))
y1_test = pickle.load(open('./pickle-dumps/y1_test', 'rb'))
											
print(len(y1))
print(len(sentences))

# generate a mapping of word to their tags for ease of study
sentences = generate_tuples(sentences, y1)
testing_sentences = generate_tuples(test_sent, y1_test)

# remove some part
to_keep = int(0.05 * len(sentences)) 
sentences = sentences[:to_keep]
y1 = y1[:to_keep]

testing_sentences = testing_sentences[:to_keep]
y1_test = y1_test[:to_keep]

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

#print(X_train[:5])
#print(len(X_train[1]))


# Fit the DictVectorizer with our set of features
dict_vectorizer = DictVectorizer(sparse=False)
dict_vectorizer.fit(X_train + X_test + X_val)

# convert dict features to vectors
X_train = dict_vectorizer.transform(X_train)
X_test = dict_vectorizer.transform(X_test)
X_val = dict_vectorizer.transform(X_val)

print(len(X_train))
print(len(X_train[2]))
print(X_train[:10])

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

model = build_model(len(X_train[1]), 512, len(y_train[1]))

if mode == 'train':
	hist = model.fit(X_train, y_train, validation_data= (X_val, y_val), 
		batch_size=256, epochs=10,  
		callbacks=[EarlyStopping(patience=5),
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

else:
	saved_weights = './model_weights/simplefeatures_MLP.hdf5'
	model.load_weights(saved_weights)

	predictions = model.predict(X_test)
	#predictions = np.argmax(predictions, axis=1)
	
	# undoing the one-hot encoding and converting list of lists to a list
	orig_labels = [item for sublist in [list(np.where(r == 1)[0]) 
	for r in y_test] for item in sublist]

	print(orig_labels[:10])
	print(predictions[:10])

	print(len(orig_labels))
	print(len(predictions))

	pickle.dump(labels, open('./pickle-dumps/labels_MLP','wb'))
	pickle.dump(predictions, open('./pickle-dumps/predictions_MLP','wb'))
	pickle.dump(orig_labels, open('./pickle-dumps/originals_MLP','wb'))

	saved_weights = 'simplefeatures_MLP.hdf5'
	model.load_weights(saved_weights)

	words = model.predict(X_test)
	predictions = np.argmax(words, axis=1)
	print(predictions)

	write_output_to_file(testing_sentences, predictions, label_encoder)

'''
'''
model_params = {
    'build_fn': build_model,
    'input_dim': len(X_train[1]),
    'hidden_neurons': 512,
    'output_dim': len(y_train[1]),
    'epochs': 7,
    'batch_size': 256,
    'verbose': 1,
    'validation_data': (X_val, y_val),
    'shuffle': True
}

clf = KerasClassifier(**model_params)

if mode == 'train':
	hist = clf.fit(X_train, y_train)

	clf.model.save('simplefeatures_MLP.h5')
	plot_model_performance(
		train_loss=hist.history.get('loss', []),
	    train_acc=hist.history.get('acc', []),
	    train_val_loss=hist.history.get('val_loss', []),
	    train_val_acc=hist.history.get('val_acc', [])
	)

else:
	model = load_model('simplefeatures_MLP.h5')
	score = model.score(X_test, y_test)
	print(score)
'''