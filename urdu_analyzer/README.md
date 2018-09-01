# Hindi Morphological Analyzer

- [parse_data.py](https://github.com/Saurav0074/hindi-morph-analysis/blob/master/parse_data.py) is the first file you need to run. This extracts the original hindi words, their roots and features and creates a pickle dump for further usage.
- [rootwords_intra](https://github.com/Saurav0074/hindi-morph-analysis/blob/master/rootwords_intra), [sentences_intra](https://github.com/Saurav0074/hindi-morph-analysis/blob/master/sentences_intra) and [features_intra](https://github.com/Saurav0074/hindi-morph-analysis/blob/master/features_intra) are the dumped outputs from the combined corpus of Intra Chunk Development and Training set.
- [load_data.py](https://github.com/Saurav0074/hindi-morph-analysis/blob/master/load_data.py) contains three different function that make use of the previously dumped pickles to segregate out individual features and building vocabulary index. All the pre-processing is carried out here. For convenience, the functions are automatically called from the main program files explained below.

# Individual Learning

- Run [onlyFeatures.py](https://github.com/Saurav0074/hindi-morph-analysis/blob/master/onlyFeatures.py) to train only the features. The input in this case are the original hindi words while outputs are individual morphological tags. The file should be edited accordingly to select over the required features.
- Run [onlyRoots.py](https://github.com/Saurav0074/hindi-morph-analysis/blob/master/onlyRoots.py) to train a basic `sequence-to-sequence model` for learning the root words.

## Multi-task Learning

- Run [multiTask.py](https://github.com/Saurav0074/hindi-morph-analysis/blob/master/multiTask.py) for training a single input multi-output model that jointly learns the root words and the features. The loss contribution imparted to each output layer is equal, i.e., 1.0.
