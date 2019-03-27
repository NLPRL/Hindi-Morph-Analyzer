# Hindi-morph-analyzer [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

This repository hosts the codes necessary to reproduce the results of our approach towards Morphological analysis of Hindi words.

- [data_preprocessing](https://github.com/Saurav0074/Hindi-morph-analyzer/tree/master/data_preprocessing) contains the files necessary to parse and load the data. 
  - The file [load_data.py](https://github.com/Saurav0074/Hindi-morph-analyzer/blob/master/data_preprocessing/load_data.py) contains functions callabe at the time of training and testing the models.
  
- [models](https://github.com/Saurav0074/Hindi-morph-analyzer/tree/master/models) hosts the various models implemented for predicting the rootword as well as the tag features.

- [result_visualization](https://github.com/Saurav0074/Hindi-morph-analyzer/tree/master/result_visualization) contains the files necessary for plotting the performance metric curves.

- [curve_outputs](https://github.com/Saurav0074/Hindi-morph-analyzer/tree/master/curve_outputs) contains the graphical outputs of each model in terms of average precision-recall scores for each classes.

- [data_and_train_info](https://github.com/Saurav0074/Hindi-morph-analyzer/tree/master/data_and_train_info) hosts the stats about the Hindi TreeBank dataset that we extracted as per our needs along with the information regarding training of the models and their performance.
