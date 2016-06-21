Implementation of the Neural Network model for word-level quality estimation, described in the [WMT15 paper] (http://www.cl.uni-heidelberg.de/~riezler/publications/papers/WMT2015.pdf) and my [thesis] (http://www.cl.uni-heidelberg.de/~kreutzer/material/KreutzerJulia_MA_16.pdf). Based on Theano and gensim. Designed for WMT data.

0. Prepare
- Install theano and gensim.
- Create a `parameters` directory. The models will be stored here.
- Create a `dict` directory, mappings from words to indices in the lookup table are stored here.
- Create a `results` directory, with subdirectories for each task (`task2`, `task1.1`). Final predictions will be stored here.
- Download the WMT data and create directories for it (`WMT14-data` and `ẀMT15-data`). Proceed with pre-processing the data to produce the format described in the following.

1. WMT data
The WMT data is expected to be stored in `WMT14-data` and `ẀMT15-data` directories. The paths are specified in the `Task.py` code. They require pre-processing (aligning, tokenization (for WMT14), lowercasing) which is not provided in this code. In the end they should have the following format:

WMT15:
- training source data, lowercased:
`WMT15-data/task2_en-es_train_comb/train.source.lc.comb`:
`0       0       we      *`, i.e. the sentence id, the word id, the source word and a placeholder
- training target data, combined with features, lowercased:
`WMT15-data/task2_en-es_train_comb/train.target.lc.comb.feat`: 
`0       0       sólo    OK      6.0     5.0     1.2     sólo    _start_ utilizamos      only    we      use     0       0       1       0       0`, i.e. sentence id, word id, target word, word-level label, and features (here: WMT15 baseline features). The use of the features is optional and not required for the QUETCH model.
- source to target alignments: 
`WMT15-data/task2_en-es_train_comb/train.align`: 
`0       1-0 2-1 3-2 4-3 5-4`, i.e. the sentence id separated with a tab from the source-target alignment indices.
If not truecased data is expected in the same paths, with missing `.lc` respectively. Dev and test data should be named and formatted analogously, with `dev` and `test` prefixes instead of `train`. If no features are used, leave out the `.comb` suffix.

WMT14 task 2:
- en-es training source data, tokenized, lowercased:
`WMT14-data/task2_en-es_train_comb/EN_ES.source.train.tok.lc.comb`:
`0.1     0       norway  *`, i.e. the sentence id (here also dependent on translation system), the word id, the source word and a placeholder
- training target data, lowercased:
`EN_ES.tgt_ann.train.lc.comb`:
`0.1     0       rakfisk OK      OK      OK`, i.e. sentence id, word id, target word, word-level label (binary), l1 word label, fine-grained word label. 
- source to target alignments:
`WMT15-data/task2_en-es_train_comb/EN_ES.train.align`:
`0.1     2-0 1-1 0-2 3-3 4-4 4-5 5-6 6-7 10-8 8-9 7-10 9-11 9-12 11-13`, i.e. the sentence id separated with a tab from the source-target alignment indices.
Note that the WMT14 requires tokenization as an additional pre-processing step.

2. Training a model
Run `python QUETCH.py 15 2 en-es`, where 15 indicates the year of the WMT task, 2 indicates the task (word-level QE) and en-es is the language pair (English to Spanish). The code also covers sentence-level predictions (task 1.1), but they are not recommended to use, since the approach is very naive. 

More model parameters:
- `-sws`: source window size
- `-tws`: target window size
- `-d`: dimensionality of word embeddings
- `-hu`: number of hidden units
- `-b`: use baseline features
- `-l`: learning rate
- `-i`: maximum number of epochs
- `-t`: threshold as stopping criterion for learning
- `-p`: use pre-trained word embeddings (trained with gensim word2vec)
- `-f`: feature indices for WMT15 task 2: index is position in combined file format. Note that this makes only sense for categorical features.
- `-a`: whether alignments are provided or not
- `-w`: weight of BAD instances
- `-r`: initial learning rate if learning rate is not constant
- `-c`: use true case data
- `-notest`: if the word embeddings of the trained model also cover the test words
- `-g`: activation function (tanh, relu, sigmoid)
- `-s`: shuffle the data before each epoch
- `-l1`: regularization constant for l1 regularizer
- `-l2`: regularization constant for l2 regularizer

The model is saved each time the score on the dev set improved. It is stored in the `parameters` directory.

3. Test a model
Run `python EvalModel.py 15 2 en-es ../parameters/mymodel.params ../results/mymodel.results -dict ../dicts/mymodel.dict` for testing a model stored in `../parameters/mymodel.params` on test data and writing the predictions to `../results/mymodel.results`. The dictionary was produced during training and is necessary to map words to vectors. Source and target window size need to be the same as during training (same flags).


