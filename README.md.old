## Implementation Project:##
# "QUality Estimation from ScraTCH (QUETCH)" #

Julia Kreutzer (kreutzer@cl.uni-heidelberg.de)

Institut für Computerlinguistik, Universität Heidelberg

Winter Term 2014/2015, Seminar "Deep Learning" (Prof. Dr. Stefan Riezler)

--------------------
Goal & Motivation
--------------------
The aim of this project is to build a Deep Learning architecture to tackle the high-level, linguistically complex task of Quality Estimation of machine-translated texts. Following Collobert et al.'s [1] "from-scratch"-philosophy the implemented classifier works without any linguistic or language-specific features.
Since the experiments of Collobert et al. prove that their Neural Network architecture performs well on NLP tasks like POS-tagging, chunking, NER and SRL, QUETCH will investigate whether their approach can be successfully applied to a more complex bilingual task in a similar manner. This will be evaluated on the ACL 2014 Ninth Workshop
on Statistical Machine Translation (WMT14) Shared-Task for Quality Estimation [2].

--------------------
 Prerequisites
--------------------
- Python version 2.7
- Additional Python libraries must be installed: Theano [3], Numpy [4], Gensim [5], Nltk [6] (including the tokenizers/punkt/english.pickle tokenizer, download via nltk.download()), Progressbar [8]

--------------------
 Architecture of the Neural Network
--------------------
- **Input**: sequence of words within a fixed-size window from target and source text (*targetWindowSize* and *sourceWindowSize*)
    - for the word-based prediction the window is centered at the position of the target word, both in target and in source sentence. Since alignment information is missing, the position is assumed to be the same. 
    - for the sentence-based prediction the windows starts at the beginning of target and source sentence
    - the token "PADDING" is used to fill up windows at sentence borders
    - words are mapped to indices via a gensim dictionary (corpus is training and test data)
- **Lookup-Table-Layer**: maps the input words to a *d_wrd* dimensional vector of floats
    - randomly initialized as described in [1] according to fan-in
- **Linear Layer(s)**: standard linear layer with tanh activation function
    - consists of *n_hidden* hidden units
    - in the experiments either one or two layers were stacked between Lookup-Table-Layer and the Output-Layer
    - Note: unlike in [1] tanh was used instead of hardtanh to guarantee differentiability
- **Output**: conditional tag probability
    - softmax operation over all possible tags
    - set of possible tasks is task-dependent
    
--------------------
 Data & Tasks
--------------------
In order to make results comparable, QUETCH is designed to classify WMT14 data (available at [7]).
The data can be found in the `WMT14` directory.

QUETCH performs the following Quality Estimation tasks (see [7] for details):

- Task 1: Sentence-based prediction of post-editing effort (quality labels: 1,2,3)
- Task 2: Word-based prediction of translation errors (binary classification, level 1 classification, and multi-class classification)

--------------------
 Experiments
--------------------
This implementation is designed to solve Task 1.1 and Task 2 of the WMT14 Quality Estimation Shared Task. Training and testing is triggered executing `src/QUETCH.py` (Note that all python scripts must be called from inside `src`).

- Hyper-parameters for the Neural Network: 
   
     | Parameter | Default | Description |
     |:---------------------|:-----------------------|:--------------------------------|
     | *SourceWindowSize* | Task 1.1: 51, Task 2: 7 | Word window size for feature extraction from source text, has to be odd |
     | *TargetWindowSize* | Task 1.1: 51, Task 2: 5 | Word window size for feature extraction from target text, has to be odd |
     | *WordEmbeddingDimensionality* | 50 | Dimensionality of feature space trained by Lookup-Table-Layer | 
     | *HiddenUnits* | 300 | Number of hidden units in Linear Layer(s) | 
     | *BaselineFeatures* | False | Only applies to Task 1.1: if True, use WMT14-provided baseline features for the initialization of the Lookup-Table-Layer |

     Note that the number of hidden layers is hard-coded in `NN.py`, default=1 

- Training parameters: 
   
     | Parameter | Default | Description |
     |:---------------------|:-----------------------|:--------------------------------|
     | *LearningRate* | 0.001 | Learning rate for the Stochastic Gradient Ascent |
     | *MaxIt* | 1000 | Convergence criterion: maximum number of training iterations | 
     | *threshold* | 0.0001 | Convergence criterion: if reduction of loss below threshold break up training |
     
     Note that batch processing is implemented, but not used here. *batch_size* is therefore set to 1 (hard-coded)

- Example for task 1.1:
     `python QUETCH.py 1 en-es -sws 7 -tws 9 -d 50 -hu 200`
- Example for task 2:
     `python QUETCH.py 2 de-en -sws 9 -tws 9 -d 50 -l 0.05`

--------------------
 Evaluation
--------------------
The evaluation follows the WMT14 Shared Task definition [2]. Task 1.1: Evaluation metrics are Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). Task 2: Evaluation metrics are label-specific and overall F1 scores (label "OK" is ignored).
The prediction output is formatted according to the WMT14 Shared Task description.

- Task 1.1: MAE and RMSE are directly printed after training. 
- Task 2: use the WMT14 evaluation script (available at http://www.statmt.org/wmt14/quality-estimation-task.html):
  `python evaluate_task2_WMT2014.py [gold standard annotation file] [a model's prediction (saved in results/ after training)]`
- To investigate the predictions of a trained model for task 2 call `EvalModel.py` with the following options:

     | Parameter |  Description |
     |:---------------------|:--------------------------------|
     | *Task* | 1 or 2 |
     | *SubTask* | for task 2: which subtask to evaluate ("bin", "l1", "multi") |
     | *LanguagePair* | language of source and target |
     | *ParameterFile* | File which contains trained model parameters |
     | *SourceWindowSize* | Word window size for feature extraction from source text |
     | *TargetWindowSize* | Word window size for feature extraction from target text |
     | *BaselineFeatures* | Whether WMT14 baseline features were used for training or not |
     | *OutputFile* | Where to store the model's predictions |

- Example for task 1.1:
     `python EvalModel.py 1 one de-en ../parameters/2015-03-14--13:54:50.278627.params myresults1 -sws 51 -tws 51` -> Predictions are stored in `myresults1`, Evaluation results are printed to stdout
- Example for task 2: 
     `python EvalModel.py 2 bin de-en ../parameters/2015-03-14--19\:30\:32.801678.params myresults2 -sws 5 -tws 7` -> Predictions are stored in `myresults`
     `python evaluate_task2_WMT2014.py ../WMT14-data/task2_de-en_test/DE_EN.tgt_ann.test myresults2` -> Evaluation results are printed to stdout
     
--------------------
 Results
--------------------
Some initial experiments with varying parameter settings were performed to get an idea how the implemented architecture performs on this task. The results and the listing of parameter files can be found in `results/results.xls`.

Some preliminary findings are summarized with the following Q&A:

- Was QUETCH able to beat the WMT14 winners and baselines?
  > Partly.
    - Yes for task 2 en-de and de-en.
    - No for the remaining tasks.
    
- Does the QUETCH architecture perform better on word-based or sentence-based tasks?
  > Clearly on word-based tasks.
  
- Does QUETCH show language-dependent strong or weak points?
  > The QUETCH architecture seems to perform better on the language pair English-German than on English-Spanish. This might be because English and German are generally more similar. Still, not enough experiments have been run to comprehensively judge this assumption. 
   
- Which parameters settings worked best?
  > The optimal parameter settings vary from language pair to language pair and from task to task. An universally optimal setting was not found (yet). The following observations were made during testing:
     - *WordEmbeddingDimensionality* = 200 (or >200) has proven to be a good choice for task 2 en-de and de-en settings
     - An increased context size appears to have a positive effect for task 2 prediction with more than two possible labels (at least for de-en).
     - For the task 2 es-en setting it was helpful to increase *HiddenUnits*.
     - Interestingly, the latter does not hold for task 1 de-en, were a reduction of *HiddenUnits* yielded better results.
     - Integrating the given WMT14 baseline features did only improve results in the task 1 en-es setting.
     
- Does the integration of hand-engineered features improve the systems's performance (as in [1])? 
  > No, suprsingly only in the task 1 en-es setting. Since this is against the expectation, it might be worth to investigate other features or try another way to integrate them into the NN architecture.
     
- Does the Lookup-Table-Layer learn "senseful" word embeddings?
  > No, at least not in a way that one would intuitively expect. `src/ParameterInspection.py` allows an insight into the learned word embeddings. For exmaple, on of the decently performing models for the task 2 de-en setting, produced the following nearest neighbours for the target word "classical": "Spiegel", "gets", "Verletzten", "denen", "Südkalifornien", "improvement", "ab", "register", "der", "Zähne". It would be impressing if those embeddings would actually allow word translations due to the bilingual characteristic of the embedding vector space.
  
--------------------
 Suggestions for Future Work 
--------------------
- Test more hyper-parameter settings (systematically).
- Integrate baseline features on word level for task 2 to allow a judgement whethere those given features improve the results. Also, un-supervised training of a language-model like in [1] should be considered.
- The WMT14 tasks are evaluated by weighted F1: correct translations (label: "OK") do not have any weight. This is why it might be sensible to adapt the classifier's loss function, such that misclassification of translation errors is punished harder than correct translations. This aspect is emphasized by the observation that models with a high accuracy on test data do not necessarily yield a high weighted F1 score.
- Since the assumption about the position of the target word in the source sentence is very naive, one could integrate alignment information or statistics to figure out the correct position and hence can more precisely set the position of the source window.
- Joint training like in [1] for all subtasks of task 2. Maybe this allows to integrate some logic, that a word cannot receive a "OK" tag on binary level and a "Accuracy" tag on level-1.
- Task 1 requires annotation on sentence level but the architecture was designed for predictions on word level. Some initial experiments have shown that the architecture optimized for word-based prediction does not perform as well on sentence-based predictions. Try to find a better setting for sentence-based predictions.

Please contact me if you wish to obtain further information or have ideas for further work.

-----------
Sources
------------
- [1]: Collobert, R., Weston, J., Bottou, L., Karlen, M., Kavukcuoglu, K., & Kuksa, P. (2011). Natural language processing (almost) from scratch. The Journal of Machine Learning Research, 12, 2493-2537. 
- [2]: Bojar, O., Buck, C., Federmann, C., Haddow, B., Koehn, P., Leveling, J., ... & Tamchyna, A. (2014, June). Findings of the 2014 workshop on statistical machine translation. In Proceedings of the Ninth Workshop on Statistical Machine Translation (pp. 12-58). Association for Computational Linguistics Baltimore, MD, USA.
- [3]: Theano Tutorial: http://deeplearning.net/tutorial/
- [4]: http://www.numpy.org/
- [5]: https://radimrehurek.com/gensim/
- [6]: http://www.nltk.org/
- [7]: http://www.statmt.org/wmt14/quality-estimation-task.html
- [8]: https://code.google.com/p/python-progressbar/

----------
License
----------
Theano:
Copyright (c) 2008–2013, Theano Development Team All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of Theano nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ‘’AS IS’’ AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.