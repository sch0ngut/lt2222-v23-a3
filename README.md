# LT2222 V23 Assignment 3 - Jonas Mack

## Part 1
From the top-level diretcory of this repository, the feature table with a test size of 20% and 1000 features can be created as follows:
```
python a3_features.py \
	--inputdir /scratch/lt2222-v23/enron_sample \
	--outputfile data.csv \
	--dims 1000 \
	--test 20
```
where
- `inputdir`: the path to the raw data
- `outputfile`: where to write the processed data to
- `dims`: the number of features
- `test` the size of the test set in %

## Part 2 & 3
To train the model, from the top-level directory of this repository simply run
```
python a3_model.py \
	--featurefile data.csv \
	--epochs 100 \
	--n_hidden 10 \
	--nonlinearity none
```
where
- `featurefile`: the path to `outputfile` from the previous step
- `epochs`: the number of epochs
- `n_hidden`: the number of nodes in the hidden layer
- `nonlinearity`: which function to use as non-linear activation function for the hidden layer. Must be one of {`tanh`, `relu`, `none`}

The command above trains the model for the given number of epochs and prints the loss for each epoch to the console. After the training completed, the overall accuracy together with a confusion matrix is logged to the console

Below an overview of the performance (accruacy in %) with different hyperparameters, each after 100 epochs. Keep in mind that the results slightly differ from run to run due to random shuffling of the training data (I chose not to use a seed).

| n_features | n_hidden | non_linearity | Accuracy in % |
|---|---|---|---|
| 50 | 50 | none | 43% |
| 50 | 50 | relu | 43% |
| 50 | 50 | tanh | 44% |
| 50 | 500 | none | 42% |
| 50 | 500 | relu | 43% |
| 50 | 500 | tanh | 44% |
| 500 | 50 | none | 72% |
| 500 | 50 | relu | 71% |
| 500 | 50 | tanh | 73% |
| 500 | 500 | none | 73% |
| 500 | 500 | relu | 74% |
| 500 | 500 | tanh | 72% |

In summary:
- The main factor for quality improvement is the number of used features in the data
- The activation function seems to have a low impact
- The number of hidden units seems to have a low impact as well, though properly has a threshold in the sense that if the numer is lower than that threshold, performance will drop.

## Part 4

The use of the Enron Corpus for machine learning and NLP tasks raises ethical questions around data privacy and informed consent. While the emails were obtained through a court subpoena and were therefore part of the public record, the Enron Corporation employees did not explicitly consent to their use in research. Moreover, the emails were originally used as evidence in financial litigation and criminal charges, which adds another layer of complexity to their use for research purposes.

In addition to these concerns around consent and privacy, the Enron Corpus also presents challenges for data bias and fairness. The emails were written by a relatively homogenous group of employees at a single company, and may not be representative of a broader population. This could lead to issues with bias and fairness in the development and deployment of machine learning models that are trained on the Enron Corpus. Researchers must be mindful of these potential biases and take steps to mitigate them through careful sampling and evaluation of their models.

Overall, the use of the Enron Corpus for machine learning and NLP research requires careful consideration of the ethical implications of using data that was not explicitly consented to for research purposes, and the potential biases and fairness issues that can arise from working with a relatively homogenous dataset. Researchers must take steps to ensure that their work is conducted in an ethical and responsible manner, and that the models they develop are fair and representative of the broader population.
