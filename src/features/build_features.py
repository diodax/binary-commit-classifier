# -*- coding: utf-8 -*-
import click
import logging
import math
import pandas as pd
import copy
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from pandas import DataFrame
from sklearn.model_selection import train_test_split

# noinspection PyUnresolvedReferences
from binary_classifier import BinaryClassifier

nltk.download('stopwords')


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Starting point of the project. Receives the location of the dataset as a
    command-line Path argument.
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final TD-IDF scores set from processed data')

    # Get the data/processed/git-refactoring-commits.csv file
    dataset = pd.read_csv(input_filepath, usecols=['Message', 'CommitSHA', 'IsRefactoring'])
    dataset = dataset.dropna()
    # dataset = dataset.sample(frac=.50)
    dataset = dataset.sample(n=10000)
    logger.info('Loaded data file ' + input_filepath + ' with ' + str(len(dataset)) + ' rows')

    # Generate new DataFrame with the TF-IDF scores
    logger.info('Saving TF-IDF scores in a new .csv file...')
    x1 = dataset['Message'].values

    vec = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words("english")).fit(x1)

    x = DataFrame(vec.transform(x1).todense(), columns=vec.get_feature_names())
    y = dataset.iloc[:, -1].values

    # Split into training and test sets
    logger.info('Split into training and test sets...')
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # Class with the libraries binary implementations
    classifier = BinaryClassifier(x_train, y_train, x_test, y_test)

    a, f = classifier.apply_gaussian_nb()
    print('Gaussian NB Approach:')
    print('Acc.: {0}, F1 Score: {1}'.format(a, f))
    a, f = classifier.apply_logistic_regression()
    print('Logistic Regression Approach:')
    print('Acc.: {0}, F1 Score: {1}'.format(a, f))
    a, f = classifier.apply_random_forest()
    print('Random Forest Approach:')
    print('Acc.: {0}, F1 Score: {1}'.format(a, f))
    a, f = classifier.apply_decision_tree()
    print('Decision Tree Classifier Approach:')
    print('Acc.: {0}, F1 Score: {1}'.format(a, f))
    a, f = classifier.apply_svm()
    print('SVM Approach:')
    print('Acc.: {0}, F1 Score: {1}'.format(a, f))
    a, f = classifier.apply_knn()
    print('kNN Approach:')
    print('Acc.: {0}, F1 Score: {1}'.format(a, f))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
