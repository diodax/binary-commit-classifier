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

    model_names = ['Gaussian Naive-Bayes', 'Logistic Regression', 'Random Forest', 'Decision Tree', 'SVM', 'kNN']
    accuracy_results = []
    f1_results = []

    nb_a, nb_f = classifier.apply_gaussian_nb()
    accuracy_results.append(nb_a)
    f1_results.append(nb_f)

    lg_a, lg_f = classifier.apply_logistic_regression()
    accuracy_results.append(lg_a)
    f1_results.append(lg_f)

    rf_a, rf_f = classifier.apply_random_forest()
    accuracy_results.append(rf_a)
    f1_results.append(rf_f)

    dt_a, dt_f = classifier.apply_decision_tree()
    accuracy_results.append(dt_a)
    f1_results.append(dt_f)

    svm_a, svm_f = classifier.apply_svm()
    accuracy_results.append(svm_a)
    f1_results.append(svm_f)

    knn_a, knn_f = classifier.apply_knn()
    accuracy_results.append(knn_a)
    f1_results.append(knn_f)

    # Model comparison
    models = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracy_results,
        'F1': f1_results
    })

    # Print table and sort by test precision
    models.sort_values(by='Accuracy', ascending=False)
    # Save the scored results on reports/model-scores.csv
    logger.info('Saved scored results on ' + output_filepath)
    models.to_csv(output_filepath, encoding='utf-8', index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
