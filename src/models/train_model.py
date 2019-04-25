# -*- coding: utf-8 -*-
import click
import logging
import math
import pandas as pd
import copy
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

nltk.download('stopwords')


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):
    """
    Receives the location of the tf-idf scores as a
    command-line Path argument.
    """
    logger = logging.getLogger(__name__)
    logger.info('Training the binary classification algorithm based on the TF-IDF scores')

    # Get the models/tf-idf-scores.csv file
    dataset = pd.read_csv(input_filepath)
    logger.info('Loaded data file ' + input_filepath + ' with ' + str(len(dataset)) + ' rows')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
