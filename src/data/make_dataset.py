# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import re
import nltk

from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')


def get_wordnet_pos(tag):
    """Map POS tag to first character lemmatize() accepts"""
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag.upper(), wordnet.NOUN)


def preprocess(text):
    """
    Applies the following pre-processing steps to a given string:
    Step 1: Convert to lower case, replace slashes (/) with spaces
    Step 1: Tokenize the strings
    Step 3: Retain only nouns, verbs, adjectives and adverbs (using a POS-tagger)
    Step 4: Remove default English stopwords
    Step 5: Remove custom domain stopwords
    Step 6: Lemmatize each word
    """
    # Remove all the special characters
    text = re.sub(r'\W', ' ', str(text))

    # remove all single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    # Converting to Lowercase
    text = text.lower()

    # Tokenize
    tokens = nltk.word_tokenize(text.lower().replace('/', ' '))
    tokens = nltk.pos_tag(tokens)
    filtered_tokens = [t for t in tokens if t[1] in ["NN", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ", "RB"]]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in filtered_tokens if word[0] not in stop_words]

    # Lemmatize
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [wordnet_lemmatizer.lemmatize(w[0], get_wordnet_pos(w[1])) for w in filtered_tokens]
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Get the data/raw/git-refactoring-commits-raw.csv file
    dataset = pd.read_csv(input_filepath, usecols=['Message', 'CommitSHA', 'IsRefactoring'])
    logger.info('Loaded data file ' + input_filepath + ' with ' + str(len(dataset)) + ' rows')

    logger.info('Applying pre-processing steps on the "Message" column...')
    dataset['Message'] = dataset['Message'].apply(preprocess)

    # Save the processed subset on data data/processed/git-refactoring-commits.csv
    logger.info('Saved processed results on ' + output_filepath + ' with ' + str(len(dataset)) + ' rows')
    dataset.to_csv(output_filepath, encoding='utf-8', index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
