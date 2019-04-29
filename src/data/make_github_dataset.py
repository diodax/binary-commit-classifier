# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import re
import nltk
import json
import base64
import os

from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError, HTTPError
from collections import Counter

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

USERNAME = ''
PASSWORD = ''


def apply_commit_info(row):
    owner = row['App'].split('$')[0]
    repo = row['App'].split('$')[1]
    commit_sha = row['CommitSHA']
    data = get_commit_info(owner, repo, commit_sha)

    if data is None:
        return "0,0,0,0,0"

    line_additions = data['stats']['additions']
    line_deletions = data['stats']['deletions']
    files = data['files']

    files_added_list = [f for f in files if f['status'] == 'added']
    files_deleted_list = [f for f in files if f['status'] == 'removed']
    files_modified_list = [f for f in files if f['status'] == 'modified']

    return "{0},{1},{2},{3},{4}".format(
        line_additions, line_deletions, len(files_added_list), len(files_deleted_list), len(files_modified_list))


def get_commit_info(owner, repo, commit_sha):
    url = 'https://api.github.com/repos/{owner}/{repo}/commits/{sha}'.format(owner=owner, repo=repo, sha=commit_sha)

    request = Request(url)
    if USERNAME != '' and PASSWORD != '':
        base64string = base64.b64encode((str(USERNAME) + ':' + str(PASSWORD)).encode())
        request.add_header("Authorization", "Basic " + str(base64string))
    print('Making request to ' + str(url) + ' ...')

    try:
        response = urlopen(request).read()
    except HTTPError as e:
        print('Error code: ', e.code)
        return None

    data = json.loads(response.decode())
    return data


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

    if os.path.exists(output_filepath):
        dataset = pd.read_csv(output_filepath,
                              usecols=['App', 'Message', 'CommitSHA', 'IsRefactoring',
                                       'LineAdditions', 'LineDeletions', 'FilesAdded',
                                       'FilesDeleted', 'FilesModified'])
        logger.info('Loaded data file ' + output_filepath + ' with ' + str(len(dataset)) + ' rows')
        empty_df = dataset[(dataset.LineAdditions == 0) & (dataset.LineDeletions == 0) &
                           (dataset.FilesAdded == 0) & (dataset.FilesDeleted == 0) & (dataset.FilesModified == 0)]
        dataset = dataset[(dataset.LineAdditions != 0) | (dataset.LineDeletions != 0) |
                          (dataset.FilesAdded != 0) | (dataset.FilesDeleted != 0) | (dataset.FilesModified != 0)]

        logger.info('Getting metrics from the Github API...')
        empty_df['Metrics'] = empty_df.apply(lambda row: apply_commit_info(row), axis=1)
        metrics = empty_df['Metrics'].str.split(",", expand=True)
        empty_df['LineAdditions'] = metrics[0]
        empty_df['LineDeletions'] = metrics[1]
        empty_df['FilesAdded'] = metrics[2]
        empty_df['FilesDeleted'] = metrics[3]
        empty_df['FilesModified'] = metrics[4]

        # Dropping old Metrics columns
        empty_df.drop(columns=["Metrics"], inplace=True)

        dataset = pd.concat([empty_df, dataset])
    else:
        # Get the data/raw/git-refactoring-commits-raw.csv file
        dataset = pd.read_csv(input_filepath, usecols=['App', 'Message', 'CommitSHA', 'IsRefactoring'])
        logger.info('Loaded data file ' + input_filepath + ' with ' + str(len(dataset)) + ' rows')
        dataset = dataset.sample(n=1000)

        # logger.info('Applying pre-processing steps on the "Message" column...')
        # dataset['Message'] = dataset['Message'].apply(preprocess)

        logger.info('Getting metrics from the Github API...')
        dataset['Metrics'] = dataset.apply(lambda row: apply_commit_info(row), axis=1)
        metrics = dataset['Metrics'].str.split(",", expand=True)
        dataset['LineAdditions'] = metrics[0]
        dataset['LineDeletions'] = metrics[1]
        dataset['FilesAdded'] = metrics[2]
        dataset['FilesDeleted'] = metrics[3]
        dataset['FilesModified'] = metrics[4]

        # Dropping old Metrics columns
        dataset.drop(columns=["Metrics"], inplace=True)

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
