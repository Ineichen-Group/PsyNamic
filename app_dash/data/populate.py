from datetime import datetime, timezone, timedelta
import os
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from models import Paper, BatchRetrieval, Token, Prediction, PredictionToken
import argparse
import pandas as pd
from settings import *


parent_folder_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_folder_path)


def create_batch_retrieval(nr_new_papers: int, retrieval_time_needed: datetime):
    return BatchRetrieval(
        date=datetime.now(timezone.utc),
        number_new_papers=nr_new_papers,
        retrieval_time_needed=retrieval_time_needed
    )


def create_paper(
        ID: int,
        title: str,
        abstract: str,
        prediction_input: str,
        key_terms: str,
        doi: str, year: int,
        authors: str,
        link_to_fulltext: str,
        link_to_pubmed: str,
        retrieval_id: int
):
    return Paper(
        id=ID,
        title=title,
        abstract=abstract,
        prediction_input=prediction_input,
        key_terms=key_terms,
        doi=doi,
        year=year,
        authors=authors,
        link_to_fulltext=link_to_fulltext,
        link_to_pubmed=link_to_pubmed,
        retrieval_id=retrieval_id
    )


def create_tokens(token_list: list[str], ner_list: list[str], paper_id) -> list[Token]:
    if len(token_list) != len(ner_list):
        raise ValueError("Token and NER lists must have the same length")
    tokens = []
    for i, (t, ner) in enumerate(zip(token_list, ner_list)):
        tokens.append(Token(
            paper_id=paper_id,
            text=t,
            ner_tag=ner,
            position_id=i
        ))
    return tokens


def create_predictions(paper_id: int, task: str, label: str, probability: float, model: str, is_multilabel: bool) -> list[Prediction]:
    prediction = Prediction(
        paper_id=paper_id,
        task=task,
        label=label,
        probability=probability,
        model=model,
        is_multilabel=is_multilabel)
    return prediction


def create_prediction_tokens(token_id, prediction_id, weight):
    return PredictionToken(
        token_id=token_id,
        prediction_id=prediction_id,
        weight=weight
    )


def populate_db(prediction_file: str, studies_file: str):

    # Using the settings.py file, create a connection to the database
    DATABASE_URL = f'postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{
        DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}'
    engine = create_engine(DATABASE_URL, echo=True)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    pred_data = pd.read_csv(prediction_file)
    studies_data = pd.read_csv(studies_file)
    nr_studies = len(studies_data)

    # duration of the retrieval, set to 0 for now, but datetime.timedelta
    retrieval_time_needed = timedelta(0)
    # Convert timedelta to datetime
    retrieval_time_needed_datetime = timedelta(0)
    batch = create_batch_retrieval(nr_studies, retrieval_time_needed_datetime)
    session.add(batch)
    session.commit()
    batch_id = batch.id

    # replace the NaN values with empty strings
    studies_data = studies_data.fillna('')

    # iterate through the studies data
    for i, row in studies_data.iterrows():
        abstract = row['abstract']
        title = row['title']
        prediction_input = title + '^\n' + abstract

        paper = create_paper(
            ID=row['id'],
            title=title,
            abstract=abstract,
            prediction_input=prediction_input,
            key_terms=row['keywords'],
            doi=row['doi'],
            year=row['year'],
            authors='',
            link_to_fulltext='',
            link_to_pubmed='',
            retrieval_id=batch_id
        )
        session.add(paper)
    session.commit()

    for i, row in pred_data.iterrows():
        pred = create_predictions(
            paper_id=row['id'],
            task=row['task'],
            label=row['label'],
            probability=row['probability'],
            model=row['model'],
            is_multilabel=row['is_multilabel']
        )
        session.add(pred)
    session.commit()

    session.close()


# args parser with the two files
def init_args_parser():
    """Initialize and return the argument parser for the script."""
    arg_parser = argparse.ArgumentParser(
        description='Populate the database with data')
    # add short and long arguments
    arg_parser.add_argument('-p', '--predictions_file', type=str,
                            help='Path to the predictions file', required=True)
    arg_parser.add_argument('-s', '--studies_file', type=str,
                            help='Path to the studies file', required=True)
    return arg_parser


if __name__ == '__main__':
    parser = init_args_parser()
    args = parser.parse_args()
    populate_db(args.predictions_file, args.studies_file)
