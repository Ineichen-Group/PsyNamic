from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import sys
import os
import pandas as pd

from .models import Base, Paper, BatchRetrieval, Token, Prediction, PredictionToken

# Add the parent folder to the Python search path
parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_folder_path)
from settings import *

# Set up the database connection
DATABASE_URL = f'postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}'
engine = create_engine(DATABASE_URL, echo=True)
Session = sessionmaker(bind=engine)

def extract_filters(input_dict: dict):
    filters = []
    buttons = input_dict['props']['children']
    # Traverse the structure to extract the necessary information
    for button_info in buttons:
        values = button_info['props']['value']
        filters.append(values)
    return filters

def get_studies(filters: list[dict[str, str]] = None) -> list[dict]:
    """
    Retrieves studies from the database based on the provided filters.

    Parameters:
    filters (list of dict): List of filters to apply to the study data.

    Returns:
    list of dict: A list of dictionaries containing the studies that match the filters.
    """
    session = Session()
    try:
        query = session.query(Paper)

        if filters:
            filters = extract_filters(filters)
            for f in filters:
                task = f['category']
                label = f['value']
                paper_ids = get_ids(task, label)
                query = query.filter(Paper.id.in_(paper_ids))

        studies = query.limit(20).all()
        result = []
        for study in studies:
            result.append({
                'id': study.id,
                'title': study.title,
                'abstract': study.abstract,
                'key_terms': study.key_terms,
                'doi': study.doi,
                'year': study.year,
                'authors': study.authors,
                'link_to_fulltext': study.link_to_fulltext,
                'link_to_pubmed': study.link_to_pubmed
            })
        return result
    finally:
        session.close()

def get_freq(task: str, filter_task: str = None, filter_task_label: str = None, threshold: float = 0.1) -> pd.DataFrame:
    """Get the prediction data for a given task and filter the data based on the filter task and label."""
    session = Session()
    try:
        subquery = session.query(Prediction.paper_id).filter(
            Prediction.task == filter_task,
            Prediction.label == filter_task_label,
            Prediction.probability >= threshold
        ).subquery()

        query = session.query(Prediction.label, func.count(Prediction.id).label('Frequency')).filter(
            Prediction.task == task,
            Prediction.probability >= threshold
        )

        if filter_task and filter_task_label:
            query = query.filter(Prediction.paper_id.in_(subquery))

        query = query.group_by(Prediction.label).order_by('Frequency')
        result = pd.read_sql(query.statement, session.bind)
        result.rename(columns={'label': task, 'Frequency': 'Frequency'}, inplace=True)
        return result
    finally:
        session.close()

def get_ids(task: str, label: str, threshold: float = 0.1) -> list[int]:
    """Get the ids of the papers that have a specific label for a given task."""
    session = Session()
    try:
        query = session.query(Prediction.paper_id).filter(
            Prediction.task == task,
            Prediction.label == label,
            Prediction.probability >= threshold
        )
        ids = [item.paper_id for item in query.all()]
        return ids
    finally:
        session.close()
