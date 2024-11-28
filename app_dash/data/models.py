from sqlalchemy import create_engine, Column, Integer, String, Text, Float, Boolean, ForeignKey, TIMESTAMP, Interval
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
import sys
import os

# Add the parent folder to the Python search path
parent_folder_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_folder_path)
from settings import *

# Base class for all models
Base = declarative_base()


class Paper(Base):
    __tablename__ = 'paper'
    # Primary Key
    id = Column(Integer, primary_key=True)
    # Columns
    title = Column(Text, nullable=False)
    abstract = Column(Text, nullable=False)
    prediction_input = Column(Text, nullable=False)  # Title + Abstract
    key_terms = Column(Text, nullable=True)
    doi = Column(String(100), nullable=True)
    year = Column(Integer, nullable=True)
    authors = Column(String(255), nullable=True)
    link_to_fulltext = Column(String(255), nullable=True)
    link_to_pubmed = Column(String(255), nullable=True)

    retrieval_id = Column(Integer, ForeignKey(
        'batch_retrieval.id'), nullable=False)

    # Relationship to BatchRetrieval (Many-to-One)
    batch_retrieval = relationship('BatchRetrieval', back_populates='papers')
    # Relationship to Token (One-to-Many)
    tokens = relationship('Token', back_populates='paper')
    # Relationship to Prediction (One-to-Many)
    predictions = relationship('Prediction', back_populates='paper')

    def __repr__(self):
        return f"<Paper(id={self.id}, title={self.title}, authors={self.authors})>"


class BatchRetrieval(Base):
    __tablename__ = 'batch_retrieval'

    # Primary Key
    id = Column(Integer, primary_key=True)

    # Columns
    date = Column(TIMESTAMP, default=datetime.utcnow)
    number_new_papers = Column(Integer, nullable=False)
    retrieval_time_needed = Column(Interval, nullable=False)

    # Relationship to Paper (One-to-Many)
    papers = relationship('Paper', back_populates='batch_retrieval')

    def __repr__(self):
        return f"<BatchRetrieval(id={self.id}, date={self.date}, number_new_papers={self.number_new_papers})>"


class Token(Base):
    __tablename__ = 'token'

    # Primary Key
    id = Column(Integer, primary_key=True)

    # Foreign Key to Paper
    paper_id = Column(Integer, ForeignKey('paper.id'), nullable=False)

    # Columns
    text = Column(String(255), nullable=False)
    ner_tag = Column(String(255), nullable=True)  # Can be null
    position_id = Column(Integer, nullable=False)

    # Relationship to Paper (Many-to-One)
    paper = relationship('Paper', back_populates='tokens')

    # Relationship to Prediction_Token (One-to-Many)
    prediction_tokens = relationship('PredictionToken', back_populates='token')

    def __repr__(self):
        return f"<Token(id={self.id}, text={self.text[:50]}, position_id={self.position_id})>"


class Prediction(Base):
    __tablename__ = 'prediction'

    # Primary Key
    id = Column(Integer, primary_key=True)

    # Foreign Key to Paper
    paper_id = Column(Integer, ForeignKey('paper.id'), nullable=False)

    # Columns
    task = Column(String(255), nullable=False)
    label = Column(String(255), nullable=False)
    probability = Column(Float, nullable=False)
    model = Column(String(255), nullable=False)
    is_multilabel = Column(Boolean, default=False)

    # Relationship to Paper (Many-to-One)
    paper = relationship('Paper', back_populates='predictions')

    # Relationship to Prediction_Token (One-to-Many)
    prediction_tokens = relationship(
        'PredictionToken', back_populates='prediction')

    def __repr__(self):
        return f"<Prediction(id={self.id}, task={self.task}, label={self.label}, probability={self.probability})>"


class PredictionToken(Base):
    __tablename__ = 'prediction_token'

    # Primary Key
    id = Column(Integer, primary_key=True)

    # Foreign Keys
    token_id = Column(Integer, ForeignKey('token.id'), nullable=False)
    prediction_id = Column(Integer, ForeignKey(
        'prediction.id'), nullable=False)

    # Columns
    weight = Column(Float, nullable=False)

    # Relationship to Token (Many-to-One)
    token = relationship('Token', back_populates='prediction_tokens')

    # Relationship to Prediction (Many-to-One)
    prediction = relationship('Prediction', back_populates='prediction_tokens')

    def __repr__(self):
        return f"<PredictionToken(id={self.id}, weight={self.weight})>"


def init_db():
    # Names from the settings are used
    DATABASE_URL = f'postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{
        DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}'

    engine = create_engine(DATABASE_URL, echo=True)

    Base.metadata.create_all(engine)


if __name__ == '__main__':
    init_db()
