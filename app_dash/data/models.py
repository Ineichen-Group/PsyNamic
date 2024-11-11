from sqlalchemy import create_engine, Column, Integer, String, Text, Float, Boolean, ForeignKey, TIMESTAMP
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.ext.declarative import declared_attr
from datetime import datetime

# Base class for all models
Base = declarative_base()


class Paper(Base):
    __tablename__ = 'paper'

    # Primary Key
    id = Column(Integer, primary_key=True)

    # Columns
    title = Column(String(255), nullable=False)
    abstract = Column(Text, nullable=False)
    prediction_input = Column(Text, nullable=False)  # Title + Abstract
    key_terms = Column(String(255), nullable=True)
    doi = Column(String(255), nullable=True)
    year = Column(Integer, nullable=True)
    authors = Column(String(255), nullable=True)
    link_to_fulltext = Column(String(255), nullable=True)
    link_to_pubmed = Column(String(255), nullable=True)

    # Foreign Key to BatchRetrival
    retrieval_id = Column(Integer, ForeignKey(
        'batch_retrival.id'), nullable=False)

    # Relationships
    batch_retrival = relationship('BatchRetrival', back_populates='papers')
    tokens = relationship('Token', back_populates='paper')
    predictions = relationship('Prediction', back_populates='paper')

    def __repr__(self):
        return f"<Paper(id={self.id}, title={self.title}, authors={self.authors})>"


class BatchRetrival(Base):
    __tablename__ = 'batch_retrival'

    # Primary Key
    id = Column(Integer, primary_key=True)

    # Columns
    date = Column(TIMESTAMP, default=datetime.utcnow)
    number_new_papers = Column(Integer, nullable=False)
    retrieval_time_needed = Column(TIMESTAMP, nullable=False)

    # Relationship to Paper (One-to-Many)
    papers = relationship('Paper', back_populates='batch_retrival')

    def __repr__(self):
        return f"<BatchRetrival(id={self.id}, date={self.date}, number_new_papers={self.number_new_papers})>"


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


# Create the database engine and session
# Replace with your PostgreSQL URL
DATABASE_URL = 'postgresql://username:password@localhost/mydatabase'

engine = create_engine(DATABASE_URL, echo=True)

# Create the tables in the database
Base.metadata.create_all(engine)
