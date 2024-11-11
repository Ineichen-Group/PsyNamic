from sqlalchemy.orm import sessionmaker
from datetime import datetime
import random

# Import the classes from your models
from models import Paper, BatchRetrival, Token, Prediction, PredictionToken, engine

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Function to create dummy batch retrieval
def create_batch_retrieval():
    return BatchRetrival(
        date=datetime.utcnow(),
        number_new_papers=random.randint(5, 20),  # Random number of papers
        retrieval_time_needed=datetime.utcnow()
    )

# Function to create dummy paper
def create_paper(batch_retrival_id):
    return Paper(
        title=f"Sample Paper {random.randint(1, 100)}",
        abstract="This is a dummy abstract of the paper.",
        prediction_input="Sample Title + Abstract",
        key_terms="NLP, Machine Learning",
        doi=f"10.1234/doi_{random.randint(1, 10000)}",
        year=random.randint(2000, 2024),
        authors="John Doe, Jane Smith",
        link_to_fulltext=None,
        link_to_pubmed=None,
        retrieval_id=batch_retrival_id
    )

# Function to create dummy tokens
def create_tokens(paper_id):
    tokens = []
    for i in range(random.randint(5, 15)):  # Random number of tokens
        tokens.append(Token(
            paper_id=paper_id,
            text=f"Token_{i}_{random.randint(1, 100)}",
            ner_tag=None,  # You can set NER tags as needed
            position_id=i
        ))
    return tokens

# Function to create dummy predictions
def create_predictions(paper_id):
    predictions = []
    for i in range(random.randint(1, 3)):  # Random number of predictions per paper
        predictions.append(Prediction(
            paper_id=paper_id,
            task=f"Task_{random.randint(1, 5)}",  # Random task name
            label=f"Label_{random.randint(1, 5)}",  # Random label
            probability=random.random(),  # Random probability
            model=f"Model_{random.randint(1, 3)}",  # Random model name
            is_multilabel=random.choice([True, False])  # Random multilabel status
        ))
    return predictions

# Function to create dummy prediction tokens
def create_prediction_tokens(token_id, prediction_id):
    return PredictionToken(
        token_id=token_id,
        prediction_id=prediction_id,
        weight=random.random()  # Random weight
    )

# Insert dummy data into the database
def insert_dummy_data():
    # Create a batch retrieval record
    batch = create_batch_retrival()
    session.add(batch)
    session.commit()  # Commit the batch to get its ID

    # Create dummy papers linked to the batch retrieval
    for _ in range(batch.number_new_papers):
        paper = create_paper(batch.id)
        session.add(paper)
        session.commit()  # Commit to get the paper's ID

        # Create tokens for the paper
        tokens = create_tokens(paper.id)
        session.add_all(tokens)
        session.commit()  # Commit tokens to get their IDs

        # Create predictions for the paper
        predictions = create_predictions(paper.id)
        session.add_all(predictions)
        session.commit()  # Commit predictions to get their IDs

        # Create prediction tokens for each token and prediction
        for prediction in predictions:
            for token in tokens:
                prediction_token = create_prediction_tokens(token.id, prediction.id)
                session.add(prediction_token)
        session.commit()  # Commit prediction tokens

    print("Dummy data inserted successfully!")

# Run the insert dummy data function
insert_dummy_data()

# Close the session
session.close()
