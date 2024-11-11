from data.prepare_data import PREDICTIONS
import pandas as pd
from dash import html, dcc
from plotly import express as px
import dash_bootstrap_components as dbc
from components.graphs import dropdown


def dual_task_graphs():
    # Default values
    default_task1 = 'Substances'
    default_task2 = 'Condition'
    # Define the identifiers
    dropdown1_id = "jux_dropdown1"
    dropdown2_id = "jux_dropdown2"
    message_id = "validation-message"

    df_task1 = get_prediction_data(default_task1)
    df_task2 = get_prediction_data(default_task2)

    my_div = html.Div([
        html.H1("Inspect two Classification Tasks", className="my-4"),
        html.Div(id=message_id, className="mt-4 text-danger"),
        dbc.Row([
            dbc.Col([
                dropdown(['Substances', 'Condition'], identifier=dropdown1_id,
                         default=default_task1, label="Select a Task", width="50%"),
                dcc.Graph(id='task1-pie-graph',
                          figure=px.pie(df_task1, values='Frequency', names=default_task1, title=f'Task 1: {default_task1}'))
            ], width=6),
            dbc.Col([
                dropdown(['Substances', 'Condition'], identifier=dropdown2_id,
                         default=default_task2, label="Select a Task", width="50%"),
                dcc.Graph(id='task2-bar-graph',
                          figure=px.bar(df_task2, x='Frequency', y=default_task2, title=f'Task 2: {default_task2}', orientation='h'))
            ], width=6)
        ]),
    ], className="container")

    return my_div


def get_prediction_data(task: str, filter_task: str = None, filter_task_label: str = None, threshold: float = 0.1) -> pd.DataFrame:
    """Get the prediction data for a given task and filter the data based on the filter task and label.
    """
    df = pd.read_csv(PREDICTIONS)

    # E.g. filter_task = 'Substances', filter_task_label = 'LSD' --> filter all predictions for 'Cocaine' in the 'Substances' task
    if filter_task and filter_task_label:
        df_filter_task = df[df['task'] == filter_task]
        df_filter_task = df_filter_task[df_filter_task['probability']
                                        >= threshold]
        df_filter_task = df_filter_task[df_filter_task['label']
                                        == filter_task_label]
        ids = df_filter_task['id'].tolist()
        # Keep only the papers that have the label
        df = df[df['id'].isin(ids)]

    # Filter the data based on the task
    df = df[df['task'] == task]
    df = df[df['probability'] >= threshold]
    df = df[['label', 'id']]
    task_df = df.groupby('label').count().reset_index().rename(
        columns={'id': 'Frequency', 'label': task})
    task_df = task_df.sort_values('Frequency', ascending=True)
    return task_df
