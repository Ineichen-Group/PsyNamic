from data.prepare_data import PREDICTIONS
import pandas as pd
from dash import html, dcc
from plotly import express as px
import dash_bootstrap_components as dbc


def substance_condition_graphs() -> html.Div:
    df_substance = get_substance_data()
    df_condition = get_condition_data()

    return html.Div([
        html.H1("Substances and Conditions in Studies", className="my-4"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id='substance-pie-graph',
                          figure=px.pie(df_substance, values='Frequency', names='Substance', title='Substances')),
                width=6
            ),
            dbc.Col(
                dcc.Graph(id='condition-bar-graph',
                          figure=px.bar(df_condition, x='Frequency', y='Condition', title='Conditions', orientation='h')),
                width=6
            )
        ])
    ], className="container")


def get_substance_data() -> pd.DataFrame:
    task = 'Substances'
    probability_threshold = 0.1
    df = pd.read_csv(PREDICTIONS)
    # filter for task
    df = df[df['task'] == task]
    # keep all rows with probability above threshold
    df = df[df['probability'] >= probability_threshold]
    # keep Substances and Frequency columns
    df = df[['label', 'id']]
    # group by label and count
    substance_df = df.groupby('label').count().reset_index().rename(
        columns={'id': 'Frequency', 'label': 'Substance'})
    return substance_df


def get_condition_data(substance_filter: str = None) -> pd.DataFrame:
    task = 'Condition'
    probability_threshold = 0.1
    df = pd.read_csv(PREDICTIONS)
    if substance_filter:
        # get substance data
        substance_df = df[df['task'] == 'Substances']
        # probability threshold
        substance_df = substance_df[substance_df['probability']
                                    >= probability_threshold]
        # filter for substances
        substance_df = substance_df[substance_df['label']
                                    == substance_filter]
        # get ids
        ids = substance_df['id'].tolist()
        # filter for ids
        df = df[df['id'].isin(ids)]

    # filter for task
    df = df[df['task'] == task]
    # keep all rows with probability above threshold
    df = df[df['probability'] >= probability_threshold]
    # keep Conditions and Frequency columns
    df = df[['label', 'id']]
    # group by label and count
    condition_df = df.groupby('label').count().reset_index().rename(
        columns={'id': 'Frequency', 'label': 'Condition'})
    # Order by frequency
    condition_df = condition_df.sort_values('Frequency', ascending=True)
    return condition_df
