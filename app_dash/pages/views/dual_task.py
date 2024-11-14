from data.prepare_data import PREDICTIONS
import pandas as pd
from dash import html, dcc
from plotly import express as px
import dash_bootstrap_components as dbc
from components.graphs import dropdown
from data.queries import get_freq


def dual_task_graphs():
    # Default values
    default_task1 = 'Substances'
    default_task2 = 'Condition'
    # Define the identifiers
    dropdown1_id = "jux_dropdown1"
    dropdown2_id = "jux_dropdown2"
    message_id = "validation-message"

    df_task1 = get_freq(default_task1)
    df_task2 = get_freq(default_task2)

    my_div = html.Div([
        html.H1("Dual Task Analysis", className="my-4"),
        html.P("In this view, you can select two classifications tasks from the dropdown menus — one for each chart. The left chart (Task 1) displays data in a pie chart, and the right chart (Task 2) shows the data in a bar chart. If you click on a segment in the pie chart, the bar chart will update to show related information, allowing you to filter Task 2 based on your selection in Task 1."),
        html.Div(id=message_id, className="mt-4 text-danger"),
        dbc.Row([
            dbc.Col([
                dropdown(['Substances', 'Condition'], identifier=dropdown1_id,
                         default=default_task1, label="Select a Task", width="50%"),
                dcc.Graph(id='task1-pie-graph',
                          figure=px.pie(df_task1, values='Frequency', names=default_task1, title=f'Task 1: {default_task1}')),

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


