
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import time


def time_graph(df: pd.DataFrame) -> html.Div:
    current_year = time.localtime().tm_year
    return html.Div([
        html.H1("Yearly Frequency Bar Plot"),
        
        # Input fields for start and end year
        html.Label("Start Year:"),
        dcc.Input(id='start-year', type='number', value=1955, min=df['Year'].min(), max=df['Year'].max()),
        
        html.Label("End Year:"),
        dcc.Input(id='end-year', type='number', value=current_year, min=df['Year'].min(), max=df['Year'].max()),
        
        # Bar plot
        dcc.Graph(id='bar-plot'),
    ])

