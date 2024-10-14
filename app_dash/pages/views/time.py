import dash
from dash import dcc, html
import pandas as pd
import time
from data.prepare_data import STUDIES

def time_graph() -> html.Div:
    df = get_time_data()
    current_year = time.localtime().tm_year
    return html.Div([
        html.H1("Number of publications over time", className="my-4"),
        
        # Input fields for start and end year in a single row
        html.Div([
            html.Div([
                html.Label("Start Year:", className="form-label pe-4"),
                dcc.Input(id='start-year', type='number', value=1955, min=df['Year'].min(), max=df['Year'].max(), className="form-control year-input"),
            ], className="col-md-3"),
                    
            html.Div([
                html.Label("End Year:", className="form-label pe-4"),
                dcc.Input(id='end-year', type='number', value=current_year, min=df['Year'].min(), max=df['Year'].max(), className="form-control year-input"),
            ], className="col-md-3"),
        ], className="row g-3 mb-3"),
        
        dcc.Graph(id='time-plot'),
    ], className="container")

def get_time_data() -> pd.DataFrame:
    df = pd.read_csv(STUDIES)
    # use year and id columns
    df = df[['id', 'year']]
    # count IDs per year, rename columns to Year and Frequency
    frequency_df = df.groupby('year').count().reset_index().rename(columns={'id': 'Frequency', 'year': 'Year'})
    return frequency_df
