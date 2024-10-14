# layouts.py
import dash
from dash import dcc, html
import pandas as pd

# callbacks.py
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

def register_callbacks(app, frequency_df):
    @app.callback(
        Output('bar-plot', 'figure'),
        Input('start-year', 'value'),
        Input('end-year', 'value')
    )
    def update_graph(start_year, end_year):
        # Filter data based on input years
        filtered_df = frequency_df[(frequency_df['Year'] >= start_year) & (frequency_df['Year'] <= end_year)]
        
        # Create the bar plot
        fig = px.bar(filtered_df, x='Year', y='Frequency', title='Frequency of IDs per Year', labels={'Frequency': 'Frequency'})
        
        return fig