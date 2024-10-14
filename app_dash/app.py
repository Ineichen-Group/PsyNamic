import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from pages.about import about_layout
from pages.contact import contact_layout
from pages.home import home_layout
from components.layout import header_layout, footer_layout
from pages.views.time import time_graph
from callbacks import register_callbacks
import pandas as pd

STUDIES = '/home/vera/Documents/Arbeit/CRS/PsychNER/app_dash/data/studies.csv'
PREDICTION = '/home/vera/Documents/Arbeit/CRS/PsychNER/app_dash/data/prediction.csv'

# from callbacks.callbacks import register_callbacks

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    header_layout(),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', className='mx-5 my-2'),
    footer_layout()
],
    )


df = pd.read_csv(STUDIES)
# use year and id columns
df = df[['id', 'year']]
# count IDs per year, rename columns to Year and Frequency
frequency_df = df.groupby('year').count().reset_index().rename(columns={'id': 'Frequency', 'year': 'Year'})
print(frequency_df.head())
@app.callback(dash.Output('page-content', 'children'),
              [dash.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/about':
        return about_layout()
    elif pathname == '/contact':
        return contact_layout()
    elif pathname == '/view/time':
        return time_graph(frequency_df)
    else:
        return home_layout()

# Register all callbacks
register_callbacks(app, frequency_df)


if __name__ == '__main__':
    app.run_server(debug=True)
