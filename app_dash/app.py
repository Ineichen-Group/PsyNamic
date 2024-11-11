import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

from pages.about import about_layout
from pages.contact import contact_layout
from pages.home import home_layout
from pages.views.dual_task import dual_task_graphs
from pages.views.time import time_graph, get_time_data

from components.layout import header_layout, footer_layout, search_filter_component, studies_display
from callbacks import register_callbacks

# Load data
frequency_df = get_time_data()

# Initialize the Dash app with suppress_callback_exceptions=True
app = dash.Dash(__name__, external_stylesheets=[
                dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True)

app.layout = html.Div([
    header_layout(),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', className='mx-5 my-2'),
    # search_filter_component(),
    # studies_display(),
    footer_layout()
])


@app.callback(dash.Output('page-content', 'children'),
              [dash.Input('url', 'pathname')])
def display_page(pathname: str):
    if pathname == '/about':
        return about_layout()
    elif pathname == '/contact':
        return contact_layout()
    elif pathname == '/view/time':
        return time_graph()
    elif pathname == '/view/sub_cond':
        return dual_task_graphs()
    else:
        return home_layout()


# Register all callbacks and pass data
register_callbacks(app, {'frequency_df': frequency_df})

if __name__ == '__main__':
    app.run_server(debug=True)
