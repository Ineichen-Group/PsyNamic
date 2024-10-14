import dash
from dash import html
from components.layout import header_layout, footer_layout


def home_layout():
    return html.Div([
        html.H1('Hello, World!'),
        html.P('Welcome to the About page.'),
    ],
        )
