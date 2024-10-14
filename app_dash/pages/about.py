import dash
from dash import html
from components.layout import header_layout, footer_layout


def about_layout():
    return html.Div([
        html.H1('Hello, World!'),
        html.P('Welcome to the About page.'),
        # add image
        html.Img(src="assets/pipeline.png", style={'width': '80%'}),
    ],)
