from dash import html


def home_layout():
    return html.Div([
        html.H1('Hello, World!'),
        html.P('Welcome to the About page.'),
    ],
        )
